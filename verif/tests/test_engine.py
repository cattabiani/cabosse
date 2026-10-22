# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/engine.sv running MATVEC, SCORES and VALUES (PLAN.md, M5) against
golden.dot.dot and the golden attention, bit for bit, through
verif/bulk/engine.cpp: one memory model per port, the x buffers as a RAM,
every write checked as it comes (the oldest command's dst, its rows only,
each row once). Configurations of 4, 8 and 20 ports (16, 32 and 80
lanes; 80 divides no SmolLM2 size), and the 128-lane engine (32 ports,
marked wide).

Shapes: rows around the lanes (fewer than a port's group, a partial group
and pass, a single row) and columns around a beat and the lane's full-rate
row (a masked last beat), all as back-to-back commands with FP32 and BF16
destinations, two x buffers and matrices across 4 KB boundaries. Then the
same under the memory's latency, limits, bandwidth and pauses, one slow
port and sparse commands; consecutive commands leave no gap; commands that
wait for their go (D-042) read their weights ahead and never step early;
the ports' FIFO depth against pauses; an error response reported on its
port. Attention: heads, grouping, head dimensions and positions around a
tile, with stale NaN in the cache and in p past t, between MATVECs; the
same under the memory behaviours and with gos; back-to-back units at the
perf model's cycles. Slow, at 128 lanes: SmolLM2's MATVECs with real data,
cycles against the perf model, and the bandwidth reached on large
matrices."""

import dataclasses

import bulk
import commands
import engine as eng
import numpy as np
import paths
import pytest
import torch
from engine import LANES_PER_PORT, Attention, Matvec
from golden import decoder
from lane import FULL_RATE_LEN

import rtl

SEED = 20261011
PORTS = (4, 8, 20)  # the smallest, plain, and a factor 5 (owner, 2026-10-15)
WIDE = 32  # 128 lanes, SmolLM2's engine
ALL_PORTS = (*PORTS, pytest.param(WIDE, marks=pytest.mark.wide))
# A command's last result after its start, past its engine cycles and the
# memory's latency: 62 at 128 lanes (measured; 6 of them the scale's
# multiply on the write path), with margin.
STARTUP_CYCLES = 70
# Around a beat and the lane's shortest full-rate row.
COLS = (1, 3, 4, 5, FULL_RATE_LEN - 1, FULL_RATE_LEN, FULL_RATE_LEN + 1, 64, 130)

# (query heads, KV heads): groups of 2, one group, plain, and SmolLM2's 3
# (its 2-head p.V pass (2, 3) moves to the next KV head inside a group).
HEADS = ((4, 2), (3, 1), (5, 5), (9, 3))
SCALES = (0.125, 0.3)  # SmolLM2's, and one whose products round
# Head dimensions: the smallest and the largest the engine takes, and
# SmolLM2's (2 heads per p.V pass at 128 lanes); p.V only those up to L.
HEAD_DIMS = (16, 64, 128)


def rows_around(lanes: int) -> list[int]:
    lpp = LANES_PER_PORT
    return sorted({1, lpp - 1, lpp, lpp + 1, lanes - 1, lanes, lanes + 1, 2 * lanes + 3})


def random_bf16(gen: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=gen).to(torch.bfloat16)


def shape_batch(ports: int, seed: int) -> tuple[list[Matvec], dict[int, torch.Tensor], list[int]]:
    """Every rows × cols pair, as commands alternating x buffer, dst and
    format; each matrix after a random gap of 0 to 200 beats."""
    gen = torch.Generator().manual_seed(seed)
    lanes = ports * LANES_PER_PORT
    xs = {0: random_bf16(gen, max(COLS)), 1: random_bf16(gen, max(COLS))}
    mats = [
        Matvec(random_bf16(gen, r, c), a=i % 2, dst=2 + i % 3, bf16=i % 2 == 1)
        for i, (r, c) in enumerate((r, c) for r in rows_around(lanes) for c in COLS)
    ]
    rng = np.random.default_rng(seed)
    gaps = rng.integers(0, 200, len(mats)).tolist()
    return mats, xs, eng.place(mats, ports, gaps)


def describe(c: eng.Command) -> str:
    if isinstance(c, Matvec):
        return f"MATVEC {tuple(c.w.shape)}, bf16={c.bf16}"
    return f"{'VALUES' if c.values else 'SCORES'} {c.heads} heads, t={c.t}, bf16={c.bf16}"


def check_bits(
    mats: list[eng.Command], xs: dict[int, torch.Tensor], result: eng.Result, label: str
) -> None:
    for i, (m, got) in enumerate(zip(mats, result.out, strict=True)):
        eng.assert_bits(got, eng.expected(m, xs[m.a]), f"{label}: command {i} ({describe(m)})")
    assert result.err == 0, f"{label}: err_o {result.err:#x}"
    assert result.r_stalls == 0, f"{label}: a memory waited {result.r_stalls} cycles on RREADY"


@rtl.needs_verilator
@pytest.mark.parametrize("ports", ALL_PORTS)
def test_shapes(ports: int) -> None:
    mats, xs, addrs = shape_batch(ports, SEED + ports)
    check_bits(mats, xs, eng.fetch(mats, xs, ports, addrs, latency=20), f"seed {SEED + ports}")


MEMORIES = {
    "latency 1": dict(latency=1),
    "latency 80, 2 bursts": dict(latency=80, outstanding=2),
    "rate 30%": dict(rate=300),
    "pauses": dict(latency=20, pause=50, pause_max=40),
    "slow port": dict(latency=20, slow_port=0, slow_rate=150),
    "sparse commands": dict(latency=20, offer=20),
}


@rtl.needs_verilator
@pytest.mark.parametrize("memory", MEMORIES)
@pytest.mark.parametrize("ports", ALL_PORTS)
def test_memories(ports: int, memory: str) -> None:
    seed = SEED + 10 * ports + list(MEMORIES).index(memory)
    mats, xs, addrs = shape_batch(ports, seed)
    mats, addrs = mats[::4], addrs[::4]  # a quarter of the shapes, still every row count
    plusargs = MEMORIES[memory] | {"seed": seed}
    check_bits(mats, xs, eng.fetch(mats, xs, ports, addrs, **plusargs), f"{memory}, seed {seed}")


STALE = (float("inf"), float("-inf"), float("nan"), -0.0)  # what an x buffer may hold past m


@rtl.needs_verilator
@pytest.mark.parametrize("ports", ALL_PORTS)
def test_stale_x(ports: int) -> None:
    """The x buffer holds infinities and NaN past the matrix's columns: the
    last beat's missing elements are masked, not multiplied by the zero
    weights the layout pads with (0 · inf is NaN)."""
    gen = torch.Generator().manual_seed(SEED)
    lanes = ports * LANES_PER_PORT
    cols = [c for c in COLS if c % commands.E]
    stale = torch.tensor(STALE, dtype=torch.bfloat16)
    xs = {i: torch.cat([random_bf16(gen, c), stale]) for i, c in enumerate(cols)}
    mats = [Matvec(random_bf16(gen, lanes + 1, c), a=i) for i, c in enumerate(cols)]
    check_bits(mats, xs, eng.fetch(mats, xs, ports, latency=5), f"stale x, seed {SEED}")


# (passes, cols): rows of full-rate length. A command's last result follows
# its last beat by 3 cycles more when the tree's adder is busy (64 or 60
# columns, after another command) than when it is idle (128 columns, or
# the first command; measured): the run ends as it starts, idle.
BACK_TO_BACK = ((1, 64), (3, 128), (2, FULL_RATE_LEN)) * 4 + ((3, 128),)


@rtl.needs_verilator
@pytest.mark.parametrize("ports", ALL_PORTS)
def test_back_to_back(ports: int) -> None:
    """With the memory at full rate, the commands after the first take
    exactly their perf-model cycles over the whole run: no gap between
    commands, the next matrix's reads hidden behind the current one."""
    gen = torch.Generator().manual_seed(SEED)
    lanes = ports * LANES_PER_PORT
    mats = [Matvec(random_bf16(gen, p * lanes, c)) for p, c in BACK_TO_BACK]
    xs = {0: random_bf16(gen, 128)}
    result = eng.fetch(mats, xs, ports, latency=40, outstanding=8)
    label = f"back to back, seed {SEED}"
    check_bits(mats, xs, result, label)
    model = [eng.model_cycles(m, ports) for m in mats]
    assert result.done[0] <= model[0] + 40 + STARTUP_CYCLES, (label, result.done[0], model[0])
    extra = int(result.done[-1] - result.done[0]) - sum(model[1:])
    assert extra == 0, (label, extra, np.diff(result.done).tolist(), model)


# (wait, passes, cols): runs of waiting and independent commands, long and
# short, more than the engine's queues hold.
GO_COMMANDS = (
    (True, 2, 64), (False, 1, 130), (True, 3, 64), (True, 1, 5),
    (False, 2, 64), (False, 1, 3), (True, 2, 130), (True, 1, 64),
)  # fmt: skip
GO_LATENCY = 120  # longer than a waiting command's start-up (design v0's engine_startup_cycles)
# (permille of gos that come as soon as their command is taken, the most
# cycles a go comes late): on time; late, so the FIFOs fill and wait; early,
# so credits wait while earlier commands step; both.
GOS = ((0, 0), (0, 300), (1000, 0), (500, 40))


@rtl.needs_verilator
@pytest.mark.parametrize(("go_early", "go_delay"), GOS)
@pytest.mark.parametrize("ports", ALL_PORTS)
def test_go(ports: int, go_early: int, go_delay: int) -> None:
    """A waiting command (D-042) gets its go, in order, once every earlier
    command is done or (go_early) once it is taken, up to go_delay cycles
    late; its x reads as NaN until then, so a step before the go writes
    wrong bits. Its weights are read ahead: with the go on time, it takes
    fewer cycles over its engine cycles than the memory's latency (a read
    after the go would add the latency and the pipeline). The memory never
    waits on the FIFOs."""
    seed = SEED + 100 * ports + go_delay + go_early
    gen = torch.Generator().manual_seed(seed)
    lanes = ports * LANES_PER_PORT
    mats = [
        Matvec(random_bf16(gen, p * lanes - i % 2, c), a=i, dst=2 + i % 3, bf16=i % 3 == 1, wait=w)
        for i, (w, p, c) in enumerate(GO_COMMANDS)
    ]
    xs = {i: random_bf16(gen, c) for i, (_, _, c) in enumerate(GO_COMMANDS)}
    gos = dict(go_early=go_early, go_delay=go_delay)
    result = eng.fetch(mats, xs, ports, latency=GO_LATENCY, seed=seed, **gos)
    label = f"{gos}, seed {seed}"
    check_bits(mats, xs, result, label)
    if (go_early, go_delay) == (0, 0):
        extra = [
            c - eng.model_cycles(m, ports, startup=False)
            for m, c in zip(mats, eng.command_cycles(result), strict=True)
        ]
        late = [(i, e) for i, e in enumerate(extra) if i > 0 and mats[i].wait and e >= GO_LATENCY]
        assert not late, (label, late, extra)


DEPTHS = (16, 64, 128, 1024)  # 1024: deep enough that the pauses alone limit the rate
DEFAULT_DEPTH = 128  # rtl/engine.sv's Depth: its run reuses the plain 4-port build
PAUSES = dict(latency=20, outstanding=8, pause=20, pause_max=40)  # ports paused ~29% of cycles
# Share of the 1024-beat FIFO's rate, at least (or at most for 16). Measured
# on these shapes and pauses, memory seeds 1 to 5 (each port its own stream):
# 0.357-0.375 (16), 0.884-0.933 (64), 0.941-0.993 (128), at 4 ports.
MIN_SHARE = {64: 0.80, 128: 0.90}
MAX_SHARE_16 = 0.5


@rtl.needs_verilator
def test_fifo_depth() -> None:
    """Four ports pausing at random, independently, in lockstep: each
    port's FIFO must cover both the memory's latency and the other ports'
    pauses. Rates against a FIFO deep enough not to matter."""
    ports, gen = 4, torch.Generator().manual_seed(SEED)
    mats = [Matvec(random_bf16(gen, 8 * ports * LANES_PER_PORT, 512)) for _ in range(4)]
    xs = {0: random_bf16(gen, 512)}
    beats = sum(eng.n_beats(m, ports) for m in mats)
    rate = {}
    for depth in DEPTHS:
        params = () if depth == DEFAULT_DEPTH else (("Depth", depth),)
        result = eng.fetch(mats, xs, ports, params=params, seed=SEED, **PAUSES)
        check_bits(mats, xs, result, f"depth {depth}, seed {SEED}")
        rate[depth] = beats / (result.done[-1] - result.taken[0])
    share = {d: r / rate[1024] for d, r in rate.items()}
    print(f"share of the deep FIFO's rate by depth: {share}")
    assert all(share[d] >= v for d, v in MIN_SHARE.items()), (SEED, share)
    assert share[16] <= MAX_SHARE_16, (SEED, share)


@rtl.needs_verilator
@pytest.mark.parametrize("bad_port", (0, 1))
def test_error_response(bad_port: int) -> None:
    gen = torch.Generator().manual_seed(SEED)
    mats = [Matvec(random_bf16(gen, 16, 64))]
    xs = {0: random_bf16(gen, 64)}
    result = eng.fetch(mats, xs, 4, bad_port=bad_port, bad=3)
    assert result.err == 1 << bad_port, (SEED, result.err)


@pytest.mark.slow
@pytest.mark.wide
@rtl.needs_verilator
@pytest.mark.skipif(not paths.SMOLLM2.exists(), reason=f"needs the checkpoint in {paths.SMOLLM2}")
def test_smollm2() -> None:
    """The first and last layers' MATVECs and the classifier, real weights
    and activations (position 63), waiting where the controller would, at
    F2's latency: bit for bit, and each command within 10% of the perf
    model's cycles (PLAN.md, M5), the first one included."""
    found, result = eng.run_smollm2(decoder.load(paths.SMOLLM2), WIDE, **eng.F2_MEMORY)
    assert result.err == 0 and result.r_stalls == 0, (result.err, result.r_stalls)
    off = {
        f.name: c / eng.model_cycles(f.matvec, WIDE) - 1
        for f, c in zip(found, eng.command_cycles(result), strict=True)
    }
    print(f"cycles against the perf model: {off}")
    assert all(abs(v) <= 0.10 for v in off.values()), off


@pytest.mark.slow
@pytest.mark.wide
@rtl.needs_verilator
@pytest.mark.parametrize("rate", eng.RATES)
def test_bandwidth(rate: int) -> None:
    """Each large matrix alone, from its start to its done (start-up and F2's
    latency included), reads at least 90% of min(the memory's bytes per
    cycle, 2·E·L) (PLAN.md, M5: 1024 B/cycle at full rate), times the perf
    model's engine use (1 for whole passes)."""
    share = {}

    def one(shape: tuple[int, int]) -> None:
        b = eng.bandwidth(*shape, WIDE, rate, SEED)
        share[shape] = b["share"] / b["engine_use"]

    bulk.in_parallel(one, eng.LARGE)
    print(f"read share over the engine use: {share}")
    assert all(v >= 0.9 for v in share.values()), (SEED, share)


# Attention ------------------------------------------------------------------


def positions_around(lanes: int) -> list[int]:
    return sorted({1, lanes - 1, lanes, lanes + 1, 2 * lanes + 3})


def stale_attention(
    values: bool,
    heads: int,
    kv_heads: int,
    d: int,
    t: int,
    gen: torch.Generator,
    *,
    lanes: int,
    **kw,
) -> tuple[Attention, torch.Tensor]:
    """A command and its x buffer, with NaN in the cache past t, in p past
    t, and cap the whole tiles holding t (so t = cap when t is a whole
    number of tiles) plus one more, at random."""
    cap = lanes * (-(-t // lanes) + int(torch.randint(2, (1,), generator=gen)))
    cache = random_bf16(gen, kv_heads, cap, d)
    cache[:, t:] = float("nan")
    if values:
        x = random_bf16(gen, heads, cap)
        x[:, t:] = float("nan")
        x = x.reshape(-1)
    else:
        x = random_bf16(gen, heads * d)
    return Attention(values, cache, heads, t, **kw), x


def attention_batch(ports: int, seed: int) -> tuple[list[eng.Command], dict, list[int]]:
    """Every op × head dimension × heads × positions, each with its own x
    buffer, alternating dst, format and scale, a MATVEC after every fifth,
    each placed after a random gap of 0 to 50 beats."""
    gen = torch.Generator().manual_seed(seed)
    lanes = ports * LANES_PER_PORT
    cmds: list[eng.Command] = []
    xs: dict[int, torch.Tensor] = {}
    shapes = [
        (v, h, d, t)
        for v in (False, True)
        for d in HEAD_DIMS
        if not v or d <= lanes
        for h in HEADS
        for t in positions_around(lanes)
    ]
    for i, (values, (heads, kv), d, t) in enumerate(shapes):
        kw = dict(a=len(xs), dst=2 + i % 3, bf16=i % 4 == 1, scale=SCALES[i % 2], lanes=lanes)
        c, xs[len(xs)] = stale_attention(values, heads, kv, d, t, gen, **kw)
        cmds.append(c)
        if i % 5 == 4:
            xs[len(xs)] = random_bf16(gen, 64)
            cmds.append(Matvec(random_bf16(gen, lanes + 1, 64), a=len(xs) - 1, dst=5))
    gaps = np.random.default_rng(seed).integers(0, 50, len(cmds)).tolist()
    return cmds, xs, eng.place(cmds, ports, gaps)


@rtl.needs_verilator
@pytest.mark.parametrize("ports", ALL_PORTS)
def test_attention_shapes(ports: int) -> None:
    """Every shape, back to back with MATVECs between: bit for bit, and the
    stale NaN past t never reaches a result (masked, not multiplied by 0)."""
    seed = SEED + 1000 + ports
    cmds, xs, addrs = attention_batch(ports, seed)
    check_bits(cmds, xs, eng.fetch(cmds, xs, ports, addrs, latency=20), f"seed {seed}")


@rtl.needs_verilator
@pytest.mark.parametrize("memory", MEMORIES)
@pytest.mark.parametrize("ports", ALL_PORTS)
def test_attention_memories(ports: int, memory: str) -> None:
    seed = SEED + 2000 + 10 * ports + list(MEMORIES).index(memory)
    cmds, xs, _ = attention_batch(ports, seed)
    cmds = cmds[::5]  # every positions count, both ops
    xs = {c.a: xs[c.a] for c in cmds}
    plusargs = MEMORIES[memory] | {"seed": seed}
    result = eng.fetch(cmds, xs, ports, **plusargs)
    check_bits(cmds, xs, result, f"{memory}, seed {seed}")


@rtl.needs_verilator
@pytest.mark.parametrize(("go_early", "go_delay"), GOS)
@pytest.mark.parametrize("ports", ALL_PORTS)
def test_attention_go(ports: int, go_early: int, go_delay: int) -> None:
    """SCORES and VALUES that wait for their go (D-044) between independent
    ones and MATVECs: their tiles are read ahead, they never step early (x
    reads as NaN until the go)."""
    seed = SEED + 3000 + 100 * ports + go_delay + go_early
    gen = torch.Generator().manual_seed(seed)
    lanes = ports * LANES_PER_PORT
    cmds: list[eng.Command] = []
    xs: dict[int, torch.Tensor] = {}
    for i, wait in enumerate((True, False, True, True, False, True)):
        kw = dict(a=len(xs), dst=2 + i % 3, wait=wait, lanes=lanes)
        values = i % 2 == 1
        d = min(64, lanes) if values else 64  # p.V: d <= L
        c, xs[len(xs)] = stale_attention(values, 4, 2, d, 2 * lanes + 3, gen, **kw)
        cmds.append(c)
        if i == 2:
            xs[len(xs)] = random_bf16(gen, 64)
            cmds.append(Matvec(random_bf16(gen, lanes, 64), a=len(xs) - 1, wait=True))
    gos = dict(go_early=go_early, go_delay=go_delay)
    result = eng.fetch(cmds, xs, ports, latency=GO_LATENCY, seed=seed, **gos)
    check_bits(cmds, xs, result, f"{gos}, seed {seed}")


# ports -> (SCORES' d, VALUES' d): heads of whole p.V passes, d <= L, so the
# perf model counts the engine's passes; each unit's fill as long as its step.
BALANCED = {8: (64, 16), WIDE: (64, 64)}


@rtl.needs_verilator
@pytest.mark.parametrize("ports", (8, pytest.param(WIDE, marks=pytest.mark.wide)))
def test_attention_back_to_back(ports: int) -> None:
    """Three SCORES, then three VALUES, t a whole number of tiles, the memory
    at full rate: after the first of each kind, each takes its perf-model
    cycles plus at most 1 (the command boundary): units leave no gap, the
    next tile fills while the current one steps."""
    gen = torch.Generator().manual_seed(SEED)
    lanes = ports * LANES_PER_PORT
    d_scores, d_values = BALANCED[ports]
    cmds = [
        Attention(v, random_bf16(gen, 3, 4 * lanes, d), 6, 3 * lanes)
        for v, d in [(False, d_scores)] * 3 + [(True, d_values)] * 3
    ]
    xs = {0: random_bf16(gen, 6 * 4 * lanes)}
    result = eng.fetch(cmds, xs, ports, latency=40, outstanding=32)
    check_bits(cmds, xs, result, f"back to back, seed {SEED}")
    took = np.diff(result.done).tolist()
    model = [eng.model_cycles(c, ports) for c in cmds]
    steady = [(i, took[i - 1], model[i]) for i in (1, 2, 4, 5)]
    print(f"cycles (command, measured, perf model): {steady}")
    assert all(m <= c <= m + 1 for _, c, m in steady), (SEED, steady)


# (name, the command's change, the harness's message): 4 ports, L = 16.
BAD_ATTENTION = [
    ("values d > L", dict(values=True, d=32), "d <= L"),
    ("d not a power of two", dict(d=24), "power of two"),
    ("d below 16", dict(d=8), "power of two"),
    ("cap not whole tiles", dict(cap=True), "cap a multiple of L"),
    ("t = 0", dict(t=0), "t <= cap"),
    ("group not dividing heads", dict(heads=3), "dividing heads"),
]


@pytest.mark.parametrize(
    ("change", "message"), [b[1:] for b in BAD_ATTENTION], ids=[b[0] for b in BAD_ATTENTION]
)
def test_attention_outside_the_engine_s_shapes(change: dict, message: str) -> None:
    """The engine does not check these, so the harness refuses a command
    past them before running the driver."""
    gen = torch.Generator().manual_seed(SEED)
    k = dict(values=False, d=16, heads=2) | change
    c, x = stale_attention(k["values"], k["heads"], 2, k["d"], 16, gen, lanes=16)
    if "cap" in change:  # 4 positions past whole tiles
        c = dataclasses.replace(c, cache=torch.cat([c.cache, c.cache[:, :4]], dim=1))
    if "t" in change:
        c = dataclasses.replace(c, t=change["t"])
    with pytest.raises(AssertionError, match=message):
        eng.fetch([c], {0: x}, 4)


@rtl.needs_verilator
def test_attention_fill_order() -> None:
    """Fill runs ahead of step: a short SCORES (one unit) fills while step is
    still on a long one (short rows: step waits for the lanes' final sums,
    fill does not), so fill reaches the MATVECs after it before step does.
    Fill must pass each MATVEC only when step takes that one, and start the
    next attention command only once step has read the MATVECs' beats (they
    come first in the ports' FIFOs)."""
    gen, ports = torch.Generator().manual_seed(SEED), 4
    lanes = ports * LANES_PER_PORT
    shapes = [(False, 6, 3, 16, 4 * lanes), (False, 1, 1, 16, lanes - 1)]
    cmds: list[eng.Command] = []
    xs: dict[int, torch.Tensor] = {}
    for values, heads, kv, d, t in shapes:
        c, xs[len(xs)] = stale_attention(values, heads, kv, d, t, gen, a=len(xs), lanes=lanes)
        cmds.append(c)
    for _ in range(2):
        xs[len(xs)] = random_bf16(gen, 64)
        cmds.append(Matvec(random_bf16(gen, 2 * lanes, 64), a=len(xs) - 1, dst=5))
    c, xs[len(xs)] = stale_attention(False, 4, 2, 64, 2 * lanes, gen, a=len(xs), lanes=lanes)
    cmds.append(c)
    check_bits(cmds, xs, eng.fetch(cmds, xs, ports, latency=5), f"seed {SEED}")
