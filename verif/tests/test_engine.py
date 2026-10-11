# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/engine.sv running MATVEC (PLAN.md, M5) against golden.dot.dot, bit for
bit, through verif/bulk/engine.cpp: one memory model per port, the x
buffers as a RAM, every write checked as it comes (the oldest command's dst,
rows below its row count only, each row once). Configurations of 1, 2 and 5
ports (4, 8 and 20 lanes) and the 128-lane engine (32 ports, marked wide).

Shapes: rows around the lanes (fewer than a port's group, a partial group
and pass, a single row) and columns around a beat and the lane's full-rate
row (a masked last beat), all as back-to-back commands with FP32 and BF16
destinations, two x buffers and matrices across 4 KB boundaries. Then the
same under the memory's latency, limits, bandwidth and pauses, one slow
port and sparse commands; consecutive commands leave no gap; commands that
wait for their go (D-042) read their weights ahead and never step early;
the ports' FIFO depth against pauses; an error response reported on its
port. Slow, at 128 lanes: SmolLM2's MATVECs with real data, cycles against
the perf model, and the bandwidth reached on large matrices."""

import bulk
import commands
import engine as eng
import numpy as np
import paths
import pytest
import torch
from engine import LANES_PER_PORT, Matvec
from golden import decoder
from lane import FULL_RATE_LEN

import rtl

SEED = 20261011
PORTS = (1, 2, 5)  # PLAN.md, M5: degenerate, plain, prime
WIDE = 32  # 128 lanes, SmolLM2's engine
ALL_PORTS = (*PORTS, pytest.param(WIDE, marks=pytest.mark.wide))
STARTUP_CYCLES = 60  # a command's first result after its start: latency 1 (measured, with margin)
# Around a beat and the lane's shortest full-rate row.
COLS = (1, 3, 4, 5, FULL_RATE_LEN - 1, FULL_RATE_LEN, FULL_RATE_LEN + 1, 64, 130)


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


def check_bits(
    mats: list[Matvec], xs: dict[int, torch.Tensor], result: eng.Result, label: str
) -> None:
    for i, (m, got) in enumerate(zip(mats, result.out, strict=True)):
        command = f"{label}: command {i} ({tuple(m.w.shape)}, bf16={m.bf16})"
        eng.assert_bits(got, eng.expected(m, xs[m.a]), command)
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
PAUSES = dict(latency=20, outstanding=8, pause=20, pause_max=40)  # ports paused ~29% of cycles
# Share of the 1024-beat FIFO's rate, at least (or at most for 16). Measured
# on these shapes and pauses, memory seeds 1 to 5 (each port its own stream):
# 0.347-0.357 (16), 0.865-0.931 (64), 0.941-0.993 (128).
MIN_SHARE = {64: 0.80, 128: 0.90}
MAX_SHARE_16 = 0.5


@rtl.needs_verilator
def test_fifo_depth() -> None:
    """Five ports pausing at random, independently, in lockstep: each
    port's FIFO must cover both the memory's latency and the other ports'
    pauses. Rates against a FIFO deep enough not to matter."""
    ports, gen = 5, torch.Generator().manual_seed(SEED)
    mats = [Matvec(random_bf16(gen, 8 * ports * LANES_PER_PORT, 512)) for _ in range(4)]
    xs = {0: random_bf16(gen, 512)}
    beats = sum(eng.n_beats(m, ports) for m in mats)
    rate = {}
    for depth in DEPTHS:
        result = eng.fetch(mats, xs, ports, params=(("Depth", depth),), seed=SEED, **PAUSES)
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
    result = eng.fetch(mats, xs, 2, bad_port=bad_port, bad=3)
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
    found, result = eng.run_smollm2(
        decoder.load(paths.SMOLLM2), WIDE, latency=eng.F2_LATENCY_CYCLES
    )
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
        read, use = eng.bandwidth(*shape, WIDE, rate, SEED)
        share[shape] = read / use

    bulk.in_parallel(one, eng.LARGE)
    print(f"read share over the engine use: {share}")
    assert all(v >= 0.9 for v in share.values()), (SEED, share)
