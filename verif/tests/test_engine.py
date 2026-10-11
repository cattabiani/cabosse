# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/engine.sv running MATVEC (PLAN.md, M5) against golden.dot.dot, bit for
bit, through verif/bulk/engine.cpp: one memory model per port, the x
buffers as a RAM, every write checked as it comes (the oldest command's dst,
rows below its row count only, each row once). Configurations of 1, 2 and 5
ports (4, 8 and 20 lanes); the 128-lane engine is the next step's.

Shapes: rows around the lanes (fewer than a port's group, a partial group
and pass, a single row) and columns around a beat and the lane's full-rate
row (a masked last beat), all as back-to-back commands with FP32 and BF16
destinations, two x buffers and matrices across 4 KB boundaries. Then the
same under the memory's latency, limits, bandwidth and pauses, one slow
port and sparse commands; consecutive commands leave no gap; the ports'
FIFO depth against pauses; an error response reported on its port."""

import commands
import engine as eng
import numpy as np
import perf
import pytest
import torch
from engine import LANES_PER_PORT, Matvec

import rtl

SEED = 20261011
PORTS = (1, 2, 5)  # PLAN.md, M5: degenerate, plain, prime
DESIGN = {"lanes": 0, "macs_per_lane": commands.E, "accumulators": 16}  # lanes set per test
STARTUP_CYCLES = 60  # a command's first result after its start: latency 1 (measured, with margin)
# A command's last result follows its last beat by a few cycles more when
# the tree's adder is busy (rows near 60 columns) than when it is not
# (measured: 3 more for 64 columns than for 128), so done_o's spacing moves
# by that much; over a run the moves cancel but for the first and last.
TAIL_CYCLES = 3


def rows_around(lanes: int) -> list[int]:
    lpp = LANES_PER_PORT
    return sorted({1, lpp - 1, lpp, lpp + 1, lanes - 1, lanes, lanes + 1, 2 * lanes + 3} - {0})


COLS = (1, 3, 4, 5, 59, 60, 61, 64, 130)


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
        want = eng.expected(m, xs[m.a])
        bad = np.flatnonzero(got != want)
        assert not len(bad), (
            f"{label}: command {i} ({tuple(m.w.shape)}, bf16={m.bf16}): {len(bad)} rows differ, "
            f"first row {bad[0]}: got 0x{got[bad[0]]:08X}, want 0x{want[bad[0]]:08X}"
        )
    assert result.err == 0, f"{label}: err_o {result.err:#x}"
    assert result.r_stalls == 0, f"{label}: a memory waited {result.r_stalls} cycles on RREADY"


def model_cycles(m: Matvec, ports: int) -> int:
    return perf.engine_cycles((tuple(m.w.shape),), DESIGN | {"lanes": ports * LANES_PER_PORT})


@rtl.needs_verilator
@pytest.mark.parametrize("ports", PORTS)
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
@pytest.mark.parametrize("ports", PORTS)
def test_memories(ports: int, memory: str) -> None:
    seed = SEED + 10 * ports + list(MEMORIES).index(memory)
    mats, xs, addrs = shape_batch(ports, seed)
    mats, addrs = mats[::4], addrs[::4]  # a quarter of the shapes, still every row count
    plusargs = MEMORIES[memory] | {"seed": seed}
    check_bits(mats, xs, eng.fetch(mats, xs, ports, addrs, **plusargs), f"{memory}, seed {seed}")


STALE = (float("inf"), float("-inf"), float("nan"), -0.0)  # what an x buffer may hold past m


@rtl.needs_verilator
@pytest.mark.parametrize("ports", PORTS)
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
    check_bits(mats, xs, eng.fetch(mats, xs, ports, latency=5), "stale x")


BACK_TO_BACK = ((1, 64), (3, 128), (2, 60)) * 4  # (passes, cols): rows of full-rate length


@rtl.needs_verilator
@pytest.mark.parametrize("ports", PORTS)
def test_back_to_back(ports: int) -> None:
    """With the memory at full rate, the commands after the first take
    their perf-model cycles, to within TAIL_CYCLES over the whole run: no
    gap between commands (one cycle each would be 11), the next matrix's
    reads hidden behind the current one."""
    gen = torch.Generator().manual_seed(SEED)
    lanes = ports * LANES_PER_PORT
    mats = [Matvec(random_bf16(gen, p * lanes, c)) for p, c in BACK_TO_BACK]
    xs = {0: random_bf16(gen, 128)}
    result = eng.fetch(mats, xs, ports, latency=40, outstanding=8)
    check_bits(mats, xs, result, "back to back")
    model = [model_cycles(m, ports) for m in mats]
    assert result.done[0] <= model[0] + 40 + STARTUP_CYCLES, (result.done[0], model[0])
    extra = int(result.done[-1] - result.done[0]) - sum(model[1:])
    assert abs(extra) <= TAIL_CYCLES, (extra, np.diff(result.done).tolist(), model)


DEPTHS = (16, 64, 128, 1024)  # 1024: deep enough that the pauses alone limit the rate
PAUSES = dict(latency=20, outstanding=8, pause=20, pause_max=40)  # ports paused ~29% of cycles
# Share of the 1024-beat FIFO's rate, at least (or at most for 16). Measured
# over seeds 1 to 5: 0.35-0.37, 0.89-0.91, 0.98-0.99 (PLAN.md, M5).
MIN_SHARE = {64: 0.85, 128: 0.95}
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
        check_bits(mats, xs, result, f"depth {depth}")
        rate[depth] = beats / (result.done[-1] - result.taken[0])
    share = {d: r / rate[1024] for d, r in rate.items()}
    print(f"share of the deep FIFO's rate by depth: {share}")
    assert all(share[d] >= v for d, v in MIN_SHARE.items()), share
    assert share[16] <= MAX_SHARE_16, share


@rtl.needs_verilator
@pytest.mark.parametrize("bad_port", (0, 1))
def test_error_response(bad_port: int) -> None:
    gen = torch.Generator().manual_seed(SEED)
    mats = [Matvec(random_bf16(gen, 16, 64))]
    xs = {0: random_bf16(gen, 64)}
    result = eng.fetch(mats, xs, 2, bad_port=bad_port, bad=3)
    assert result.err == 1 << bad_port, result.err
