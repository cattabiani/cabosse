# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/dot_lane.sv (PLAN.md, M4) against golden.dot.dot, bit for bit, through
its valid/ready handshake: bubbles on the input one by one, in runs and at
random; random ready on the output; every row length up to 20 and random ones
to 300; missing elements (mask) anywhere in a row; special values. With the
output always ready, rows of 64 elements or more never see ready drop, so a
bubble costs exactly its own cycle; shorter rows may, with the same bits.

The arithmetic itself is tested exhaustively in test_bf16_mac; here it is the
wiring, the rotation of the partial sums, the keep rule, the tree and the
handshake."""

from collections.abc import Callable
from dataclasses import dataclass

import cocotb
import numpy as np
import pytest
import torch
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly
from fp_inputs import BF16_SPECIAL_BITS
from golden import arith, dot

import rtl

SEED = 20261008
E = 4  # pairs per beat: dot_lane's default
FULL_RATE_LEN = 64  # rows at least this long keep E pairs per cycle
MAX_CYCLES = 1_000_000  # a hung handshake fails instead of running forever
# dot_lane's product registers: the same bits and handshake either way.
PIPELINES = {"mulregs0": (("MulRegs", 0),), "mulregs1": (), "mulregs2": (("MulRegs", 2),)}

Ready = Callable[[int], bool]  # cycle -> out_ready_i


@rtl.needs_verilator
@pytest.mark.parametrize("pipeline", PIPELINES)
def test_dot_lane(pipeline: str) -> None:
    rtl.simulate("dot_lane", params=PIPELINES[pipeline])


@dataclass
class Row:
    """One dot product: BF16 bits w, x and which elements are there."""

    w: np.ndarray  # uint16
    x: np.ndarray  # uint16
    valid: np.ndarray  # bool

    def expected(self) -> int:
        """golden.dot.dot of the row, as FP32 bits."""
        w, x = (
            arith.bf16_from_bits(torch.from_numpy(v.astype(np.int64))) for v in (self.w, self.x)
        )
        y = dot.dot(w, x, valid=torch.from_numpy(self.valid))
        return int(arith.bits_f32(y))

    def beats(self) -> list[tuple[int, int, int, bool]]:
        """(w, x, mask, last) per beat, E elements each, element e of a beat
        in bits 16e .. 16e + 15; a short last beat is masked."""
        n_beats = -(-len(self.w) // E)
        out = []
        for b in range(n_beats):
            w = x = mask = 0
            for e in range(E):
                k = b * E + e
                if k < len(self.w):
                    w |= int(self.w[k]) << (16 * e)
                    x |= int(self.x[k]) << (16 * e)
                    mask |= int(self.valid[k]) << e
            out.append((w, x, mask, b == n_beats - 1))
        return out


@dataclass
class Run:
    results: list[int]
    stalls: int  # cycles with in_valid_i high and in_ready_o low
    cycles: int


def bf16_bits(values: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(values.astype(np.float32)).to(torch.bfloat16)
    return arith.bits_bf16(t).numpy().astype(np.uint16)


def normal_row(rng: np.random.Generator, n: int, masked: float = 0.0) -> Row:
    """Values like a model's: normal, a spread of scales, both signs."""
    w = bf16_bits(rng.normal(0, 1, n) * 2.0 ** rng.integers(-8, 8, n))
    x = bf16_bits(rng.normal(0, 1, n) * 2.0 ** rng.integers(-8, 8, n))
    return Row(w, x, rng.random(n) >= masked)


def start_clock(dut) -> None:
    """Once per test: every run() of the test shares the clock."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())


def always(_: int) -> bool:
    return True


async def run(
    dut,
    rows: list[Row],
    gaps: list[int] | None = None,
    out_ready: Ready = always,
    seed: int = SEED,
) -> Run:
    """Reset, then stream the rows' beats, gaps[i] idle cycles before beat i
    (default none), with out_ready_i from out_ready(cycle). Inputs carry
    random bits while in_valid_i is low. The clock runs (start_clock)."""
    rng = np.random.default_rng(seed)
    beats = [b for row in rows for b in row.beats()]
    gaps = gaps or [0] * len(beats)
    assert len(gaps) == len(beats)
    await FallingEdge(dut.clk_i)  # a previous run ends in ReadOnly
    dut.rst_ni.value = 0
    dut.in_valid_i.value = 0
    dut.out_ready_i.value = 0
    for _ in range(3):
        await FallingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    results: list[int] = []
    i, idle, holding, stalls, cycle = 0, 0, False, 0, 0
    while len(results) < len(rows):
        await FallingEdge(dut.clk_i)
        if not holding and i < len(beats) and idle < gaps[i]:
            idle += 1
            valid = False
        else:
            valid = i < len(beats)
        if valid:
            w, x, mask, last = beats[i]
        else:
            w, x = (int.from_bytes(rng.bytes(2 * E), "little") for _ in range(2))
            mask, last = int(rng.integers(0, 1 << E)), bool(rng.integers(0, 2))
        dut.in_valid_i.value = valid
        dut.w_i.value, dut.x_i.value, dut.mask_i.value, dut.last_i.value = w, x, mask, last
        dut.out_ready_i.value = out_ready(cycle)
        await ReadOnly()
        if valid:
            if dut.in_ready_o.value:
                i, idle, holding = i + 1, 0, False
            else:
                stalls, holding = stalls + 1, True
        if dut.out_valid_o.value and dut.out_ready_i.value:
            results.append(int(dut.out_o.value))
        cycle += 1
        assert cycle < MAX_CYCLES, f"hung after {len(results)} of {len(rows)} results"
    return Run(results, stalls, cycle)


def check_bits(rows: list[Row], got: Run, label: str) -> None:
    want = [r.expected() for r in rows]
    bad = [k for k, (g, w) in enumerate(zip(got.results, want, strict=True)) if g != w]
    if bad:
        k = bad[0]
        raise AssertionError(
            f"{label} (seed {SEED}): {len(bad)} of {len(rows)} rows differ; row {k} "
            f"(length {len(rows[k].w)}): got 0x{got.results[k]:08X}, want 0x{want[k]:08X}"
        )


@cocotb.test()
async def single_rows(dut) -> None:
    """Every length 1 .. 20, then lengths 64 and 576, one row at a time."""
    start_clock(dut)
    rng = np.random.default_rng(SEED)
    for n in [*range(1, 21), 64, 576]:
        rows = [normal_row(rng, n)]
        check_bits(rows, await run(dut, rows), f"length {n}")


@cocotb.test()
async def every_short_length_back_to_back(dut) -> None:
    """Lengths 1 .. 20, three rows each, with no gaps: rows end before the
    tree is free, so ready drops, and no result may be lost or reordered."""
    start_clock(dut)
    rng = np.random.default_rng(SEED + 1)
    rows = [normal_row(rng, n) for n in range(1, 21) for _ in range(3)]
    got = await run(dut, rows)
    check_bits(rows, got, "short rows")
    assert got.stalls > 0, "short rows never stalled: the throttle is untested"


@cocotb.test()
async def full_rate_without_gaps(dut) -> None:
    """Rows of 64 and of random lengths up to 300, back to back: ready never
    drops."""
    start_clock(dut)
    rng = np.random.default_rng(SEED + 2)
    lengths = [FULL_RATE_LEN] * 20 + list(rng.integers(FULL_RATE_LEN, 301, 20))
    rows = [normal_row(rng, int(n)) for n in lengths]
    got = await run(dut, rows)
    check_bits(rows, got, "full rate")
    assert got.stalls == 0, f"{got.stalls} stalls on rows of {FULL_RATE_LEN} or more"


def bubble_cases() -> list[tuple[str, list[int]]]:
    """Gaps before each beat of four 64-long rows (16 beats each): one bubble
    at each beat of the second row, runs of 1 .. 5 and of 40 bubbles there,
    and a bubble before every beat."""
    n_beats = 4 * FULL_RATE_LEN // E
    cases = []
    for b in range(16, 32):
        gaps = [0] * n_beats
        gaps[b] = 1
        cases.append((f"one bubble before beat {b}", gaps))
    for run_len in (1, 2, 3, 4, 5, 40):
        for b in (16, 21, 31):
            gaps = [0] * n_beats
            gaps[b] = run_len
            cases.append((f"{run_len} bubbles before beat {b}", gaps))
    cases.append(("a bubble before every beat", [1] * n_beats))
    return cases


@cocotb.test()
async def bubbles(dut) -> None:
    """Each bubble pattern of bubble_cases: same bits, and ready never drops,
    so the bubbles cost their own cycles only."""
    start_clock(dut)
    rng = np.random.default_rng(SEED + 3)
    for label, gaps in bubble_cases():
        rows = [normal_row(rng, FULL_RATE_LEN) for _ in range(4)]
        got = await run(dut, rows, gaps)
        check_bits(rows, got, label)
        assert got.stalls == 0, f"{label}: {got.stalls} stalls"


@cocotb.test()
async def random_bubbles(dut) -> None:
    """Random gaps at three densities on rows of 64 .. 300: no stalls."""
    start_clock(dut)
    rng = np.random.default_rng(SEED + 4)
    for p_gap in (0.05, 0.3, 0.7):
        rows = [normal_row(rng, int(n)) for n in rng.integers(FULL_RATE_LEN, 301, 30)]
        n_beats = sum(len(r.beats()) for r in rows)
        gaps = list(rng.geometric(1 - p_gap, n_beats) - 1)
        got = await run(dut, rows, gaps)
        check_bits(rows, got, f"gap probability {p_gap}")
        assert got.stalls == 0, f"gap probability {p_gap}: {got.stalls} stalls"


@cocotb.test()
async def random_backpressure(dut) -> None:
    """Random lengths 1 .. 300, random gaps and a random output ready, from
    nearly always ready to nearly never: every result, in order."""
    start_clock(dut)
    rng = np.random.default_rng(SEED + 5)
    for p_ready in (0.9, 0.5, 0.05):
        rows = [normal_row(rng, int(n), masked=0.1) for n in rng.integers(1, 301, 60)]
        n_beats = sum(len(r.beats()) for r in rows)
        gaps = list(rng.geometric(0.6, n_beats) - 1)
        ready = rng.random(MAX_CYCLES) < p_ready
        got = await run(dut, rows, gaps, out_ready=lambda c, r=ready: bool(r[c]))
        check_bits(rows, got, f"ready probability {p_ready}")


@cocotb.test()
async def output_held(dut) -> None:
    """The output not ready for long stretches: the lane fills, stops taking
    rows, and loses nothing."""
    start_clock(dut)
    rng = np.random.default_rng(SEED + 6)
    rows = [normal_row(rng, int(n)) for n in rng.integers(1, 200, 40)]
    got = await run(dut, rows, out_ready=lambda c: c % 500 >= 450)
    check_bits(rows, got, "output held")
    assert got.stalls > 0


@cocotb.test()
async def masks(dut) -> None:
    """Missing elements anywhere: rows with a quarter, half and all elements
    masked, and a row missing whole groups of A."""
    start_clock(dut)
    rng = np.random.default_rng(SEED + 7)
    rows = [normal_row(rng, int(n), masked=m) for m in (0.25, 0.5, 1.0) for n in (5, 16, 70, 300)]
    gap_row = normal_row(rng, 96)
    gap_row.valid[16:48] = False
    rows.append(gap_row)
    check_bits(rows, await run(dut, rows), "masks")


def signed_zero_rows() -> list[Row]:
    """Products that underflow to -0 (-2^-80 * 2^-80): with A of them every
    partial sum is -0 and so is the result. A missing element must keep a -0
    partial sum; adding a zero product instead makes it +0 (numerics.md,
    section 3). Fewer than A of them leave +0 partial sums, so +0."""
    neg, pos = 0xD780, 0x1780  # BF16 -2^-80, 2^-80
    rows = []
    for n, n_valid in ((16, 16), (20, 17), (32, 16), (64, 16), (8, 8)):
        valid = np.arange(n) < n_valid
        rows.append(Row(np.full(n, neg, np.uint16), np.full(n, pos, np.uint16), valid))
    return rows


def special_rows(rng: np.random.Generator) -> list[Row]:
    """Rows of the BF16 special values (zeros, subnormals, the largest
    value, infinities, NaNs) mixed into normal values; rows whose products
    overflow FP32 one by one but not added (the largest values of opposite
    signs); rows that cancel to exactly zero; and rows of subnormal partial
    sums."""
    specials = np.array(BF16_SPECIAL_BITS, dtype=np.uint16)
    rows = []
    for n in (1, 7, 64, 130):
        for density in (0.05, 0.5):
            row = normal_row(rng, n)
            for v in (row.w, row.x):
                pick = rng.random(n) < density
                v[pick] = rng.choice(specials, int(pick.sum()))
            rows.append(row)
    big = np.full(64, 0x7F7F, np.uint16)  # largest BF16: products above FP32's range
    sign = np.where(np.arange(64) % 32 < 16, 0, 0x8000).astype(np.uint16)
    rows.append(Row(big, big ^ sign, np.ones(64, bool)))
    rows.append(Row(big, big, np.ones(64, bool)))
    w = bf16_bits(rng.normal(0, 1, 64))
    rows.append(Row(np.concatenate([w, w]), np.concatenate([w, w ^ 0x8000]), np.ones(128, bool)))
    tiny = bf16_bits(rng.normal(0, 1, 80) * 2.0**-70)
    rows.append(Row(tiny, bf16_bits(rng.normal(0, 1, 80) * 2.0**-70), np.ones(80, bool)))
    return rows


@cocotb.test()
async def special_values(dut) -> None:
    start_clock(dut)
    rng = np.random.default_rng(SEED + 8)
    rows = signed_zero_rows() + special_rows(rng)
    check_bits(rows, await run(dut, rows), "special values")
