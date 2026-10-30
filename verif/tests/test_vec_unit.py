# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/vec_unit.sv's buffers and elementwise commands against
commands.vector_command, every buffer bit for bit after the commands (so a
write past a command's elements shows), through verif/bulk/vec_unit.cpp.

Each command alone in every shape the walk has (packed rows, one flat row,
rows of t at the stride, ROTATE_HALF's halves; rows shorter than, equal to
and longer than the lanes), random sequences of commands back to back and
with dependencies (a command that shares a buffer with an earlier one waits
for it, as the controller does), and the speed of back-to-back commands."""

import dataclasses

import commands
import numpy as np
import pytest
import torch
import vector
from commands import Buf, Command, Flag, Op, f32_bits
from fp_inputs import F32_SPECIAL_BITS, with_exponents
from golden import arith
from vector import Step

import rtl

SEED = 20261023
LENGTH = 256  # elements of every buffer in the fast tests
ELEMENTWISE = (Op.ADD, Op.MUL, Op.FMA, Op.EXP, Op.RECIP, Op.RSQRT)
REDUCTIONS = (Op.SUM, Op.SUMSQ, Op.MAX)
BINARY = (Op.ADD, Op.MUL, Op.FMA)
SOURCES = vector.FP32_BUFS
# Row lengths around the lanes, and the shapes of the walk.
LENGTHS = (1, 3, 15, 16, 17, 31, 32, 33, 64)
# A reduction's rows also around its partials (128) and their groups (8).
REDUCE_LENGTHS = (*LENGTHS, 113, 127, 128, 129, 255)


def scalar(rng: np.random.Generator) -> int:
    return f32_bits(float(rng.uniform(-4, 4)))


def random_command(rng: np.random.Generator, length: int) -> Step:
    """A random vector-unit command, its buffers, flags and shape within
    `length` elements."""
    op = Op(rng.choice([*ELEMENTWISE, *REDUCTIONS, Op.ROTATE_HALF]))
    a, b, c = (Buf(x) for x in rng.choice(SOURCES, 3, replace=False))
    flags = Flag.NONE
    if op in ELEMENTWISE and rng.random() < 0.5:
        flags |= Flag.NEG_A
    b_mode = "none"
    if op in BINARY:
        b_mode = rng.choice(["buffer", "scalar", "per_row", "row"])
        flags |= {"per_row": Flag.B_PER_ROW, "row": Flag.B_ROW}.get(b_mode, Flag.NONE)
        if b_mode != "scalar" and rng.random() < 0.5:
            flags |= Flag.NEG_B
    len_t = op != Op.ROTATE_HALF and rng.random() < 0.4
    n = int(rng.integers(1, 5))
    t, stride = 1, 16
    if len_t:
        flags |= Flag.LEN_T
        t = int(rng.choice(LENGTHS))
        stride = 16 * int(rng.integers(-(-t // 16), -(-t // 16) + 3))
        n = min(n, length // stride)
        m = 0
    else:
        m = int(rng.choice(REDUCE_LENGTHS if op in REDUCTIONS else LENGTHS))
        if op == Op.ROTATE_HALF:
            m = 2 * max(1, m // 2)
        n = min(n, length // m)
    dst = a if rng.random() < 0.3 else Buf(rng.choice([b2 for b2 in Buf if b2 != Buf.NONE]))
    cmd = Command(
        op,
        dst,
        a,
        b if b_mode in ("buffer", "per_row", "row") else Buf.NONE,
        c if op == Op.FMA else Buf.NONE,
        flags,
        n=n,
        m=m,
        scalar=scalar(rng) if b_mode == "scalar" else 0,
    )
    if not commands.overlap_ok(cmd):  # a dst the unit may not write: a free buffer
        cmd = dataclasses.replace(cmd, dst=next(x for x in SOURCES if x not in (a, b, c)))
    return Step(cmd, t=t, stride=stride)


def with_waits(steps: list[Step]) -> list[Step]:
    """A command waits for every earlier one to be done when it reads a
    buffer that one still running may write (the controller's stop, D-045);
    after a wait nothing earlier runs. A write after a read or after a write
    needs no wait: the unit reads a command's groups before the next one's,
    and writes in order."""
    written: set[Buf] = set()  # by the commands since the last wait
    for s in steps:
        s.wait = bool({s.cmd.a, s.cmd.b, s.cmd.c} & written)
        written = ({s.cmd.dst} | (set() if s.wait else written)) - {Buf.NONE}
    return steps


@rtl.needs_verilator
@pytest.mark.parametrize("op", [*ELEMENTWISE, Op.ROTATE_HALF])
@pytest.mark.parametrize("m", LENGTHS)
def test_one_command(op: Op, m: int) -> None:
    """3 packed rows of m (ROTATE_HALF: m rounded up to even)."""
    rng = np.random.default_rng(SEED + m)
    m = 2 * -(-m // 2) if op == Op.ROTATE_HALF else m
    b = Buf.U if op in BINARY else Buf.NONE
    c = Buf.E if op == Op.FMA else Buf.NONE
    cmd = Command(op, Buf.T, Buf.G, b, c, n=3, m=m)
    vector.check([Step(cmd)], vector.random_buffers(rng, LENGTH), LENGTH, f"{op.name}, m {m}")


@rtl.needs_verilator
@pytest.mark.parametrize("op", REDUCTIONS)
@pytest.mark.parametrize("m", REDUCE_LENGTHS)
def test_one_reduction(op: Op, m: int) -> None:
    """Up to 3 packed rows of m: partials reached or not, a partial last
    group, rows past the partials (128) and their groups (8)."""
    rng = np.random.default_rng(SEED + m)
    cmd = Command(op, Buf.R, Buf.G, n=max(1, min(3, LENGTH // m)), m=m)
    vector.check([Step(cmd)], vector.random_buffers(rng, LENGTH), LENGTH, f"{op.name}, m {m}")


def special_rows(rng: np.random.Generator, m: int) -> torch.Tensor:
    """Rows of m FP32 values that test a reduction's order and rules: the
    shared special values (fp_inputs: zeros, subnormal edges, the largest
    value, infinities, NaNs) mixed into moderate values at two densities, a
    large cancellation, all -0 and all NaN."""
    specials = np.array(F32_SPECIAL_BITS, dtype=np.uint32)
    mixed = []
    for density in (0.05, 0.5):
        row = with_exponents(rng, m, 117, 137, 32)
        pick = rng.random(m) < density
        row[pick] = rng.choice(specials, int(pick.sum()))
        mixed.append(arith.f32_from_bits(torch.from_numpy(row.astype(np.int64))))
    big = torch.tensor([2.0**100, 1.0, -(2.0**100), 3.0])
    rows = [
        *mixed,
        big.repeat(m // 4 + 1)[:m],  # sums depend on the order
        torch.full((m,), -0.0),  # -0 sums: +0 partials count
        torch.full((m,), torch.nan),
    ]
    return torch.stack(rows).to(torch.float32)


@rtl.needs_verilator
@pytest.mark.parametrize("op", REDUCTIONS)
@pytest.mark.parametrize("m", (1, 16, 37, 128, 200))
def test_reduction_special_values(op: Op, m: int) -> None:
    rows = special_rows(np.random.default_rng(SEED + m), m)
    length = max(LENGTH, -(-rows.numel() // vector.LANES) * vector.LANES)
    start = vector.random_buffers(np.random.default_rng(SEED), length)
    start[Buf.G][: rows.numel()] = rows.flatten()
    cmd = Command(op, Buf.R, Buf.G, n=rows.shape[0], m=m)
    vector.check([Step(cmd)], start, length, f"{op.name}, m {m}")


@rtl.needs_verilator
@pytest.mark.parametrize("t", LENGTHS)
def test_softmax_at_the_stride(t: int) -> None:
    """Softmax as the command list writes it: rows of t at the stride, b one
    value per row, each command waiting for the one before."""
    rng = np.random.default_rng(SEED + t)
    stride = 16 * (-(-t // 16) + 1)
    rows = Flag.LEN_T
    steps = [
        Step(Command(Op.MAX, Buf.R, Buf.S, flags=rows, n=3), t, stride),
        Step(
            Command(
                Op.ADD, Buf.S, Buf.S, Buf.R, flags=Flag.LEN_T | Flag.B_PER_ROW | Flag.NEG_B, n=3
            ),
            t,
            stride,
            wait=True,
        ),
        Step(Command(Op.EXP, Buf.S, Buf.S, flags=Flag.LEN_T, n=3), t, stride, wait=True),
        Step(Command(Op.SUM, Buf.W, Buf.S, flags=rows, n=3), t, stride, wait=True),
        Step(Command(Op.RECIP, Buf.R, Buf.W, n=3, m=1), wait=True),
        Step(
            Command(Op.MUL, Buf.P, Buf.S, Buf.R, flags=Flag.LEN_T | Flag.B_PER_ROW, n=3),
            t,
            stride,
            wait=True,
        ),
    ]
    vector.check(steps, vector.random_buffers(rng, LENGTH), LENGTH, f"t {t}, stride {stride}")


@rtl.needs_verilator
@pytest.mark.parametrize("seed", range(4))
def test_random_sequences(seed: int) -> None:
    """60 random commands: dependent ones wait, the rest run back to back,
    offered at random cycles in half of the runs."""
    rng = np.random.default_rng(SEED + seed)
    steps = with_waits([random_command(rng, LENGTH) for _ in range(60)])
    offer = 1000 if seed % 2 else 300
    vector.check(
        steps,
        vector.random_buffers(rng, LENGTH),
        LENGTH,
        f"seed {SEED + seed}",
        offer=offer,
        seed=seed,
    )


@rtl.needs_verilator
def test_back_to_back_speed() -> None:
    """Independent commands of 16k elements back to back: a group a cycle,
    so each takes its elements / 16 cycles after the first."""
    length = 1 << 14
    rng = np.random.default_rng(SEED)
    ops = [(Op.ADD, Buf.H, Buf.Q), (Op.MUL, Buf.K, Buf.V), (Op.EXP, Buf.KR, Buf.NONE)]
    steps = [
        Step(Command(op, dst, Buf.T if i % 2 else Buf.G, b, n=1, m=length))
        for i, (op, dst, b) in enumerate(ops)
    ]
    result = vector.check(steps, vector.random_buffers(rng, length), length, "speed")
    took = np.diff(result.done).tolist()
    assert took == [length // vector.LANES] * 2, took


@rtl.needs_verilator
@pytest.mark.parametrize("op", REDUCTIONS)
def test_reduction_trees_overlap_the_rows(op: Op) -> None:
    """Rows of 4096 elements (256 groups) are longer than a row's tree, so
    each row after the first costs its groups alone: the tree of a row runs
    while the slots take the next (D-050)."""
    length = 1 << 14
    rng = np.random.default_rng(SEED)
    steps = [Step(Command(op, Buf.R, Buf.G, n=n, m=4096)) for n in (1, 4)]
    steps[1].wait = True
    result = vector.check(steps, vector.random_buffers(rng, length), length, op.name)
    one = result.done[0] - result.taken[0]
    four = result.done[1] - result.taken[1]
    assert four - one == 3 * 4096 // vector.LANES, (one, four)
