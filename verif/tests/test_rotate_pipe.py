# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/rotate_pipe.sv (D-051) against a Python rotation, cycle by cycle: a
new random vector and shift every cycle, each output checked against the
input Regs cycles before it. Entry counts a power of two and not (the
engine at 20 ports has 80 lanes), 0 to 2 registers, one copy of the shift
and one per 4 entries (the engine's), the copies all equal as the engine
gives them."""

import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly

import rtl

SEED = 20261031
W = 8
CYCLES = 300
# (N, Regs, Groups)
SHAPES = [(8, 0, 1), (16, 2, 4), (32, 2, 8), (80, 2, 20), (80, 1, 1), (128, 2, 32)]


@rtl.needs_verilator
@pytest.mark.parametrize(
    ("n", "regs", "groups"), SHAPES, ids=[f"n{n}-r{r}-g{g}" for n, r, g in SHAPES]
)
def test_rotate_pipe(n: int, regs: int, groups: int) -> None:
    rtl.simulate("rotate_pipe", params=(("N", n), ("W", W), ("Regs", regs), ("Groups", groups)))


@cocotb.test()
async def rotates(dut) -> None:
    n, regs, groups = int(dut.N.value), int(dut.Regs.value), int(dut.Groups.value)
    sw = max(1, (n - 1).bit_length())
    rng = np.random.default_rng(SEED)
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    sent: list[tuple[list[int], int]] = []
    for cycle in range(CYCLES):
        await FallingEdge(dut.clk_i)
        data = [int(v) for v in rng.integers(0, 1 << W, n)]
        shift = int(rng.integers(0, n))
        dut.in_i.value = sum(v << (W * i) for i, v in enumerate(data))
        dut.shift_i.value = sum(shift << (sw * g) for g in range(groups))
        sent.append((data, shift))
        await ReadOnly()
        if cycle >= regs:
            data, shift = sent[cycle - regs]
            want = [data[(i + shift) % n] for i in range(n)]
            out = int(dut.out_o.value)
            got = [(out >> (W * i)) & ((1 << W) - 1) for i in range(n)]
            assert got == want, (f"seed {SEED}, cycle {cycle}, shift {shift}", got, want)
