# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/fp32_to_bf16.sv against golden.arith.bf16, bit for bit: every one of
the 65,536 high halves with each low half that decides the rounding (zero,
just below a tie, the tie, just above, all ones) and a random one, and the
special values."""

import cocotb
import numpy as np
import torch
from cocotb.triggers import Timer
from fp_inputs import special_tuples
from golden import arith

import rtl

SEED = 20261011
LOW_HALVES = (0x0000, 0x7FFF, 0x8000, 0x8001, 0xFFFF)


def inputs() -> np.ndarray:
    high = np.arange(1 << 16, dtype=np.uint32) << 16
    rng = np.random.default_rng(SEED)
    lows = [np.full(1 << 16, v, dtype=np.uint32) for v in LOW_HALVES]
    lows.append(rng.integers(0, 1 << 16, 1 << 16, dtype=np.uint32))
    return np.concatenate([*(high | low for low in lows), special_tuples(1).ravel()])


@rtl.needs_verilator
def test_fp32_to_bf16() -> None:
    rtl.simulate("fp32_to_bf16")


@cocotb.test()
async def every_high_half(dut) -> None:
    a = inputs()
    want = arith.bits_bf16(arith.bf16(torch.from_numpy(a.view(np.float32)))).numpy()
    for x, y in zip(a.tolist(), want.tolist(), strict=True):
        dut.a_i.value = x
        await Timer(1, unit="ns")
        got = int(dut.y_o.value)
        assert got == y, f"fp32 0x{x:08X}: got 0x{got:04X}, want 0x{y:04X} (seed {SEED})"
