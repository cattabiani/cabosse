# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/bf16_to_fp32.sv against golden.arith.up, every one of the 65,536 BF16
inputs."""

import cocotb
import torch
from cocotb.triggers import Timer
from golden import arith

import rtl


@rtl.needs_verilator
def test_bf16_to_fp32() -> None:
    rtl.simulate("bf16_to_fp32")


@cocotb.test()
async def every_input(dut) -> None:
    bits = torch.arange(2**16, dtype=torch.int64)
    want = arith.bits_f32(arith.up(arith.bf16_from_bits(bits))).tolist()
    for a, y in zip(bits.tolist(), want, strict=True):
        dut.a_i.value = a
        await Timer(1, unit="ns")
        got = int(dut.y_o.value)
        assert got == y, f"bf16 0x{a:04X}: got 0x{got:08X}, want 0x{y:08X}"
