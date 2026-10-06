# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/fp32_fma.sv (CVFPU's FMA, D-033) against golden.arith.fma, add and
mul: every triple of special values, and random operands. A smoke test; the
high-volume runs need the bulk harness (PLAN.md, M3)."""

import cocotb
import torch
from cocotb.triggers import Timer
from golden import arith

import rtl

SEED = 20261007
N_RANDOM = 3000  # per operation
OPS = {0: "fma", 1: "add", 2: "mul"}  # op_i, rtl/fp32_fma.sv
# Zeros, the subnormal and normal boundaries, one, the largest finite value,
# infinities, the canonical NaN and a signalling NaN; both signs where it
# applies.
SPECIALS = [
    sign | bits
    for bits in (
        0x0000_0000,
        0x0000_0001,
        0x007F_FFFF,
        0x0080_0000,
        0x3F80_0000,
        0x7F7F_FFFF,
        0x7F80_0000,
    )
    for sign in (0, 0x8000_0000)
] + [0x7FC0_0000, 0x7F80_0001]


def golden(op: str, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> list[int]:
    x, y, z = (arith.f32_from_bits(t) for t in (a, b, c))
    result = {
        "fma": lambda: arith.fma(x, y, z),
        "add": lambda: arith.add(x, y),
        "mul": lambda: arith.mul(x, y),
    }[op]()
    return arith.bits_f32(result).tolist()


def cases() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s = torch.tensor(SPECIALS, dtype=torch.int64)
    a, b, c = (t.reshape(-1) for t in torch.meshgrid(s, s, s, indexing="ij"))
    g = torch.Generator().manual_seed(SEED)
    r = torch.randint(0, 2**32, (3, N_RANDOM), generator=g, dtype=torch.int64)
    return torch.cat([a, r[0]]), torch.cat([b, r[1]]), torch.cat([c, r[2]])


@rtl.needs_verilator
def test_fp32_fma() -> None:
    rtl.simulate("fp32_fma")


@cocotb.test()
async def specials_and_random(dut) -> None:
    a, b, c = cases()
    dut.valid_i.value = 1
    for op_i, op in OPS.items():
        want = golden(op, a, b, c)
        dut.op_i.value = op_i
        for x, y, z, w in zip(a.tolist(), b.tolist(), c.tolist(), want, strict=True):
            dut.a_i.value, dut.b_i.value, dut.c_i.value = x, y, z
            await Timer(1, unit="ns")
            got = int(dut.y_o.value)
            assert got == w, (
                f"{op}(0x{x:08X}, 0x{y:08X}, 0x{z:08X}): got 0x{got:08X}, want 0x{w:08X}"
            )
