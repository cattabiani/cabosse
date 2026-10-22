# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""What the vector unit's tests share: its buffers as the golden model holds
them, the driver verif/bulk/vec_unit.cpp run on commands, and the bit-exact
comparison of every buffer afterwards (rtl/vec_unit.sv)."""

from dataclasses import dataclass

import bulk
import commands
import numpy as np
import torch
from commands import Buf, Command
from golden import arith

LANES = 16  # rtl/vec_unit.sv's default
BUFS = len(Buf)  # ids 0 (NONE) to 21
CMD_FIELDS = 12  # the driver's words per command
# Each buffer's format (commands.buffers; the RTL's default Bf16Mask).
BF16_BUFS = frozenset({Buf.X, Buf.QR, Buf.P, Buf.ATT, Buf.M})
FP32_BUFS = tuple(b for b in Buf if b != Buf.NONE and b not in BF16_BUFS)


def dtype(b: Buf) -> torch.dtype:
    return torch.bfloat16 if b in BF16_BUFS else torch.float32


def bits(x: torch.Tensor) -> np.ndarray:
    """A buffer's raw bits, uint32 (BF16 in the low half)."""
    raw = arith.bits_bf16(x) if x.dtype == torch.bfloat16 else arith.bits_f32(x)
    return raw.numpy().astype(np.uint32)


def from_bits(raw: np.ndarray, b: Buf) -> torch.Tensor:
    t = torch.from_numpy(raw.astype(np.int64))
    return arith.bf16_from_bits(t) if dtype(b) == torch.bfloat16 else arith.f32_from_bits(t)


def random_buffers(rng: np.random.Generator, length: int) -> dict[Buf, torch.Tensor]:
    """Every buffer of `length` elements, random bit patterns: the sources'
    values and the stale data a command must leave alone. NaN and infinite
    patterns are kept to a few percent, so most results are numbers."""
    out = {}
    for b in Buf:
        if b == Buf.NONE:
            continue
        width = 16 if dtype(b) == torch.bfloat16 else 32
        raw = rng.integers(0, 2**width, length, dtype=np.uint64)
        # Most values moderate: an exponent near the middle of the range.
        frac = 7 if width == 16 else 23
        moderate = rng.random(length) < 0.95
        exp = rng.integers(127 - 20, 127 + 20, length, dtype=np.uint64)
        sign = rng.integers(0, 2, length, dtype=np.uint64)
        mant = rng.integers(0, 2**frac, length, dtype=np.uint64)
        nice = (sign << (width - 1)) | (exp << frac) | mant
        raw = np.where(moderate, nice, raw)
        out[b] = from_bits(raw.astype(np.uint32), b)
    return out


@dataclass
class Step:
    """One command as the driver takes it: the command, the registers it
    runs with, and whether it waits for every earlier command to be done."""

    cmd: Command
    t: int = 1
    stride: int = 16
    wait: bool = False


@dataclass
class Result:
    buffers: dict[Buf, np.ndarray]  # raw bits, uint32
    taken: np.ndarray  # per command, the cycle it was taken
    done: np.ndarray  # per command, the cycle of its done_o
    cycles: int


def golden(steps: list[Step], start: dict[Buf, torch.Tensor]) -> dict[Buf, np.ndarray]:
    """The buffers after the commands, by commands.vector_command, as raw bits."""
    buf = {b: x.clone() for b, x in start.items()}
    for s in steps:
        commands.vector_command(s.cmd, buf, s.t, s.stride)
    return {b: bits(x) for b, x in buf.items()}


def words(steps: list[Step], start: dict[Buf, torch.Tensor], length: int) -> np.ndarray:
    """The driver's input: every buffer written, the commands, every buffer read."""
    groups = range(0, length, LANES)
    out = [len(start) * len(groups)]
    for b, x in start.items():
        raw = np.zeros(-(-length // LANES) * LANES, dtype=np.uint32)
        raw[:length] = bits(x)
        for g in groups:
            mask = (1 << min(LANES, length - g)) - 1
            out += [int(b), g, mask, *raw[g : g + LANES].tolist()]
    out.append(len(steps))
    for s in steps:
        c = s.cmd
        out += [c.op, c.dst, c.a, c.b, c.c, c.flags, c.n, c.m, c.scalar, s.t, s.stride, int(s.wait)]
    out.append(len(start) * len(groups))
    for b in start:
        out += [x for g in groups for x in (int(b), g)]
    return np.array(out, dtype=np.uint32)


def params(length: int) -> tuple[tuple[str, int], ...]:
    return (("Len", length),)


def run(steps: list[Step], start: dict[Buf, torch.Tensor], length: int, **plusargs: int) -> Result:
    """The commands on the unit, every buffer filled with `start` first."""
    result = bulk.run_driver("vec_unit", words(steps, start, length), params(length), **plusargs)
    assert result.returncode == 0, result.stderr.decode()
    out = np.frombuffer(result.stdout, dtype="<u4")
    times = out[: 2 * len(steps)].reshape(-1, 2)
    reads = out[2 * len(steps) : -1].reshape(len(start), -1)[:, :length]
    buffers = {b: r.copy() for b, r in zip(start, reads, strict=True)}
    return Result(buffers, times[:, 0], times[:, 1], int(out[-1]))


def assert_same(got: dict[Buf, np.ndarray], want: dict[Buf, np.ndarray], label: str) -> None:
    """Every buffer bit for bit; else the first buffer and element that differ."""
    for b, w in want.items():
        bad = np.flatnonzero(got[b] != w)
        assert not len(bad), (
            f"{label}: {b.name}: {len(bad)} elements differ, first {bad[0]}: "
            f"got 0x{got[b][bad[0]]:08X}, want 0x{w[bad[0]]:08X}"
        )


def check(
    steps: list[Step], start: dict[Buf, torch.Tensor], length: int, label: str, **plusargs: int
) -> Result:
    result = run(steps, start, length, **plusargs)
    assert_same(result.buffers, golden(steps, start), label)
    return result
