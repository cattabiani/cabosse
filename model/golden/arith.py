# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Arithmetic primitives of the numerics spec (docs/numerics.md, section 2).

Values are torch tensors: BF16 values as `torch.bfloat16`, FP32 values as
`torch.float32`. Every operation rounds exactly once to nearest, ties to even.
Subnormals are kept (D-016), and any NaN result is the canonical NaN (D-012).

`flush_to_zero()` switches to flushing subnormal inputs and outputs to signed
zero. It is for experiments only and is not part of the spec.
"""

from collections.abc import Generator
from contextlib import contextmanager

import torch

NAN_BF16_BITS = 0x7FC0
NAN_F32_BITS = 0x7FC00000
F32_MIN_NORMAL = 2.0**-126

_ftz = False


@contextmanager
def flush_to_zero(enabled: bool = True) -> Generator[None]:
    """Flush subnormal inputs and outputs to signed zero inside this block."""
    global _ftz
    old, _ftz = _ftz, enabled
    try:
        yield
    finally:
        _ftz = old


# --- bit patterns -----------------------------------------------------------
# Bit patterns are held in int64 tensors (torch's unsigned types support few
# operations), always in the unsigned range.


def bits_f32(x: torch.Tensor) -> torch.Tensor:
    """FP32 values -> their 32-bit patterns, as int64 in [0, 2**32)."""
    return x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def f32_from_bits(u: torch.Tensor) -> torch.Tensor:
    """32-bit patterns (int64 in [0, 2**32)) -> FP32 values."""
    return torch.where(u >= 2**31, u - 2**32, u).to(torch.int32).view(torch.float32)


def bits_bf16(b: torch.Tensor) -> torch.Tensor:
    """BF16 values -> their 16-bit patterns, as int64 in [0, 2**16)."""
    return b.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF


def bf16_from_bits(u: torch.Tensor) -> torch.Tensor:
    """16-bit patterns (int64 in [0, 2**16)) -> BF16 values."""
    return torch.where(u >= 2**15, u - 2**16, u).to(torch.int16).view(torch.bfloat16)


# --- helpers ----------------------------------------------------------------


def _check(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if x.dtype != dtype:
        raise TypeError(f"expected {dtype}, got {x.dtype}")
    return x


def _flush_f32(x: torch.Tensor) -> torch.Tensor:
    """Subnormal -> signed zero, only when flush-to-zero is enabled."""
    if not _ftz:
        return x
    subnormal = (x != 0) & (x.abs() < F32_MIN_NORMAL)
    return torch.where(subnormal, torch.copysign(torch.zeros_like(x), x), x)


def _finish_f32(x: torch.Tensor) -> torch.Tensor:
    """Apply the output rules to an FP32 result: canonical NaN, optional FTZ."""
    nan = f32_from_bits(torch.tensor(NAN_F32_BITS, dtype=torch.int64))
    return _flush_f32(torch.where(torch.isnan(x), nan, x))


# --- primitives (docs/numerics.md, section 2) -------------------------------


def bf16(x: torch.Tensor) -> torch.Tensor:
    """FP32 -> BF16, round to nearest even. Overflow rounds to infinity."""
    x = _flush_f32(_check(x, torch.float32))
    u = bits_f32(x)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
    r = torch.where(torch.isnan(x), NAN_BF16_BITS, r)
    if _ftz:
        subnormal = ((r >> 7) & 0xFF == 0) & (r & 0x7F != 0)
        r = torch.where(subnormal, r & 0x8000, r)
    return bf16_from_bits(r)


def up(b: torch.Tensor) -> torch.Tensor:
    """BF16 -> FP32, exact (the 16 low bits become zero)."""
    u = bits_bf16(_check(b, torch.bfloat16))
    return _finish_f32(f32_from_bits(u << 16))


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP32 a + b, one rounding."""
    a, b = _flush_f32(_check(a, torch.float32)), _flush_f32(_check(b, torch.float32))
    return _finish_f32(a + b)


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP32 a * b, one rounding."""
    a, b = _flush_f32(_check(a, torch.float32)), _flush_f32(_check(b, torch.float32))
    return _finish_f32(a * b)


def fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """FP32 a * b + c with one rounding (D-019).

    torch has no exact fused multiply-add for tensors, so it is emulated:
    1. a*b is exact in float64 (24 + 24 significant bits fit in 53, and the
       exponent range fits).
    2. p + c is rounded to float64 and its exact error is recovered with
       TwoSum. The float64 sum is then moved to "round to odd": if inexact
       and its last bit is even, step one float64 ulp toward the exact value.
    3. Rounding a round-to-odd float64 to FP32 gives the correctly rounded
       result, because float64 has at least 2 more bits than FP32 (Boldo &
       Melquiond, 2008). Plain float64 would round twice and be wrong in rare
       cases.
    """
    a, b, c = (_flush_f32(_check(t, torch.float32)) for t in (a, b, c))
    a64, b64, c64 = (t.to(torch.float64) for t in (a, b, c))
    p = a64 * b64
    s = p + c64
    bv = s - p  # TwoSum (Knuth): e = p + c - s, exactly
    e = (p - (s - bv)) + (c64 - bv)
    inexact = torch.isfinite(s) & (e != 0)
    even = (s.view(torch.int64) & 1) == 0
    toward = torch.where(e > 0, torch.inf, -torch.inf).to(torch.float64)
    s = torch.where(inexact & even, torch.nextafter(s, toward), s)
    return _finish_f32(s.to(torch.float32))


def mac(w: torch.Tensor, x: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
    """Lane step: BF16 w * BF16 x + FP32 acc, one rounding."""
    return fma(up(w), up(x), acc)
