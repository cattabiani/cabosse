# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Arithmetic primitives of the numerics spec (docs/numerics.md, section 2).

Values are torch tensors: BF16 values as `torch.bfloat16`, FP32 values as
`torch.float32`. Every operation rounds exactly once to nearest, ties to even.
Subnormals are kept (D-016), and any NaN result is the canonical NaN (D-012).

`settings.override(ftz=True)` flushes subnormal inputs and outputs to signed
zero. It is for experiments only and is not part of the spec.
"""

import torch

from golden import settings

NAN_BF16_BITS = 0x7FC0
NAN_F32_BITS = 0x7FC00000
F32_MIN_NORMAL = 2.0**-126


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


def apply_ftz(x: torch.Tensor) -> torch.Tensor:
    """FP32 subnormal -> signed zero when the flush-to-zero setting is on, else x.

    Every operation applies this to its inputs and outputs. Functions built on
    the primitives (golden.funcs) also apply it to their inputs.
    """
    if not settings.current().ftz:
        return x
    subnormal = (x != 0) & (x.abs() < F32_MIN_NORMAL)
    return torch.where(subnormal, torch.copysign(torch.zeros_like(x), x), x)


_NAN_F32 = f32_from_bits(torch.tensor(NAN_F32_BITS, dtype=torch.int64))


def _finish_f32(x: torch.Tensor) -> torch.Tensor:
    """Apply the output rules to an FP32 result: canonical NaN, optional FTZ."""
    return apply_ftz(torch.where(torch.isnan(x), _NAN_F32, x))


# --- primitives (docs/numerics.md, section 2) -------------------------------


def bf16(x: torch.Tensor) -> torch.Tensor:
    """FP32 -> BF16, round to nearest even. Overflow rounds to infinity."""
    assert x.dtype == torch.float32, x.dtype
    x = apply_ftz(x)
    u = bits_f32(x)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
    r = torch.where(torch.isnan(x), NAN_BF16_BITS, r)
    if settings.current().ftz:
        subnormal = ((r >> 7) & 0xFF == 0) & (r & 0x7F != 0)
        r = torch.where(subnormal, r & 0x8000, r)
    return bf16_from_bits(r)


def up(b: torch.Tensor) -> torch.Tensor:
    """BF16 -> FP32, exact (the 16 low bits become zero)."""
    assert b.dtype == torch.bfloat16, b.dtype
    u = bits_bf16(b)
    return _finish_f32(f32_from_bits(u << 16))


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP32 a + b, one rounding."""
    assert a.dtype == b.dtype == torch.float32, (a.dtype, b.dtype)
    return _finish_f32(apply_ftz(a) + apply_ftz(b))


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP32 a * b, one rounding."""
    assert a.dtype == b.dtype == torch.float32, (a.dtype, b.dtype)
    return _finish_f32(apply_ftz(a) * apply_ftz(b))


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
    assert a.dtype == b.dtype == c.dtype == torch.float32, (a.dtype, b.dtype, c.dtype)
    a, b, c = apply_ftz(a), apply_ftz(b), apply_ftz(c)
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


def maximum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP32 max(a, b): exact. max(-0, +0) = +0, and NaN if a or b is NaN.

    torch.maximum is not used: it returns -0 for maximum(-0, +0).
    """
    assert a.dtype == b.dtype == torch.float32, (a.dtype, b.dtype)
    a, b = apply_ftz(a), apply_ftz(b)
    y = torch.where(a > b, a, b)
    y = torch.where((a == 0) & (b == 0) & ~torch.signbit(a), a, y)  # +0 wins over -0
    y = torch.where(torch.isnan(a), a, y)  # a > b is false for NaN: pass it on
    return _finish_f32(y)


def mac(w: torch.Tensor, x: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
    """Lane step: BF16 w * BF16 x + FP32 acc, one rounding (= fma(up(w), up(x), acc))."""
    return mac_f32(up(w), up(x), acc)


def mac_f32(a: torch.Tensor, b: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
    """mac on inputs already widened: a and b are FP32 values that are exactly
    BF16 (outputs of up). Lets a caller widen once and reuse the result.

    Fast path: a BF16 x BF16 product has at most 16 significant bits, so the
    FP32 product is exact unless it underflows or overflows. When it is exact,
    a plain FP32 add rounds the exact sum once, which is what fma does. The
    slow fma emulation runs only for the rare inexact products.
    """
    p = mul(a, b)
    exact = p.to(torch.float64) == a.to(torch.float64) * b.to(torch.float64)
    fast = add(p, acc)
    if bool(exact.all()):
        return fast
    return torch.where(exact, fast, fma(a, b, acc))
