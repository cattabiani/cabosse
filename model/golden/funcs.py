# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Function approximations of the numerics spec (docs/numerics.md, section 5).

rsqrt and recip: a bit-trick first guess (integer arithmetic on the FP32 bit
pattern, as in Quake III) refined by Newton steps. exp: 2^i through the
exponent field times a polynomial for 2^f. All floating-point arithmetic goes
through the primitives in `golden.arith`, so every rounding is the spec's.
Accuracy (measured, see tests): max relative error < 2^-17 for all three,
far below the BF16 rounding (2^-9) that follows every use.
"""

import torch

from golden import arith

# --- parameters (docs/numerics.md, section 7) ----------------------------------

R_RSQRT = 0x5F3759DF  # first-guess constant for 1/sqrt(x) (the Quake III constant)
N_RSQRT = 2  # Newton steps: 2^-4.9 -> 2^-9.2 -> 2^-17.7
R_RECIP = 0x7EF311C3  # first-guess constant for 1/x
N_RECIP = 2  # Newton steps: 2^-4.3 -> 2^-8.6 -> 2^-17.2
LOG2E_BITS = 0x3FB8AA3B  # f32(log2(e))
# 2^f on [-0.5, 0.5], degree 4, c0..c4 as FP32 bit patterns (Chebyshev
# interpolation rounded to FP32; pinned so every platform uses the same bits).
EXP2_COEFF_BITS = (0x3F800000, 0x3F317061, 0x3E75FD26, 0x3D650E71, 0x3C1E5FB0)
EXP2_COEFFS = list(torch.tensor(EXP2_COEFF_BITS, dtype=torch.int32).view(torch.float32))

F32_MIN_NORMAL = 2.0**-126
# Below this, rsqrt scales x by 2^24 first. 2^-125 rather than 2^-126 keeps
# h = x/2 normal, so the Newton steps also work with flush-to-zero on.
RSQRT_SCALE_BELOW = 2.0**-125


def _f32(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


def _from_bits(bits: int) -> torch.Tensor:
    return arith.f32_from_bits(torch.tensor(bits, dtype=torch.int64))


def _nan() -> torch.Tensor:
    return _from_bits(arith.NAN_F32_BITS)


# --- rsqrt ------------------------------------------------------------------------


def rsqrt(x: torch.Tensor) -> torch.Tensor:
    """1/sqrt(x). x < 2^-125 is scaled by 2^24 first and the result by 2^12."""
    assert x.dtype == torch.float32, x.dtype
    x = arith.apply_ftz(x)
    small = (x > 0) & (x < RSQRT_SCALE_BELOW)
    xs = torch.where(small, arith.mul(x, _f32(2.0**24)), x)
    y = arith.f32_from_bits((R_RSQRT - (arith.bits_f32(xs) >> 1)) & 0xFFFFFFFF)
    h = arith.mul(_f32(0.5), xs)
    for _ in range(N_RSQRT):
        t = arith.fma(-arith.mul(h, y), y, _f32(1.5))  # unary minus: exact sign flip
        y = arith.mul(y, t)
    y = torch.where(small, arith.mul(y, _f32(2.0**12)), y)
    y = torch.where(x == 0, torch.copysign(_f32(torch.inf), x), y)  # rsqrt(-0) = -inf
    y = torch.where(x == torch.inf, _f32(0.0), y)
    return torch.where((x < 0) | torch.isnan(x), _nan(), y)


# --- recip ------------------------------------------------------------------------


def recip(x: torch.Tensor) -> torch.Tensor:
    """1/x. Works on |x|; |x| < 2^-126 is scaled by 2^24 and |x| >= 2^125 by
    2^-24 first (and the result by the same factor), so the bit trick only
    sees [2^-126, 2^125)."""
    assert x.dtype == torch.float32, x.dtype
    x = arith.apply_ftz(x)
    ax = x.abs()
    small, big = ax < F32_MIN_NORMAL, ax >= 2.0**125
    xs = torch.where(small, arith.mul(ax, _f32(2.0**24)), ax)
    xs = torch.where(big, arith.mul(ax, _f32(2.0**-24)), xs)
    y = arith.f32_from_bits((R_RECIP - arith.bits_f32(xs)) & 0xFFFFFFFF)
    for _ in range(N_RECIP):
        e = arith.fma(-xs, y, _f32(1.0))
        y = arith.fma(y, e, y)
    y = torch.where(small, arith.mul(y, _f32(2.0**24)), y)
    y = torch.where(big, arith.mul(y, _f32(2.0**-24)), y)
    y = torch.copysign(y, x)
    y = torch.where(x == 0, torch.copysign(_f32(torch.inf), x), y)
    y = torch.where(torch.isinf(x), torch.copysign(_f32(0.0), x), y)
    return torch.where(torch.isnan(x), _nan(), y)


# --- exp --------------------------------------------------------------------------


def exp2_poly(f: torch.Tensor) -> torch.Tensor:
    """2^f for f in [-0.5, 0.5]: degree-4 polynomial, Horner's rule with FMA."""
    p = EXP2_COEFFS[-1]
    for c in reversed(EXP2_COEFFS[:-1]):
        p = arith.fma(p, f, c)
    return p


def exp(x: torch.Tensor) -> torch.Tensor:
    """e^x = 2^i * 2^f with t = x*log2(e), i = t rounded to nearest even, f = t - i.

    Results above the FP32 range are +inf. Results below the normal range are
    rounded once into the subnormal range (or to +0), as in IEEE arithmetic.
    """
    assert x.dtype == torch.float32, x.dtype
    x = arith.apply_ftz(x)
    t = arith.mul(x, _from_bits(LOG2E_BITS))
    # Saturate: |t| > 150 means a result above the FP32 range or below 2^-150
    # (which rounds to 0). At the clamp f = 0, so the checks below give +inf or
    # +0, also for x = ±inf.
    t = torch.where(torch.isnan(t), _f32(0.0), t.clamp(-150.0, 150.0))
    i = torch.round(t)  # round half to even
    f = arith.add(t, -i)  # exact (Sterbenz)
    p = exp2_poly(f)
    i_int = i.to(torch.int64)
    p_bits = arith.bits_f32(p)
    exponent = ((p_bits >> 23) & 0xFF) + i_int  # exponent field of p * 2^i
    y = arith.f32_from_bits((p_bits + (i_int << 23)) & 0xFFFFFFFF)
    # Below the normal range: build p * 2^(i+64) (still normal, exact), then one
    # multiply by 2^-64 rounds it once into the subnormal range.
    lifted = arith.f32_from_bits((p_bits + ((i_int + 64) << 23)) & 0xFFFFFFFF)
    y = torch.where(exponent <= 0, arith.mul(lifted, _f32(2.0**-64)), y)
    y = torch.where(exponent >= 255, _f32(torch.inf), y)
    return torch.where(torch.isnan(x), _nan(), y)
