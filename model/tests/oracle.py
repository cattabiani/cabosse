# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Slow, independent reference for FP32 rounding: exact rational arithmetic
(`fractions.Fraction`) rounded to FP32 by direct comparison. Shares no code
with `golden.arith`."""

from fractions import Fraction

import numpy as np

F32 = np.float32
_MAX = Fraction(float(np.finfo(F32).max))
_OVERFLOW = _MAX + Fraction(2) ** 103  # max + half an ulp: rounds to inf (ties to even)


def _even(x: np.float32) -> bool:
    return int(np.array(x, F32).view(np.uint32)) & 1 == 0


def round_f32(v: Fraction, zero_sign_negative: bool = False) -> np.float32:
    """Correctly rounded (to nearest, ties to even) FP32 value of exact `v`.

    `zero_sign_negative` gives the sign of an exact zero result.
    """
    if v == 0:
        return F32(-0.0) if zero_sign_negative else F32(0.0)
    if abs(v) >= _OVERFLOW:
        return F32(np.inf) if v > 0 else F32(-np.inf)
    guess = F32(min(max(float(v), -float(_MAX)), float(_MAX)))  # within 1 ulp
    with np.errstate(over="ignore"):  # stepping past max gives inf, filtered below
        cands = [guess, np.nextafter(guess, F32(-np.inf)), np.nextafter(guess, F32(np.inf))]
    cands = [c for c in cands if np.isfinite(c)]
    best = min(cands, key=lambda c: (abs(Fraction(float(c)) - v), not _even(c)))
    if best == 0:  # underflow to zero keeps the sign of v
        best = F32(-0.0) if v < 0 else F32(0.0)
    return best


def _neg(x: np.float32) -> bool:
    return bool(np.signbit(x))


# The exact arithmetic cannot represent inf or NaN. With a non-finite operand
# the result is inf or NaN, and IEEE float64 arithmetic gives the same one as
# FP32: the inf/NaN rules do not depend on precision.


def _finite(*xs: np.float32) -> bool:
    return all(np.isfinite(x) for x in xs)


def fma_ref(a: np.float32, b: np.float32, c: np.float32) -> np.float32:
    """Correctly rounded a*b + c."""
    if not _finite(a, b, c):
        with np.errstate(invalid="ignore", over="ignore"):
            return F32(np.float64(a) * np.float64(b) + np.float64(c))
    p = Fraction(float(a)) * Fraction(float(b))
    v = p + Fraction(float(c))
    p_neg = _neg(a) != _neg(b)
    return round_f32(v, zero_sign_negative=(p == 0 and p_neg and c == 0 and _neg(c)))


def add_ref(a: np.float32, b: np.float32) -> np.float32:
    if not _finite(a, b):
        with np.errstate(invalid="ignore"):
            return F32(np.float64(a) + np.float64(b))
    v = Fraction(float(a)) + Fraction(float(b))
    return round_f32(v, zero_sign_negative=(a == 0 and b == 0 and _neg(a) and _neg(b)))


def mul_ref(a: np.float32, b: np.float32) -> np.float32:
    if not _finite(a, b):
        with np.errstate(invalid="ignore", over="ignore"):
            return F32(np.float64(a) * np.float64(b))
    v = Fraction(float(a)) * Fraction(float(b))
    return round_f32(v, zero_sign_negative=(_neg(a) != _neg(b)))
