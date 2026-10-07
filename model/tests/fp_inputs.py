# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""FP32 test inputs shared by the golden-model tests (test_arith) and the RTL
tests (verif/tests): special values as bit patterns, and random families
that stress rounding."""

import numpy as np
import torch
from golden import arith

F32 = np.float32
# Zeros, the subnormal and normal boundaries, one and its neighbour, the
# largest finite value, infinities, the canonical NaN and a signalling NaN;
# both signs where it applies.
F32_SPECIAL_BITS = [
    sign | bits
    for bits in (
        0x0000_0000,  # zero
        0x0000_0001,  # smallest subnormal
        0x007F_FFFF,  # largest subnormal
        0x0080_0000,  # smallest normal
        0x3F80_0000,  # 1
        0x3F80_0001,  # 1 + 2^-23
        0x7F7F_FFFF,  # largest finite
        0x7F80_0000,  # infinity
    )
    for sign in (0, 0x8000_0000)
] + [arith.NAN_F32_BITS, 0x7F80_0001]  # canonical NaN, a signalling NaN


def special_triples() -> np.ndarray:
    """Every (a, b, c) triple of the special values, as uint32 rows."""
    s = np.array(F32_SPECIAL_BITS, dtype=np.uint32)
    return np.stack([x.ravel() for x in np.meshgrid(s, s, s, indexing="ij")], axis=1)


def all_bf16_widened() -> np.ndarray:
    """golden.arith.up of every BF16 bit pattern, as FP32 bits (index = the
    BF16 bits)."""
    v = arith.bf16_from_bits(torch.arange(2**16, dtype=torch.int64))
    return arith.bits_f32(arith.up(v)).numpy().astype(np.uint32)


def random_bits(rng: np.random.Generator, shape) -> np.ndarray:
    """Uniformly random 32-bit patterns."""
    return rng.integers(0, 2**32, shape, dtype=np.uint32)


def moderate(rng: np.random.Generator, n: int) -> np.ndarray:
    """Values around 1 with random signs: the common case."""
    sign = rng.choice([-1.0, 1.0], n)
    return F32(sign * rng.uniform(1, 2, n) * 2.0 ** rng.integers(-20, 21, n))


def any_finite(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniformly random finite bit patterns: extreme magnitudes, subnormals."""
    x = random_bits(rng, n).view(F32)
    return np.where(np.isfinite(x), x, F32(1.0))


def families(rng: np.random.Generator, n: int) -> dict[str, tuple[np.ndarray, ...]]:
    """(a, b, c) triples that stress different parts of the rounding."""

    def sign() -> np.ndarray:
        return rng.choice([-1.0, 1.0], n)

    def mag(lo: int, hi: int) -> np.ndarray:
        """Random magnitudes in [2**lo, 2**hi)."""
        return rng.uniform(1, 2, n) * 2.0 ** rng.integers(lo, hi, n)

    a, b, c = moderate(rng, n), moderate(rng, n), moderate(rng, n)
    fam = {"moderate": (a, b, c), "any_finite": tuple(any_finite(rng, n) for _ in range(3))}

    # Cancellation: c is close to -(a*b), so most leading bits cancel.
    p = (a.astype(np.float64) * b).astype(F32)
    k = rng.integers(-3, 4, n)
    fam["cancellation"] = (a, b, step_ulps(-p, k))

    # Exact ties: a*b is exactly half an ulp of c (a is a power of two).
    e = rng.integers(-100, 100, n)
    c_t = F32(sign() * rng.uniform(1, 2, n) * 2.0**e)
    i = rng.integers(-10, 11, n)
    a_t = F32(2.0**i)
    b_t = F32(sign() * 2.0 ** (e - 24 - i))
    fam["ties"] = (a_t, b_t, c_t)

    # Near ties: b moved by one of its own ulps, so a*b is just above or below
    # half an ulp of c.
    fam["near_ties"] = (a_t, step_ulps(b_t, rng.choice([-1, 1], n)), c_t)

    # Double-rounding traps: a*b = (1 + 2**-23)(1 - 2**-23) * half-ulp(c)
    # = half-ulp(c) * (1 - 2**-46). The exact sum is a hair from a tie, closer
    # than float64 can see, so a plain float64 FMA rounds twice and fails.
    a_dr = F32((1 + 2.0**-23) * 2.0**i)
    b_dr = F32(sign() * (1 - 2.0**-23) * 2.0 ** (e - 24 - i))
    fam["double_rounding"] = (a_dr, b_dr, c_t)

    # Subnormal results and products that underflow FP32.
    c_tiny = F32(sign() * 2.0 ** rng.integers(-149, -126, n))
    fam["tiny"] = (F32(mag(-80, -60)), F32(sign() * mag(-80, -60)), c_tiny)

    # Results near the overflow threshold.
    fam["huge"] = (F32(mag(60, 68)), F32(mag(60, 68)), F32(sign() * 2.0**127))
    return fam


def step_ulps(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Move each x by k[i] FP32 ulps."""
    out = np.asarray(x, F32).copy()
    for direction in (1, -1):
        for _ in range(int(np.abs(k).max())):
            m = (np.sign(k) == direction) & (np.abs(k) > 0)
            out[m] = np.nextafter(out[m], F32(direction * np.inf))
            k = np.where(m, k - direction, k)
    return out
