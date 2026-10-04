# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Tests for golden.funcs: accuracy, special values, and bit-exact agreement
with an independent restatement of docs/numerics.md section 5."""

import math

import numpy as np
import pytest
import torch
from golden import arith, funcs, settings
from oracle import add_ref, fma_ref, mul_ref

SEED = 20261002
F32 = np.float32
INF = F32(np.inf)
EXP2_COEFFS = list(
    np.array([0x3F800000, 0x3F317061, 0x3E75FD26, 0x3D650E71, 0x3C1E5FB0], np.uint32).view(F32)
)


def t32(x) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, F32))


def all_floats(lo: float, hi: float) -> torch.Tensor:
    """Every FP32 value in [lo, hi), for 0 <= lo < hi."""
    a, b = (int(np.array(v, F32).view(np.uint32)) for v in (lo, hi))
    return arith.f32_from_bits(torch.arange(a, b, dtype=torch.int64))


def log_uniform(rng: np.random.Generator, n: int, lo: int, hi: int) -> np.ndarray:
    """Positive values with exponents spread over [lo, hi)."""
    return F32(rng.uniform(1, 2, n) * 2.0 ** rng.integers(lo, hi, n))


def max_rel_err(y: torch.Tensor, ref: torch.Tensor) -> float:
    return ((y.double() - ref) / ref).abs().max().item()


# --- independent restatement of the spec (scalar, exact oracle arithmetic) --------
# Constants are copied from docs/numerics.md on purpose, not imported.


def _bits(x) -> int:
    return int(np.array(x, F32).view(np.uint32))


def _from_bits(u: int) -> np.float32:
    return np.array(u & 0xFFFFFFFF, np.uint32).view(F32)[()]


def spec_rsqrt(x: np.float32) -> np.float32:
    if np.isnan(x) or x < 0:
        return F32(np.nan)
    if x == 0:
        return np.copysign(INF, x)
    if x == INF:
        return F32(0.0)
    small = x < 2.0**-125
    xs = mul_ref(x, F32(2.0**24)) if small else x
    y = _from_bits(0x5F3759DF - (_bits(xs) >> 1))
    h = mul_ref(F32(0.5), xs)
    for _ in range(2):
        y = mul_ref(y, fma_ref(-mul_ref(h, y), y, F32(1.5)))
    return mul_ref(y, F32(2.0**12)) if small else y


def spec_recip(x: np.float32) -> np.float32:
    if np.isnan(x):
        return F32(np.nan)
    if x == 0:
        return np.copysign(INF, x)
    if np.isinf(x):
        return np.copysign(F32(0.0), x)
    ax = abs(x)
    scale = F32(2.0**24) if ax < 2.0**-126 else F32(2.0**-24) if ax >= 2.0**125 else None
    xs = mul_ref(ax, scale) if scale is not None else ax
    y = _from_bits(0x7EF311C3 - _bits(xs))
    for _ in range(2):
        y = fma_ref(y, fma_ref(-xs, y, F32(1.0)), y)
    y = mul_ref(y, scale) if scale is not None else y
    return np.copysign(y, x)


def spec_exp(x: np.float32) -> np.float32:
    if np.isnan(x):
        return F32(np.nan)
    # inf * log2(e) is exactly inf (the oracle cannot represent inf).
    t = np.copysign(INF, x) if np.isinf(x) else mul_ref(x, _from_bits(0x3FB8AA3B))
    t = F32(min(max(t, -150.0), 150.0))
    i = round(float(t))  # Python rounds half to even
    f = add_ref(t, F32(-i))
    p = EXP2_COEFFS[4]
    for c in reversed(EXP2_COEFFS[:4]):
        p = fma_ref(p, f, c)
    exponent = ((_bits(p) >> 23) & 0xFF) + i
    if exponent >= 255:
        return INF
    if exponent <= 0:  # one rounding into the subnormal range
        return mul_ref(_from_bits(_bits(p) + ((i + 64) << 23)), F32(2.0**-64))
    return _from_bits(_bits(p) + (i << 23))


def _conformance_inputs(name: str) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    specials = F32([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 2.0**-149, 2.0**-126])
    if name == "exp":
        return np.concatenate(
            [F32(rng.uniform(-110, 95, 3000)), F32(rng.uniform(-1, 1, 500)), specials]
        )
    x = np.concatenate(
        [log_uniform(rng, 2000, -126, 128), log_uniform(rng, 300, -149, -126), specials]
    )
    return np.concatenate([x, -x]) if name == "recip" else x


@pytest.mark.parametrize(
    "name, golden, spec",
    [
        ("rsqrt", funcs.rsqrt, spec_rsqrt),
        ("recip", funcs.recip, spec_recip),
        ("exp", funcs.exp, spec_exp),
    ],
)
def test_matches_spec_bit_exactly(name, golden, spec) -> None:
    x = _conformance_inputs(name)
    got = arith.bits_f32(golden(t32(x))).numpy()
    want = np.array([spec(v) for v in x], F32)
    want_bits = np.where(np.isnan(want), arith.NAN_F32_BITS, want.view(np.uint32))
    bad = np.nonzero(got != want_bits)[0]
    assert not len(bad), (
        f"{name}: {len(bad)} mismatches (seed {SEED}), first x={x[bad[0]]!r}: "
        f"golden {got[bad[0]]:#010x}, spec {want_bits[bad[0]]:#010x}"
    )


# --- accuracy ----------------------------------------------------------------------
# Bounds: 2^-17. rsqrt and recip are exhaustive over one period of the bit-trick
# error (measured maxima 2^-17.69 and 2^-17.22).


def test_rsqrt_accuracy_exhaustive_period() -> None:
    x = all_floats(1.0, 4.0)  # the error pattern repeats every two binades
    assert max_rel_err(funcs.rsqrt(x), 1 / x.double().sqrt()) < 2**-17


def test_recip_accuracy_exhaustive_period() -> None:
    x = all_floats(1.0, 2.0)  # the error pattern repeats every binade
    assert max_rel_err(funcs.recip(x), 1 / x.double()) < 2**-17


def test_rsqrt_and_recip_accuracy_full_range() -> None:
    """Including the scaled paths (subnormal inputs; huge inputs for recip)."""
    rng = np.random.default_rng(SEED)
    x = t32(
        np.concatenate(
            [log_uniform(rng, 1 << 18, -126, 128), log_uniform(rng, 1 << 16, -149, -126)]
        )
    )
    assert max_rel_err(funcs.rsqrt(x), 1 / x.double().sqrt()) < 2**-17
    ref = 1 / x.double()
    err = (funcs.recip(x).double() - ref).abs()
    # Results below 2^-126 are subnormal and have fewer bits: allow half a
    # subnormal ulp of rounding on top of the relative bound. Overflow is inf.
    finite = ref < float(np.finfo(F32).max)
    assert (err[finite] <= 2**-17 * ref[finite] + 2**-150).all()
    assert torch.isinf(funcs.recip(x)[~finite]).all()


def test_exp_accuracy() -> None:
    """Random x with normal results (the exhaustive version is below)."""
    rng = np.random.default_rng(SEED)
    x = t32(F32(rng.uniform(-87.3, 88.7, 1 << 22)))
    assert max_rel_err(funcs.exp(x), torch.exp(x.double())) < 2**-17


def test_exp_accuracy_subnormal_results_exhaustive() -> None:
    """Every FP32 x in [-104, -87.3] (about 2.2M inputs; results subnormal or
    near the bottom of the normal range).

    Here |t| = |x*log2(e)| >= 128, so t is rounded to a coarser grid (spacing
    2^-16) and exp's relative error grows to 2^-16.63 (measured). Bound:
    2^-16.5 relative plus half a subnormal ulp (2^-150) for the final rounding.
    """
    lo, hi = (int(np.array(v, F32).view(np.uint32)) for v in (87.3, 104.0))
    x = -arith.f32_from_bits(torch.arange(lo, hi + 1, dtype=torch.int64))
    ref = torch.exp(x.double())
    err = (funcs.exp(x).double() - ref).abs()
    assert (err <= 2**-16.5 * ref + 2**-150).all()


@pytest.mark.slow
def test_exp_accuracy_exhaustive() -> None:
    """Every FP32 x in [-87.3, 88.7] (normal results), in chunks: about 2 billion
    inputs, a few minutes. Measured max: 2^-17.08 at x = -81.44."""
    for sign, hi in ((1.0, 88.7), (-1.0, 87.3)):
        top = int(np.array(hi, F32).view(np.uint32))
        for start in range(0, top + 1, 1 << 24):
            u = torch.arange(start, min(start + (1 << 24), top + 1), dtype=torch.int64)
            x = arith.f32_from_bits(u) * sign
            assert max_rel_err(funcs.exp(x), torch.exp(x.double())) < 2**-17, (sign, start)


def test_exp_range_limits() -> None:
    x = t32(F32([88.7, 88.8, -87.3, -88.0, -103.0, -104.0, -1000.0, 1000.0]))
    y = funcs.exp(x).tolist()
    assert math.isfinite(y[0]) and y[1] == math.inf
    assert y[2] >= 2.0**-126  # still normal
    assert 0 < y[3] < 2.0**-126 and 0 < y[4] < 2.0**-126  # subnormal, not flushed
    assert y[5] == 0.0  # e^-104 < 2^-150: rounds to 0
    assert y[6] == 0.0 and y[7] == math.inf


# --- special values -------------------------------------------------------------------


def _bits_of(y: torch.Tensor) -> list[int]:
    return arith.bits_f32(y).tolist()


def test_special_values() -> None:
    nan = arith.NAN_F32_BITS
    x = t32(F32([0.0, -0.0, np.inf, -np.inf, np.nan]))
    assert _bits_of(funcs.rsqrt(x)) == [0x7F800000, 0xFF800000, 0, nan, nan]
    assert _bits_of(funcs.rsqrt(t32(F32([-1.0])))) == [nan]
    assert _bits_of(funcs.recip(x)) == [0x7F800000, 0xFF800000, 0, 0x80000000, nan]
    e = funcs.exp(x)
    assert _bits_of(e)[:5] == [0x3F800000, 0x3F800000, 0x7F800000, 0, nan]


def test_torch_round_is_half_to_even() -> None:
    """exp relies on torch.round rounding ties to even."""
    x = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5], dtype=torch.float32)
    assert torch.round(x).tolist() == [0.0, 2.0, 2.0, -0.0, -2.0]


def test_recip_is_approximate_even_at_one() -> None:
    """x = 1 is the worst case of the bit trick: 1/1 = 0.9999935 (109 ulps low,
    error 2^-17.2). Sign is kept. After BF16 rounding it is exactly 1."""
    y = funcs.recip(t32(F32([1.0, -1.0])))
    assert abs(abs(y[0].item()) - 1) < 2**-17 and y[1].item() == -y[0].item()
    assert arith.bf16(y)[0].item() == 1.0


def test_flush_to_zero_does_not_change_normal_results() -> None:
    """With FTZ on, inputs and results that are normal must give the same bits:
    no intermediate value may be subnormal. Covers the lowest binades for rsqrt,
    where h = x/2 would otherwise be subnormal."""
    rng = np.random.default_rng(SEED)
    normal = log_uniform(rng, 1 << 16, -126, 126)
    cases = [
        (funcs.rsqrt, torch.cat([all_floats(2.0**-126, 2.0**-123), t32(normal)])),
        (funcs.recip, t32(np.concatenate([normal, -normal]))),
        (funcs.exp, t32(F32(rng.uniform(-87.3, 88.7, 1 << 16)))),
    ]
    for func, x in cases:
        plain = func(x)
        with settings.override(ftz=True):
            flushed = func(x)
        keep = plain.abs() >= 2.0**-126  # normal results only
        same = arith.bits_f32(plain)[keep] == arith.bits_f32(flushed)[keep]
        assert same.all(), f"{func.__name__}: {int((~same).sum())} results differ under FTZ"
