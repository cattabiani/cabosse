# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Tests for golden.funcs: accuracy, special values, and bit-exact agreement
with an independent restatement of docs/numerics.md section 5."""

import math

import numpy as np
import pytest
import torch
from golden import arith, funcs
from oracle import add_ref, fma_ref, mul_ref

SEED = 20261002
F32 = np.float32
INF = F32(np.inf)


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
    small = x < 2.0**-126
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


EXP2_COEFFS = [_from_bits(b) for b in (0x3F800000, 0x3F317061, 0x3E75FD26, 0x3D650E71, 0x3C1E5FB0)]


def spec_exp(x: np.float32) -> np.float32:
    if np.isnan(x):
        return F32(np.nan)
    # inf * log2(e) is exactly inf (the oracle cannot represent inf).
    t = np.copysign(INF, x) if np.isinf(x) else mul_ref(x, _from_bits(0x3FB8AA3B))
    t = F32(min(max(t, -200.0), 200.0))
    i = round(float(t))  # Python rounds half to even
    f = add_ref(t, F32(-i))
    p = EXP2_COEFFS[4]
    for c in reversed(EXP2_COEFFS[:4]):
        p = fma_ref(p, f, c)
    exponent = ((_bits(p) >> 23) & 0xFF) + i
    if exponent >= 255:
        return INF
    if exponent <= 0:
        return F32(0.0)
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
    """Random x over the normal-result range. Measured max: 2^-17.1 on 20M samples;
    the bound has a little slack for this smaller sample."""
    rng = np.random.default_rng(SEED)
    x = t32(F32(rng.uniform(-87.3, 88.7, 1 << 22)))
    assert max_rel_err(funcs.exp(x), torch.exp(x.double())) < 2**-16.5


def test_exp_range_limits() -> None:
    x = t32(F32([88.7, 88.8, -87.3, -88.0, -1000.0, 1000.0]))
    y = funcs.exp(x)
    assert math.isfinite(y[0].item()) and y[1].item() == math.inf
    assert y[2].item() > 0 and y[3].item() == 0.0  # below 2^-126: +0
    assert y[4].item() == 0.0 and y[5].item() == math.inf


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
