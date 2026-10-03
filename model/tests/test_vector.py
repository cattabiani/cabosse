# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Tests for arith.maximum and golden.vector: bit-exact agreement with an
independent restatement of docs/numerics.md sections 2 and 4, and accuracy
against float64."""

import functools
from fractions import Fraction

import numpy as np
import pytest
import torch
from golden import arith, settings, vector
from oracle import add_ref, fma_ref, mul_ref, round_f32
from test_funcs import max_rel_err, spec_exp, spec_recip, spec_rsqrt, t32

SEED = 20261003
F32 = np.float32
NAN_BITS = arith.NAN_F32_BITS


def bits_of(y) -> list[int]:
    """FP32 values (tensor or array) -> bit patterns, NaN -> canonical NaN."""
    y = np.asarray(y, F32).reshape(-1)
    return [NAN_BITS if np.isnan(v) else int(v.view(np.uint32)) for v in y]


def moderate(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """FP32 values around 1 with random signs, plus some signed zeros."""
    v = (
        rng.choice([-1.0, 1.0], shape)
        * rng.uniform(1, 2, shape)
        * 2.0 ** rng.integers(-8, 9, shape)
    )
    v[rng.random(shape) < 0.05] = 0.0
    v[rng.random(shape) < 0.02] = -0.0
    return F32(v)


def extreme(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """FP32 values over the whole exponent range (sums and squares overflow and
    underflow), subnormals, a few infinities and NaNs."""
    v = (
        rng.choice([-1.0, 1.0], shape)
        * rng.uniform(1, 2, shape)
        * 2.0 ** rng.integers(-149, 128, shape)
    )
    v[rng.random(shape) < 0.02] = np.inf
    v[rng.random(shape) < 0.02] = -np.inf
    v[rng.random(shape) < 0.01] = np.nan
    return F32(v)


# --- independent restatement of the spec (scalar, exact oracle arithmetic) --------


def spec_max(a: np.float32, b: np.float32) -> np.float32:
    if np.isnan(a) or np.isnan(b):
        return F32(np.nan)
    if a == 0 and b == 0:
        return F32(-0.0) if np.signbit(a) and np.signbit(b) else F32(0.0)
    return a if a > b else b


def spec_reduce(x: np.ndarray, step, width: int) -> np.float32:
    acc = [F32(0.0)] * width
    for i, v in enumerate(x):
        acc[i % width] = step(v, acc[i % width])
    while len(acc) > 1:
        acc = [add_ref(acc[i], acc[i + 1]) for i in range(0, len(acc), 2)]
    return acc[0]


def spec_sum(x: np.ndarray, width: int) -> np.float32:
    return spec_reduce(x, add_ref, width)


def spec_sum_squares(x: np.ndarray, width: int) -> np.float32:
    return spec_reduce(x, lambda v, acc: fma_ref(v, v, acc), width)


def spec_rmsnorm(x: np.ndarray, g: np.ndarray, eps: float, width: int) -> list[np.float32]:
    """g: FP32 values that are exactly BF16 (up() is exact)."""
    inv_n = round_f32(Fraction(1, len(x)))  # f32(1/n), exact rational rounded once
    var = mul_ref(spec_sum_squares(x, width), inv_n)
    r = spec_rsqrt(add_ref(var, F32(eps)))
    return [mul_ref(gi, mul_ref(xi, r)) for xi, gi in zip(x, g, strict=True)]


def spec_softmax(s: np.ndarray, width: int) -> list[np.float32]:
    m = functools.reduce(spec_max, s)
    e = [spec_exp(add_ref(si, -m)) for si in s]
    z = spec_sum(np.array(e, F32), width)
    rz = spec_recip(z)
    return [mul_ref(ei, rz) for ei in e]


# --- max ----------------------------------------------------------------------------

SPECIALS = F32(
    [-np.inf, -3.0, -1.0, -(2.0**-149), -0.0, 0.0, 2.0**-149, 2.0**-126, 1.0, 3.0, np.inf, np.nan]
)


def test_maximum_all_pairs_of_special_values() -> None:
    a, b = np.meshgrid(SPECIALS, SPECIALS)
    a, b = a.reshape(-1), b.reshape(-1)
    got = bits_of(arith.maximum(t32(a), t32(b)))
    assert got == bits_of([spec_max(x, y) for x, y in zip(a, b, strict=True)])


def test_maximum_signed_zeros_and_nan() -> None:
    """The cases torch.maximum gets differently (it gives -0 for (-0, +0))."""
    neg_nan = arith.f32_from_bits(torch.tensor([0xFFFFFFFF]))  # what torch's own ops produce
    assert bits_of(arith.maximum(t32([-0.0]), t32([0.0]))) == [0]
    assert bits_of(arith.maximum(t32([0.0]), t32([-0.0]))) == [0]
    assert bits_of(arith.maximum(t32([-0.0]), t32([-0.0]))) == [0x80000000]
    assert bits_of(arith.maximum(neg_nan, t32([1.0]))) == [NAN_BITS]
    assert bits_of(arith.maximum(t32([1.0]), neg_nan)) == [NAN_BITS]


@pytest.mark.parametrize("family", ["moderate", "extreme"])
def test_reduce_max_matches_spec_in_any_order(family: str) -> None:
    rng = np.random.default_rng(SEED)
    make = moderate if family == "moderate" else extreme
    for length in range(1, 40):
        x = make(rng, (6, length))
        got = bits_of(vector.reduce_max(t32(x)))
        for row in range(6):
            want = bits_of([functools.reduce(spec_max, rng.permutation(x[row]))])
            assert got[row] == want[0], f"{family}, n={length}, row {row} (seed {SEED})"


def test_reduce_max_signed_zeros_and_single_nan() -> None:
    assert bits_of(vector.reduce_max(t32([-0.0, 0.0, -0.0]))) == [0]
    assert bits_of(vector.reduce_max(t32([-0.0, -0.0]))) == [0x80000000]
    neg_nan = arith.f32_from_bits(torch.tensor([0xFFFFFFFF]))
    assert bits_of(vector.reduce_max(neg_nan)) == [NAN_BITS]  # n = 1: still canonical


# --- sums -------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["moderate", "extreme"])
@pytest.mark.parametrize("width", [1, 2, 4, 8, 16])
@pytest.mark.parametrize(
    "golden, spec",
    [(vector.reduce_sum, spec_sum), (vector.reduce_sum_squares, spec_sum_squares)],
    ids=["sum", "sum_squares"],
)
def test_sums_match_spec_bit_exactly(golden, spec, width: int, family: str) -> None:
    """Lengths 1..40 include n < S and n not a multiple of S."""
    rng = np.random.default_rng(SEED + width)
    make = moderate if family == "moderate" else extreme
    for length in range(1, 41):
        x = make(rng, (4, length))
        got = bits_of(golden(t32(x), width))
        want = bits_of([spec(row, width) for row in x])
        assert got == want, f"S={width}, n={length}, {family} (seed {SEED + width})"


def test_sum_order_is_the_spec_order() -> None:
    """Values where the order matters. With S = 2: partials (2^24 + 1) -> 2^24
    and (1 - 2^24) exact, then 2^24 + (1 - 2^24) = 1. Left to right gives 0."""
    x = t32([2.0**24, 1.0, 1.0, -(2.0**24)])
    assert vector.reduce_sum(x, 2).item() == 1.0
    assert vector.reduce_sum(x, 1).item() == 0.0


def test_reductions_reject_bad_arguments() -> None:
    x = torch.zeros(4, dtype=torch.float32)
    for bad in (
        lambda: vector.reduce_sum(x, 6),  # not a power of two
        lambda: vector.reduce_sum(x[:0]),  # n = 0 is not defined
        lambda: vector.reduce_max(x[:0]),
        lambda: vector.reduce_sum(x.to(torch.bfloat16)),
    ):
        with pytest.raises(AssertionError):
            bad()


# --- RMSNorm ----------------------------------------------------------------------


def bf16_weights(rng: np.random.Generator, n: int) -> torch.Tensor:
    return arith.bf16(t32(rng.uniform(0.5, 2.0, n) * rng.choice([-1.0, 1.0], n)))


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("width", [1, 8])
def test_rmsnorm_matches_spec_bit_exactly(width: int, eps: float) -> None:
    """n = 576 is SmolLM2's hidden size (1/576 is not exact in FP32)."""
    rng = np.random.default_rng(SEED + width)
    for n in (1, 3, 8, 37, 64, 576):
        x, g = moderate(rng, (2, n)), bf16_weights(rng, n)
        got = bits_of(vector.rmsnorm(t32(x), g, eps, width))
        g32 = g.to(torch.float32).numpy()
        want = bits_of([v for row in x for v in spec_rmsnorm(row, g32, eps, width)])
        assert got == want, f"S={width}, n={n}, eps={eps} (seed {SEED + width})"


def test_rmsnorm_accuracy() -> None:
    """Relative error per element <= 2^-16 against float64 with the same eps.

    Bound: the sum of squares has only positive terms, so its relative error is
    at most gamma_k with k = n/S + log2(S) = 75 roundings (2^-17.8); 1/n, the
    multiply and the add add 3 * 2^-24; rsqrt adds 2^-17.7 and halves the error
    of its input; the two final multiplies add 2 * 2^-24. Total about 2^-17;
    2^-16 leaves a factor-2 margin."""
    rng = np.random.default_rng(SEED)
    n, eps = 576, 1e-5
    x = F32(rng.standard_normal((64, n)) * 2.0 ** rng.integers(-6, 7, (64, 1)))
    g = bf16_weights(rng, n)
    y = vector.rmsnorm(t32(x), g, eps).double()
    x64, g64 = torch.from_numpy(x).double(), g.double()
    ref = g64 * x64 / torch.sqrt((x64 * x64).mean(-1, keepdim=True) + eps)
    assert max_rel_err(y, ref) <= 2.0**-16


def test_rmsnorm_of_zero_vector_is_zero() -> None:
    """ss = 0, so r = rsqrt(eps) is finite and every output is a zero."""
    y = vector.rmsnorm(torch.zeros(576), arith.bf16(torch.ones(576)), 1e-5)
    assert (y == 0).all()


def test_rmsnorm_rejects_weight_of_wrong_shape() -> None:
    """A [1] weight would broadcast silently over all elements."""
    x = torch.ones(8)
    for g in (torch.ones(1), torch.ones(4), torch.ones(2, 8)):
        with pytest.raises(AssertionError):
            vector.rmsnorm(x, g.to(torch.bfloat16), 1e-5)


# --- softmax ----------------------------------------------------------------------


@pytest.mark.parametrize("width", [1, 8])
def test_softmax_matches_spec_bit_exactly(width: int) -> None:
    rng = np.random.default_rng(SEED + width)
    for n in (1, 2, 5, 8, 33, 100):
        s = F32(rng.uniform(-30, 30, (3, n)))
        got = bits_of(vector.softmax(t32(s), width))
        want = bits_of([v for row in s for v in spec_softmax(row, width)])
        assert got == want, f"S={width}, n={n} (seed {SEED + width})"


def test_softmax_accuracy() -> None:
    """Relative error per element <= 2^-15 against float64, for scores within
    30 of the maximum (every exp result is normal).

    Bound: s - m is rounded (relative error of exp(s - m) <= 30 * 2^-24 =
    2^-19.1), exp 2^-17.08, the sum of positive terms gamma_k with k = n/S + 3
    (2^-18.9 for n = 256), recip 2^-17.2, the multiply 2^-24. Total 2^-15.8;
    2^-15 leaves a factor-1.7 margin."""
    rng = np.random.default_rng(SEED)
    for n in (1, 7, 64, 256):
        s = F32(rng.uniform(-15, 15, (32, n)))
        y = vector.softmax(t32(s)).double()
        ref = torch.softmax(torch.from_numpy(s).double(), -1)
        assert max_rel_err(y, ref) <= 2.0**-15, f"n={n}"


def test_softmax_of_largest_score_and_single_element() -> None:
    """exp(+0) = 1 exactly, so z >= 1. The result is not exactly normalized:
    softmax of one score is recip(1) = 0.9999935, not 1."""
    p = vector.softmax(t32([3.0]))
    assert p.item() == pytest.approx(0.9999935, abs=1e-7) and p.item() < 1.0
    assert arith.bf16(p).item() == 1.0  # attention rounds p to BF16 (D-011)


def test_softmax_non_finite_scores() -> None:
    """-inf gets p = 0; +inf, NaN, or all -inf give NaN everywhere."""
    p = vector.softmax(t32([-np.inf, 1.0, 0.0]))
    assert bits_of(p) == bits_of(spec_softmax(F32([-np.inf, 1.0, 0.0]), vector.REDUCE_WIDTH))
    assert p[0].item() == 0.0 and torch.isfinite(p).all()
    s = t32([[1.0, np.inf, 0.0], [1.0, np.nan, 0.0], [-np.inf, -np.inf, -np.inf]])
    assert torch.isnan(vector.softmax(s)).all()


# --- flush to zero ----------------------------------------------------------------


def test_flush_to_zero_does_not_change_normal_results() -> None:
    """With FTZ on, inputs and results in the normal range give the same bits."""
    rng = np.random.default_rng(SEED)
    x, g = t32(moderate(rng, (32, 576))), bf16_weights(rng, 576)
    s = t32(rng.uniform(-30, 30, (32, 200)))  # every exp result is normal
    plain = (vector.rmsnorm(x, g, 1e-5), vector.softmax(s), vector.reduce_sum(x))
    with settings.override(ftz=True):
        flushed = (vector.rmsnorm(x, g, 1e-5), vector.softmax(s), vector.reduce_sum(x))
    for a, b in zip(plain, flushed, strict=True):
        assert torch.equal(arith.bits_f32(a), arith.bits_f32(b))
