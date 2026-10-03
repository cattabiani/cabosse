# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Tests for golden.dot: bit-exact agreement with an independent restatement
of docs/numerics.md section 3, the classical error bound, and shapes."""

import math

import numpy as np
import pytest
import torch
from golden import arith, dot
from oracle import add_ref, fma_ref

SEED = 20261002
F32 = np.float32


def random_bf16(rng: np.random.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    """BF16 values around 1 with random signs, plus some signed zeros."""
    v = (
        rng.choice([-1.0, 1.0], shape)
        * rng.uniform(1, 2, shape)
        * 2.0 ** rng.integers(-8, 9, shape)
    )
    v[rng.random(shape) < 0.05] = 0.0
    v[rng.random(shape) < 0.02] = -0.0
    return arith.bf16(torch.from_numpy(F32(v)))


def extreme_bf16(rng: np.random.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    """BF16 values with exponents over most of the range, so products underflow
    (to subnormals or signed zeros) and overflow; a few infinities and NaNs."""
    sign = rng.choice([-1.0, 1.0], shape)
    v = sign * rng.uniform(1, 2, shape) * 2.0 ** rng.integers(-100, 100, shape)
    v[rng.random(shape) < 0.02] = np.inf
    v[rng.random(shape) < 0.02] = -np.inf
    v[rng.random(shape) < 0.01] = np.nan
    return arith.bf16(torch.from_numpy(F32(v)))


# --- independent restatement of section 3 (scalar, exact oracle arithmetic) --------


def spec_dot(w: np.ndarray, x: np.ndarray, accumulators: int) -> np.float32:
    """w, x: FP32 values that are exactly BF16 (the widening is exact)."""
    acc = [F32(0.0)] * accumulators
    for k in range(len(w)):
        acc[k % accumulators] = fma_ref(w[k], x[k], acc[k % accumulators])
    while len(acc) > 1:
        acc = [add_ref(acc[i], acc[i + 1]) for i in range(0, len(acc), 2)]
    return acc[0]


def spec_bits(w: torch.Tensor, x: torch.Tensor, accumulators: int) -> list[int]:
    """spec_dot for each row, as bit patterns (NaN -> canonical NaN)."""
    w32, x32 = w.to(torch.float32).numpy(), x.to(torch.float32).numpy()
    out = []
    for row in range(w32.shape[0]):
        y = spec_dot(w32[row], x32[row], accumulators)
        out.append(arith.NAN_F32_BITS if np.isnan(y) else int(np.array(y).view(np.uint32)))
    return out


@pytest.mark.parametrize("family", ["moderate", "extreme"])
@pytest.mark.parametrize("accumulators", [1, 2, 4, 8, 16])
def test_matches_spec_bit_exactly(accumulators: int, family: str) -> None:
    """Lengths 1..48 include K < A and K not a multiple of A. The extreme family
    has underflowing and overflowing products, infinities, and NaNs."""
    rng = np.random.default_rng(SEED + accumulators)
    make = random_bf16 if family == "moderate" else extreme_bf16
    for length in range(1, 49):
        w, x = make(rng, (4, length)), make(rng, (4, length))
        got = arith.bits_f32(dot.dot(w, x, accumulators)).tolist()
        want = spec_bits(w, x, accumulators)
        assert got == want, (
            f"A={accumulators}, K={length}, {family}: golden {[hex(g) for g in got]}, "
            f"spec {[hex(s) for s in want]} (seed {SEED + accumulators})"
        )


def test_partial_group_is_not_zero_padded() -> None:
    """A tiny negative product underflows to -0 in its accumulator. Feeding
    zeros to the missing elements of the last group would turn it into +0
    (mac(0, 0, -0) = +0), so the spec says they keep their value."""
    w = arith.bf16(torch.full((9,), -(2.0**-80)))
    x = arith.bf16(torch.full((9,), 2.0**-80))
    assert arith.bits_f32(dot.dot(w, x, 8)).item() == 0x80000000  # -0
    assert spec_bits(w[None], x[None], 8) == [0x80000000]
    zeros = torch.zeros(7, dtype=torch.bfloat16)
    padded = dot.dot(torch.cat([w, zeros]), torch.cat([x, zeros]), 8)
    assert arith.bits_f32(padded).item() == 0x00000000  # +0: padding is not a no-op


def test_cancellation_matches_spec() -> None:
    """Large terms that cancel: the summation order decides the result."""
    rng = np.random.default_rng(SEED)
    big = rng.choice([-1.0, 1.0], 64) * 2.0 ** rng.integers(10, 20, 64)
    small = rng.uniform(-1, 1, 64)
    w = arith.bf16(torch.from_numpy(F32(np.concatenate([big, -big, small]))))
    x = arith.bf16(torch.ones(192))
    for accumulators in (1, 8):
        got = arith.bits_f32(dot.dot(w, x, accumulators)).item()
        assert [got] == spec_bits(w[None], x[None], accumulators)


# --- accuracy: the classical bound for floating-point summation ----------------------


@pytest.mark.parametrize("accumulators", [1, 8, 16])
def test_error_within_classical_bound(accumulators: int) -> None:
    """|computed - exact| <= gamma_n * sum |w_k x_k|, gamma_n = n u / (1 - n u),
    u = 2^-24. Products are exact (fma), and each term goes through at most
    n = ceil(K/A) + log2(A) roundings: its accumulator chain, then the tree.

    A sanity check on accuracy only: any reasonable summation order passes it.
    The bit-exact spec tests are the ones that pin the order."""
    rng = np.random.default_rng(SEED)
    u = 2.0**-24
    for length in (576, 1536, 4096):
        w, x = random_bf16(rng, (256, length)), random_bf16(rng, (length,))
        w64, x64 = w.to(torch.float64), x.to(torch.float64)
        exact = (w64 * x64).sum(-1)  # float64: error far below the FP32 bound
        n = math.ceil(length / accumulators) + int(math.log2(accumulators))
        bound = n * u / (1 - n * u) * (w64 * x64).abs().sum(-1)
        err = (dot.dot(w, x, accumulators).to(torch.float64) - exact).abs()
        assert (err <= bound).all(), f"K={length}: worst ratio {(err / bound).max().item():.3f}"


# --- shapes and the tree ------------------------------------------------------------


def test_matrix_vector_equals_row_by_row() -> None:
    rng = np.random.default_rng(SEED)
    w, x = random_bf16(rng, (37, 100)), random_bf16(rng, (100,))
    rows = torch.stack([dot.dot(w[i], x) for i in range(37)])
    assert torch.equal(arith.bits_f32(dot.dot(w, x)), arith.bits_f32(rows))


def test_tree_sum_pairs_neighbours() -> None:
    """Values where the order matters. Tree: (1 + 2^24) rounds to 2^24 (tie to
    even), (1 - 2^24) is exact, and their sum is 1. Left to right would give
    ((1 + 2^24) + 1) - 2^24 = 0."""
    parts = torch.tensor([1.0, 2.0**24, 1.0, -(2.0**24)], dtype=torch.float32)
    assert dot.tree_sum(parts).item() == 1.0


def test_tree_sum_rejects_non_power_of_two() -> None:
    with pytest.raises(AssertionError):
        dot.tree_sum(torch.zeros(3, dtype=torch.float32))


def test_rejects_bad_arguments() -> None:
    w = torch.zeros(4, dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        dot.dot(w, w, accumulators=6)  # not a power of two: fails before the loop
    with pytest.raises(AssertionError):
        dot.dot(w[:0], w[:0])  # K = 0 is not defined
