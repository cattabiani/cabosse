# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/fp32_max.sv (our own, D-039) against golden.arith.maximum
(maximumNumber, D-037), bit for bit, through the bulk harness
(verif/bulk.py): every pair of the special values, near pairs (ties, signed
zeros, neighbours, NaN payloads) and random bit patterns. The fast run checks
a few hundred thousand inputs; the slow run 10⁸ (PLAN.md, M3 exit
criterion)."""

import bulk
import numpy as np
import pytest
import torch
from fp_inputs import random_bits, special_tuples
from golden import arith

import rtl

SEED = 20261007
N_NEAR, N_RANDOM = 100_000, 100_000  # near pairs, random bits: fast run
N_NEAR_SLOW, N_RANDOM_SLOW = 500_000, 500_000  # per chunk of the slow run
N_SLOW = 10**8
SIGN = np.uint32(0x8000_0000)


def near_pairs(rng: np.random.Generator, n: int) -> np.ndarray:
    """(a, b) with b = a, its last two bits changed at random and its sign
    flipped half the time: equal values, ±x, neighbours, signed zeros and
    NaNs with different payloads occur often."""
    a = random_bits(rng, n)
    a[rng.random(n) < 0.1] &= SIGN  # some zeros
    b = a ^ rng.integers(0, 4, n, dtype=np.uint32)
    b[rng.random(n) < 0.5] ^= SIGN
    return np.stack([a, b], axis=1)


def batch(rng: np.random.Generator, n_near: int, n_random: int) -> np.ndarray:
    return np.concatenate([near_pairs(rng, n_near), random_bits(rng, (n_random, 2))])


def check(ab: np.ndarray, seed: int | str) -> None:
    a, b = torch.from_numpy(ab.view(np.float32)).unbind(1)
    want = arith.bits_f32(arith.maximum(a, b)).numpy().astype(np.uint32)
    bulk.check("fp32_max", ab, want, f"seed {seed}")


@rtl.needs_verilator
def test_specials() -> None:
    check(special_tuples(2), "specials")


@rtl.needs_verilator
def test_random() -> None:
    check(batch(np.random.default_rng(SEED), N_NEAR, N_RANDOM), SEED)


@pytest.mark.slow
@rtl.needs_verilator
def test_random_slow() -> None:
    """10⁸ inputs in chunks of 10⁶, each with its own seed (printed on
    failure), bulk.WORKERS at a time."""
    seeds = bulk.chunk_seeds(SEED, N_SLOW, N_NEAR_SLOW + N_RANDOM_SLOW)
    bulk.in_parallel(
        lambda seed: check(batch(np.random.default_rng(seed), N_NEAR_SLOW, N_RANDOM_SLOW), seed),
        seeds,
    )
