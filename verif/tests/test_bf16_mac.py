# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/bf16_mac.sv (our own BF16 multiply-add, D-038) against
golden.arith.mac_f32, bit for bit, through the bulk harness (verif/bulk.py):
every combination of the special values, accumulators that cancel the
product or sit at any distance from it, and random bits. The slow run takes
every BF16 pair, 2^32, with a random accumulator each."""

import bulk
import numpy as np
import pytest
import torch
from fp_inputs import BF16_SPECIAL_BITS, F32_SPECIAL_BITS, all_bf16_widened, random_bits
from golden import arith

import rtl

SEED = 20261007
N_NEAR, N_SCALED, N_RANDOM = 100_000, 100_000, 100_000  # fast run, per family
W_PER_CHUNK = 16  # exhaustive run: 16 values of w against all 2^16 x
UP = all_bf16_widened()  # up(b) as FP32 bits, indexed by the BF16 bits


def bf16_bits(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 2**16, n, dtype=np.uint32)


def products(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """up(w) * up(x), exact, as float64 (NaN or Inf where they are)."""
    with np.errstate(invalid="ignore", over="ignore"):
        return UP[w].view(np.float32).astype(np.float64) * UP[x].view(np.float32)


def near_cancelling(rng: np.random.Generator, n: int) -> np.ndarray:
    """acc = -(w * x) rounded to FP32, its bits moved by up to 4 ulps: exact
    cancellation, and sums that lose most of their leading bits."""
    w, x = bf16_bits(rng, n), bf16_bits(rng, n)
    with np.errstate(invalid="ignore", over="ignore"):
        acc = (-products(w, x)).astype(np.float32).view(np.uint32)
    acc = acc + rng.integers(-4, 5, n).astype(np.uint32)  # wraps like the bits do
    return np.stack([w, x, acc], axis=1)


def scaled(rng: np.random.Generator, n: int) -> np.ndarray:
    """acc of either sign at 2^-40 .. 2^40 times the product, random low bits:
    every alignment distance, including past the sticky bit."""
    w, x = bf16_bits(rng, n), bf16_bits(rng, n)
    scale = 2.0 ** rng.integers(-40, 41, n) * rng.choice([-1.0, 1.0], n)
    with np.errstate(invalid="ignore", over="ignore"):
        acc = (products(w, x) * scale).astype(np.float32).view(np.uint32)
    acc ^= rng.integers(0, 2**12, n, dtype=np.uint32)
    return np.stack([w, x, acc], axis=1)


def specials() -> np.ndarray:
    """Every (w, x, acc) of the BF16 and FP32 special values."""
    b = np.array(BF16_SPECIAL_BITS, dtype=np.uint32)
    f = np.array(F32_SPECIAL_BITS, dtype=np.uint32)
    return np.stack([v.ravel() for v in np.meshgrid(b, b, f, indexing="ij")], axis=1)


def batch(rng: np.random.Generator) -> np.ndarray:
    rand = np.stack(
        [bf16_bits(rng, N_RANDOM), bf16_bits(rng, N_RANDOM), random_bits(rng, N_RANDOM)], 1
    )
    return np.concatenate([near_cancelling(rng, N_NEAR), scaled(rng, N_SCALED), rand])


def check(wxa: np.ndarray, label: str) -> None:
    w, x, acc = (
        torch.from_numpy(v.view(np.float32)) for v in (UP[wxa[:, 0]], UP[wxa[:, 1]], wxa[:, 2])
    )
    want = arith.bits_f32(arith.mac_f32(w, x, acc)).numpy().astype(np.uint32)
    bulk.check("bf16_mac", wxa, want, label)


@rtl.needs_verilator
def test_specials() -> None:
    check(specials(), "specials")


@rtl.needs_verilator
def test_random() -> None:
    check(batch(np.random.default_rng(SEED)), f"seed {SEED}")


@pytest.mark.slow
@rtl.needs_verilator
def test_bf16_pairs_exhaustive() -> None:
    """Every pair of BF16 inputs, 2^32, each with a random FP32 accumulator,
    and the near-cancelling and scaled families once per chunk."""
    x = np.tile(np.arange(2**16, dtype=np.uint32), W_PER_CHUNK)

    def one(chunk: int) -> None:
        seed = SEED + chunk
        rng = np.random.default_rng(seed)
        w = np.repeat(
            np.arange(chunk * W_PER_CHUNK, (chunk + 1) * W_PER_CHUNK, dtype=np.uint32), 2**16
        )
        pairs = np.stack([w, x, random_bits(rng, len(x))], axis=1)
        check(
            np.concatenate([pairs, near_cancelling(rng, 2**14), scaled(rng, 2**14)]),
            f"w chunk {chunk}, seed {seed}",
        )

    bulk.in_parallel(one, range(2**16 // W_PER_CHUNK))
