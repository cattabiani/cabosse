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
N_NEAR, N_SCALED, N_EDGE, N_RANDOM = 100_000, 100_000, 50_000, 100_000  # fast run, per family
W_PER_CHUNK = 16  # exhaustive run: 16 values of w against all 2^16 x
UP = all_bf16_widened()  # up(b) as FP32 bits, indexed by the BF16 bits
# bf16_mac's product registers: the same bits either way. The default (1) is
# built without parameters, the build the slow run uses.
PIPELINES = {"mulregs0": (("MulRegs", 0),), "mulregs1": (), "mulregs2": (("MulRegs", 2),)}


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
    # On the bits, not by ulps: a step can cross zero into the other sign, or
    # the largest finite value into Inf and NaN, which is wanted here.
    acc = acc + rng.integers(-4, 5, n).astype(np.uint32)
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


def with_exponents(rng: np.random.Generator, n: int, lo: int, hi: int, width: int) -> np.ndarray:
    """Random bits of a `width`-bit float whose exponent field is in lo..hi."""
    frac_bits = 7 if width == 16 else 23
    sign = rng.integers(0, 2, n, dtype=np.uint32) << (width - 1)
    exp = rng.integers(lo, hi + 1, n, dtype=np.uint32) << frac_bits
    return sign | exp | rng.integers(0, 2**frac_bits, n, dtype=np.uint32)


def range_edges(rng: np.random.Generator, n: int) -> np.ndarray:
    """Products far below FP32's range against accumulators near zero (the
    sticky shift to exponent 1, subnormal results), and products above it
    against accumulators near the largest value (overflow, or cancellation
    back into range)."""
    tiny = [with_exponents(rng, n, 0, 69, 16), with_exponents(rng, n, 0, 69, 16)]
    huge = [with_exponents(rng, n, 190, 254, 16), with_exponents(rng, n, 190, 254, 16)]
    return np.concatenate([
        np.stack([*tiny, with_exponents(rng, n, 0, 5, 32)], axis=1),
        np.stack([*huge, with_exponents(rng, n, 240, 254, 32)], axis=1),
    ])  # fmt: skip


def specials() -> np.ndarray:
    """Every (w, x, acc) of the BF16 and FP32 special values."""
    b = np.array(BF16_SPECIAL_BITS, dtype=np.uint32)
    f = np.array(F32_SPECIAL_BITS, dtype=np.uint32)
    return np.stack([v.ravel() for v in np.meshgrid(b, b, f, indexing="ij")], axis=1)


def batch(rng: np.random.Generator) -> np.ndarray:
    rand = np.stack(
        [bf16_bits(rng, N_RANDOM), bf16_bits(rng, N_RANDOM), random_bits(rng, N_RANDOM)], 1
    )
    edges = range_edges(rng, N_EDGE)
    return np.concatenate([near_cancelling(rng, N_NEAR), scaled(rng, N_SCALED), edges, rand])


def check(wxa: np.ndarray, label: str, params: tuple = ()) -> None:
    w, x, acc = (
        torch.from_numpy(v.view(np.float32)) for v in (UP[wxa[:, 0]], UP[wxa[:, 1]], wxa[:, 2])
    )
    want = arith.bits_f32(arith.mac_f32(w, x, acc)).numpy().astype(np.uint32)
    bulk.check("bf16_mac", wxa, want, label, params)


@rtl.needs_verilator
@pytest.mark.parametrize("pipeline", PIPELINES)
def test_specials(pipeline: str) -> None:
    check(specials(), "specials", PIPELINES[pipeline])


@rtl.needs_verilator
@pytest.mark.parametrize("pipeline", PIPELINES)
def test_random(pipeline: str) -> None:
    check(batch(np.random.default_rng(SEED)), f"seed {SEED}", PIPELINES[pipeline])


@pytest.mark.slow
@rtl.needs_verilator
def test_bf16_pairs_exhaustive() -> None:
    """Every pair of BF16 inputs, 2^32, each with a random FP32 accumulator,
    and the near-cancelling, scaled and range-edge families once per chunk."""
    x = np.tile(np.arange(2**16, dtype=np.uint32), W_PER_CHUNK)

    def one(chunk: int) -> None:
        seed = SEED + chunk
        rng = np.random.default_rng(seed)
        w = np.repeat(
            np.arange(chunk * W_PER_CHUNK, (chunk + 1) * W_PER_CHUNK, dtype=np.uint32), 2**16
        )
        pairs = np.stack([w, x, random_bits(rng, len(x))], axis=1)
        check(
            np.concatenate(
                [pairs, near_cancelling(rng, 2**14), scaled(rng, 2**14), range_edges(rng, 2**13)]
            ),
            f"w chunk {chunk}, seed {seed}",
        )

    bulk.in_parallel(one, range(2**16 // W_PER_CHUNK))
