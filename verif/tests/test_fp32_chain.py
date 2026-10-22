# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/fp32_chain.sv (one vector-unit slot, D-048) against golden.arith.add,
mul, fma and golden.funcs.exp, recip, rsqrt, bit for bit, through the bulk
harness (verif/bulk.py).

Fast: every special value (every triple for add, mul and fma), the edges of
each function's integer steps (exp's clamp, rounding of t and subnormal
results; the inputs recip and rsqrt scale), rounding families and random
bits. Slow: 10⁸ random inputs per operation, and the M1 golden model's
exhaustive ranges for recip and rsqrt (one period of the bit-trick error;
the scaled inputs). On request (CABOSSE_SWEEP=1, tens of minutes): every
FP32 input of exp, recip and rsqrt (PLAN.md, M6 answers)."""

import os
from collections.abc import Iterable

import bulk
import numpy as np
import pytest
import torch
from fp_inputs import random_bits, rounding_batch, special_tuples, with_exponents
from golden import arith, funcs

import rtl

SEED = 20261022
# op_i (fp_pkg::chain_op_e) and the golden function (a, b, c -> result).
OPS = {
    "add": (0, lambda a, b, _: arith.add(a, b)),
    "mul": (1, lambda a, b, _: arith.mul(a, b)),
    "fma": (2, arith.fma),
    "exp": (3, lambda a, *_: funcs.exp(a)),
    "recip": (4, lambda a, *_: funcs.recip(a)),
    "rsqrt": (5, lambda a, *_: funcs.rsqrt(a)),
}
BINARY, UNARY = ("add", "mul", "fma"), ("exp", "recip", "rsqrt")
N_FAMILY, N_RANDOM = 2_000, 20_000  # per rounding family, random bits: fast run
N_EDGE = 2_000  # per exponent band of an edge, fast run
N_SLOW, PER_CHUNK = 10**8, 1 << 21  # per operation, slow run; records per chunk
LOG2E = float(np.uint32(funcs.LOG2E_BITS).view(np.float32))
ULPS = 8  # exp: ulps on each side of x where t = x * log2(e) meets an edge
SIGN = np.uint32(0x8000_0000)
# (operation, first, end): the M1 golden model's exhaustive ranges (one
# period of the bit-trick error: test_funcs) and every input each function
# scales, as FP32 bit patterns; recip's both signs.
PERIODS = {
    "rsqrt [1, 4)": ("rsqrt", 0x3F80_0000, 0x4080_0000),
    "rsqrt below 2^-124": ("rsqrt", 0, 0x0100_0000),
    "recip [1, 2)": ("recip", 0x3F80_0000, 0x4000_0000),
    "recip subnormal": ("recip", 0, 0x0080_0000),
    "recip from 2^125": ("recip", 0x7E00_0000, 0x7F80_0000),
}
SWEEP = os.environ.get("CABOSSE_SWEEP") == "1"  # the every-input run, on request
SWEEP_CHUNK = 1 << 22  # inputs per chunk of it (memory: bulk.WORKERS at once)


def records(code: int | np.ndarray, abc: np.ndarray) -> np.ndarray:
    """The driver's rows: op_i (one, or one per row), then a, b, c."""
    codes = np.broadcast_to(np.asarray(code, dtype=np.uint32), (len(abc),))
    return np.concatenate([codes[:, None], abc], axis=1)


def expected(op: str, abc: np.ndarray) -> np.ndarray:
    """The golden function on (a, b, c) rows, as FP32 bits."""
    a, b, c = torch.from_numpy(abc.view(np.float32)).unbind(1)
    return arith.bits_f32(OPS[op][1](a, b, c)).numpy().astype(np.uint32)


def check(op: str, abc: np.ndarray, label: str | int) -> None:
    bulk.check("fp32_chain", records(OPS[op][0], abc), expected(op, abc), f"{op}, {label}")


def unary(x: np.ndarray) -> np.ndarray:
    """(a, b, c) rows for a one-input operation: x, and zeros (the
    interleaved test sends it random b and c)."""
    x = np.asarray(x, dtype=np.uint32).ravel()
    return np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)


def around(x: np.ndarray, ulps: int) -> np.ndarray:
    """Every FP32 bit pattern within `ulps` of each value x (same sign)."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.int64)
    near = bits[:, None] + np.arange(-ulps, ulps + 1)
    return near.clip(0, 2**32 - 1).astype(np.uint32).ravel()


def bands(rng: np.random.Generator, fields: Iterable[int], n: int) -> np.ndarray:
    """n random values of either sign in each exponent field of `fields`."""
    return np.concatenate([with_exponents(rng, n, f, f, 32) for f in fields])


def edges(op: str, rng: np.random.Generator) -> np.ndarray:
    """Inputs at each function's integer steps (docs/numerics.md, section 5;
    the stage table in rtl/fp32_chain.sv)."""
    if op == "exp":
        # t = x * log2(e) near an integer (where i changes), near a half (the
        # ties of round(t)), at the clamp (+-150) and at the ends of the
        # normal and subnormal results (t near -126, -150 and 128).
        k = np.arange(-151, 152, dtype=np.float64)
        targets = np.concatenate([k, k + 0.5]) / LOG2E
        return np.concatenate(
            [
                around(targets.astype(np.float32), ULPS),
                bands(rng, range(127), 50),  # |x| < 1, subnormals too
                rng.uniform(-110, 95, N_EDGE).astype(np.float32).view(np.uint32),
            ]
        )
    if op == "recip":
        # Scaled by 2^24 (field 0) and 2^-24 (from 2^125, field 252); the
        # largest results overflow, the smallest are subnormal.
        return bands(rng, [0, 1, 2, 125, 126, 127, 128, 250, 251, 252, 253, 254], N_EDGE // 4)
    # rsqrt: scaled by 2^24 below 2^-125 (fields 0 and 1); negatives are NaN.
    positive = bands(rng, [0, 1, 2, 3, 126, 127, 128, 254], N_EDGE // 4) & ~SIGN
    return np.concatenate([positive, positive[:N_EDGE] | SIGN])


def all_between(first: int, end: int) -> np.ndarray:
    """Every FP32 bit pattern from first up to (not including) end <= 2^32."""
    return np.arange(first, end, dtype=np.uint64).astype(np.uint32)


@rtl.needs_verilator
@pytest.mark.parametrize("op", BINARY)
def test_binary_specials(op: str) -> None:
    check(op, special_tuples(3), "specials")


@rtl.needs_verilator
@pytest.mark.parametrize("op", BINARY)
def test_binary_random(op: str) -> None:
    check(op, rounding_batch(np.random.default_rng(SEED), N_FAMILY, N_RANDOM), SEED)


@rtl.needs_verilator
@pytest.mark.parametrize("op", UNARY)
def test_unary_specials_and_edges(op: str) -> None:
    rng = np.random.default_rng(SEED)
    x = np.concatenate([special_tuples(1).ravel(), edges(op, rng), random_bits(rng, N_RANDOM)])
    check(op, unary(x), f"specials, edges, seed {SEED}")


@rtl.needs_verilator
def test_operations_interleave() -> None:
    """Every operation in a random order, one per cycle, random bits in the
    inputs an operation ignores: a stage's state follows its own input,
    whatever the operations around it."""
    rng = np.random.default_rng(SEED)
    n = 6 * N_FAMILY
    ops = rng.choice(list(OPS), n)
    batch = rounding_batch(rng, N_FAMILY // 4, n)
    abc = batch[rng.choice(len(batch), n, replace=False)]
    abc[:, 0] = np.where(np.isin(ops, UNARY), random_bits(rng, n), abc[:, 0])
    want = np.empty(n, dtype=np.uint32)
    for op in OPS:
        want[ops == op] = expected(op, abc[ops == op])
    codes = np.array([OPS[o][0] for o in ops])
    bulk.check("fp32_chain", records(codes, abc), want, f"interleaved, seed {SEED}")


@pytest.mark.slow
@rtl.needs_verilator
@pytest.mark.parametrize("op", list(OPS))
def test_random_slow(op: str) -> None:
    """10⁸ inputs in chunks, each with its own seed (printed on failure),
    bulk.WORKERS at a time: rounding families and random bits for add, mul
    and fma; edges and random bits for the functions."""

    def one(seed: int) -> None:
        rng = np.random.default_rng(seed)
        if op in BINARY:
            abc = rounding_batch(rng, PER_CHUNK // 20, PER_CHUNK // 2)
        else:
            abc = unary(np.concatenate([edges(op, rng), random_bits(rng, PER_CHUNK)]))
        check(op, abc, seed)

    bulk.in_parallel(one, bulk.chunk_seeds(SEED, N_SLOW, PER_CHUNK))


@pytest.mark.slow
@rtl.needs_verilator
@pytest.mark.parametrize("name", list(PERIODS))
def test_exhaustive_ranges(name: str) -> None:
    op, first, end = PERIODS[name]
    signs = (0, SIGN) if op == "recip" else (0,)
    chunks = [(s, k) for s in signs for k in range(first, end, PER_CHUNK)]

    def one(chunk: tuple[int, int]) -> None:
        sign, k = chunk
        check(op, unary(all_between(k, min(k + PER_CHUNK, end)) | sign), f"{name}, {chunk}")

    bulk.in_parallel(one, chunks)


@pytest.mark.skipif(not SWEEP, reason="every FP32 input, tens of minutes: CABOSSE_SWEEP=1")
@rtl.needs_verilator
@pytest.mark.parametrize("op", UNARY)
def test_every_input(op: str) -> None:
    """Every FP32 bit pattern, 2^32, in chunks of SWEEP_CHUNK, bulk.WORKERS
    at a time."""

    def one(k: int) -> None:
        check(op, unary(all_between(k, k + SWEEP_CHUNK)), f"from 0x{k:08X}")

    bulk.in_parallel(one, range(0, 2**32, SWEEP_CHUNK))
