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
the scaled inputs). Sweep, on request: every FP32 input of exp, recip and
rsqrt (PLAN.md, M6 answers)."""

import bulk
import numpy as np
import pytest
import torch
from fp_inputs import families, random_bits, special_tuples, with_exponents
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
F32_ONE, F32_TWO, F32_FOUR = 0x3F80_0000, 0x4000_0000, 0x4080_0000
SIGN = np.uint32(0x8000_0000)
SWEEP_CHUNK = 1 << 22  # inputs per chunk of the every-input run (memory: 8 at once)


def records(op: str, abc: np.ndarray) -> np.ndarray:
    """The driver's rows: op_i, then a, b, c."""
    return np.concatenate([np.full((len(abc), 1), OPS[op][0], dtype=np.uint32), abc], axis=1)


def expected(op: str, abc: np.ndarray) -> np.ndarray:
    """The golden function on (a, b, c) rows, as FP32 bits."""
    a, b, c = torch.from_numpy(abc.view(np.float32)).unbind(1)
    return arith.bits_f32(OPS[op][1](a, b, c)).numpy().astype(np.uint32)


def check(op: str, abc: np.ndarray, label: str | int) -> None:
    bulk.check("fp32_chain", records(op, abc), expected(op, abc), f"{op}, {label}")


def unary(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(a, b, c) rows for a one-input operation: x, and random bits in b and
    c, which it must ignore."""
    x = np.asarray(x, dtype=np.uint32).ravel()
    return np.stack([x, *random_bits(rng, (2, len(x)))], axis=1)


def binary_batch(rng: np.random.Generator, n_family: int, n_random: int) -> np.ndarray:
    """(a, b, c) bit patterns: every rounding family, then random bits."""
    abc = [
        np.stack([x.view(np.uint32) for x in t], axis=1) for t in families(rng, n_family).values()
    ]
    abc.append(random_bits(rng, (n_random, 3)))
    return np.concatenate(abc)


def around(x: np.ndarray, ulps: int) -> np.ndarray:
    """Every FP32 bit pattern within `ulps` of each value x (same sign)."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.int64)
    near = bits[:, None] + np.arange(-ulps, ulps + 1)
    return near.clip(0, 2**32 - 1).astype(np.uint32).ravel()


def bands(rng: np.random.Generator, fields: list[int], n: int, signs: bool = True) -> np.ndarray:
    """n random values in each exponent field of `fields`, either sign (or
    positive only)."""
    x = np.concatenate([with_exponents(rng, n, f, f, 32) for f in fields])
    return x if signs else x & ~SIGN


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
                bands(rng, list(range(0, 127)), 50),  # |x| < 1, subnormals too
                (rng.uniform(-110, 95, N_EDGE).astype(np.float32)).view(np.uint32),
            ]
        )
    if op == "recip":
        # Scaled by 2^24 (field 0) and 2^-24 (from 2^125, field 252); the
        # largest results overflow, the smallest are subnormal.
        return bands(rng, [0, 1, 2, 125, 126, 127, 128, 250, 251, 252, 253, 254], N_EDGE // 4)
    # rsqrt: scaled by 2^24 below 2^-125 (fields 0 and 1); negatives are NaN.
    positive = bands(rng, [0, 1, 2, 3, 126, 127, 128, 254], N_EDGE // 4, signs=False)
    return np.concatenate([positive, positive[:N_EDGE] | SIGN])


@rtl.needs_verilator
@pytest.mark.parametrize("op", BINARY)
def test_binary_specials(op: str) -> None:
    check(op, special_tuples(3), "specials")


@rtl.needs_verilator
@pytest.mark.parametrize("op", BINARY)
def test_binary_random(op: str) -> None:
    check(op, binary_batch(np.random.default_rng(SEED), N_FAMILY, N_RANDOM), SEED)


@rtl.needs_verilator
@pytest.mark.parametrize("op", UNARY)
def test_unary_specials_and_edges(op: str) -> None:
    rng = np.random.default_rng(SEED)
    x = np.concatenate([special_tuples(1).ravel(), edges(op, rng), random_bits(rng, N_RANDOM)])
    check(op, unary(x, rng), f"specials, edges, seed {SEED}")


@rtl.needs_verilator
def test_operations_interleave() -> None:
    """Every operation in a random order, one per cycle: a stage's state
    follows its own input, whatever the operations around it."""
    rng = np.random.default_rng(SEED)
    n = 6 * N_FAMILY
    ops = rng.choice(list(OPS), n)
    batch = binary_batch(rng, N_FAMILY // 4, n)
    abc = batch[rng.choice(len(batch), n, replace=False)]
    abc[:, 0] = np.where(np.isin(ops, UNARY), random_bits(rng, n), abc[:, 0])
    rows = np.concatenate([np.array([OPS[o][0] for o in ops], np.uint32)[:, None], abc], axis=1)
    want = np.empty(n, dtype=np.uint32)
    for op in OPS:
        m = ops == op
        want[m] = expected(op, abc[m])
    bulk.check("fp32_chain", rows, want, f"interleaved, seed {SEED}")


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
            abc = binary_batch(rng, PER_CHUNK // 20, PER_CHUNK // 2)
        else:
            abc = unary(np.concatenate([edges(op, rng), random_bits(rng, PER_CHUNK)]), rng)
        check(op, abc, seed)

    bulk.in_parallel(one, bulk.chunk_seeds(SEED, N_SLOW, PER_CHUNK))


def all_between(lo: int, hi: int) -> np.ndarray:
    """Every FP32 bit pattern from lo up to (not including) hi <= 2^32."""
    return np.arange(lo, hi, dtype=np.uint64).astype(np.uint32)


# (operation, inputs): the M1 golden model's exhaustive ranges (one period
# of the bit-trick error: test_funcs) and every input each function scales,
# both signs for recip.
PERIODS = {
    "rsqrt [1, 4)": ("rsqrt", lambda: all_between(F32_ONE, F32_FOUR)),
    "rsqrt subnormal and below 2^-125": ("rsqrt", lambda: all_between(0, 0x0100_0000)),
    "recip [1, 2)": ("recip", lambda: all_between(F32_ONE, F32_TWO)),
    "recip subnormal": ("recip", lambda: all_between(0, 0x0080_0000)),
    "recip from 2^125": ("recip", lambda: all_between(0x7E00_0000, 0x7F80_0000)),
}


@pytest.mark.slow
@rtl.needs_verilator
@pytest.mark.parametrize("name", list(PERIODS))
def test_exhaustive_ranges(name: str) -> None:
    op, inputs = PERIODS[name]
    x = inputs()
    if op == "recip":
        x = np.concatenate([x, x | SIGN])
    chunks = np.array_split(x, -(-len(x) // PER_CHUNK))

    def one(k: int) -> None:
        check(op, unary(chunks[k], np.random.default_rng(SEED + k)), f"{name}, chunk {k}")

    bulk.in_parallel(one, range(len(chunks)))


@pytest.mark.sweep
@rtl.needs_verilator
@pytest.mark.parametrize("op", UNARY)
def test_every_input(op: str) -> None:
    """Every FP32 bit pattern, 2^32, in chunks of SWEEP_CHUNK run
    bulk.WORKERS at a time."""

    def one(k: int) -> None:
        x = all_between(k * SWEEP_CHUNK, (k + 1) * SWEEP_CHUNK)
        abc = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
        check(op, abc, f"chunk {k}")

    bulk.in_parallel(one, range(2**32 // SWEEP_CHUNK))
