# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/fp32_fma.sv (CVFPU's FMA, D-033) against golden.arith.fma, add and mul,
bit for bit, through the bulk harness (verif/bulk.py): every triple of the
special values, the rounding families of the golden-model tests, and random
bit patterns. The fast run checks a few hundred thousand inputs per
operation; the slow run 10⁸ (PLAN.md, M3 exit criterion)."""

import bulk
import numpy as np
import pytest
import torch
from fp_inputs import F32_SPECIAL_BITS, families
from golden import arith

import rtl

SEED = 20261007
# op_i of rtl/fp32_fma.sv and the golden function (a, b, c -> result).
OPS = {
    "fma": (0, arith.fma),
    "add": (1, lambda a, b, _: arith.add(a, b)),
    "mul": (2, lambda a, b, _: arith.mul(a, b)),
}
N_FAMILY = 20_000  # per rounding family, fast run
N_RANDOM = 100_000  # random bit patterns, fast run
N_SLOW = 10**8  # per operation, slow run
PIPELINED = (("NumPipeRegs", 3),)  # the same bits with registers in the pipe


def specials() -> np.ndarray:
    """Every (a, b, c) triple of the special values."""
    s = np.array(F32_SPECIAL_BITS, dtype=np.uint32)
    return np.stack([x.ravel() for x in np.meshgrid(s, s, s, indexing="ij")], axis=1)


def batch(rng: np.random.Generator, n_family: int, n_random: int) -> np.ndarray:
    """(a, b, c) bit patterns: every rounding family, then random bits."""
    abc = [
        np.stack([x.view(np.uint32) for x in t], axis=1) for t in families(rng, n_family).values()
    ]
    abc.append(rng.integers(0, 2**32, (n_random, 3), dtype=np.uint64).astype(np.uint32))
    return np.concatenate(abc)


def check(op: str, abc: np.ndarray, seed: int | str, params: tuple = ()) -> None:
    op_i, golden = OPS[op]
    records = np.concatenate([np.full((len(abc), 1), op_i, dtype=np.uint32), abc], axis=1)
    got = bulk.run("fp32_fma", records, params)
    x, y, z = (arith.f32_from_bits(torch.from_numpy(abc[:, k].astype(np.int64))) for k in range(3))
    want = arith.bits_f32(golden(x, y, z)).numpy().astype(np.uint32)
    report = bulk.mismatches(got, want, records)
    assert not report, f"{op}, seed {seed}: {report}"


@rtl.needs_verilator
@pytest.mark.parametrize("op", OPS)
def test_specials(op: str) -> None:
    check(op, specials(), "specials")


@rtl.needs_verilator
@pytest.mark.parametrize("op", OPS)
def test_specials_pipelined(op: str) -> None:
    """The driver collects results by valid_o, whatever the pipeline depth."""
    check(op, specials(), "specials", PIPELINED)


@rtl.needs_verilator
@pytest.mark.parametrize("op", OPS)
def test_random(op: str) -> None:
    check(op, batch(np.random.default_rng(SEED), N_FAMILY, N_RANDOM), SEED)


@pytest.mark.slow
@rtl.needs_verilator
@pytest.mark.parametrize("op", OPS)
def test_random_slow(op: str) -> None:
    """10⁸ inputs in chunks of 10⁶, each with its own seed (printed on failure)."""
    n_family, n_random = 100_000, 200_000  # 8 families + random bits
    per_chunk = 8 * n_family + n_random
    for chunk in range(-(-N_SLOW // per_chunk)):
        seed = SEED + 1 + chunk
        check(op, batch(np.random.default_rng(seed), n_family, n_random), seed)
