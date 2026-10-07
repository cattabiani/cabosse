# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/fp32_fma.sv (CVFPU's FMA, D-033) against golden.arith.fma, add and mul,
bit for bit, through the bulk harness (verif/bulk.py): every triple of the
special values, the rounding families of the golden-model tests, and random
bit patterns. The fast run checks a few hundred thousand inputs per
operation; the slow run 10⁸ (PLAN.md, M3 exit criterion)."""

from concurrent.futures import ThreadPoolExecutor

import bulk
import numpy as np
import pytest
import torch
from fp_inputs import families, random_bits, special_triples
from golden import arith

import rtl

SEED = 20261007
# op_i of rtl/fp32_fma.sv and the golden function (a, b, c -> result).
OPS = {
    "fma": (0, arith.fma),
    "add": (1, lambda a, b, _: arith.add(a, b)),
    "mul": (2, lambda a, b, _: arith.mul(a, b)),
}
N_FAMILY, N_RANDOM = 20_000, 100_000  # per rounding family, random bits: fast run
N_FAMILY_SLOW, N_RANDOM_SLOW = 100_000, 200_000  # per chunk of the slow run
N_SLOW = 10**8  # per operation, slow run
WORKERS = 8  # slow-run chunks checked at once (each a driver process)
PIPELINES = {"comb": (), "pipe3": (("NumPipeRegs", 3),)}  # the same bits either way


def batch(rng: np.random.Generator, n_family: int, n_random: int) -> np.ndarray:
    """(a, b, c) bit patterns: every rounding family, then random bits."""
    abc = [
        np.stack([x.view(np.uint32) for x in t], axis=1) for t in families(rng, n_family).values()
    ]
    abc.append(random_bits(rng, (n_random, 3)))
    return np.concatenate(abc)


def check(op: str, abc: np.ndarray, seed: int | str, params: tuple = ()) -> None:
    op_i, golden = OPS[op]
    records = np.concatenate([np.full((len(abc), 1), op_i, dtype=np.uint32), abc], axis=1)
    got = bulk.run("fp32_fma", records, params)
    x, y, z = torch.from_numpy(abc.view(np.float32)).unbind(1)
    want = arith.bits_f32(golden(x, y, z)).numpy().astype(np.uint32)
    report = bulk.mismatches(got, want, records)
    assert not report, f"{op}, seed {seed}: {report}"


@rtl.needs_verilator
@pytest.mark.parametrize("pipeline", PIPELINES)
@pytest.mark.parametrize("op", OPS)
def test_specials(op: str, pipeline: str) -> None:
    """Every triple of special values; the driver collects results by
    valid_o, whatever the pipeline depth."""
    check(op, special_triples(), "specials", PIPELINES[pipeline])


@rtl.needs_verilator
@pytest.mark.parametrize("op", OPS)
def test_random(op: str) -> None:
    check(op, batch(np.random.default_rng(SEED), N_FAMILY, N_RANDOM), SEED)


@pytest.mark.slow
@rtl.needs_verilator
@pytest.mark.parametrize("op", OPS)
def test_random_slow(op: str) -> None:
    """10⁸ inputs in chunks of about 10⁶, each with its own seed (printed on
    failure), WORKERS at a time."""
    per_chunk = len(batch(np.random.default_rng(0), N_FAMILY_SLOW, N_RANDOM_SLOW))
    seeds = [SEED + 1 + k for k in range(-(-N_SLOW // per_chunk))]

    def one(seed: int) -> None:
        check(op, batch(np.random.default_rng(seed), N_FAMILY_SLOW, N_RANDOM_SLOW), seed)

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(one, seeds))
