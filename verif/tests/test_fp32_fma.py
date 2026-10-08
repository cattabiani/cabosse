# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/fp32_fma.sv (our own FMA, D-039) against golden.arith.fma, add and mul,
bit for bit, through the bulk harness (verif/bulk.py): every triple of the
special values, the rounding families of the golden-model tests, and random
bit patterns. The fast run checks a few hundred thousand inputs per
operation; the slow run 10⁸ (PLAN.md, M3 exit criterion)."""

import bulk
import numpy as np
import pytest
import torch
from fp_inputs import all_bf16_widened, families, random_bits, special_tuples
from golden import arith, settings

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
W_PER_CHUNK = 16  # exhaustive BF16 run: 16 values of w against all 2^16 x
# fp32_fma's product registers: the same bits either way. The default (2) is
# built without parameters, the build the slow run uses.
PIPELINES = {"mulregs0": (("MulRegs", 0),), "mulregs1": (("MulRegs", 1),), "mulregs2": ()}
FTZ = (("Ftz", 1),)  # the flush-to-zero variant (D-016's cost), default pipeline
N_FLUSHED = 500  # results the flushes change, at least, per operation (add: 881)


def batch(rng: np.random.Generator, n_family: int, n_random: int) -> np.ndarray:
    """(a, b, c) bit patterns: every rounding family, then random bits."""
    abc = [
        np.stack([x.view(np.uint32) for x in t], axis=1) for t in families(rng, n_family).values()
    ]
    abc.append(random_bits(rng, (n_random, 3)))
    return np.concatenate(abc)


def check(op: str, abc: np.ndarray, seed: int | str, params: tuple = (), golden=None) -> None:
    """Run (a, b, c) rows through the block as `op`, against OPS[op]'s golden
    function, or `golden` (a, b, c -> result) when given."""
    op_i, default = OPS[op]
    golden = golden or default
    records = np.concatenate([np.full((len(abc), 1), op_i, dtype=np.uint32), abc], axis=1)
    x, y, z = torch.from_numpy(abc.view(np.float32)).unbind(1)
    want = arith.bits_f32(golden(x, y, z)).numpy().astype(np.uint32)
    bulk.check("fp32_fma", records, want, f"{op}, seed {seed}", params)


@rtl.needs_verilator
@pytest.mark.parametrize("pipeline", PIPELINES)
@pytest.mark.parametrize("op", OPS)
def test_specials(op: str, pipeline: str) -> None:
    """Every triple of special values; the driver collects results by
    valid_o, whatever the pipeline depth."""
    check(op, special_tuples(3), "specials", PIPELINES[pipeline])


@rtl.needs_verilator
@pytest.mark.parametrize("pipeline", PIPELINES)
@pytest.mark.parametrize("op", OPS)
def test_random(op: str, pipeline: str) -> None:
    check(op, batch(np.random.default_rng(SEED), N_FAMILY, N_RANDOM), SEED, PIPELINES[pipeline])


@rtl.needs_verilator
@pytest.mark.parametrize("op", OPS)
def test_flush_to_zero(op: str) -> None:
    """The variant with Ftz = 1 against the golden model's ftz switch, on the
    specials and the random batch. The inputs reach the flushes: at least
    N_FLUSHED results differ from the default mode's."""
    abc = np.concatenate(
        [special_tuples(3), batch(np.random.default_rng(SEED), N_FAMILY, N_RANDOM)]
    )
    x, y, z = torch.from_numpy(abc.view(np.float32)).unbind(1)
    golden = OPS[op][1]
    with settings.override(ftz=True):
        flushed = arith.bits_f32(golden(x, y, z))
        check(op, abc, f"ftz, {SEED}", FTZ)
    n_changed = int((flushed != arith.bits_f32(golden(x, y, z))).sum())
    assert n_changed >= N_FLUSHED, n_changed


@pytest.mark.slow
@rtl.needs_verilator
@pytest.mark.parametrize("op", ["mul", "fma"])
def test_bf16_products_exhaustive(op: str) -> None:
    """Every pair of BF16 inputs, 2^32, widened to FP32 (golden.arith.up):
    up(w) * up(x) (mul), and up(w) * up(x) + acc (fma) with a random FP32 acc per pair,
    checked against golden.arith.mac_f32, the lane's own function. A BF16
    product is exact in FP32 unless it underflows or overflows, so those
    products are where this can fail."""
    up = all_bf16_widened()
    x = np.tile(up, W_PER_CHUNK)
    n = len(x)
    golden = arith.mac_f32 if op == "fma" else None

    def one(chunk: int) -> None:
        seed = SEED + chunk
        w = np.repeat(up[chunk * W_PER_CHUNK : (chunk + 1) * W_PER_CHUNK], 2**16)
        acc = random_bits(np.random.default_rng(seed), n) if op == "fma" else np.zeros(n, np.uint32)
        check(op, np.stack([w, x, acc], axis=1), f"w chunk {chunk}, seed {seed}", golden=golden)

    bulk.in_parallel(one, range(2**16 // W_PER_CHUNK))


@pytest.mark.slow
@rtl.needs_verilator
@pytest.mark.parametrize("op", OPS)
def test_random_slow(op: str) -> None:
    """10⁸ inputs in chunks of about 10⁶, each with its own seed (printed on
    failure), bulk.WORKERS at a time."""
    per_chunk = len(batch(np.random.default_rng(0), N_FAMILY_SLOW, N_RANDOM_SLOW))
    seeds = bulk.chunk_seeds(SEED, N_SLOW, per_chunk)

    def one(seed: int) -> None:
        check(op, batch(np.random.default_rng(seed), N_FAMILY_SLOW, N_RANDOM_SLOW), seed)

    bulk.in_parallel(one, seeds)
