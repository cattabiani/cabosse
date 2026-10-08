# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/dot_lane.sv (PLAN.md, M4) against golden.dot.dot, bit for bit, in
volume, through the bulk driver verif/bulk/dot_lane.cpp: model-like rows at
SmolLM2's lengths (64, 576, 1536, and p.V rows to 8192), random lengths,
random bits, products near the ends of FP32's range, cancelling rows, missing
elements and -0 rows, with input gaps and a random output ready. The fast run
checks a few hundred thousand pairs and the cycle count of full-rate rows;
the slow run checks 10^8 pairs, and real SmolLM2 rows with the activations
of a decode step. The flush-to-zero build (Ftz = 1) against the golden
model's ftz switch.

The handshake's corner cases are in test_dot_lane (cocotb); here it is
volume."""

import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import bulk
import numpy as np
import paths
import pytest
import torch
from golden import arith, decoder, dot, settings

import rtl

SEED = 20261009
E = 4  # pairs per beat: dot_lane's default
FULL_RATE_LEN = 60  # (A - 1) * E, as test_dot_lane
SMOLLM2_LENGTHS = (64, 576, 1536)  # head dimension, hidden, intermediate
LONG_LEN = 8192  # p.V rows at SmolLM2's full window
FTZ = (("Ftz", 1),)
N_FAST = 300_000  # pairs per fast test
N_SLOW = 10**8
PAIRS_PER_CHUNK = 2_000_000  # slow run: one driver run per chunk
DRAIN_CYCLES = 64  # more than the lane's latency from last beat to result
N_FLUSHED = 50  # rows the ftz switch changes, at least
# Output ready, permille of cycles, cycled through the chunks.
READY_PERMILLE = (1000, 500, 50)


@dataclass
class Rows:
    """Rows of one length: BF16 bits w, x [rows, K] and which elements are
    there."""

    w: np.ndarray  # uint16
    x: np.ndarray  # uint16
    valid: np.ndarray  # bool

    @property
    def n_pairs(self) -> int:
        return self.w.size


@dataclass
class Run:
    results: np.ndarray  # uint32, one per row
    cycles: int
    stalls: int


def bf16_bits(values: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(torch.bfloat16)
    return arith.bits_bf16(t).numpy().astype(np.uint16)


def expected(groups: list[Rows]) -> np.ndarray:
    """golden.dot.dot of every row, as FP32 bits, in order."""
    out = []
    for g in groups:
        w, x = (arith.bf16_from_bits(torch.from_numpy(v.astype(np.int64))) for v in (g.w, g.x))
        y = dot.dot(w, x, valid=torch.from_numpy(g.valid))
        out.append(arith.bits_f32(y).numpy().astype(np.uint32))
    return np.concatenate(out)


def records(groups: list[Rows], gaps: np.ndarray | None = None) -> np.ndarray:
    """The driver's beats for every row, in order (verif/bulk/dot_lane.cpp):
    E elements a beat, a short last beat masked, gaps[i] idle cycles before
    beat i (default none)."""
    out = []
    for g in groups:
        n_rows, k = g.w.shape
        n_beats = -(-k // E)
        pad = n_beats * E - k
        w, x = (
            np.pad(v, ((0, 0), (0, pad))).astype(np.uint32).reshape(n_rows, n_beats, E)
            for v in (g.w, g.x)
        )
        valid = np.pad(g.valid, ((0, 0), (0, pad))).reshape(n_rows, n_beats, E)
        mask = (valid.astype(np.uint32) << np.arange(E, dtype=np.uint32)).sum(-1, dtype=np.uint32)
        last = np.zeros((n_rows, n_beats), np.uint32)
        last[:, -1] = 1
        rec = np.stack(
            [w[..., 0] | w[..., 1] << 16, w[..., 2] | w[..., 3] << 16,
             x[..., 0] | x[..., 1] << 16, x[..., 2] | x[..., 3] << 16, mask | last << 4],
            axis=-1,
        )  # fmt: skip
        out.append(rec.reshape(-1, 5))
    rec = np.concatenate(out)
    if gaps is not None:
        rec[:, 4] |= gaps.astype(np.uint32) << 8
    return rec


def run(rec: np.ndarray, params: tuple = (), ready_permille: int = 1000, seed: int = SEED) -> Run:
    """Stream the beats through the lane; its results, cycles and stalls."""
    binary = bulk.build("dot_lane", params)
    argv = [str(binary), f"+ready={ready_permille}", f"+seed={seed}"]
    data = memoryview(np.ascontiguousarray(rec, dtype="<u4"))
    result = subprocess.run(argv, input=data, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    words = np.frombuffer(result.stdout, dtype="<u4")
    return Run(words[:-2], int(words[-2]), int(words[-1]))


def check(groups: list[Rows], label: str, params: tuple = (), ready: int = 1000, gaps=None) -> Run:
    """Run the rows and compare every result with the golden model."""
    rec = records(groups, gaps)
    got = run(rec, params, ready)
    want = expected(groups)
    assert len(got.results) == len(want), (
        f"{label}: {len(got.results)} results for {len(want)} rows"
    )
    bad = np.flatnonzero(got.results != want)
    if len(bad):
        lengths = np.concatenate([np.full(g.w.shape[0], g.w.shape[1]) for g in groups])
        first = ", ".join(
            f"row {i} (length {lengths[i]}): got 0x{got.results[i]:08X}, want 0x{want[i]:08X}"
            for i in bad[:5]
        )
        raise AssertionError(f"{label}: {len(bad)} of {len(want)} rows differ; {first}")
    return got


# --- Families: each makes rows of about n pairs ---------------------------------


def lengths_for(
    rng: np.random.Generator, n: int, choices: tuple[int, ...] | None, longest: int = 300
) -> Iterator[int]:
    """Row lengths summing to about n: from `choices`, or random 1 .. longest."""
    total = 0
    while total < n:
        k = int(rng.choice(choices)) if choices else int(rng.integers(1, longest + 1))
        total += k
        yield k


def grouped(make: Callable[[int, int], Rows], ks: Iterator[int]) -> list[Rows]:
    """make(k, rows) for each distinct length, as many rows as it occurs."""
    counts: dict[int, int] = {}
    for k in ks:
        counts[k] = counts.get(k, 0) + 1
    return [make(k, r) for k, r in counts.items()]


def model_like(rng: np.random.Generator, n: int, choices=SMOLLM2_LENGTHS) -> list[Rows]:
    """Normal values at a spread of scales, both signs, as a model's."""

    def make(k: int, r: int) -> Rows:
        w = bf16_bits(rng.normal(0, 1, (r, k)) * 2.0 ** rng.integers(-8, 8, (r, k)))
        x = bf16_bits(rng.normal(0, 1, (r, k)) * 2.0 ** rng.integers(-8, 8, (r, k)))
        return Rows(w, x, np.ones((r, k), bool))

    return grouped(make, lengths_for(rng, n, choices))


def random_bits(rng: np.random.Generator, n: int) -> list[Rows]:
    """Any BF16 bits: NaNs, infinities, subnormals and zeros among them. Rows
    are short (1 .. 32): a long one almost always holds a NaN or Inf * 0."""

    def make(k: int, r: int) -> Rows:
        w, x = (rng.integers(0, 2**16, (r, k)).astype(np.uint16) for _ in range(2))
        return Rows(w, x, np.ones((r, k), bool))

    return grouped(make, lengths_for(rng, n, None, longest=32))


def with_exponents(rng: np.random.Generator, shape: tuple, lo: int, hi: int) -> np.ndarray:
    """BF16 bits of either sign with an exponent field in lo .. hi."""
    sign = rng.integers(0, 2, shape).astype(np.uint16) << 15
    exp = rng.integers(lo, hi + 1, shape).astype(np.uint16) << 7
    return sign | exp | rng.integers(0, 128, shape).astype(np.uint16)


def range_edges(rng: np.random.Generator, n: int) -> list[Rows]:
    """Products below FP32's normal range (factors of 2^-72 .. 2^-64, so
    products of 2^-144 .. 2^-127: subnormal partial sums and results, some
    underflowing to zero) and near its largest (factors of 2^53 .. 2^73:
    large finite sums, overflow to Inf, Inf - Inf), half each."""

    def make(k: int, r: int) -> Rows:
        lo, hi = (55, 63) if rng.random() < 0.5 else (180, 200)
        w, x = (with_exponents(rng, (r, k), lo, hi) for _ in range(2))
        return Rows(w, x, np.ones((r, k), bool))

    return grouped(make, lengths_for(rng, n, None))


def cancelling(rng: np.random.Generator, n: int) -> list[Rows]:
    """Rows [a, a] . [b, -b]: the second half cancels the first, exactly
    for small integers (half the rows), up to rounding otherwise, so most
    leading bits vanish in the partial sums or in the final tree."""

    def make(k: int, r: int) -> Rows:
        half = (k + 1) // 2
        if rng.random() < 0.5:
            a, b = (bf16_bits(rng.integers(-8, 9, (r, half))) for _ in range(2))
        else:
            a, b = (bf16_bits(rng.normal(0, 1, (r, half))) for _ in range(2))
        w = np.concatenate([a, a], axis=1)[:, :k]
        x = np.concatenate([b, b ^ 0x8000], axis=1)[:, :k]
        return Rows(w, x, np.ones((r, k), bool))

    return grouped(make, lengths_for(rng, n, None))


def masked(rng: np.random.Generator, n: int) -> list[Rows]:
    """Model-like rows with missing elements at random, a missing element
    carrying zeros (feeding it would show); and rows of products that
    underflow to -0 with missing elements, whose result is -0 only if every
    partial sum kept its -0."""
    groups = model_like(rng, n // 2, None)
    for g in groups:
        g.valid = rng.random(g.w.shape) >= rng.choice([0.1, 0.5, 0.9])
        g.w[~g.valid] = 0
        g.x[~g.valid] = 0

    def make(k: int, r: int) -> Rows:
        valid = rng.random((r, k)) >= 0.3
        w = np.where(valid, 0x9780, 0).astype(np.uint16)  # -2^-80, or a zero
        return Rows(w, np.full((r, k), 0x1780, np.uint16), valid)  # 2^-80

    return groups + grouped(make, lengths_for(rng, n - n // 2, None))


FAMILIES = (model_like, random_bits, range_edges, cancelling, masked)


def mixed(rng: np.random.Generator, n: int) -> list[Rows]:
    """Every family, about n / len(FAMILIES) pairs each, and a few long p.V
    rows."""
    groups = [g for fam in FAMILIES for g in fam(rng, n // len(FAMILIES))]
    return groups + model_like(rng, LONG_LEN, (LONG_LEN,))


def random_gaps(rng: np.random.Generator, groups: list[Rows], p_gap: float) -> np.ndarray:
    """Idle cycles before each beat: geometric, a gap of k with probability
    (1 - p_gap) * p_gap^k."""
    n_beats = sum(g.w.shape[0] * -(-g.w.shape[1] // E) for g in groups)
    return rng.geometric(1 - p_gap, n_beats) - 1


# --- Fast tests ------------------------------------------------------------------


@rtl.needs_verilator
@pytest.mark.parametrize("ready", READY_PERMILLE)
def test_families(ready: int) -> None:
    rng = np.random.default_rng([SEED, ready])
    groups = mixed(rng, N_FAST)
    check(groups, f"ready {ready}‰, seed {SEED}", ready=ready, gaps=random_gaps(rng, groups, 0.2))


@rtl.needs_verilator
@pytest.mark.parametrize("p_gap", [0.0, 0.3])
def test_full_rate(p_gap: float) -> None:
    """Rows of SmolLM2's lengths and of FULL_RATE_LEN .. 300, with gaps and the
    output always ready: no stalls, so the run takes exactly the beats plus
    the gaps, plus the last row's latency."""
    rng = np.random.default_rng([SEED, int(p_gap * 10)])
    groups = model_like(rng, N_FAST) + model_like(
        rng, N_FAST // 4, tuple(range(FULL_RATE_LEN, 301))
    )
    gaps = random_gaps(rng, groups, p_gap)
    got = check(groups, f"full rate, gap probability {p_gap}", gaps=gaps)
    n_beats = len(gaps)
    assert got.stalls == 0, f"{got.stalls} stalls"
    extra = got.cycles - n_beats - int(gaps.sum())
    assert 0 < extra <= DRAIN_CYCLES, (
        f"{got.cycles} cycles for {n_beats} beats and {gaps.sum()} gaps"
    )


def ftz_rows(rng: np.random.Generator) -> list[Rows]:
    """Rows whose products and sums cross FP32's subnormal range."""

    def make(k: int, r: int) -> Rows:
        w, x = (with_exponents(rng, (r, k), 0, 70) for _ in range(2))
        return Rows(w, x, np.ones((r, k), bool))

    return grouped(make, lengths_for(rng, N_FAST // 2, None)) + model_like(rng, N_FAST // 4, None)


@rtl.needs_verilator
def test_ftz() -> None:
    """The Ftz = 1 build against the golden model's ftz switch, on rows the
    switch changes (at least N_FLUSHED of them)."""
    rng = np.random.default_rng([SEED, 1])
    groups = ftz_rows(rng)
    with settings.override(ftz=True):
        flushed = expected(groups)
        check(groups, f"ftz, seed {SEED}", FTZ, ready=500, gaps=random_gaps(rng, groups, 0.2))
    n_changed = int((flushed != expected(groups)).sum())
    assert n_changed >= N_FLUSHED, f"only {n_changed} rows change with ftz"


# --- Slow tests ------------------------------------------------------------------


@pytest.mark.slow
@rtl.needs_verilator
def test_volume() -> None:
    """N_SLOW pairs in chunks, each with its own seed, output ready and gaps."""
    seeds = bulk.chunk_seeds(SEED, N_SLOW, PAIRS_PER_CHUNK)

    def one(seed: int) -> None:
        rng = np.random.default_rng(seed)
        ready = READY_PERMILLE[seed % len(READY_PERMILLE)]
        groups = mixed(rng, PAIRS_PER_CHUNK)
        gaps = random_gaps(rng, groups, 0.2 if seed % 2 else 0.0)
        check(groups, f"chunk seed {seed}, ready {ready}‰", ready=ready, gaps=gaps)

    bulk.in_parallel(one, seeds)


def smollm2_rows(model: decoder.Model) -> dict[str, list[Rows]]:
    """The dot products of one SmolLM2 decode step, as the lanes see them:
    every matrix of the first and last layers against its real input, the
    attention scores and p.V rows of those layers, and the first 4096 rows
    of the classifier. The step runs at position 63 after a 63-token prompt."""
    seen: dict[str, list[Rows]] = {}
    layers = {0, len(model.layers) - 1}
    calls = {"matvec": 0, "scores": 0, "values": 0}
    matrices = ("q", "k", "v", "o", "gate", "up", "down")  # a layer's matvec calls, in order

    def bits(t: torch.Tensor) -> np.ndarray:
        return arith.bits_bf16(t.contiguous()).numpy().astype(np.uint16)

    def add(name: str, w: torch.Tensor, x: torch.Tensor, valid: torch.Tensor | None) -> None:
        w, x = torch.broadcast_tensors(w, x)
        k = w.shape[-1]
        v = np.ones(w.shape, bool) if valid is None else torch.broadcast_to(valid, w.shape).numpy()
        seen.setdefault(name, []).append(
            Rows(bits(w).reshape(-1, k), bits(x).reshape(-1, k), v.reshape(-1, k))
        )

    matvec, scores, values = dot.matvec, decoder.attention_scores, decoder.attention_values

    def rec_matvec(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        i, calls["matvec"] = calls["matvec"], calls["matvec"] + 1
        layer = i // len(matrices)
        if layer in layers or layer == len(model.layers):
            name = (
                "classifier"
                if layer == len(model.layers)
                else f"layer {layer} {matrices[i % len(matrices)]}"
            )
            rows = w[:4096] if layer == len(model.layers) else w
            add(name, rows, x[-1][None, :], None)
        return matvec(w, x)

    def rec_scores(q: torch.Tensor, k: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        i, calls["scores"] = calls["scores"], calls["scores"] + 1
        if i in layers:
            add(f"layer {i} scores", k, q[..., None, :], None)
        return scores(q, k, scale)

    def rec_values(
        p: torch.Tensor, v: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        i, calls["values"] = calls["values"], calls["values"] + 1
        if i in layers:
            mask = None if valid is None else valid[..., None, :]
            add(f"layer {i} p.V", v.transpose(-1, -2), p[..., None, :], mask)
        return values(p, v, valid)

    prompt = list(range(1000, 1063))
    cache = decoder.KVCache.empty(model, 64)
    decoder.step(model, cache, prompt)
    with pytest.MonkeyPatch.context() as mp:  # only the decode step is recorded
        mp.setattr(dot, "matvec", rec_matvec)
        mp.setattr(decoder, "attention_scores", rec_scores)
        mp.setattr(decoder, "attention_values", rec_values)
        decoder.decode_step(model, cache, 1063)
    return seen


@pytest.mark.slow
@rtl.needs_verilator
@pytest.mark.skipif(not paths.SMOLLM2.exists(), reason=f"needs the checkpoint in {paths.SMOLLM2}")
def test_smollm2_rows() -> None:
    """Real weights against real activations, every row of each matrix."""
    found = smollm2_rows(decoder.load(paths.SMOLLM2))
    assert len(found) == 2 * 9 + 1, sorted(found)
    for name, groups in found.items():
        check(groups, name, ready=500)
