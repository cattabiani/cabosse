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

import itertools
import subprocess
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import bulk
import numpy as np
import paths
import pytest
import torch
from fp_inputs import bf16_rounded, f32_classes, with_exponents
from golden import arith, decoder, dot, settings
from lane import FULL_RATE_LEN, E, golden_dot, random_gaps

import rtl

SEED = 20261009
SMOLLM2_LENGTHS = (64, 576, 1536)  # head dimension, hidden, intermediate
LONG_LEN = 8192  # p.V rows at SmolLM2's full window
FTZ = (("Ftz", 1),)
FIELDS = E + 1  # words per beat record: E / 2 of w, E / 2 of x, then control
N_FAST = 300_000  # pairs per fast test
N_SLOW = 10**8
PAIRS_PER_CHUNK = 2_000_000  # slow run: one driver run per chunk
MAX_LATENCY_CYCLES = 64  # from a row's last beat to its result, more than the lane's
N_FLUSHED = 50  # rows the ftz switch changes, at least
# Output ready, permille of cycles, cycled through the chunks.
READY_PERMILLE = (1000, 500, 50)


@dataclass
class Rows:
    """Rows of one length: BF16 bits w, x [rows, K] and which elements are
    there (default: all)."""

    w: np.ndarray  # uint16
    x: np.ndarray  # uint16
    valid: np.ndarray | None = None  # bool

    def __post_init__(self) -> None:
        if self.valid is None:
            self.valid = np.ones(self.w.shape, bool)


@dataclass
class Run:
    results: np.ndarray  # uint32, one per row
    cycles: int
    stalls: int
    n_beats: int
    n_gaps: int  # idle cycles requested before the beats


def expected(groups: list[Rows]) -> np.ndarray:
    """golden.dot.dot of every row, as FP32 bits, in order."""
    return np.concatenate([golden_dot(g.w, g.x, g.valid) for g in groups])


def records(groups: list[Rows]) -> np.ndarray:
    """The driver's beats for every row, in order (verif/bulk/dot_lane.cpp):
    E elements a beat, two BF16 values to a word, a short last beat masked;
    no gaps yet."""
    out = []
    for g in groups:
        n_rows, k = g.w.shape
        n_beats = -(-k // E)
        pad = ((0, 0), (0, n_beats * E - k))
        w, x = (np.pad(v, pad).astype(np.uint32).reshape(n_rows, n_beats, E) for v in (g.w, g.x))
        valid = np.pad(g.valid, pad).reshape(n_rows, n_beats, E)
        mask = (valid.astype(np.uint32) << np.arange(E, dtype=np.uint32)).sum(-1, dtype=np.uint32)
        last = np.zeros((n_rows, n_beats), np.uint32)
        last[:, -1] = 1
        words = [v[..., 0::2] | v[..., 1::2] << 16 for v in (w, x)]
        out.append(np.concatenate([*words, (mask | last << 4)[..., None]], -1).reshape(-1, FIELDS))
    return np.concatenate(out)


def check(
    groups: list[Rows],
    label: str,
    params: tuple = (),
    ready: int = 1000,
    p_gap: float = 0.0,
    seed: int = SEED,
    want: np.ndarray | None = None,
) -> Run:
    """Stream the rows through the lane, with random gaps (lane.random_gaps)
    and output ready both drawn from `seed`, and compare every result with
    the golden model's, or with `want` when given."""
    rec = records(groups)
    gaps = random_gaps(np.random.default_rng(seed), len(rec), p_gap)
    rec[:, -1] |= gaps.astype(np.uint32) << 8
    argv = [str(bulk.build("dot_lane", params)), f"+ready={ready}", f"+seed={seed}"]
    result = subprocess.run(argv, input=memoryview(rec), capture_output=True)
    assert result.returncode == 0, f"{label}: {result.stderr.decode()}"
    words = np.frombuffer(result.stdout, dtype="<u4")
    got = Run(words[:-2], int(words[-2]), int(words[-1]), len(rec), int(gaps.sum()))

    want = expected(groups) if want is None else want
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
    return [make(k, r) for k, r in Counter(ks).items()]


def model_like(
    rng: np.random.Generator, n: int, choices: tuple[int, ...] | None = SMOLLM2_LENGTHS
) -> list[Rows]:
    """Normal values at a spread of scales, both signs, as a model's."""

    def make(k: int, r: int) -> Rows:
        scaled = (rng.normal(0, 1, (r, k)) * 2.0 ** rng.integers(-8, 8, (r, k)) for _ in range(2))
        return Rows(*(bf16_rounded(v) for v in scaled))

    return grouped(make, lengths_for(rng, n, choices))


def random_bits(rng: np.random.Generator, n: int) -> list[Rows]:
    """Any BF16 bits: NaNs, infinities, subnormals and zeros among them. Rows
    are short (1 .. 32): a long one almost always holds a NaN or Inf * 0."""

    def make(k: int, r: int) -> Rows:
        return Rows(*(rng.integers(0, 2**16, (r, k)).astype(np.uint16) for _ in range(2)))

    return grouped(make, lengths_for(rng, n, None, longest=32))


def range_edges(rng: np.random.Generator, n: int) -> list[Rows]:
    """Products below FP32's normal range (factors of 2^-72 .. 2^-64, so
    products of 2^-144 .. 2^-127: subnormal partial sums and results, some
    underflowing to zero) and near its largest (factors of 2^53 .. 2^73:
    large finite sums, overflow to Inf, Inf - Inf), half each."""

    def make(k: int, r: int) -> Rows:
        lo, hi = (55, 63) if rng.random() < 0.5 else (180, 200)
        return Rows(*(with_exponents(rng, (r, k), lo, hi, 16).astype(np.uint16) for _ in range(2)))

    return grouped(make, lengths_for(rng, n, None))


def cancelling(rng: np.random.Generator, n: int) -> list[Rows]:
    """Rows [a, a] . [b, -b]: the second half cancels the first, exactly
    for small integers (half the rows), up to rounding otherwise, so most
    leading bits vanish in the partial sums or in the final tree."""

    def make(k: int, r: int) -> Rows:
        half = (k + 1) // 2
        if rng.random() < 0.5:
            a, b = (bf16_rounded(rng.integers(-8, 9, (r, half))) for _ in range(2))
        else:
            a, b = (bf16_rounded(rng.normal(0, 1, (r, half))) for _ in range(2))
        w = np.concatenate([a, a], axis=1)[:, :k]
        x = np.concatenate([b, b ^ 0x8000], axis=1)[:, :k]
        return Rows(w, x)

    return grouped(make, lengths_for(rng, n, None))


def masked(rng: np.random.Generator, n: int) -> list[Rows]:
    """Model-like rows with missing elements at random, a missing element
    carrying random bits (feeding it would show); and rows of products that
    underflow to -0 with missing elements carrying zeros, whose result is -0
    only if every partial sum kept its -0 (numerics.md, section 3)."""
    groups = model_like(rng, n // 2, None)
    for g in groups:
        g.valid = rng.random(g.w.shape) >= rng.choice([0.1, 0.5, 0.9])
        n_missing = int((~g.valid).sum())
        g.w[~g.valid] = rng.integers(0, 2**16, n_missing)
        g.x[~g.valid] = rng.integers(0, 2**16, n_missing)

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


# The results each family must reach, at least, on N_REACH pairs: a family
# that never reaches its case tests nothing (REVIEW.md).
N_REACH = 60_000
REACHES = {
    random_bits: {"nan": 100, "inf": 100, "normal": 100},
    range_edges: {"subnormal": 50, "nan": 50, "normal": 20},
    cancelling: {"+0": 50},
    masked: {"-0": 50},
}


@pytest.mark.parametrize("family", REACHES, ids=lambda f: f.__name__)
def test_families_reach_their_cases(family: Callable) -> None:
    """Golden model only: the expected results' classes, counted."""
    counts = f32_classes(expected(family(np.random.default_rng(SEED), N_REACH)))
    short = {c: n for c, n in REACHES[family].items() if counts[c] < n}
    assert not short, f"{family.__name__} reaches {dict(counts)}, needs {REACHES[family]}"


# --- Fast tests ------------------------------------------------------------------


@rtl.needs_verilator
@pytest.mark.parametrize("ready", READY_PERMILLE)
def test_families(ready: int) -> None:
    seed = SEED + ready
    groups = mixed(np.random.default_rng(seed), N_FAST)
    check(groups, f"ready {ready}‰, seed {seed}", ready=ready, p_gap=0.2, seed=seed)


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
    got = check(groups, f"full rate, gap probability {p_gap}", p_gap=p_gap)
    assert got.stalls == 0, f"{got.stalls} stalls"
    latency = got.cycles - got.n_beats - got.n_gaps
    assert 0 < latency <= MAX_LATENCY_CYCLES, (
        f"{got.cycles} cycles for {got.n_beats} beats and {got.n_gaps} gaps"
    )


def ftz_rows(rng: np.random.Generator) -> list[Rows]:
    """Rows whose products and sums cross FP32's subnormal range."""

    def make(k: int, r: int) -> Rows:
        return Rows(*(with_exponents(rng, (r, k), 0, 70, 16).astype(np.uint16) for _ in range(2)))

    return grouped(make, lengths_for(rng, N_FAST // 2, None)) + model_like(rng, N_FAST // 4, None)


@rtl.needs_verilator
def test_ftz() -> None:
    """The Ftz = 1 build against the golden model's ftz switch, on rows the
    switch changes (at least N_FLUSHED of them)."""
    rng = np.random.default_rng([SEED, 1])
    groups = ftz_rows(rng)
    with settings.override(ftz=True):
        flushed = expected(groups)
    check(groups, f"ftz, seed {SEED}", FTZ, ready=500, p_gap=0.2, want=flushed)
    n_changed = int((flushed != expected(groups)).sum())
    assert n_changed >= N_FLUSHED, f"only {n_changed} rows change with ftz"


# --- Slow tests ------------------------------------------------------------------


def volume_seeds() -> list[int]:
    """One seed per chunk of the volume run."""
    return bulk.chunk_seeds(SEED, N_SLOW, PAIRS_PER_CHUNK)


def volume_chunk(seed: int) -> list[Rows]:
    """The rows of one chunk of the volume run: about PAIRS_PER_CHUNK pairs."""
    return mixed(np.random.default_rng(seed), PAIRS_PER_CHUNK)


@pytest.mark.slow
@rtl.needs_verilator
def test_volume() -> None:
    """N_SLOW pairs in chunks, each with its own seed, output ready and gaps."""

    def one(seed: int) -> None:
        ready = READY_PERMILLE[seed % len(READY_PERMILLE)]
        groups = volume_chunk(seed)
        p_gap = 0.2 if seed % 2 else 0.0
        check(groups, f"chunk seed {seed}, ready {ready}‰", ready=ready, p_gap=p_gap, seed=seed)

    bulk.in_parallel(one, volume_seeds())


def smollm2_rows(model: decoder.Model) -> dict[str, list[Rows]]:
    """The dot products of one SmolLM2 decode step, as the lanes see them:
    every matrix of the first and last layers against its real input, the
    attention scores and p.V rows of those layers, and the first 4096 rows
    of the classifier. The step runs at position 63 after a 63-token prompt."""
    seen: dict[str, list[Rows]] = {}
    n_layers = len(model.layers)
    layers = {0, n_layers - 1}
    matrices = ("q", "k", "v", "o", "gate", "up", "down")  # a layer's matvec calls, in order
    n_matvec, n_scores, n_values = itertools.count(), itertools.count(), itertools.count()

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
        layer, j = divmod(next(n_matvec), len(matrices))
        if layer == n_layers:
            add("classifier", w[:4096], x[-1][None, :], None)
        elif layer in layers:
            add(f"layer {layer} {matrices[j]}", w, x[-1][None, :], None)
        return matvec(w, x)

    def rec_scores(q: torch.Tensor, k: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if (i := next(n_scores)) in layers:
            add(f"layer {i} scores", k, q[..., None, :], None)
        return scores(q, k, scale)

    def rec_values(
        p: torch.Tensor, v: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        if (i := next(n_values)) in layers:
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
