# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Tests for golden.arith against torch and against the exact oracle."""

import numpy as np
import pytest
import torch
from golden import arith, settings
from oracle import add_ref, fma_ref, mul_ref

SEED = 20261002
N_ORACLE = 20_000  # per input family; the Fraction oracle is slow
F32 = np.float32


def bits(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, F32).view(np.uint32)


def t32(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, F32))


def assert_same_bits(got: torch.Tensor, want: np.ndarray, inputs: tuple, what: str) -> None:
    """Bit-exact comparison; NaNs must be the canonical NaN."""
    got_bits = bits(got.numpy())
    want = np.asarray(want, F32)
    want_bits = np.where(np.isnan(want), arith.NAN_F32_BITS, bits(want))
    bad = np.nonzero(got_bits != want_bits)[0]
    if len(bad):
        i = bad[0]
        args = ", ".join(f"{np.asarray(x, F32)[i]!r} ({bits(x)[i]:#010x})" for x in inputs)
        pytest.fail(
            f"{what}: {len(bad)}/{len(want)} mismatches (seed {SEED}). First: "
            f"{what}({args}) = {got_bits[i]:#010x}, expected {want_bits[i]:#010x}"
        )


# --- input families ----------------------------------------------------------


def moderate(rng: np.random.Generator, n: int) -> np.ndarray:
    """Values around 1 with random signs: the common case."""
    sign = rng.choice([-1.0, 1.0], n)
    return F32(sign * rng.uniform(1, 2, n) * 2.0 ** rng.integers(-20, 21, n))


def any_finite(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniformly random finite bit patterns: extreme magnitudes, subnormals."""
    x = rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32).view(F32)
    return np.where(np.isfinite(x), x, F32(1.0))


def families(rng: np.random.Generator, n: int) -> dict[str, tuple[np.ndarray, ...]]:
    """(a, b, c) triples that stress different parts of the rounding."""

    def sign() -> np.ndarray:
        return rng.choice([-1.0, 1.0], n)

    def mag(lo: int, hi: int) -> np.ndarray:
        """Random magnitudes in [2**lo, 2**hi)."""
        return rng.uniform(1, 2, n) * 2.0 ** rng.integers(lo, hi, n)

    a, b, c = moderate(rng, n), moderate(rng, n), moderate(rng, n)
    fam = {"moderate": (a, b, c), "any_finite": tuple(any_finite(rng, n) for _ in range(3))}

    # Cancellation: c is close to -(a*b), so most leading bits cancel.
    p = (a.astype(np.float64) * b).astype(F32)
    k = rng.integers(-3, 4, n)
    fam["cancellation"] = (a, b, _step_ulps(-p, k))

    # Exact ties: a*b is exactly half an ulp of c (a is a power of two).
    e = rng.integers(-100, 100, n)
    c_t = F32(sign() * rng.uniform(1, 2, n) * 2.0**e)
    i = rng.integers(-10, 11, n)
    a_t = F32(2.0**i)
    b_t = F32(sign() * 2.0 ** (e - 24 - i))
    fam["ties"] = (a_t, b_t, c_t)

    # Near ties: b moved by one of its own ulps, so a*b is just above or below
    # half an ulp of c.
    fam["near_ties"] = (a_t, _step_ulps(b_t, rng.choice([-1, 1], n)), c_t)

    # Double-rounding traps: a*b = (1 + 2**-23)(1 - 2**-23) * half-ulp(c)
    # = half-ulp(c) * (1 - 2**-46). The exact sum is a hair from a tie, closer
    # than float64 can see, so a plain float64 FMA rounds twice and fails.
    a_dr = F32((1 + 2.0**-23) * 2.0**i)
    b_dr = F32(sign() * (1 - 2.0**-23) * 2.0 ** (e - 24 - i))
    fam["double_rounding"] = (a_dr, b_dr, c_t)

    # Subnormal results and products that underflow FP32.
    c_tiny = F32(sign() * 2.0 ** rng.integers(-149, -126, n))
    fam["tiny"] = (F32(mag(-80, -60)), F32(sign() * mag(-80, -60)), c_tiny)

    # Results near the overflow threshold.
    fam["huge"] = (F32(mag(60, 68)), F32(mag(60, 68)), F32(sign() * 2.0**127))
    return fam


def _step_ulps(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Move each x by k[i] FP32 ulps."""
    out = np.asarray(x, F32).copy()
    for direction in (1, -1):
        for _ in range(int(np.abs(k).max())):
            m = (np.sign(k) == direction) & (np.abs(k) > 0)
            out[m] = np.nextafter(out[m], F32(direction * np.inf))
            k = np.where(m, k - direction, k)
    return out


SPECIALS = F32(
    [
        0.0,
        -0.0,
        1.0,
        -1.0,
        np.inf,
        -np.inf,
        np.nan,
        2.0**-149,
        -(2.0**-149),
        2.0**-126,
        np.finfo(F32).max,
        -np.finfo(F32).max,
        1.0 + 2.0**-23,
    ]
)


# --- bf16 / up -----------------------------------------------------------------


def _check_bf16(x: np.ndarray) -> None:
    got = arith.bits_bf16(arith.bf16(t32(x))).numpy()
    want = arith.bits_bf16(t32(x).to(torch.bfloat16)).numpy()
    nan = np.isnan(x)
    assert (got[nan] == arith.NAN_BF16_BITS).all(), "NaN must become the canonical BF16 NaN"
    bad = np.nonzero(got[~nan] != want[~nan])[0]
    assert not len(bad), (
        f"bf16 differs from torch on {len(bad)} inputs, first "
        f"{bits(x[~nan])[bad[0]]:#010x} (seed {SEED})"
    )


def test_bf16_random_and_specials() -> None:
    rng = np.random.default_rng(SEED)
    x = np.concatenate(
        [
            any_finite(rng, 1 << 22),
            moderate(rng, 1 << 20),
            SPECIALS,
            rng.integers(0x7F800001, 0x80000000, 1000, dtype=np.uint64).astype(np.uint32).view(F32),
        ]
    )  # NaN payloads
    _check_bf16(x)


@pytest.mark.slow
def test_bf16_exhaustive() -> None:
    """All 2**32 FP32 inputs (about two minutes)."""
    chunk = 1 << 26
    for start in range(0, 1 << 32, chunk):
        _check_bf16(np.arange(start, start + chunk, dtype=np.uint64).astype(np.uint32).view(F32))


def test_up_exhaustive() -> None:
    """All 2**16 BF16 inputs: exact widening, canonical NaN."""
    u = torch.arange(1 << 16, dtype=torch.int64)
    b = arith.bf16_from_bits(u)
    got = arith.bits_f32(arith.up(b))
    want = torch.where(torch.isnan(b), arith.NAN_F32_BITS, u << 16)
    assert torch.equal(got, want)


def test_bf16_then_up_is_identity_on_bf16_values() -> None:
    b = arith.bf16_from_bits(torch.arange(1 << 16, dtype=torch.int64))
    keep = ~torch.isnan(b)
    assert torch.equal(arith.bits_bf16(arith.bf16(arith.up(b)))[keep], arith.bits_bf16(b)[keep])


# --- add / mul / fma against the exact oracle --------------------------------------


@pytest.mark.parametrize("op", ["add", "mul", "fma"])
def test_against_oracle(op: str) -> None:
    rng = np.random.default_rng(SEED)
    for name, (a, b, c) in families(rng, N_ORACLE).items():
        if op == "fma":
            got = arith.fma(t32(a), t32(b), t32(c))
            want = [fma_ref(*t) for t in zip(a, b, c, strict=True)]
            inputs = (a, b, c)
        elif op == "mul":
            got = arith.mul(t32(a), t32(b))
            want = [mul_ref(*t) for t in zip(a, b, strict=True)]
            inputs = (a, b)
        else:
            got = arith.add(t32(a), t32(c))
            want = [add_ref(*t) for t in zip(a, c, strict=True)]
            inputs = (a, c)
        assert_same_bits(got, np.array(want, F32), inputs, f"{op}[{name}]")


def test_fma_special_values() -> None:
    """All combinations of special inputs; non-finite cases follow IEEE via float64."""
    a, b, c = (x.ravel() for x in np.meshgrid(SPECIALS, SPECIALS, SPECIALS, indexing="ij"))
    got = arith.fma(t32(a), t32(b), t32(c))
    finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    want = np.empty_like(a)
    want[finite] = [fma_ref(*t) for t in zip(a[finite], b[finite], c[finite], strict=True)]
    with np.errstate(invalid="ignore", over="ignore"):
        want[~finite] = (a[~finite].astype(np.float64) * b[~finite] + c[~finite]).astype(F32)
    assert_same_bits(got, want, (a, b, c), "fma[specials]")


def test_fma_rounds_once() -> None:
    """A case where a separate multiply and add round twice and give a different answer."""
    a = t32(F32([1.0 + 2.0**-12]))
    b = t32(F32([1.0 + 2.0**-12]))
    c = t32(F32([-1.0]))
    exact = (1 + 2**-12) ** 2 - 1  # 2**-11 + 2**-24, representable
    assert arith.fma(a, b, c).item() == exact
    assert arith.add(arith.mul(a, b), c).item() != exact


@pytest.mark.parametrize("sign", [0, 0x80000000], ids=["positive", "negative"])
def test_fma_avoids_float64_double_rounding(sign: int) -> None:
    """A case where a plain float64 FMA is wrong by one ulp, in both signs.

    The exact a*b + c lies 0.4999999999999929 ulp above |result| 0x555d95cb, a
    hair below the tie. float64 rounds it to exactly the tie, and the second
    rounding (ties to even) then goes to 0x555d95cc. Correct: 0x555d95cb.
    The negative case flips the signs of a and c, so the result is mirrored.
    """
    a, b, c = (
        np.array([x], np.uint32).view(F32)
        for x in (0x41000001 | sign, 0x477FFFFE, 0x555D95CB | sign)
    )
    plain = (a.astype(np.float64) * b + c).astype(F32)
    assert bits(plain)[0] == 0x555D95CC | sign  # the wrong answer: the case is a real trap
    assert int(bits(fma_ref(a[0], b[0], c[0]))) == 0x555D95CB | sign
    assert arith.bits_f32(arith.fma(t32(a), t32(b), t32(c))).item() == 0x555D95CB | sign


# --- mac -------------------------------------------------------------------------


def test_mac_equals_fma_of_widened_inputs() -> None:
    rng = np.random.default_rng(SEED)
    w = arith.bf16(t32(moderate(rng, 1 << 16)))
    x = arith.bf16(t32(any_finite(rng, 1 << 16)))
    acc = t32(moderate(rng, 1 << 16))
    want = arith.fma(arith.up(w), arith.up(x), acc)
    assert torch.equal(arith.bits_f32(arith.mac(w, x, acc)), arith.bits_f32(want))


def test_mac_product_is_exact_for_normal_range() -> None:
    """For BF16 inputs the product is exact in FP32, so mac == add(mul(...))."""
    rng = np.random.default_rng(SEED)
    w = arith.bf16(t32(moderate(rng, 1 << 16)))
    x = arith.bf16(t32(moderate(rng, 1 << 16)))
    acc = t32(moderate(rng, 1 << 16))
    separate = arith.add(arith.mul(arith.up(w), arith.up(x)), acc)
    assert torch.equal(arith.bits_f32(arith.mac(w, x, acc)), arith.bits_f32(separate))


# --- flush to zero (experiments only) ----------------------------------------------


def test_flush_to_zero() -> None:
    sub = t32(F32([2.0**-140, -(2.0**-140)]))
    one = t32(F32([1.0, 1.0]))
    assert torch.equal(arith.add(sub, sub * 0), sub)  # default: kept
    with settings.override(ftz=True):
        out = arith.add(sub, sub * 0)
        assert (out == 0).all() and torch.equal(torch.signbit(out), torch.tensor([False, True]))
        assert (arith.mul(sub, one) == 0).all()
        assert (arith.up(arith.bf16_from_bits(torch.tensor([0x0001]))) == 0).all()
    assert torch.equal(arith.mul(sub, one), sub)  # restored afterwards


def test_override_rejects_unknown_settings() -> None:
    with pytest.raises(TypeError), settings.override(fzt=True):
        pass


def test_mac_fast_path_falls_back_when_product_is_inexact() -> None:
    """Tiny BF16 products underflow FP32 (subnormal with lost bits): there a
    plain multiply-then-add rounds twice. mac must still equal the one-rounding fma."""
    rng = np.random.default_rng(SEED)
    n = 1 << 14
    w = arith.bf16(t32(F32(rng.uniform(1, 2, n) * 2.0 ** rng.integers(-80, -70, n))))
    x = arith.bf16(t32(F32(rng.choice([-1.0, 1.0], n) * rng.uniform(1, 2, n) * 2.0**-70)))
    acc = t32(F32(rng.choice([-1.0, 1.0], n) * 2.0 ** rng.integers(-130, -120, n)))
    a, b = arith.up(w), arith.up(x)
    want = arith.bits_f32(arith.fma(a, b, acc))
    assert torch.equal(arith.bits_f32(arith.mac(w, x, acc)), want)
    naive = arith.bits_f32(arith.add(arith.mul(a, b), acc))
    assert (naive != want).any()  # the inputs really exercise the slow path


def test_mac_fast_path_falls_back_on_overflow() -> None:
    """2^100 * 2^100 overflows FP32 but is finite and exact in the fma: with
    acc = -inf the result is -inf. A multiply-then-add would give inf + -inf = NaN."""
    big = arith.bf16(t32(F32([2.0**100])))
    acc = t32(F32([-np.inf]))
    assert arith.mac(big, big, acc).item() == -np.inf
    assert torch.isnan(arith.add(arith.mul(arith.up(big), arith.up(big)), acc)).all()
