# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Tests for RoPE, SiLU and SwiGLU (golden.vector) and the RoPE tables
(golden.host): bit-exact agreement with an independent restatement of
docs/numerics.md section 4, accuracy, and agreement with `transformers`."""

import numpy as np
import paths
import pytest
import torch
from golden import arith, host, settings, vector
from oracle import add_ref, fma_ref, mul_ref
from test_funcs import max_rel_err, spec_exp, spec_recip, t32
from test_vector import bits_of, extreme, moderate
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

SEED = 20261004
F32 = np.float32


def llama_config(head_dim: int) -> LlamaConfig:
    """SmolLM2-135M's RoPE settings (theta, positions) with any head size."""
    return LlamaConfig(
        hidden_size=9 * head_dim,
        num_attention_heads=9,
        num_key_value_heads=3,
        head_dim=head_dim,
        rope_theta=100000.0,
        max_position_embeddings=8192,
    )


def tables(rng: np.random.Generator, head_dim: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Tables for n random positions in [0, 8192), one row per position."""
    return host.rope_tables(llama_config(head_dim), torch.from_numpy(rng.integers(0, 8192, n)))


# --- independent restatement of the spec (scalar, exact oracle arithmetic) --------


def spec_rope(x: np.ndarray, c: np.ndarray, s: np.ndarray) -> list[np.float32]:
    half = len(x) // 2
    r = np.concatenate([-x[half:], x[:half]])
    return [fma_ref(x[i], c[i], mul_ref(r[i], s[i])) for i in range(len(x))]


def spec_silu(a: np.float32) -> np.float32:
    return mul_ref(a, spec_recip(add_ref(F32(1.0), spec_exp(-a))))


def spec_swiglu(a: np.float32, b: np.float32) -> np.float32:
    return mul_ref(spec_silu(a), b)


# --- RoPE tables ------------------------------------------------------------------


@pytest.mark.skipif(not paths.REFERENCE_PLATFORM, reason=paths.REFERENCE_PLATFORM_NOTE)
def test_rope_tables_pinned() -> None:
    """SmolLM2's tables for every position, pinned by hash.

    They come from torch's float32 cos/sin, which are not correctly rounded
    (about 5% of the values are 1 ulp off), so other platforms may give
    different tables (D-023). A torch or transformers upgrade can also
    change them: then check the change and update the hash.
    """
    cos, sin = host.rope_tables(llama_config(64), torch.arange(8192))
    assert cos.shape == sin.shape == (8192, 64) and cos.dtype == torch.float32
    assert (arith.bits_f32(cos[0]) == 0x3F800000).all() and (arith.bits_f32(sin[0]) == 0).all()
    digest = arith.bits_sha256(cos, sin)
    assert digest == "f9daa3f71dd859b63c24f7abc7e075c4d7ea9831ff81779a710077a8472728af", digest


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Number of FP32 values between a and b (finite inputs; -0 and +0 count as one step)."""

    def ordered(t: torch.Tensor) -> torch.Tensor:
        i = arith.bits_f32(t)
        return torch.where(i >= 2**31, -(i - 2**31), i)  # monotonic in the value

    return (ordered(a) - ordered(b)).abs()


@pytest.mark.parametrize("head_dim", [2, 8, 64])
def test_rope_tables_within_one_ulp(head_dim: int) -> None:
    """Every platform: the tables are at most 1 ulp from correct rounding.

    inv_freq is within 1 ulp of f32(theta^(-2i/d)). The angle is
    pos * inv_freq with one FP32 rounding, the two halves of a row are equal,
    and cos/sin are within 1 ulp of the correctly rounded cos/sin of that
    angle (float64 then rounded once). 1 ulp is the known behaviour of
    torch's float32 math (D-020, D-023); more would be a real error.
    """
    config = llama_config(head_dim)
    inv_freq = LlamaRotaryEmbedding(config).inv_freq
    exact_inv = F32(100000.0 ** (-np.arange(0, head_dim, 2) / head_dim))
    assert ulp_distance(inv_freq, t32(exact_inv)).max().item() <= 1

    positions = torch.arange(8192)
    cos, sin = host.rope_tables(config, positions)
    half = head_dim // 2
    assert torch.equal(cos[:, :half], cos[:, half:]) and torch.equal(sin[:, :half], sin[:, half:])
    angle = arith.mul(positions.to(torch.float32)[:, None], inv_freq[None, :]).double().numpy()
    for table, f in ((cos, np.cos), (sin, np.sin)):
        dist = ulp_distance(table[:, :half], t32(f(angle)))
        assert dist.max().item() <= 1, f"{f.__name__}: {dist.max().item()} ulp"


# --- RoPE -------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["moderate", "extreme"])
@pytest.mark.parametrize("head_dim", [2, 8, 64])
def test_rope_matches_spec_bit_exactly(head_dim: int, family: str) -> None:
    rng = np.random.default_rng(SEED + head_dim)
    make = moderate if family == "moderate" else extreme
    x = make(rng, (40, head_dim))
    cos, sin = tables(rng, head_dim, 40)
    got = bits_of(vector.rope(t32(x), cos, sin))
    c, s = cos.numpy(), sin.numpy()
    want = bits_of([v for i in range(40) for v in spec_rope(x[i], c[i], s[i])])
    assert got == want, f"d={head_dim}, {family} (seed {SEED + head_dim})"


def test_rope_at_position_zero() -> None:
    """C = 1, S = 0: the output equals x, -0 may become +0, and the partner of
    an infinity becomes NaN (Inf * 0)."""
    cos, sin = host.rope_tables(llama_config(8), torch.tensor([0]))
    x = t32([[1.5, -0.0, 3.0, 0.0, -2.0, 7.0, -0.0, 0.25]])
    y = vector.rope(x, cos, sin)
    assert torch.equal(y, x)  # numerically equal (== treats -0 and +0 as equal)
    # element 1: partner r = -x[5] = -7, r * 0 = -0, and -0 + -0 = -0
    # element 6: partner r = x[2] = 3, r * 0 = +0, and -0 + +0 = +0
    assert bits_of(y[0, [1, 6]]) == [0x80000000, 0]
    x_inf = t32([[np.inf, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    y_inf = vector.rope(x_inf, cos, sin)
    assert y_inf[0, 0].item() == np.inf and torch.isnan(y_inf[0, 4])


def test_rope_accuracy() -> None:
    """|error| <= 2^-22 * (|x C| + |r S|) per element, against float64 with
    the same tables. Bound: mul(r, S) rounds once (u |r S|), the fma rounds
    once (u times the result, at most |x C| + |r S| plus that first error),
    u = 2^-24: 2u(1 + u), so 2^-22 leaves a factor-2 margin. The output can
    cancel to near 0, so a relative bound per element would not hold."""
    rng = np.random.default_rng(SEED)
    x = moderate(rng, (2000, 64))
    cos, sin = tables(rng, 64, 2000)
    x64, c64, s64 = torch.from_numpy(x).double(), cos.double(), sin.double()
    r64 = torch.cat([-x64[:, 32:], x64[:, :32]], dim=-1)
    exact = x64 * c64 + r64 * s64
    scale = (x64 * c64).abs() + (r64 * s64).abs()
    err = (vector.rope(t32(x), cos, sin).double() - exact).abs()
    assert (err <= 2.0**-22 * scale).all()


def test_rope_agrees_with_transformers() -> None:
    """Against `transformers`' apply_rotary_pos_emb in FP32 (separate
    multiplies and add, error <= 2u per element by the same argument as
    above): |difference| <= 2^-21 * (|x C| + |r S|), a factor-2 margin over
    the sum of both errors."""
    rng = np.random.default_rng(SEED)
    x = moderate(rng, (500, 64))
    cos, sin = tables(rng, 64, 500)
    theirs, _ = apply_rotary_pos_emb(t32(x)[None, None], t32(x)[None, None], cos[None], sin[None])
    x64, c64, s64 = torch.from_numpy(x).double(), cos.double(), sin.double()
    r64 = torch.cat([-x64[:, 32:], x64[:, :32]], dim=-1)
    scale = (x64 * c64).abs() + (r64 * s64).abs()
    diff = (vector.rope(t32(x), cos, sin).double() - theirs[0, 0].double()).abs()
    assert (diff <= 2.0**-21 * scale).all()


def test_rope_rejects_bad_shapes() -> None:
    x = torch.zeros(6)
    for c, s in ((torch.ones(4), torch.zeros(4)), (torch.ones(6), torch.zeros(3))):
        with pytest.raises(AssertionError):
            vector.rope(x, c, s)
    with pytest.raises(AssertionError):
        vector.rope(torch.zeros(5), torch.ones(5), torch.zeros(5))  # odd d


# --- SiLU / SwiGLU ----------------------------------------------------------------

GATE_INPUTS = {
    "moderate": lambda rng, n: F32(rng.uniform(-10, 10, n)),
    # exp(-a) overflows (a < -88.7) or recip gives a subnormal (a < -87.3)
    "wide": lambda rng, n: F32(rng.uniform(-120, 120, n)),
    "extreme": lambda rng, n: extreme(rng, (n,)),
}


@pytest.mark.parametrize("family", list(GATE_INPUTS))
def test_swiglu_matches_spec_bit_exactly(family: str) -> None:
    rng = np.random.default_rng(SEED)
    a = GATE_INPUTS[family](rng, 3000)
    b = moderate(rng, (3000,)) if family != "extreme" else extreme(rng, (3000,))
    assert bits_of(vector.silu(t32(a))) == bits_of([spec_silu(v) for v in a])
    got = bits_of(vector.swiglu(t32(a), t32(b)))
    assert got == bits_of([spec_swiglu(u, v) for u, v in zip(a, b, strict=True)]), family


def test_silu_accuracy() -> None:
    """Relative error <= 2^-15 for |a| <= 80, against float64.

    Bound: exp 2^-17.08 (its error is scaled by e/(1+e) <= 1 in the
    quotient), add 2^-24, recip 2^-17.2, mul 2^-24: total 2^-16.1, so 2^-15
    leaves a factor-2 margin. Below -87.3 the reciprocal is subnormal and
    loses relative precision; the absolute error stays below 2^-149."""
    rng = np.random.default_rng(SEED)
    a = F32(rng.uniform(-80, 80, 100_000))
    a = a[a != 0]
    a64 = torch.from_numpy(a).double()
    ref = a64 / (1 + torch.exp(-a64))
    assert max_rel_err(vector.silu(t32(a)), ref) <= 2.0**-15


def test_silu_edges_match_torch() -> None:
    """exp(-a) overflows below -88.7: -0, as in torch. silu(-inf) is NaN
    (-inf * 0) and silu(inf) is inf, as in torch. Large positive a gives
    a * recip(1) = a * 0.9999935, not a."""
    a = t32([-100.0, -np.inf, np.inf])
    assert bits_of(vector.silu(a)) == [0x80000000, arith.NAN_F32_BITS, 0x7F800000]
    torch_silu = torch.nn.functional.silu(a)
    assert torch.signbit(torch_silu[0]) and torch_silu[0] == 0 and torch.isnan(torch_silu[1])
    big = t32([1000.0])
    assert vector.silu(big).item() == pytest.approx(1000 * 0.9999935, rel=1e-7)


# --- flush to zero ----------------------------------------------------------------


def test_flush_to_zero_does_not_change_normal_results() -> None:
    rng = np.random.default_rng(SEED)
    x = t32(moderate(rng, (64, 64)))
    cos, sin = tables(rng, 64, 64)
    a, b = t32(rng.uniform(-20, 20, 4096)), t32(moderate(rng, (4096,)))
    plain = (vector.rope(x, cos, sin), vector.swiglu(a, b))
    with settings.override(ftz=True):
        flushed = (vector.rope(x, cos, sin), vector.swiglu(a, b))
    for p, f in zip(plain, flushed, strict=True):
        assert torch.equal(arith.bits_f32(p), arith.bits_f32(f))
