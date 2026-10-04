# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Vector unit operations (docs/numerics.md, section 4): reductions, RMSNorm,
softmax, RoPE, and SiLU/SwiGLU. The residual add is `arith.add`.

The vector unit works on FP32 vectors. Its adder has a latency like a lane's,
so a sum keeps S partial sums in rotation (element i goes to partial i mod S)
and ends with the pairwise tree of section 3. Functions work on the last
dimension and treat the others as independent vectors.
"""

import torch

from golden import arith, dot, funcs

# S: interleaved partial sums of the vector unit. Provisional until M2 derives
# it from the adder pipeline depth; every function takes it as a parameter.
REDUCE_WIDTH = 8


def reduce_sum(
    x: torch.Tensor, width: int = REDUCE_WIDTH, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """sum(x): partial sums acc = add(x[i], acc). valid: see dot.interleaved_sum."""
    assert x.dtype == torch.float32, x.dtype
    return dot.interleaved_sum(arith.add, width, x, valid=valid)


def reduce_sum_squares(x: torch.Tensor, width: int = REDUCE_WIDTH) -> torch.Tensor:
    """sum_squares(x): partial sums acc = fma(x[i], x[i], acc)."""
    assert x.dtype == torch.float32, x.dtype
    return dot.interleaved_sum(lambda v, acc: arith.fma(v, v, acc), width, x)


def reduce_max(x: torch.Tensor) -> torch.Tensor:
    """max(x). max is exact, commutative and associative, so the order (here
    halving, odd element carried) does not change the result."""
    assert x.dtype == torch.float32, x.dtype
    assert x.shape[-1] > 0, "empty reduction (not defined by the spec)"
    x = arith.maximum(x, x)  # max(a, a) = a: applies the NaN and FTZ rules for n = 1
    while x.shape[-1] > 1:
        half = x.shape[-1] // 2
        y = arith.maximum(x[..., :half], x[..., half : 2 * half])
        x = torch.cat([y, x[..., 2 * half :]], dim=-1)
    return x[..., 0]


def rmsnorm(
    x: torch.Tensor, g: torch.Tensor, eps: float, width: int = REDUCE_WIDTH
) -> torch.Tensor:
    """x / sqrt(mean(x^2) + eps) * g, with FP32 x, BF16 weight g, FP32 result.

    eps is the model config's rms_norm_eps. It and 1/n are constants rounded to
    FP32 once (on the host, in hardware).
    """
    assert x.dtype == torch.float32 and g.dtype == torch.bfloat16, (x.dtype, g.dtype)
    n = x.shape[-1]
    assert g.shape == (n,), (g.shape, n)  # one weight per element, no broadcasting
    inv_n = torch.tensor(1.0, dtype=torch.float32) / n  # IEEE division: f32(1/n), one rounding
    eps32 = torch.tensor(eps, dtype=torch.float32)  # f32(eps), one rounding
    var = arith.mul(reduce_sum_squares(x, width), inv_n)
    r = funcs.rsqrt(arith.add(var, eps32))
    return arith.mul(arith.up(g), arith.mul(x, r[..., None]))


def softmax(
    s: torch.Tensor, width: int = REDUCE_WIDTH, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """exp(s - max(s)) / sum, with the division as a multiply by recip(sum).
    FP32 in and out; attention rounds the result to BF16 (D-011).

    valid (bool, broadcastable to s): softmax over the valid entries only, with
    the bits of softmax over the valid prefix; invalid entries give +0. At
    least one entry per vector must be valid.
    """
    if valid is not None:
        s = torch.where(valid, s, -torch.inf)  # max(x, -inf) = x, exactly
    m = reduce_max(s)
    e = funcs.exp(arith.add(s, -m[..., None]))  # unary minus: exact sign flip
    z = reduce_sum(e, width, valid)
    p = arith.mul(e, funcs.recip(z)[..., None])
    return p if valid is None else torch.where(valid, p, 0.0)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding over the last dimension (d even, halves not
    interleaved): out = fma(x, cos, mul(rotate_half(x), sin)). cos and sin are
    the host's FP32 tables for this position (golden.host.rope_tables)."""
    assert x.dtype == cos.dtype == sin.dtype == torch.float32, (x.dtype, cos.dtype, sin.dtype)
    d = x.shape[-1]
    assert d % 2 == 0 and cos.shape == sin.shape and cos.shape[-1] == d, (x.shape, cos.shape)
    half = d // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)  # unary minus: exact sign flip
    return arith.fma(x, cos, arith.mul(rotated, sin))


def silu(a: torch.Tensor) -> torch.Tensor:
    """a * sigmoid(a) = mul(a, recip(add(1, exp(-a))))."""
    assert a.dtype == torch.float32, a.dtype
    one = torch.tensor(1.0, dtype=torch.float32)
    return arith.mul(a, funcs.recip(arith.add(one, funcs.exp(-a))))  # unary minus: exact


def swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """mul(silu(a), b) with a = gate.x and b = up.x. FP32 in and out; the
    result is rounded to BF16 before the down projection (D-011)."""
    assert a.dtype == b.dtype == torch.float32, (a.dtype, b.dtype)
    return arith.mul(silu(a), b)
