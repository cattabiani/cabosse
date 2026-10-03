# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Vector unit operations (docs/numerics.md, section 4): reductions, RMSNorm,
and softmax.

The vector unit works on FP32 vectors. Its adder has a latency like a lane's,
so a sum keeps S partial sums in rotation (element i goes to partial i mod S)
and ends with the pairwise tree of section 3. Functions work on the last
dimension and treat the others as independent vectors.
"""

from collections.abc import Callable

import torch

from golden import arith, dot, funcs

# S: interleaved partial sums of the vector unit. Provisional until M2 derives
# it from the adder pipeline depth; every function takes it as a parameter.
REDUCE_WIDTH = 8

Step = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _reduce(x: torch.Tensor, step: Step, width: int) -> torch.Tensor:
    """acc[i mod width] = step(x[i], acc[i mod width]) in increasing i, then the tree."""
    assert x.dtype == torch.float32, x.dtype
    assert width > 0 and width & (width - 1) == 0, width
    n = x.shape[-1]
    assert n > 0, "empty reduction (not defined by the spec)"
    acc = torch.zeros((*x.shape[:-1], width), dtype=torch.float32)  # all +0.0
    for start in range(0, n, width):
        m = min(width, n - start)  # partial last group: the rest keep their value
        acc[..., :m] = step(x[..., start : start + m], acc[..., :m])
    return dot.tree_sum(acc)


def reduce_sum(x: torch.Tensor, width: int = REDUCE_WIDTH) -> torch.Tensor:
    """sum(x): partial sums acc = add(x[i], acc)."""
    return _reduce(x, arith.add, width)


def reduce_sum_squares(x: torch.Tensor, width: int = REDUCE_WIDTH) -> torch.Tensor:
    """sum_squares(x): partial sums acc = fma(x[i], x[i], acc)."""
    return _reduce(x, lambda v, acc: arith.fma(v, v, acc), width)


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


def softmax(s: torch.Tensor, width: int = REDUCE_WIDTH) -> torch.Tensor:
    """exp(s - max(s)) / sum, with the division as a multiply by recip(sum).
    FP32 in and out; attention rounds the result to BF16 (D-011)."""
    m = reduce_max(s)
    e = funcs.exp(arith.add(s, -m[..., None]))  # unary minus: exact sign flip
    z = reduce_sum(e, width)
    return arith.mul(e, funcs.recip(z)[..., None])
