# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Dot products as the lanes compute them (docs/numerics.md, section 3).

A lane takes one BF16 pair per cycle. Its FP32 adder needs several cycles per
sum, so it keeps A partial sums in rotation: element k goes to accumulator
k mod A. At the end the A partial sums are added in a fixed pairwise tree.
This fixes the summation order, and the golden model must follow it exactly.
"""

from collections.abc import Callable

import torch

from golden import arith

# One step of a partial sum: step(*elements, acc) -> new acc.
Step = Callable[..., torch.Tensor]

# A: interleaved accumulators per lane. Provisional until M2 derives it from
# the adder pipeline depth; every function takes it as a parameter.
ACCUMULATORS = 8


def tree_sum(parts: torch.Tensor) -> torch.Tensor:
    """Sum over the last dimension (length a power of two) in a balanced tree,
    pairing neighbours level by level. For 8 parts:
    ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))."""
    n = parts.shape[-1]
    assert n > 0 and n & (n - 1) == 0, f"tree_sum needs a power of two, got {n}"
    while parts.shape[-1] > 1:
        parts = arith.add(parts[..., 0::2], parts[..., 1::2])
    return parts[..., 0]


def interleaved_sum(step: Step, width: int, *operands: torch.Tensor) -> torch.Tensor:
    """Sum over the last dimension in the order of section 3: element k goes to
    partial sum k mod width, acc = step(*operands[..., k], acc) in increasing k,
    then tree_sum. The operands have the same shape. Shared by the lanes (dot)
    and the vector unit (golden.vector)."""
    assert width > 0 and width & (width - 1) == 0, f"width must be a power of two, got {width}"
    n = operands[0].shape[-1]
    assert n > 0, "empty sum (not defined by the spec)"
    acc = torch.zeros((*operands[0].shape[:-1], width), dtype=torch.float32)  # all +0.0
    for start in range(0, n, width):
        m = min(width, n - start)  # partial last group: the rest keep their value
        acc[..., :m] = step(*(o[..., start : start + m] for o in operands), acc[..., :m])
    return tree_sum(acc)


def dot(w: torch.Tensor, x: torch.Tensor, accumulators: int = ACCUMULATORS) -> torch.Tensor:
    """Dot products over the last dimension: sum_k w[..., k] * x[..., k].

    w and x are BF16 and broadcast against each other, so a matrix-vector
    product is dot(W, x) with W of shape [rows, K] and x of shape [K]. Returns
    FP32 with the broadcast shape minus the last dimension.
    """
    assert w.dtype == x.dtype == torch.bfloat16, (w.dtype, x.dtype)
    return interleaved_sum(arith.mac, accumulators, *torch.broadcast_tensors(w, x))
