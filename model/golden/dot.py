# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Dot products as the lanes compute them (docs/numerics.md, section 3).

A lane takes one BF16 pair per cycle. Its FP32 adder needs several cycles per
sum, so it keeps A partial sums in rotation: element k goes to accumulator
k mod A. At the end the A partial sums are added in a fixed pairwise tree.
This fixes the summation order, and the golden model must follow it exactly.
"""

import torch

from golden import arith

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


def dot(w: torch.Tensor, x: torch.Tensor, accumulators: int = ACCUMULATORS) -> torch.Tensor:
    """Dot products over the last dimension: sum_k w[..., k] * x[..., k].

    w and x are BF16 and broadcast against each other, so a matrix-vector
    product is dot(W, x) with W of shape [rows, K] and x of shape [K]. Returns
    FP32 with the broadcast shape minus the last dimension.
    """
    assert w.dtype == x.dtype == torch.bfloat16, (w.dtype, x.dtype)
    assert accumulators > 0 and accumulators & (accumulators - 1) == 0, accumulators
    w, x = torch.broadcast_tensors(w, x)
    k = w.shape[-1]
    assert k > 0, "empty dot product (not defined by the spec)"
    acc = torch.zeros((*w.shape[:-1], accumulators), dtype=torch.float32)  # all +0.0
    for start in range(0, k, accumulators):
        n = min(accumulators, k - start)  # partial last group: the rest keep their value
        acc[..., :n] = arith.mac(w[..., start : start + n], x[..., start : start + n], acc[..., :n])
    return tree_sum(acc)
