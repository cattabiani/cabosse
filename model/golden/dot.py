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


def interleaved_sum(
    step: Step, width: int, *operands: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """Sum over the last dimension in the order of section 3: element k goes to
    partial sum k mod width, acc = step(*operands[..., k], acc) in increasing k,
    then tree_sum. The operands have the same shape. Shared by the lanes (dot)
    and the vector unit (golden.vector).

    valid (bool, broadcastable to the operands): where it is False, the partial
    sum keeps its value, as for the missing elements of a partial group (a
    write enable in hardware). So a masked vector gives the bits of the same
    vector truncated to its valid prefix; feeding zeros instead would not.
    """
    assert width > 0 and width & (width - 1) == 0, f"width must be a power of two, got {width}"
    n = operands[0].shape[-1]
    assert n > 0, "empty sum (not defined by the spec)"
    acc = torch.zeros((*operands[0].shape[:-1], width), dtype=torch.float32)  # all +0.0
    for start in range(0, n, width):
        m = min(width, n - start)  # partial last group: the rest keep their value
        new = step(*(o[..., start : start + m] for o in operands), acc[..., :m])
        if valid is not None:
            new = torch.where(valid[..., start : start + m], new, acc[..., :m])
        acc[..., :m] = new
    return tree_sum(acc)


def dot(
    w: torch.Tensor,
    x: torch.Tensor,
    accumulators: int = ACCUMULATORS,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dot products over the last dimension: sum_k w[..., k] * x[..., k].

    w and x are BF16 and broadcast against each other, so a matrix-vector
    product is dot(W, x) with W of shape [rows, K] and x of shape [K]. Returns
    FP32 with the broadcast shape minus the last dimension. valid: see
    interleaved_sum.
    """
    assert w.dtype == x.dtype == torch.bfloat16, (w.dtype, x.dtype)
    # Widen once, before broadcasting: up() is exact, so this is mac's result
    # without repeating the widening for every row or every vector.
    a, b = torch.broadcast_tensors(arith.up(w), arith.up(x))
    return interleaved_sum(arith.mac_f32, accumulators, a, b, valid=valid)


# matvec blocks: rows per block, and rows x vectors per block. Blocks keep a
# slice of a large matrix (the 49152-row vocabulary projection) in cache while
# it meets every vector, and bound the temporaries of the emulated arithmetic.
MATVEC_ROWS = 2048
MATVEC_BLOCK = 1 << 20


def matvec(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dot(w, x[t]) for every vector x[t] of x [T, K]: FP32 [T, rows]. Each
    element is the same dot product as dot(w, x[t]); only the batching differs."""
    rows, n = w.shape[0], x.shape[0]
    block_rows = min(rows, MATVEC_ROWS)
    block_vectors = max(1, MATVEC_BLOCK // block_rows)
    return torch.cat(
        [
            torch.cat(
                [
                    dot(w[r : r + block_rows], x[t : t + block_vectors, None, :])
                    for r in range(0, rows, block_rows)
                ],
                dim=1,
            )
            for t in range(0, n, block_vectors)
        ]
    )
