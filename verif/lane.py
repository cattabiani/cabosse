# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""What the lane's tests share (test_dot_lane, test_dot_lane_bulk): its
default shape, the golden result and input gaps."""

import numpy as np
import torch
from golden import arith, dot

E = 4  # pairs per beat: dot_lane's default
FULL_RATE_LEN = 60  # (A - 1) * E: shorter rows end before their tree's adds are issued


def golden_dot(w: np.ndarray, x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """golden.dot.dot over the last dimension of BF16 bits w and x (uint16)
    with the elements that are there, as FP32 bits (uint32)."""
    w_t, x_t = (arith.bf16_from_bits(torch.from_numpy(v.astype(np.int64))) for v in (w, x))
    y = dot.dot(w_t, x_t, valid=torch.from_numpy(valid))
    return arith.bits_f32(y).numpy().astype(np.uint32)


def random_gaps(rng: np.random.Generator, n_beats: int, p_gap: float) -> np.ndarray:
    """Idle cycles before each of n_beats beats: geometric, a gap of k with
    probability (1 - p_gap) * p_gap^k."""
    return rng.geometric(1 - p_gap, n_beats) - 1
