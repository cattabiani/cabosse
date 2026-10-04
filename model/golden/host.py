# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Values the host computes once and hands to the accelerator (not hardware).

RoPE tables come from `transformers`' own rotary embedding (D-020), so no
table code is reimplemented here.
"""

import torch
from transformers import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


def rope_tables(
    config: PretrainedConfig, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin tables for the given positions, FP32, shape [len(positions), head_dim].

    Computed by `transformers`' LlamaRotaryEmbedding in FP32: it casts its
    result to the dtype of its probe input, so an FP32 probe gives the tables
    before any cast to the model dtype.
    """
    assert positions.dim() == 1 and not positions.is_floating_point(), positions
    rotary = LlamaRotaryEmbedding(config)
    probe = torch.zeros(1, dtype=torch.float32)  # only its dtype and device are used
    cos, sin = rotary(probe, positions[None, :])
    return cos[0], sin[0]
