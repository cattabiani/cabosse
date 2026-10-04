# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Tiny random-weight Llama configs: fast tests for the golden model now and
for the RTL later (PLAN.md, model ladder)."""

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM


def tiny_config(n_kv_heads: int = 2, head_dim: int = 16, tied: bool = True) -> LlamaConfig:
    """2 layers, 4 query heads, hidden 64, vocab 256, 128 positions. tied:
    the output projection is the embedding table (as in SmolLM2)."""
    n_heads = 4
    return LlamaConfig(
        vocab_size=256,
        hidden_size=n_heads * head_dim,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=tied,
    )


def random_weights(config: LlamaConfig, seed: int) -> dict[str, torch.Tensor]:
    """BF16 weights under `transformers` names. Matrices are N(0, 1/fan_in) so
    activations stay near 1 through the layers; norm weights are 1 + N(0, 0.1)
    so they are not all equal."""
    gen = torch.Generator().manual_seed(seed)
    hidden, inter = config.hidden_size, config.intermediate_size
    d = config.head_dim
    q_dim, kv_dim = config.num_attention_heads * d, config.num_key_value_heads * d

    def matrix(rows: int, cols: int) -> torch.Tensor:
        return (torch.randn(rows, cols, generator=gen) / cols**0.5).to(torch.bfloat16)

    def norm(n: int) -> torch.Tensor:
        return (1 + 0.1 * torch.randn(n, generator=gen)).to(torch.bfloat16)

    embed = torch.randn(config.vocab_size, hidden, generator=gen).to(torch.bfloat16)  # N(0, 1)
    state = {"model.embed_tokens.weight": embed}
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}."
        state |= {
            p + "input_layernorm.weight": norm(hidden),
            p + "self_attn.q_proj.weight": matrix(q_dim, hidden),
            p + "self_attn.k_proj.weight": matrix(kv_dim, hidden),
            p + "self_attn.v_proj.weight": matrix(kv_dim, hidden),
            p + "self_attn.o_proj.weight": matrix(hidden, q_dim),
            p + "post_attention_layernorm.weight": norm(hidden),
            p + "mlp.gate_proj.weight": matrix(inter, hidden),
            p + "mlp.up_proj.weight": matrix(inter, hidden),
            p + "mlp.down_proj.weight": matrix(hidden, inter),
        }
    state["model.norm.weight"] = norm(hidden)
    if not config.tie_word_embeddings:
        state["lm_head.weight"] = matrix(config.vocab_size, hidden)
    return state


def transformers_model(
    config: LlamaConfig, state: dict[str, torch.Tensor], dtype: torch.dtype
) -> LlamaForCausalLM:
    """The same weights in `transformers`' LlamaForCausalLM, in FP32 (widened,
    exact) or BF16, for comparisons. Checks that every weight was loaded.

    Built the way from_pretrained builds a BF16 model: the weights in dtype,
    the RoPE frequency buffers in FP32 (a plain .to(dtype) would round those
    to BF16 too and make the BF16 baseline worse than the real one)."""
    model = AutoModelForCausalLM.from_config(config, dtype=dtype).eval()
    missing, unexpected = model.load_state_dict(
        {k: v.to(dtype) for k, v in state.items()}, strict=False
    )
    assert unexpected == [] and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
    model.tie_weights()
    return model
