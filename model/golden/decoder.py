# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Llama-style decoder, one token per step (docs/numerics.md, section 4,
"Decode step"): weights, KV cache, and the decode step.

Weights are the checkpoint's BF16 tensors under their `transformers` names,
so the same state dict loads here and into `LlamaForCausalLM`.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import LlamaConfig

from golden import arith, dot, host, vector


@dataclass(frozen=True)
class Layer:
    attn_norm: torch.Tensor  # [hidden]
    q: torch.Tensor  # [n_heads * head_dim, hidden]
    k: torch.Tensor  # [n_kv_heads * head_dim, hidden]
    v: torch.Tensor  # [n_kv_heads * head_dim, hidden]
    o: torch.Tensor  # [hidden, n_heads * head_dim]
    mlp_norm: torch.Tensor  # [hidden]
    gate: torch.Tensor  # [intermediate, hidden]
    up: torch.Tensor  # [intermediate, hidden]
    down: torch.Tensor  # [hidden, intermediate]


@dataclass(frozen=True)
class Model:
    config: LlamaConfig
    embed: torch.Tensor  # [vocab, hidden]
    layers: list[Layer]
    final_norm: torch.Tensor  # [hidden]
    lm_head: torch.Tensor  # [vocab, hidden]; the same tensor as embed when tied
    cos: torch.Tensor  # RoPE tables from the host, FP32 [max_positions, head_dim]
    sin: torch.Tensor
    scale: torch.Tensor  # attention scale from the host, FP32 scalar


def from_state_dict(config: LlamaConfig, state: dict[str, torch.Tensor]) -> Model:
    """Build the model from `transformers`-named BF16 tensors."""
    assert not config.attention_bias and not config.mlp_bias, "biases are not supported in v0"
    assert config.hidden_act == "silu", config.hidden_act
    assert all(t.dtype == torch.bfloat16 for t in state.values()), "weights must be BF16"

    def layer(i: int) -> Layer:
        p = f"model.layers.{i}."
        return Layer(
            attn_norm=state[p + "input_layernorm.weight"],
            q=state[p + "self_attn.q_proj.weight"],
            k=state[p + "self_attn.k_proj.weight"],
            v=state[p + "self_attn.v_proj.weight"],
            o=state[p + "self_attn.o_proj.weight"],
            mlp_norm=state[p + "post_attention_layernorm.weight"],
            gate=state[p + "mlp.gate_proj.weight"],
            up=state[p + "mlp.up_proj.weight"],
            down=state[p + "mlp.down_proj.weight"],
        )

    embed = state["model.embed_tokens.weight"]
    lm_head = embed if config.tie_word_embeddings else state["lm_head.weight"]
    cos, sin = host.rope_tables(config, torch.arange(config.max_position_embeddings))
    return Model(
        config=config,
        embed=embed,
        layers=[layer(i) for i in range(config.num_hidden_layers)],
        final_norm=state["model.norm.weight"],
        lm_head=lm_head,
        cos=cos,
        sin=sin,
        scale=host.attention_scale(config.head_dim),
    )


def load(path: Path) -> Model:
    """Load a checkpoint directory (config.json + model.safetensors)."""
    config = LlamaConfig.from_pretrained(path)
    return from_state_dict(config, load_file(path / "model.safetensors"))


@dataclass
class KVCache:
    """K and V in BF16, [layers, n_kv_heads, max_positions, head_dim]; `length`
    positions are filled. The next token goes to position `length`.

    One flat array allocated once, no paging or wrap-around (D-024): the model
    is not trained beyond max_position_embeddings anyway."""

    k: torch.Tensor
    v: torch.Tensor
    length: int = 0

    @staticmethod
    def empty(model: Model, max_positions: int | None = None) -> KVCache:
        c = model.config
        n = c.max_position_embeddings if max_positions is None else max_positions
        assert 1 <= n <= c.max_position_embeddings, (n, c.max_position_embeddings)
        shape = (c.num_hidden_layers, c.num_key_value_heads, n, c.head_dim)
        return KVCache(
            torch.zeros(shape, dtype=torch.bfloat16), torch.zeros(shape, dtype=torch.bfloat16)
        )


def attention_scores(q: torch.Tensor, k: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """s_t = mul(dot(q, K[t]), scale): q BF16 [..., d], k BF16 [..., T, d] with
    leading dimensions that broadcast against q's. Returns FP32 [..., T]."""
    return arith.mul(dot.dot(k, q[..., None, :]), scale)


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Attention over positions 0..T-1 for one query head (q [d], k and v
    [T, d]) or a batch of heads (leading dimensions broadcast, each head
    computed independently). BF16 inputs, FP32 o [..., d]."""
    p = arith.bf16(vector.softmax(attention_scores(q, k, scale)))  # D-011: p rounded before p.V
    return dot.dot(v.transpose(-1, -2), p[..., None, :])  # o_i = sum over t of p_t V[t][i]


def step(model: Model, cache: KVCache, tokens: list[int]) -> torch.Tensor:
    """Run `tokens` at positions cache.length, cache.length + 1, ...; return
    their FP32 logits [len(tokens), vocab] and write their K and V into the
    cache.

    Every position computes what one decode step at that position computes:
    the matrix-vector products just handle all positions together, and
    attention runs position by position because it is causal.
    """
    c = model.config
    start, n, d = cache.length, len(tokens), c.head_dim
    assert start + n <= cache.k.shape[2], "KV cache is full"
    n_heads, n_kv = c.num_attention_heads, c.num_key_value_heads
    group = n_heads // n_kv  # query heads per KV head (GQA)
    eps = c.rms_norm_eps
    positions = slice(start, start + n)
    cos, sin = model.cos[positions, None], model.sin[positions, None]  # [T, 1, d]

    h = arith.up(model.embed[tokens])  # [T, hidden]
    for i, layer in enumerate(model.layers):
        x = arith.bf16(vector.rmsnorm(h, layer.attn_norm, eps))
        q = dot.matvec(layer.q, x).reshape(n, n_heads, d)
        k = dot.matvec(layer.k, x).reshape(n, n_kv, d)
        v = dot.matvec(layer.v, x).reshape(n, n_kv, d)
        cache.k[i, :, positions] = arith.bf16(vector.rope(k, cos, sin)).transpose(0, 1)
        cache.v[i, :, positions] = arith.bf16(v).transpose(0, 1)
        q_hat = arith.bf16(vector.rope(q, cos, sin))
        # query head j uses KV head j // group: [n_kv, group, d] against [n_kv, 1, T, d]
        keys, values = cache.k[i, :, None], cache.v[i, :, None]
        o = torch.stack(
            [
                attention(
                    q_hat[t].reshape(n_kv, group, d),
                    keys[..., : start + t + 1, :],
                    values[..., : start + t + 1, :],
                    model.scale,
                ).reshape(-1)
                for t in range(n)
            ]
        )
        h = arith.add(h, dot.matvec(layer.o, arith.bf16(o)))
        x = arith.bf16(vector.rmsnorm(h, layer.mlp_norm, eps))
        mlp = vector.swiglu(dot.matvec(layer.gate, x), dot.matvec(layer.up, x))
        h = arith.add(h, dot.matvec(layer.down, arith.bf16(mlp)))
    cache.length = start + n
    x = arith.bf16(vector.rmsnorm(h, model.final_norm, eps))
    return dot.matvec(model.lm_head, x)


def decode_step(model: Model, cache: KVCache, token: int) -> torch.Tensor:
    """One token at position cache.length (the hardware's mode, Q-20): FP32
    logits [vocab]."""
    return step(model, cache, [token])[0]


def forward(model: Model, tokens: list[int]) -> torch.Tensor:
    """Teacher forcing from an empty cache: FP32 logits [len(tokens), vocab],
    the same bits as one decode_step per token. A golden-model shortcut for
    comparisons."""
    return step(model, KVCache.empty(model, len(tokens)), tokens)
