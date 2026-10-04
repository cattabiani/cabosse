# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Tests for golden.decoder: bit-exact agreement with a restatement of the
"Decode step" of docs/numerics.md section 4, a wiring check against
`transformers`, the KV cache, and loading checkpoints."""

from pathlib import Path

import compare
import paths
import pytest
import torch
from golden import arith, decoder, dot, host, tiny, vector
from transformers import LlamaConfig, LlamaForCausalLM

SEED = 20261005
WEIGHTS = paths.SMOLLM2


def random_tokens(n: int, vocab: int, seed: int = SEED) -> list[int]:
    return torch.randint(0, vocab, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def golden_logits(model: decoder.Model, tokens: list[int]) -> torch.Tensor:
    cache = decoder.KVCache.empty(model, len(tokens))
    return torch.stack([decoder.decode_step(model, cache, t) for t in tokens])


def transformers_logits(config: LlamaConfig, state: dict, tokens: list[int]) -> torch.Tensor:
    """FP32 reference: the same BF16 weights, widened, in LlamaForCausalLM."""
    return compare.transformers_logits(
        tiny.transformers_model(config, state, torch.float32), tokens
    )


# --- restatement of the "Decode step" (written from the spec, per head and per
# position, with the golden primitives that the other tests check bit-exactly) ---


def spec_logits(config: LlamaConfig, state: dict, tokens: list[int]) -> torch.Tensor:
    d, n_heads, n_kv = config.head_dim, config.num_attention_heads, config.num_key_value_heads
    eps, scale = config.rms_norm_eps, torch.tensor(d**-0.5, dtype=torch.float32)
    w = state.__getitem__
    keys = [[[] for _ in range(n_kv)] for _ in range(config.num_hidden_layers)]
    values = [[[] for _ in range(n_kv)] for _ in range(config.num_hidden_layers)]
    out = []
    for pos, token in enumerate(tokens):
        cos, sin = (t[0] for t in host.rope_tables(config, torch.tensor([pos])))
        h = arith.up(w("model.embed_tokens.weight")[token])
        for layer in range(config.num_hidden_layers):
            p = f"model.layers.{layer}."
            x = arith.bf16(vector.rmsnorm(h, w(p + "input_layernorm.weight"), eps))
            q = dot.dot(w(p + "self_attn.q_proj.weight"), x)
            k = dot.dot(w(p + "self_attn.k_proj.weight"), x)
            v = dot.dot(w(p + "self_attn.v_proj.weight"), x)
            for g in range(n_kv):
                keys[layer][g].append(arith.bf16(vector.rope(k[g * d : (g + 1) * d], cos, sin)))
                values[layer][g].append(arith.bf16(v[g * d : (g + 1) * d]))
            heads = []
            for head in range(n_heads):
                g = head // (n_heads // n_kv)
                q_hat = arith.bf16(vector.rope(q[head * d : (head + 1) * d], cos, sin))
                s = torch.stack([arith.mul(dot.dot(q_hat, key), scale) for key in keys[layer][g]])
                prob = arith.bf16(vector.softmax(s))
                vt = torch.stack(values[layer][g])  # [positions, d]
                heads.append(torch.stack([dot.dot(prob, vt[:, i]) for i in range(d)]))
            o = arith.bf16(torch.cat(heads))
            h = arith.add(h, dot.dot(w(p + "self_attn.o_proj.weight"), o))
            x = arith.bf16(vector.rmsnorm(h, w(p + "post_attention_layernorm.weight"), eps))
            a = dot.dot(w(p + "mlp.gate_proj.weight"), x)
            b = dot.dot(w(p + "mlp.up_proj.weight"), x)
            mlp = arith.bf16(vector.swiglu(a, b))
            h = arith.add(h, dot.dot(w(p + "mlp.down_proj.weight"), mlp))
        x = arith.bf16(vector.rmsnorm(h, w("model.norm.weight"), eps))
        head = "model.embed_tokens.weight" if config.tie_word_embeddings else "lm_head.weight"
        out.append(dot.dot(w(head), x))
    return torch.stack(out)


# 4 query heads: MQA, GQA, MHA; head size 64 as in SmolLM2 (scale 0.125); an
# untied output projection
@pytest.mark.parametrize(
    "n_kv_heads, head_dim, n_tokens, tied",
    [(1, 16, 20, True), (2, 16, 20, True), (4, 16, 20, True), (2, 64, 6, True), (2, 16, 12, False)],
)
def test_matches_spec_bit_exactly(
    n_kv_heads: int, head_dim: int, n_tokens: int, tied: bool
) -> None:
    config = tiny.tiny_config(n_kv_heads, head_dim, tied)
    state = tiny.random_weights(config, SEED + n_kv_heads)
    tokens = random_tokens(n_tokens, config.vocab_size)
    got = golden_logits(decoder.from_state_dict(config, state), tokens)
    want = spec_logits(config, state, tokens)
    assert torch.equal(arith.bits_f32(got), arith.bits_f32(want)), (
        f"n_kv={n_kv_heads}, d={head_dim}"
    )


def spec_scores(q: torch.Tensor, k: torch.Tensor, scale) -> torch.Tensor:
    return torch.stack([arith.mul(dot.dot(q, key), scale) for key in k])


def spec_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale) -> torch.Tensor:
    s = spec_scores(q, k, scale)
    prob = arith.bf16(vector.softmax(s))
    return torch.stack([dot.dot(prob, v[:, i]) for i in range(v.shape[1])])


def test_attention_scores_match_spec_bit_exactly() -> None:
    """K spread over many exponents, so q.K sums round and their order over d
    matters. Checked on the scores themselves: a 1-ulp change in a score
    rarely survives the BF16 rounding of p."""
    gen = torch.Generator().manual_seed(SEED)
    q = torch.randn(64, generator=gen).to(torch.bfloat16)
    spread = 2.0 ** torch.randint(-12, 3, (200, 64), generator=gen)
    k = (torch.randn(200, 64, generator=gen) * spread).to(torch.bfloat16)
    scale = torch.tensor(0.125)
    got = decoder.attention_scores(q, k, scale)
    assert torch.equal(arith.bits_f32(got), arith.bits_f32(spec_scores(q, k, scale)))


@pytest.mark.parametrize("positions", [1, 7, 100, 300])
def test_attention_matches_spec_bit_exactly(positions: int) -> None:
    """Long sequences and V spread over many exponents, so that p.V sums
    round and their order over the positions matters (in the tiny decoder
    they are mostly exact, which would hide a wrong order)."""
    gen = torch.Generator().manual_seed(SEED + positions)
    q = torch.randn(64, generator=gen).to(torch.bfloat16)
    k = torch.randn(positions, 64, generator=gen).to(torch.bfloat16)
    spread = 2.0 ** torch.randint(-12, 12, (positions, 64), generator=gen)
    v = (torch.randn(positions, 64, generator=gen) * spread).to(torch.bfloat16)
    scale = torch.tensor(0.125)
    got = decoder.attention(q, k, v, scale)
    assert torch.equal(arith.bits_f32(got), arith.bits_f32(spec_attention(q, k, v, scale)))


def test_attention_scale_is_correctly_rounded() -> None:
    """host.attention_scale (float64 then FP32, as transformers computes it)
    equals the correctly rounded 1/sqrt(d) (docs/numerics.md, attention).
    Exact check: 1/sqrt(d) > m for an FP32 midpoint m exactly when m^2 d < 1."""
    from fractions import Fraction

    import numpy as np

    for d in range(1, 4097):
        y = host.attention_scale(d).numpy()
        up, down = np.nextafter(y, np.float32(np.inf)), np.nextafter(y, np.float32(0))
        mid_up = (Fraction(float(y)) + Fraction(float(up))) / 2
        mid_down = (Fraction(float(y)) + Fraction(float(down))) / 2
        assert mid_up**2 * d > 1 and mid_down**2 * d < 1, f"d={d}"


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
@pytest.mark.parametrize("small_blocks", [False, True])
def test_forward_equals_decode_steps(
    n_kv_heads: int, small_blocks: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teacher forcing gives the logits of one decode_step per token, bit for
    bit, and so does a prompt run in two parts. Small blocks: matvec splits
    rows and vectors into many blocks, attention splits positions."""
    if small_blocks:
        monkeypatch.setattr(dot, "MATVEC_ROWS", 24)
        monkeypatch.setattr(dot, "MATVEC_BLOCK", 100)
        monkeypatch.setattr(decoder, "ATTENTION_BLOCK", 7)
    config = tiny.tiny_config(n_kv_heads)
    model = decoder.from_state_dict(config, tiny.random_weights(config, SEED + n_kv_heads))
    tokens = random_tokens(30, config.vocab_size)
    want = arith.bits_f32(golden_logits(model, tokens))
    assert torch.equal(arith.bits_f32(decoder.forward(model, tokens)), want)
    cache = decoder.KVCache.empty(model, 30)
    parts = torch.cat(
        [decoder.step(model, cache, tokens[:11]), decoder.step(model, cache, tokens[11:])]
    )
    assert torch.equal(arith.bits_f32(parts), want) and cache.length == 30


# --- wiring check against transformers ------------------------------------------


def assert_wiring(ours: torch.Tensor, theirs: torch.Tensor) -> None:
    """max |golden - transformers FP32| <= 1% of the largest logit."""
    rel = ((ours - theirs).abs().max() / theirs.abs().max()).item()
    assert rel <= 0.01, f"{rel:.4f} of the largest logit"


@pytest.mark.parametrize("n_kv_heads, tied", [(1, True), (2, True), (4, True), (2, False)])
def test_agrees_with_transformers(n_kv_heads: int, tied: bool) -> None:
    """max |golden - transformers FP32| <= 1% of the largest logit.

    A wiring check, not an accuracy measurement (that is the M1 comparison
    step). Measured: 0.2% on these configs, 0.8% on SmolLM2. Wiring mistakes
    give 6% (wrong norm weight) to 50% (wrong KV head for a query head).
    """
    config = tiny.tiny_config(n_kv_heads, tied=tied)
    state = tiny.random_weights(config, SEED + n_kv_heads)
    tokens = random_tokens(24, config.vocab_size)
    ours = golden_logits(decoder.from_state_dict(config, state), tokens)
    theirs = transformers_logits(config, state, tokens)
    assert_wiring(ours, theirs)


# --- KV cache and loading ---------------------------------------------------------


def test_kv_cache_fills_one_position_per_token() -> None:
    config = tiny.tiny_config()
    model = decoder.from_state_dict(config, tiny.random_weights(config, SEED))
    cache = decoder.KVCache.empty(model, max_positions=3)
    assert cache.k.shape == (2, 2, 3, 16) and cache.k.dtype == torch.bfloat16
    for n, token in enumerate([5, 6, 7], start=1):
        decoder.decode_step(model, cache, token)
        assert cache.length == n
        assert (cache.k[:, :, :n].abs().sum(-1) > 0).all()  # filled positions are written
    with pytest.raises(AssertionError):
        decoder.decode_step(model, cache, 8)  # full
    with pytest.raises(AssertionError):
        decoder.KVCache.empty(model, max_positions=129)  # beyond the config's positions
    with pytest.raises(AssertionError):
        decoder.KVCache.empty(model, max_positions=0)  # not "use the default"
    assert decoder.KVCache.empty(model).k.shape[2] == 128  # None: the config's positions


def test_load_from_checkpoint_directory(tmp_path: Path) -> None:
    """A tiny model saved as config.json + model.safetensors loads to the same logits."""
    from safetensors.torch import save_file

    config = tiny.tiny_config()
    state = tiny.random_weights(config, SEED)
    config.save_pretrained(tmp_path)
    save_file(state, tmp_path / "model.safetensors")
    tokens = random_tokens(5, config.vocab_size)
    loaded = golden_logits(decoder.load(tmp_path), tokens)
    direct = golden_logits(decoder.from_state_dict(config, state), tokens)
    assert torch.equal(arith.bits_f32(loaded), arith.bits_f32(direct))


def test_rejects_unsupported_models() -> None:
    config = tiny.tiny_config()
    state = tiny.random_weights(config, SEED)
    for key, value in (("attention_bias", True), ("mlp_bias", True), ("hidden_act", "gelu")):
        bad = tiny.tiny_config()
        setattr(bad, key, value)
        with pytest.raises(AssertionError):
            decoder.from_state_dict(bad, state)
    with pytest.raises(AssertionError):
        decoder.from_state_dict(config, {k: v.float() for k, v in state.items()})  # not BF16


@pytest.mark.slow
@pytest.mark.skipif(not WEIGHTS.exists(), reason=f"needs the SmolLM2 checkpoint in {WEIGHTS}")
def test_smollm2_agrees_with_transformers() -> None:
    """The real model, a short prompt: the same wiring bound, and the same
    greedy next token. About 3 s per token."""
    from transformers import AutoTokenizer

    model = decoder.load(WEIGHTS)
    tokens = AutoTokenizer.from_pretrained(WEIGHTS)("The capital of France is").input_ids
    ours = golden_logits(model, tokens)
    ref = LlamaForCausalLM.from_pretrained(WEIGHTS, dtype=torch.float32).eval()
    with torch.no_grad():
        theirs = ref(torch.tensor([tokens])).logits[0]
    assert_wiring(ours, theirs)
    assert torch.equal(ours.argmax(-1), theirs.argmax(-1))
