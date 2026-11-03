# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The runtime's greedy decoding against the golden model's, on any
platform (sw/tests/test_runtime.py, verif/tests/test_core.py): the same
tokens, logits and KV cache bits."""

import torch
from golden import arith, decoder, tiny
from runtime import Runtime

SEED = 7
# SmolLM2 on the simulated core (verif/tests/test_core.py,
# scripts/measure_m7.py): F2's 32 ports; 256 positions (a multiple of the
# 128 lanes); 64 MiB channels, as channel 0 holds the engine's region and
# LOAD's tensors; buffers as long as the longest, LOGITS (the vocabulary).
SMOLLM2_PORTS, SMOLLM2_CAP, SMOLLM2_CHAN_W, SMOLLM2_LENGTH = 32, 256, 26, 49152
SMOLLM2_PROMPT = "The capital of France is"


def tiny_model(seed: int = SEED, **config: int) -> decoder.Model:
    c = tiny.tiny_config(**config)
    return decoder.from_state_dict(c, tiny.random_weights(c, seed=seed))


def check_generation(
    rt: Runtime, model: decoder.Model, prompt: list[int], n_new: int, label: str
) -> None:
    """rt.generate(prompt, n_new) against decoder.generate, and the KV cache
    rt's device holds against the golden one, from rt's start of a
    conversation. `label` (the seed, ...) starts each failure's message."""
    tokens, logits = rt.generate(prompt, n_new)
    want_tokens, want_logits = decoder.generate(model, prompt, n_new)
    assert tokens == want_tokens, f"{label}: tokens differ"
    bad = (arith.bits_f32(logits) != arith.bits_f32(want_logits)).nonzero()
    assert not len(bad), f"{label}: {len(bad)} logits differ, first at {bad[0].tolist()}"
    cache = decoder.KVCache.empty(model, rt.layout.cap)
    decoder.step(model, cache, prompt + tokens[:-1])
    got, n = rt.kv_cache(), rt.position
    for i in range(model.config.num_hidden_layers):
        for name, want in (("k_cache", cache.k[i]), ("v_cache", cache.v[i])):
            stored = got[rt.layout.addr(f"layers.{i}.{name}")][:, :n]
            assert torch.equal(arith.bits_bf16(stored), arith.bits_bf16(want[:, :n])), (
                label,
                i,
                name,
            )
