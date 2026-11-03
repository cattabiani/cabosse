# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The runtime's greedy decoding against the golden model's, on any
platform (sw/tests/test_runtime.py, verif/tests/test_core.py): the same
tokens, logits and KV cache bits."""

import torch
from golden import arith, decoder, tiny
from runtime import Runtime

SEED = 7


def tiny_model(seed: int = SEED, **config: int) -> decoder.Model:
    c = tiny.tiny_config(**config)
    return decoder.from_state_dict(c, tiny.random_weights(c, seed=seed))


def check_generation(rt: Runtime, model: decoder.Model, prompt: list[int], n_new: int) -> None:
    """rt.generate(prompt, n_new) against decoder.generate, and the KV cache
    rt's device holds against the golden one, from rt's start of a
    conversation."""
    tokens, logits = rt.generate(prompt, n_new)
    want_tokens, want_logits = decoder.generate(model, prompt, n_new)
    assert tokens == want_tokens, "tokens differ"
    bad = (arith.bits_f32(logits) != arith.bits_f32(want_logits)).nonzero()
    assert not len(bad), f"{len(bad)} logits differ, first at {bad[0].tolist()}"
    cache = decoder.KVCache.empty(model, rt.layout.cap)
    decoder.step(model, cache, prompt + tokens[:-1])
    got, n = rt.kv_cache(), rt.position
    for i in range(model.config.num_hidden_layers):
        for name, want in (("k_cache", cache.k[i]), ("v_cache", cache.v[i])):
            stored = got[rt.layout.addr(f"layers.{i}.{name}")][:, :n]
            assert torch.equal(arith.bits_bf16(stored), arith.bits_bf16(want[:, :n])), (i, name)
