# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The whole core (rtl/core.sv) driven by the runtime through the
simulation platform (PLAN.md, M7): greedy decoding on the tiny model gives
the golden model's tokens, logits and KV cache bit for bit."""

import time

import pytest
import torch
from golden import arith, decoder, tiny
from runtime import Runtime
from sim_platform import SimPlatform

import rtl

SEED = 7
CAP = 128  # the tiny model's whole window
CHAN_W = 20  # 1 MiB channels: the tiny model's layout fits
LENGTH = 1024  # buffer elements: the tiny model's largest is S, 4 heads x 128
N_NEW = 64
PROMPTS = 3


def tiny_model() -> decoder.Model:
    c = tiny.tiny_config()
    return decoder.from_state_dict(c, tiny.random_weights(c, seed=SEED))


def check_generation(rt: Runtime, model: decoder.Model, prompt: list[int], label: str) -> None:
    """rt.generate against decoder.generate, and the KV cache rt's device
    holds against the golden one."""
    rt.reset()
    start = time.monotonic()
    tokens, logits = rt.generate(prompt, N_NEW)
    seconds = time.monotonic() - start
    want_tokens, want_logits = decoder.generate(model, prompt, N_NEW)
    assert tokens == want_tokens, f"{label}: tokens differ"
    bad = (arith.bits_f32(logits) != arith.bits_f32(want_logits)).nonzero()
    assert not len(bad), f"{label}: {len(bad)} logits differ, first at {bad[0].tolist()}"
    n = len(prompt) + N_NEW - 1  # positions the runtime stored
    cache = decoder.KVCache.empty(model, CAP)
    decoder.step(model, cache, prompt + tokens[:-1])
    got = rt.kv_cache()
    for i in range(model.config.num_hidden_layers):
        for name, want in (("k_cache", cache.k[i]), ("v_cache", cache.v[i])):
            stored = got[rt.layout.addr(f"layers.{i}.{name}")][:, :n]
            assert torch.equal(arith.bits_bf16(stored), arith.bits_bf16(want[:, :n])), (
                label,
                i,
                name,
            )
    print(f"{label}: {n + 1} tokens in {seconds:.1f} s, last {rt.cycles} cycles")


def prompts(n: int, vocab: int) -> list[list[int]]:
    gen = torch.Generator().manual_seed(SEED)
    return [torch.randint(0, vocab, (4 + 6 * k,), generator=gen).tolist() for k in range(n)]


@rtl.needs_verilator
def test_tiny_decodes_bit_exact_at_4_ports() -> None:
    """3 prompts x 64 tokens (M7's criterion), one head a VALUES pass."""
    model = tiny_model()
    with SimPlatform(4, CHAN_W, LENGTH) as p:
        rt = Runtime.open(p, model, CAP)
        for k, prompt in enumerate(prompts(PROMPTS, model.config.vocab_size)):
            check_generation(rt, model, prompt, f"seed {SEED}, prompt {k}")


@pytest.mark.wide
@rtl.needs_verilator
def test_tiny_decodes_bit_exact_at_32_ports() -> None:
    """One prompt on F2's 32 ports: two heads a VALUES pass (P's second read)."""
    model = tiny_model()
    with SimPlatform(32, CHAN_W, LENGTH) as p:
        rt = Runtime.open(p, model, CAP)
        check_generation(
            rt, model, prompts(1, model.config.vocab_size)[0], f"seed {SEED}, 32 ports"
        )
