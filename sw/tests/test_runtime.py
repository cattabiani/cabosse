# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The runtime on the golden platform: decoding bit-exact with the golden
model, the end of the context, the device's errors. Validation's own tests
are model/tests/test_commands.py's."""

import commands
import pytest
import runtime
import torch
from commands import REG, Layout
from golden import arith, decoder, tiny
from golden_platform import GoldenPlatform
from runtime import Runtime

SEED = 7


def tiny_model(seed: int = SEED, **config: int) -> decoder.Model:
    c = tiny.tiny_config(**config)
    return decoder.from_state_dict(c, tiny.random_weights(c, seed=seed))


def open_golden(model: decoder.Model, ports: int, cap: int) -> tuple[Runtime, GoldenPlatform]:
    """The runtime on a golden platform of the same layout."""
    p = GoldenPlatform(Layout.of(model.config, cap, ports=ports))
    rt = Runtime.open(p, model, cap)
    assert rt.layout == p.layout
    return rt, p


@pytest.mark.parametrize("ports", [4, 32])
def test_generate_is_the_golden_models(ports: int) -> None:
    """Prompt and new tokens one decode step each, as the hardware runs
    them: the same tokens, logits and KV cache bits as decoder.generate."""
    model, cap = tiny_model(), 128
    rt, _ = open_golden(model, ports, cap)
    gen = torch.Generator().manual_seed(SEED)
    prompt = torch.randint(0, model.config.vocab_size, (5,), generator=gen).tolist()
    tokens, logits = rt.generate(prompt, 6)
    want_tokens, want_logits = decoder.generate(model, prompt, 6)
    assert tokens == want_tokens, f"seed {SEED}"
    assert torch.equal(arith.bits_f32(logits), arith.bits_f32(want_logits)), f"seed {SEED}"
    cache = decoder.KVCache.empty(model, cap)
    decoder.step(model, cache, prompt + tokens[:-1])
    got, n = rt.kv_cache(), rt.position
    for i in range(model.config.num_hidden_layers):
        for name, want in (("k_cache", cache.k[i]), ("v_cache", cache.v[i])):
            stored = got[rt.layout.addr(f"layers.{i}.{name}")][:, :n]
            assert torch.equal(arith.bits_bf16(stored), arith.bits_bf16(want[:, :n])), (i, name)


def test_a_new_conversation_starts_at_position_0() -> None:
    model = tiny_model()
    rt, _ = open_golden(model, 4, 16)
    first = rt.generate([1, 2, 3], 2)
    rt.reset()
    again = rt.generate([1, 2, 3], 2)
    assert first[0] == again[0] and torch.equal(first[1], again[1])


def test_the_end_of_the_context_stops_the_conversation() -> None:
    model = tiny_model()
    rt, _ = open_golden(model, 4, 16)
    for t in range(16):
        rt.step(t)
    with pytest.raises(runtime.ContextFull, match="longer contexts are not supported"):
        rt.step(0)


def test_an_unknown_opcode_stops_the_device() -> None:
    model = tiny_model()
    rt, p = open_golden(model, 4, 16)
    p.reg_write(REG["COMMAND_BUFFER"] + 3 * commands.COMMAND_BYTES, 99)  # op of command 3
    with pytest.raises(runtime.DeviceError, match="ERROR 0x3"):
        rt.step(0)


def test_a_device_with_another_id_is_refused() -> None:
    model = tiny_model()
    p = GoldenPlatform(Layout.of(model.config, 16, ports=4))
    p.regs[REG["VERSION"]] = commands.VERSION + 1
    with pytest.raises(runtime.DeviceError, match="VERSION"):
        Runtime.open(p, model, 16)
