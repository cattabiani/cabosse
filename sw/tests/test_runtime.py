# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The runtime on the golden platform: decoding bit-exact with the golden
model, the end of the context, and validation refusing each kind of
command the hardware cannot run."""

import dataclasses

import commands
import pytest
import runtime
import torch
from commands import REG, Buf, Command, Flag, Layout, Op
from golden import arith, decoder, tiny
from golden_platform import GoldenPlatform
from runtime import Runtime

SEED = 7


def tiny_model(seed: int = SEED, **config: int) -> decoder.Model:
    c = tiny.tiny_config(**config)
    return decoder.from_state_dict(c, tiny.random_weights(c, seed=seed))


def platform(model: decoder.Model, ports: int, cap: int) -> GoldenPlatform:
    return GoldenPlatform(Layout.of(model.config, cap, ports=ports))


@pytest.mark.parametrize("ports", [4, 32])
def test_generate_is_the_golden_models(ports: int) -> None:
    """Prompt and new tokens one decode step each, as the hardware runs
    them: the same tokens, logits and KV cache bits as decoder.generate."""
    model, cap = tiny_model(), 128
    rt = Runtime.open(platform(model, ports, cap), model, cap)
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
    rt = Runtime.open(platform(model, 4, 16), model, 16)
    first = rt.generate([1, 2, 3], 2)
    rt.reset()
    again = rt.generate([1, 2, 3], 2)
    assert first[0] == again[0] and torch.equal(first[1], again[1])


def test_the_end_of_the_context_stops_the_conversation() -> None:
    model = tiny_model()
    rt = Runtime.open(platform(model, 4, 16), model, 16)
    for t in range(16):
        rt.step(t)
    with pytest.raises(runtime.ContextFull, match="longer contexts are not supported"):
        rt.step(0)


def test_an_unknown_opcode_stops_the_device() -> None:
    model = tiny_model()
    p = platform(model, 4, 16)
    rt = Runtime.open(p, model, 16)
    p.reg_write(REG["COMMAND_BUFFER"] + 3 * commands.COMMAND_BYTES, 99)  # op of command 3
    with pytest.raises(runtime.DeviceError, match="ERROR 0x3"):
        rt.step(0)


def test_a_device_with_another_id_is_refused() -> None:
    model = tiny_model()
    p = platform(model, 4, 16)
    p.regs[REG["VERSION"]] = commands.VERSION + 1
    with pytest.raises(runtime.DeviceError, match="VERSION"):
        Runtime.open(p, model, 16)


def first(cmds: list[Command], op: Op) -> int:
    return next(i for i, c in enumerate(cmds) if c.op == op)


def change(which: Op, **fields: int) -> object:
    def bad(cmds: list[Command]) -> None:
        i = first(cmds, which)
        cmds[i] = dataclasses.replace(cmds[i], **fields)

    return bad


def end_twice(cmds: list[Command]) -> None:
    cmds.insert(0, Command(Op.END))


def too_many(cmds: list[Command]) -> None:
    cmds[:0] = [Command(Op.ADD, Buf.T, Buf.T, Buf.W, n=1, m=1)] * 2048


# Each kind of command the hardware would hang on or silently get wrong.
BAD = [
    ("rows", change(Op.MATVEC, n=0), "no rows"),
    ("columns", change(Op.MATVEC, m=0), "no columns"),
    ("vector columns", change(Op.MUL, m=0), "no columns"),
    ("odd rotate", change(Op.ROTATE_HALF, m=63), "odd row length"),
    ("cap", change(Op.SCORES, cap=20), "not a multiple of the 16 lanes"),
    ("store cap", change(Op.KV_STORE, cap=20), "not a multiple of the 16 lanes"),
    ("head dim", change(Op.SCORES, m=8), "head dimension 8"),
    ("values head dim", change(Op.VALUES, m=32), "above the 16 lanes"),
    ("groups", change(Op.SCORES, kv_heads=3), "KV heads not dividing"),
    ("flags", change(Op.SUM, flags=Flag.NEG_A), "flags"),
    ("overlap", change(Op.ROTATE_HALF, dst=Buf.K), "overlaps"),
    ("address", change(Op.MATVEC, addr=8), "beat-aligned"),
    ("width", change(Op.MATVEC, n=2**20), "20 bits"),
    ("opcode", change(Op.MUL, op=99), "unknown opcode"),
    ("end", end_twice, "END must be the last"),
    ("length", too_many, "the buffer holds"),
]


@pytest.mark.parametrize(("label", "mutate", "why"), BAD, ids=[b[0] for b in BAD])
def test_validation_refuses_what_the_hardware_cannot_run(label, mutate, why) -> None:
    layout = Layout.of(tiny.tiny_config(head_dim=16), 16, ports=4)
    cmds = commands.build(layout)
    runtime.validate(layout, cmds)  # the list as built passes
    mutate(cmds)
    with pytest.raises(ValueError, match=why):
        runtime.validate(layout, cmds)


def test_validation_refuses_a_misaligned_stride() -> None:
    layout = Layout.of(tiny.tiny_config(), 16, ports=4)
    with pytest.raises(ValueError, match="stride 8"):
        runtime.validate(dataclasses.replace(layout, stride=8), commands.build(layout))
