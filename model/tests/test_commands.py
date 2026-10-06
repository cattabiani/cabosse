# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The command list (model/commands.py): run by the reference controller, it
gives the golden decode step's bits, so every operation has a command; its
encoding round-trips; the HBM layout is consistent."""

import commands
import pytest
import torch
from golden import arith, decoder, tiny
from reporting import arch_doc

TOKENS = [3, 17, 200, 5, 99]  # one decode step each, at positions 0, 1, ...


@pytest.mark.parametrize(("n_kv_heads", "tied"), [(1, False), (2, True), (4, True)])
def test_commands_give_the_golden_decode_step(n_kv_heads: int, tied: bool) -> None:
    """Logits and KV cache bit-exact against decoder.decode_step, token by
    token, with grouped (1, 2 KV heads) and plain (4) attention."""
    c = tiny.tiny_config(n_kv_heads=n_kv_heads, tied=tied)
    model = decoder.from_state_dict(c, tiny.random_weights(c, seed=1))
    layout = commands.Layout.of(c, cap=len(TOKENS) + 2)
    hbm, cmds = commands.load(model, layout), commands.build(layout)
    cache = decoder.KVCache.empty(model, layout.cap)
    for position, token in enumerate(TOKENS):
        expected = decoder.decode_step(model, cache, token)
        got = commands.run(cmds, hbm, token, position)
        assert torch.equal(arith.bits_f32(got), arith.bits_f32(expected)), position
    for i in range(c.num_hidden_layers):
        for name, golden in (("k_cache", cache.k[i]), ("v_cache", cache.v[i])):
            stored = hbm[layout.addr(f"layers.{i}.{name}")]
            assert torch.equal(arith.bits_bf16(stored), arith.bits_bf16(golden)), (i, name)


def test_encoding_round_trips() -> None:
    assert commands.COMMAND_BYTES == 32
    layout = commands.Layout.of(tiny.tiny_config(), cap=4)
    for cmd in commands.build(layout):
        raw = cmd.encode()
        assert len(raw) == commands.COMMAND_BYTES
        assert commands.Command.decode(raw) == cmd


@pytest.mark.parametrize("tied", [True, False])
def test_layout_is_aligned_and_does_not_overlap(tied: bool) -> None:
    layout = commands.Layout.of(tiny.tiny_config(tied=tied), cap=4)
    spans = sorted(
        {(a, a + torch.Size(s).numel() * t.itemsize) for a, s, t in layout.tensors.values()}
    )
    assert all(a % commands.ALIGN == 0 for a, _ in spans)
    assert all(end <= start for (_, end), (start, _) in zip(spans, spans[1:], strict=False))
    assert (layout.addr("lm_head") == layout.addr("embed")) == tied


def test_every_opcode_is_documented_and_used() -> None:
    layout = commands.Layout.of(tiny.tiny_config(), cap=4)
    assert set(commands.OP_DOC) == set(commands.Op)
    assert {cmd.op for cmd in commands.build(layout)} == set(commands.Op)


def test_a_list_without_end_is_rejected() -> None:
    c = tiny.tiny_config()
    model = decoder.from_state_dict(c, tiny.random_weights(c, seed=1))
    layout = commands.Layout.of(c, cap=2)
    with pytest.raises(AssertionError, match="without END"):
        commands.run(commands.build(layout)[:-1], commands.load(model, layout), 3, 0)


def test_architecture_doc_is_up_to_date() -> None:
    """docs/architecture.md shows what model/commands.py defines; after a
    change, run: python scripts/report_arch.py"""
    assert arch_doc.render() == arch_doc.PATH.read_text()
