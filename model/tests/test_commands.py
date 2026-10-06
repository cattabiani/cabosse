# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The command list (model/commands.py): run by the reference controller, it
gives the golden decode step's bits, so every operation has a command; its
encoding round-trips; the HBM layout is consistent."""

import dataclasses

import commands
import perf
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
        got = commands.run(cmds, layout, hbm, token, position)
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
    spans = sorted({(a, a + commands.n_bytes(s, t)) for a, s, t in layout.tensors.values()})
    assert all(a % commands.ALIGN == 0 for a, _ in spans)
    assert all(end <= start for (_, end), (start, _) in zip(spans, spans[1:], strict=False))
    assert (layout.addr("lm_head") == layout.addr("embed")) == tied


def test_every_opcode_is_documented_and_used() -> None:
    layout = commands.Layout.of(tiny.tiny_config(), cap=4)
    assert set(commands.OPS) == set(commands.Op)
    assert {cmd.op for cmd in commands.build(layout)} == set(commands.Op)


def test_commands_set_only_the_fields_their_op_uses() -> None:
    """OPS lists each command's fields; the doc's tables rely on it."""
    for cmd in commands.build(commands.Layout.of(tiny.tiny_config(), cap=4)):
        default = commands.Command(cmd.op)
        set_fields = {
            f.name
            for f in dataclasses.fields(cmd)
            if getattr(cmd, f.name) != getattr(default, f.name)
        }
        assert set_fields <= set(commands.OPS[cmd.op][1]), cmd


# Perf model op kinds per compute command; KV_STORE, OUTPUT and END move data
# or signal, and perf.ops() charges their traffic to other ops.
PERF_KIND = {
    commands.Op.EMBED: "embed",
    commands.Op.RMSNORM: "rmsnorm",
    commands.Op.MATVEC: "matvec",
    commands.Op.ROPE: "rope",
    commands.Op.SCORES: "attention_scores",
    commands.Op.SOFTMAX: "softmax",
    commands.Op.VALUES: "attention_values",
    commands.Op.SWIGLU: "swiglu",
    commands.Op.ADD: "add",
}


@pytest.mark.parametrize("n_kv_heads", [1, 2])
def test_perf_ops_follow_the_command_list(n_kv_heads: int) -> None:
    """perf.ops() and build() are two lists of one decode step: the same
    compute operations, in the same order, on the same matrix shapes."""
    c = tiny.tiny_config(n_kv_heads=n_kv_heads)
    cmds = [x for x in commands.build(commands.Layout.of(c, cap=4)) if x.op in PERF_KIND]
    ops = perf.ops(c, position=0)
    assert [PERF_KIND[x.op] for x in cmds] == [o.kind for o in ops]
    for cmd, op in zip(cmds, ops, strict=True):
        if cmd.op == commands.Op.MATVEC:
            assert (cmd.n, cmd.m) == op.shape, (cmd, op)


@pytest.mark.parametrize("name", arch_doc.LADDER)
def test_ladder_command_lists_fit_the_command_buffer(name: str) -> None:
    c = perf.load_config(name)
    n = len(commands.build(commands.Layout.of(c, c.max_position_embeddings)))
    assert n * commands.COMMAND_BYTES <= commands.COMMAND_BUFFER_BYTES, (name, n)


def test_buffer_capacities_cover_every_ladder_model() -> None:
    caps = arch_doc.capacities()
    for layout in arch_doc.ladder_layouts().values():
        for b, (dtype, n) in commands.buffers(layout.config, layout.cap).items():
            assert caps[b][0] == dtype and n <= caps[b][1], (b, layout.config)


@pytest.mark.parametrize("drop", ["end", "all"])
def test_a_list_without_end_is_rejected(drop: str) -> None:
    """The step without its END, and the empty list."""
    c = tiny.tiny_config()
    model = decoder.from_state_dict(c, tiny.random_weights(c, seed=1))
    layout = commands.Layout.of(c, cap=2)
    cmds = commands.build(layout)[:-1] if drop == "end" else []
    with pytest.raises(AssertionError, match="without END"):
        commands.run(cmds, layout, commands.load(model, layout), 3, 0)


def test_a_list_without_output_returns_nothing() -> None:
    """A prompt token whose logits the host skips: the KV cache is written
    all the same."""
    c = tiny.tiny_config()
    model = decoder.from_state_dict(c, tiny.random_weights(c, seed=1))
    layout = commands.Layout.of(c, cap=2)
    hbm = commands.load(model, layout)
    cmds = [x for x in commands.build(layout) if x.op != commands.Op.OUTPUT]
    assert commands.run(cmds, layout, hbm, 3, 0) is None
    decoder.decode_step(model, cache := decoder.KVCache.empty(model, 2), 3)
    stored = hbm[layout.addr("layers.0.k_cache")]
    assert torch.equal(arith.bits_bf16(stored), arith.bits_bf16(cache.k[0]))


def test_architecture_doc_is_up_to_date() -> None:
    """docs/architecture.md shows what model/commands.py defines; after a
    change, run: python scripts/report_arch.py"""
    assert arch_doc.render() == arch_doc.PATH.read_text()
