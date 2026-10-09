# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The command list (model/commands.py): run by the reference controller, it
gives the golden decode step's bits, so every operation has a command; its
encoding round-trips; the HBM layout is consistent."""

import dataclasses

import commands
import paths
import perf
import pytest
import torch
from golden import arith, decoder, dot, tiny
from reporting import arch_doc

TOKENS = [3, 17, 200, 5, 99]  # one decode step each, at positions 0, 1, ...
LAYER_READS = ("q", "k", "v", "k_cache", "v_cache", "o", "gate", "up", "down")  # engine order


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


@pytest.mark.slow
@pytest.mark.skipif(not paths.SMOLLM2.exists(), reason=f"needs the checkpoint in {paths.SMOLLM2}")
def test_commands_give_the_golden_decode_step_on_smollm2() -> None:
    """The same check on the real model, 7 tokens (about 20 s)."""
    model = decoder.load(paths.SMOLLM2)
    tokens = [1, 9690, 314, 253, 1789, 28, 2]
    layout = commands.Layout.of(model.config, cap=len(tokens))
    hbm, cmds = commands.load(model, layout), commands.build(layout)
    cache = decoder.KVCache.empty(model, layout.cap)
    for position, token in enumerate(tokens):
        expected = decoder.decode_step(model, cache, token)
        got = commands.run(cmds, layout, hbm, token, position)
        assert torch.equal(arith.bits_f32(got), arith.bits_f32(expected)), position
    for i in range(model.config.num_hidden_layers):
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


def test_every_flag_is_documented_and_fits_its_byte() -> None:
    assert set(commands.FLAGS) == set(commands.Flag)
    assert max(commands.Flag) < 256


def test_built_commands_use_only_accepted_flags() -> None:
    layout = commands.Layout.of(tiny.tiny_config(), cap=4)
    assert all(commands.flags_ok(cmd) for cmd in commands.build(layout))


F = commands.Flag
BAD_FLAGS = [
    (commands.Op.SUM, F.NEG_A),  # a reduction does not negate its input
    (commands.Op.ROTATE_HALF, F.LEN_T),
    (commands.Op.MATVEC, F.NEG_A),  # engine commands take no flags
    (commands.Op.MUL, F.B_PER_ROW | F.B_ROW),  # one b shape at a time
    (commands.Op.LOAD, F.BY_TOKEN | F.BY_POSITION),
]


@pytest.mark.parametrize(("op", "flags"), BAD_FLAGS)
def test_a_flag_the_command_does_not_accept_stops_the_run(op, flags) -> None:
    """Flags outside the command's ACCEPTS are an error, not ignored."""
    c = tiny.tiny_config()
    model = decoder.from_state_dict(c, tiny.random_weights(c, seed=1))
    layout = commands.Layout.of(c, cap=2)
    cmds = commands.build(layout)
    i = next(k for k, x in enumerate(cmds) if x.op == op)
    cmds[i] = dataclasses.replace(cmds[i], flags=flags)
    with pytest.raises(AssertionError, match=f"command {i}: flags"):
        commands.run(cmds, layout, commands.load(model, layout), 3, 0)


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


@pytest.mark.parametrize("byte", [0, 1])  # the opcode, the destination buffer
def test_an_unknown_opcode_or_buffer_does_not_decode(byte: int) -> None:
    raw = bytearray(commands.Command(commands.Op.END).encode())
    raw[byte] = 0xFF
    with pytest.raises(ValueError, match="255"):
        commands.Command.decode(bytes(raw))


def test_a_command_after_end_is_rejected() -> None:
    """END is the last command: anything after it is a mistake of the host
    (a wrong count, two steps in one list), not something to skip."""
    c = tiny.tiny_config()
    model = decoder.from_state_dict(c, tiny.random_weights(c, seed=1))
    layout = commands.Layout.of(c, cap=2)
    cmds = commands.build(layout) + [commands.Command(commands.Op.END)]
    with pytest.raises(AssertionError, match="1 commands after END"):
        commands.run(cmds, layout, commands.load(model, layout), 3, 0)


def test_an_unknown_command_stops_the_run() -> None:
    """A command run() has no case for (an opcode added to Op but not to
    run()) is an error, not skipped."""
    c = tiny.tiny_config()
    model = decoder.from_state_dict(c, tiny.random_weights(c, seed=1))
    layout = commands.Layout.of(c, cap=2)
    cmds = commands.build(layout)
    cmds.insert(1, dataclasses.replace(cmds[0], op=99))
    with pytest.raises(AssertionError, match="unknown command 99"):
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


def same_bits(x: torch.Tensor, y: torch.Tensor) -> bool:
    bits = arith.bits_bf16 if x.dtype == torch.bfloat16 else arith.bits_f32
    return x.dtype == y.dtype and x.shape == y.shape and torch.equal(bits(x), bits(y))


@pytest.mark.parametrize("n_kv_heads", [1, 2])
@pytest.mark.parametrize("n_prompt", [0, 1, 4])
def test_record_step_matches_running_token_by_token(n_kv_heads: int, n_prompt: int) -> None:
    """A prompt run by decoder.step and copied into HBM gives the same step
    as running every prompt token through run(): logits and every engine
    command's inputs and output."""
    c = tiny.tiny_config(n_kv_heads=n_kv_heads)
    model = decoder.from_state_dict(c, tiny.random_weights(c, seed=2))
    prompt, token = TOKENS[:n_prompt], TOKENS[n_prompt]
    layout = commands.Layout.of(c, cap=n_prompt + 1)
    hbm, cmds = commands.load(model, layout), commands.build(layout)
    for position, t in enumerate(prompt):
        commands.run(cmds, layout, hbm, t, position)
    expected: list[commands.EngineCall] = []
    logits = commands.run(cmds, layout, hbm, token, n_prompt, expected.append)
    got_logits, got = commands.record_step(model, prompt, token)
    assert same_bits(got_logits, logits)
    assert len(got) == len(expected)
    for g, e in zip(got, expected, strict=True):
        assert (g.index, g.name, g.cmd) == (e.index, e.name, e.cmd)
        assert all(same_bits(getattr(g, f), getattr(e, f)) for f in ("a", "mem", "out")), g.name


@pytest.mark.parametrize("tied", [True, False])
def test_recorded_calls_are_the_engine_commands(tied: bool) -> None:
    """Every MATVEC, SCORES and VALUES, in order, named after what it reads;
    attention reads the cache's first t positions; a MATVEC's output is its
    matrix times its input."""
    c = tiny.tiny_config(tied=tied)
    model = decoder.from_state_dict(c, tiny.random_weights(c, seed=3))
    prompt = TOKENS[:3]
    _, calls = commands.record_step(model, prompt, TOKENS[3])
    layers = [f"layers.{i}.{r}" for i in range(c.num_hidden_layers) for r in LAYER_READS]
    assert [x.name for x in calls] == [*layers, "lm_head"]
    t = len(prompt) + 1
    for x in calls:
        assert x.cmd.op in commands.ENGINE_OPS and x.a.dtype == torch.bfloat16, x.name
        if x.cmd.op == commands.Op.MATVEC:
            assert x.mem.shape == (x.cmd.n, x.cmd.m), x.name
            assert same_bits(x.out, dot.matvec(x.mem, x.a[None])[0]), x.name
        else:
            assert x.mem.shape == (c.num_key_value_heads, t, c.head_dim), x.name


def lane_order_restated(w: torch.Tensor, lanes: int) -> list[list[list[int]]]:
    """docs/architecture.md's weight layout, element by element: each port's
    beats as BF16 bit patterns, zeros where no weight is."""
    bits = arith.bits_bf16(w).tolist()
    n, m = w.shape
    ports = lanes // 4
    out: list[list[list[int]]] = [[] for _ in range(ports)]
    for first in range(0, n, lanes):  # a pass
        for p in range(ports):
            rows = range(first + 4 * p, min(first + 4 * p + 4, n))
            for k in range(0, m, 4) if rows else ():
                beat = [0] * 16
                for i, r in enumerate(rows):
                    for e in range(min(4, m - k)):
                        beat[4 * i + e] = bits[r][k + e]
                out[p].append(beat)
    return out


# Includes SmolLM2's row counts: 192 (k, v) leaves half of 128 lanes' ports
# idle in its second pass, 576 the last of 20 lanes' ports in its last.
LANE_ORDER_SHAPES = [(1, 1), (3, 5), (4, 4), (7, 9), (21, 17), (64, 8), (192, 12), (576, 6)]


@pytest.mark.parametrize("lanes", [4, 8, 20, 128])
@pytest.mark.parametrize("shape", LANE_ORDER_SHAPES)
def test_lane_order(lanes: int, shape: tuple[int, int]) -> None:
    """Each port's beats hold its rows' weights, 4 columns per row, in pass
    order; 32 bytes per beat; zeros past the matrix; nothing for a port
    with no rows in a pass."""
    rng = torch.Generator().manual_seed(lanes * 1000 + shape[0])
    w = arith.bf16_from_bits(torch.randint(0, 2**16, shape, generator=rng))
    got = commands.lane_order(w, lanes)
    assert all(b.shape[1] * b.dtype.itemsize == commands.BEAT_BYTES for b in got)
    assert [arith.bits_bf16(b).tolist() for b in got] == lane_order_restated(w, lanes)
