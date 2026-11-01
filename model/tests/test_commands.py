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


# (KV heads, tied, cap, S and P's stride): grouped and plain attention, the
# stride above and below the cache's cap.
DECODE_CASES = [(1, False, 7, 16), (2, True, 20, 16), (4, True, 5, 32)]


@pytest.mark.parametrize(("n_kv_heads", "tied", "cap", "stride"), DECODE_CASES)
def test_commands_give_the_golden_decode_step(
    n_kv_heads: int, tied: bool, cap: int, stride: int
) -> None:
    """Logits and KV cache bit-exact against decoder.decode_step, token by
    token, with grouped (1, 2 KV heads) and plain (4) attention."""
    c = tiny.tiny_config(n_kv_heads=n_kv_heads, tied=tied)
    model = decoder.from_state_dict(c, tiny.random_weights(c, seed=1))
    layout = commands.Layout.of(c, cap=cap, stride=stride)
    hbm, cmds = commands.load(model, layout), commands.build(layout)
    cache = decoder.KVCache.empty(model, layout.cap)
    for position, token in enumerate(TOKENS):
        expected = decoder.decode_step(model, cache, token)
        got = commands.run(cmds, layout, hbm, token, position)
        assert torch.equal(arith.bits_f32(got), arith.bits_f32(expected)), position
    for i in range(c.num_hidden_layers):
        for name, golden in (("k_cache", cache.k[i]), ("v_cache", cache.v[i])):
            stored = hbm[layout.addr(f"layers.{i}.{name}")][:, : len(TOKENS)]  # the rest is stale
            assert torch.equal(
                arith.bits_bf16(stored), arith.bits_bf16(golden[:, : len(TOKENS)])
            ), (i, name)


@pytest.mark.slow
@pytest.mark.skipif(not paths.SMOLLM2.exists(), reason=f"needs the checkpoint in {paths.SMOLLM2}")
def test_commands_give_the_golden_decode_step_on_smollm2() -> None:
    """The same check on the real model, 7 tokens (about 20 s)."""
    model = decoder.load(paths.SMOLLM2)
    tokens = [1, 9690, 314, 253, 1789, 28, 2]
    layout = commands.Layout.of(model.config, cap=len(tokens), stride=16)
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
    assert layout.addr("lm_head") != layout.addr("embed")  # LOAD reads embed by rows


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


B, Op = commands.Buf, commands.Op
# Commands the vector unit cannot run as the model would: two operands from
# one buffer (one read port), dst over a source read again, no rows or no
# elements, an odd row to rotate.
BAD_VECTOR = [
    commands.Command(Op.MUL, B.T, B.G, B.G, n=3, m=16, flags=F.B_ROW),
    commands.Command(Op.MUL, B.T, B.G, B.G, n=3, m=16),
    commands.Command(Op.FMA, B.T, B.G, B.U, B.U, n=1, m=16),
    commands.Command(Op.MUL, B.U, B.G, B.U, n=3, m=16, flags=F.B_PER_ROW),
    commands.Command(Op.ROTATE_HALF, B.Q, B.Q, n=1, m=16),
    commands.Command(Op.MUL, B.T, B.H, B.R, n=0, m=16, flags=F.B_PER_ROW),
    commands.Command(Op.ADD, B.T, B.H, B.U, n=1, m=0),
    commands.Command(Op.ROTATE_HALF, B.QT, B.Q, n=1, m=5),
]


@pytest.mark.parametrize("cmd", BAD_VECTOR, ids=str)
def test_a_command_the_vector_unit_cannot_run_is_rejected(cmd: commands.Command) -> None:
    buf = commands.empty_buffers(commands.buffers(tiny.tiny_config(), 16))
    with pytest.raises(AssertionError):
        commands.vector_command(cmd, buf, 1, 16)


def test_built_commands_have_distinct_sources() -> None:
    """The ladder's lists already keep to overlap_ok (run() checks it too)."""
    for name in ("SmolLM2-135M-Instruct", "Qwen2.5-0.5B-Instruct"):
        layout = commands.Layout.of(perf.load_config(name), cap=16)
        assert all(commands.overlap_ok(cmd) for cmd in commands.build(layout)), name


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
        for b, (dtype, n) in commands.buffers(layout.config, layout.stride).items():
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
    stored = hbm[layout.addr("layers.0.k_cache")][:, :1]  # the row written; the rest is stale
    assert torch.equal(arith.bits_bf16(stored), arith.bits_bf16(cache.k[0][:, :1]))


def test_architecture_doc_is_up_to_date() -> None:
    """docs/architecture.md shows what model/commands.py defines; after a
    change, run: python scripts/report_arch.py"""
    assert arch_doc.render() == arch_doc.PATH.read_text()


def same_bits(x: torch.Tensor, y: torch.Tensor) -> bool:
    bits = arith.bits_bf16 if x.dtype == torch.bfloat16 else arith.bits_f32
    return x.dtype == y.dtype and x.shape == y.shape and torch.equal(bits(x), bits(y))


def assert_same_step(
    got: tuple[torch.Tensor, list[commands.EngineCall]],
    want: tuple[torch.Tensor, list[commands.EngineCall]],
) -> None:
    """Two recorded steps: the same logits and engine commands, bit for bit."""
    assert same_bits(got[0], want[0])
    for g, w in zip(got[1], want[1], strict=True):
        assert (g.index, g.name, g.cmd) == (w.index, w.name, w.cmd)
        assert all(same_bits(getattr(g, f), getattr(w, f)) for f in ("a", "mem", "out")), g.name


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
    assert_same_step(commands.record_step(model, prompt, token), (logits, expected))


def test_record_step_from_a_longer_cache() -> None:
    """A cache filled past the prompt and cut back to it (as the long
    positions reuse one saved cache) gives the same step as the prompt run
    by record_step itself: the rows past the prompt are never read."""
    c = tiny.tiny_config(n_kv_heads=2)
    model = decoder.from_state_dict(c, tiny.random_weights(c, seed=2))
    prompt, token = TOKENS[:4], TOKENS[4]
    cache = decoder.KVCache.empty(model, 16)
    decoder.step(model, cache, TOKENS)  # position 4 too, which the step rewrites
    cache.length = len(prompt)
    assert_same_step(
        commands.record_step(model, prompt, token, cache),
        commands.record_step(model, prompt, token),
    )


@pytest.mark.parametrize("tied", [True, False])
def test_recorded_calls_are_the_engine_commands(tied: bool) -> None:
    """Every MATVEC, SCORES and VALUES, in order, named after what it reads;
    attention records the whole cache (t = cap here); a MATVEC's output is
    its matrix times its input."""
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
            assert x.t == t and x.mem.shape == (c.num_key_value_heads, t, c.head_dim), x.name


def test_stride_is_a_multiple_of_the_vector_unit_s_width() -> None:
    """S and P's stride (the STRIDE register) is cap rounded up to 16 by
    default; one that is not a multiple of 16 is refused."""
    c = tiny.tiny_config()
    assert commands.Layout.of(c, cap=20).stride == 32
    assert commands.Layout.of(c, cap=20, stride=16).stride == 16
    for bad in (0, 8, 20):
        with pytest.raises(AssertionError):
            commands.Layout.of(c, cap=20, stride=bad)


@pytest.mark.parametrize(("cap", "stride"), [(8, 16), (20, 16), (4, 32)])
def test_record_step_with_other_cap_and_stride(cap: int, stride: int) -> None:
    """A cache of more positions than the step reads, and S and P at another
    stride than the default (cap rounded up to 16), give the same step: S
    and P hold the same rows of t at the stride, and the cache's rows past
    t are stale (NaN)."""
    c = tiny.tiny_config(n_kv_heads=2)
    model = decoder.from_state_dict(c, tiny.random_weights(c, seed=2))
    prompt, token = TOKENS[:3], TOKENS[3]
    got, want = (
        commands.record_step(model, prompt, token, cap=cap, stride=stride),
        commands.record_step(model, prompt, token),
    )
    assert same_bits(got[0], want[0])
    t = len(prompt) + 1
    for g, w in zip(got[1], want[1], strict=True):
        assert (g.index, g.name, g.t, g.cmd.op) == (w.index, w.name, w.t, w.cmd.op), g.name
        assert g.stride == stride, g.name
        # P (VALUES' a) and S (SCORES' out) as rows of t, each at its stride
        a = (g.rows(g.a), w.rows(w.a)) if g.cmd.op == commands.Op.VALUES else (g.a, w.a)
        out = (g.rows(g.out), w.rows(w.out)) if g.cmd.op == commands.Op.SCORES else (g.out, w.out)
        assert same_bits(*a) and same_bits(*out), g.name
        if g.cmd.op == commands.Op.MATVEC:
            assert same_bits(g.mem, w.mem), g.name
        else:
            assert g.cmd.cap == cap and g.mem.shape[1] == cap, g.name
            assert same_bits(g.mem[:, :t], w.mem) and g.mem[:, t:].isnan().all(), g.name


# Engine commands whose input the vector unit (or the engine command before)
# has just written: after a norm (q, gate, the classifier), RoPE (scores),
# softmax (values), attention (o) and SwiGLU (down).
WAITING_READS = ("q", "k_cache", "v_cache", "o", "gate", "down")


def test_engine_waits() -> None:
    """The engine commands that wait for a go (D-042) are the ones after a
    vector-unit result, not those sharing the input of the one before."""
    c = tiny.tiny_config()
    layout = commands.Layout.of(c, cap=8)
    cmds = commands.build(layout)
    names = {addr: name for name, (addr, _, _) in layout.tensors.items()}
    waiting = [names[x.addr] for x, w in zip(cmds, commands.engine_waits(cmds), strict=True) if w]
    layers = [f"layers.{i}.{r}" for i in range(c.num_hidden_layers) for r in WAITING_READS]
    assert waiting == [*layers, "lm_head"]


def test_engine_waits_for_every_unwaited_write() -> None:
    """An engine command between a write and its reader does not hide the
    dependency; an input no command wrote needs no go."""
    norm = commands.Command(
        commands.Op.ADD, commands.Buf.X, commands.Buf.H, commands.Buf.H, n=1, m=8
    )
    q = commands.Command(commands.Op.MATVEC, commands.Buf.Q, commands.Buf.X, n=8, m=8)
    other = commands.Command(commands.Op.MATVEC, commands.Buf.K, commands.Buf.T, n=8, m=8)
    k = commands.Command(commands.Op.MATVEC, commands.Buf.K, commands.Buf.X, n=8, m=8)
    assert commands.engine_waits([norm, other, q, k]) == [False, False, True, False]


def test_stops() -> None:
    """The controller stops at a command off the engine that reads what a
    command since the last stop wrote, an engine command's result included;
    after a stop, only later writes count."""
    Op, Buf = commands.Op, commands.Buf
    cmds = [
        commands.Command(Op.ADD, Buf.H, Buf.X, Buf.T, n=1, m=8),  # nothing written yet
        commands.Command(Op.MUL, Buf.R, Buf.X, Buf.T, n=1, m=8),  # reads no write
        commands.Command(Op.ADD, Buf.X, Buf.H, Buf.R, n=1, m=8),  # reads H: stops
        commands.Command(Op.MUL, Buf.T, Buf.H, Buf.H, n=1, m=8),  # H before the stop
        commands.Command(Op.MATVEC, Buf.Q, Buf.X, n=8, m=8),  # waits in the engine
        commands.Command(Op.KV_STORE, Buf.NONE, Buf.Q, n=1, m=8, cap=8),  # reads Q: stops
    ]
    assert commands.stops(cmds) == [False, False, True, False, False, True]


def hazard_lists() -> list[tuple[str, list[commands.Command]]]:
    """Tiny models with grouped and plain attention, and the real models'
    lists (the configs are in the repo; no weights needed)."""
    tiny_lists = [
        (f"tiny, {kv} KV heads", tiny.tiny_config(n_kv_heads=kv, tied=True)) for kv in (1, 2, 4)
    ]
    real = [(p.stem, perf.load_config(p.stem)) for p in sorted(paths.CONFIGS.glob("*.json"))]
    return [(name, commands.build(commands.Layout.of(c, cap=16))) for name, c in tiny_lists + real]


@pytest.mark.parametrize(("name", "cmds"), hazard_lists())
def test_the_command_after_the_engine_stops(name: str, cmds: list[commands.Command]) -> None:
    """The first command off the engine after an engine command reads a
    result written since the last stop, so the controller stops there: the
    vector unit's tests hold it until the engine is done, with the same
    cycles (verif/decode.py's unit_cycles)."""
    stops = commands.stops(cmds)
    for i in range(1, len(cmds)):
        before, cmd = cmds[i - 1], cmds[i]
        if before.op in commands.ENGINE_OPS and cmd.op not in commands.ENGINE_OPS:
            assert stops[i] or cmd.op in (commands.Op.OUTPUT, commands.Op.END), (name, i, cmd)


@pytest.mark.parametrize("write_after_reads", [True, False])
def test_command_list_has_no_hazards(write_after_reads: bool) -> None:
    """Under the ordering rules (D-041, D-042, D-044) no command reads a
    buffer or a cache row before its last write is done, or overwrites one
    an earlier command may still read; the current lists need no
    write-after-read rule for that (D-044's claim)."""
    for name, cmds in hazard_lists():
        assert commands.hazards(cmds, write_after_reads=write_after_reads) == [], name


def test_hazards_found_without_a_rule() -> None:
    """Each rule is needed: SCORES given without wait (D-044's first draft)
    reads q before RoPE writes it; without the stores' handover, SCORES and
    VALUES read the newest cache row before it is written."""
    for name, cmds in hazard_lists():
        layers = sum(cmd.op == commands.Op.SCORES for cmd in cmds)
        no_scores_go = [
            w and cmd.op != commands.Op.SCORES
            for cmd, w in zip(cmds, commands.engine_waits(cmds), strict=True)
        ]
        found = commands.hazards(cmds, waits=no_scores_go)
        assert {(h.kind, h.resource) for h in found} == {("read after write", commands.Buf.QR)}
        assert len(found) == 2 * layers, name  # every layer of both tokens
        found = commands.hazards(cmds, kv_after=[None] * len(cmds))
        readers = {cmds[h.cmd[1]].op for h in found}
        assert readers == {commands.Op.SCORES, commands.Op.VALUES}, name
        assert all(cmds[h.earlier[1]].op in commands.STORE_OPS for h in found), name
        assert len(found) == 2 * 2 * layers, name


def test_write_after_read_hazard() -> None:
    """A vector command overwriting the engine's input while the engine may
    still read it (and while the first write may still land) is a hazard
    unless writes wait for earlier readers and writers."""
    op, buf = commands.Op, commands.Buf
    norm = commands.Command(op.ADD, buf.X, buf.H, buf.H, n=1, m=8)
    q = commands.Command(op.MATVEC, buf.Q, buf.X, n=8, m=8)
    overwrite = commands.Command(op.MUL, buf.X, buf.H, buf.H, n=1, m=8)
    cmds = [norm, q, overwrite, commands.Command(op.END)]
    found = commands.hazards(cmds, write_after_reads=False)
    assert sorted((h.cmd, h.earlier, h.kind) for h in found) == [
        ((token, 2), (token, earlier), kind)
        for token in (0, 1)
        for earlier, kind in ((0, "write after write"), (1, "write after read"))
    ]
    assert commands.hazards(cmds) == []


def test_read_port_hazard() -> None:
    """A vector command reading the engine's input while the engine may
    still read it, or the reverse, is a hazard (one read port per buffer);
    the engine's own commands, and the vector unit's, share it in order."""
    op, buf = commands.Op, commands.Buf
    norm = commands.Command(op.ADD, buf.X, buf.H, buf.H, n=1, m=8)
    q = commands.Command(op.MATVEC, buf.Q, buf.X, n=8, m=8)
    k = commands.Command(op.MATVEC, buf.K, buf.X, n=8, m=8)
    sumsq = commands.Command(op.SUMSQ, buf.R, buf.X, n=1, m=8)
    end = commands.Command(op.END)
    cases = (([norm, q, k, sumsq, end], {(3, 1), (3, 2)}), ([norm, sumsq, q, end], {(2, 1)}))
    for cmds, pairs in cases:
        found = commands.hazards(cmds)
        assert {(h.kind, h.resource) for h in found} == {("read port", buf.X)}
        assert sorted((h.cmd, h.earlier) for h in found) == sorted(
            ((token, later), (token, earlier)) for token in (0, 1) for later, earlier in pairs
        )
    assert commands.hazards([norm, q, k, end]) == []
    assert commands.hazards([norm, sumsq, sumsq, end]) == []


def test_kv_waits() -> None:
    """SCORES is given after the KV_STORE of the K cache it reads, VALUES
    after the V_STORE of its V cache."""
    layout = commands.Layout.of(tiny.tiny_config(), cap=8)
    cmds = commands.build(layout)
    store = {commands.Op.SCORES: commands.Op.KV_STORE, commands.Op.VALUES: commands.Op.V_STORE}
    for cmd, after in zip(cmds, commands.kv_waits(cmds), strict=True):
        if cmd.op in store:
            assert cmds[after].op == store[cmd.op] and cmds[after].addr == cmd.addr
        else:
            assert after is None


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
