# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""A whole decode step of the tiny config (GQA) on rtl/vec_unit.sv: every
vector, LOAD and KV_STORE command of commands.build()'s list in one run,
the engine's commands stood in by the driver's engine steps with
commands.run()'s results (OUTPUT and END are not the unit's).

Checked bit for bit against run(): what each engine command reads (so a
divergence shows at the first engine command after it), every buffer at the
end, and the whole memory: the KV cache in channel order (D-043), every
other byte unchanged. An engine step waits for every earlier command, so
the engine never overlaps the vector unit here (on shared buffers: M7).
Both start from the same buffers, all NaN
(commands.empty_buffers), so a read of an element no command wrote shows.

The memory holds only what the unit reads and writes (PLAN.md, M6's
real-data answers; D-040's channel offsets come in M7): the test's own
map puts each KV cache at a low offset in every channel (D-043), the
tensors LOAD reads in channel 0 after them, and the rest (the engine's)
past the memory. SmolLM2's step (slow) runs at the real-data positions,
one at a time: its memory is up to 512 MiB."""

import dataclasses

import commands
import engine as eng
import numpy as np
import paths
import pytest
import torch
import vector
from commands import Command, Layout, Op
from golden import decoder, tiny
from vector import EngineStep, Step, with_waits

import rtl

SEED = 20261030
CONFIG = tiny.tiny_config(n_kv_heads=2)  # 4 query heads: 2 per KV head
LENGTH = 256  # elements of every buffer: the largest, LOGITS
CHAN_W = 17  # 128 KiB channels
# SmolLM2 at the real-data positions (PLAN.md, M6); its caches hold whole
# groups of 16 positions, S and P's rows sit at cap + 16 (so a mix-up of
# the two shows).
SMOLLM2_POSITIONS = (0, 63, 1023, 8191)
SMOLLM2_LENGTH = 9 * (8192 + 16)  # S and P at the longest stride
SMOLLM2_CHAN_W = 27  # 128 MiB: at 8191, the caches' 47 MiB a channel, then LOAD's 61 MiB
# F2's read latency for both of the unit's AXI channels: its write latency
# is not documented (to measure in M8).
F2_MEMORY = eng.F2_MEMORY | {"wlatency": eng.F2_LATENCY_CYCLES}


def mapped(layout: Layout, chan_w: int) -> Layout:
    """layout at the test's addresses: each KV cache at a low offset in
    every channel (its share of the channels' bytes apart), the tensors
    LOAD reads in channel 0 after them, the rest past the memory."""
    cmds = commands.build(layout)
    caches = {c.addr for c in cmds if c.op == Op.KV_STORE}
    loads = {c.addr for c in cmds if c.op == Op.LOAD}

    def place(addrs: set[int], at: int, share: int = 1) -> int:
        for name, (addr, shape, dtype) in layout.tensors.items():
            if addr in addrs:
                new[name] = (at, shape, dtype)
                at += -(-commands.n_bytes(shape, dtype) // share // commands.ALIGN) * commands.ALIGN
        return at

    new: dict[str, tuple[int, tuple[int, ...], torch.dtype]] = {}
    end = place(loads, place(caches, 0, vector.KV_PORTS))
    assert end <= 1 << chan_w, (end, chan_w)
    rest = {a for a, _, _ in layout.tensors.values()} - caches - loads
    place(rest, vector.KV_PORTS << chan_w)
    return dataclasses.replace(layout, tensors={n: new[n] for n in layout.tensors})


def image(
    layout: Layout, hbm: dict[int, torch.Tensor], cmds: list[Command], chan_w: int
) -> np.ndarray:
    """The memory's words (mapped()'s layout): each KV cache as KV_STORE
    writes it (D-043), the tensors in channel 0 as they are."""
    mem = np.zeros((vector.KV_PORTS << chan_w) // 4, dtype=np.uint32)
    caches = {c.addr: c for c in cmds if c.op == Op.KV_STORE}
    for addr, x in hbm.items():
        if addr in caches:
            rows = x.contiguous().view(torch.int16).numpy().view(np.uint16)
            vector.cache_view(mem, caches[addr])[...] = eng.cache_by_channel(rows, vector.KV_PORTS)
        elif addr < 1 << chan_w:
            raw = x.contiguous().view(-1).view(torch.uint8).numpy()
            mem.view(np.uint8)[addr : addr + raw.size] = raw
    return mem


def engine_step(call: commands.EngineCall) -> EngineStep:
    """The engine command's read of its input (EngineCall.a's span) and its
    result's writes: SCORES's as rows of t at the stride, as it writes them."""
    cmd = call.cmd
    reads = [(cmd.a, 0, len(call.a))]
    if cmd.op != Op.SCORES:
        return EngineStep(reads, [(cmd.dst, 0, call.out)])
    rows = call.rows(call.out)
    return EngineStep(reads, [(cmd.dst, h * call.stride, rows[h]) for h in range(cmd.n)])


def unit_steps(
    cmds: list[Command], calls: list[commands.EngineCall], token: int, t: int, stride: int
) -> list[Step | EngineStep]:
    """The list as the driver runs it: vector, LOAD and KV_STORE commands
    (with_waits' waits), engine steps for the engine's."""
    by_index = {c.index: c for c in calls}
    steps = [
        engine_step(by_index[i])
        if cmd.op in commands.ENGINE_OPS
        else Step(cmd, t=t, stride=stride, token=token)
        for i, cmd in enumerate(cmds)
        if cmd.op not in (Op.OUTPUT, Op.END)
    ]
    return with_waits(steps)


def decode_step(
    model: decoder.Model,
    cache: decoder.KVCache,
    token: int,
    cap: int,
    stride: int,
    length: int,
    chan_w: int,
    label: str,
    **plusargs: int,
) -> tuple[list[Step | EngineStep], vector.Result]:
    """The step of `token` after `cache`'s positions on the unit, checked
    bit for bit against commands.run(): what each engine command reads,
    every buffer, the whole memory."""
    position = cache.length
    layout = mapped(Layout.of(model.config, cap, stride), chan_w)
    assert max(n for _, n in commands.buffers(model.config, stride).values()) <= length
    hbm = commands.load(model, layout, cache)
    cmds = commands.build(layout)
    mem = image(layout, hbm, cmds, chan_w)

    start = commands.empty_buffers({b: (vector.dtype(b), length) for b in vector.BUFS})
    want, calls = {b: x.clone() for b, x in start.items()}, []
    commands.run(cmds, layout, hbm, token, position, calls.append, want)
    steps = unit_steps(cmds, calls, token, position + 1, stride)
    result = vector.run(steps, start, length, mem, chan_w=chan_w, **plusargs)
    del mem

    for call, (got,) in zip(calls, result.engine_reads, strict=True):
        bad = np.flatnonzero(got != vector.bits(call.a))
        assert not len(bad), (
            f"{label}: command {call.index} ({call.cmd.op.name} {call.name}) reads "
            f"{len(bad)} elements that differ, first {bad[0]}"
        )
    vector.assert_same(result.buffers, {b: vector.bits(x) for b, x in want.items()}, label)
    bad = np.flatnonzero(result.mem != image(layout, hbm, cmds, chan_w))
    assert not len(bad), f"{label}: memory: {len(bad)} words differ, first at byte 0x{4 * bad[0]:X}"
    assert not result.err, f"{label}: err_o"
    print(f"{label}: {len(steps)} steps in {result.cycles} cycles")
    return steps, result


# (position, cap, stride): position 0, and position 5 after a prompt with a
# cache of 8 rows and S and P's rows at 32.
CASES = [(0, 16, 16), (5, 8, 32)]


@rtl.needs_verilator
@pytest.mark.parametrize(("position", "cap", "stride"), CASES)
def test_decode_step(position: int, cap: int, stride: int) -> None:
    seed = SEED + position  # the tokens'
    gen = torch.Generator().manual_seed(seed)
    tokens = torch.randint(CONFIG.vocab_size, (position + 1,), generator=gen).tolist()
    model = decoder.from_state_dict(CONFIG, tiny.random_weights(CONFIG, SEED))
    cache = decoder.KVCache.empty(model, cap)
    if position:
        decoder.step(model, cache, tokens[:-1])
    label = f"position {position}, token seed {seed}"
    decode_step(model, cache, tokens[-1], cap, stride, LENGTH, CHAN_W, label)


@pytest.mark.slow
@rtl.needs_verilator
@pytest.mark.skipif(not paths.SMOLLM2.exists(), reason=f"needs the checkpoint in {paths.SMOLLM2}")
@pytest.mark.parametrize("position", SMOLLM2_POSITIONS)
def test_smollm2(position: int) -> None:
    """SmolLM2's whole step at `position` of the long cache's text
    (scripts/fill_long_cache.py), at F2's memory latency."""
    try:
        tokens, saved = eng.saved_cache()
    except eng.NoLongCache as e:
        pytest.skip(str(e))
    cache = dataclasses.replace(saved, length=position)
    cap = -(-(position + 1) // 16) * 16
    label = f"SmolLM2 at position {position}"
    decode_step(
        eng.smollm2(),
        cache,
        tokens[position],
        cap,
        cap + 16,
        SMOLLM2_LENGTH,
        SMOLLM2_CHAN_W,
        label,
        **F2_MEMORY,
    )
