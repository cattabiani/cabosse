# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""A whole decode step of the tiny config (GQA) on rtl/vec_unit.sv: every
vector, LOAD and KV_STORE command of commands.build()'s list in one run,
the engine's commands stood in by the driver's engine steps with
commands.run()'s results (OUTPUT and END are not the unit's).

Checked bit for bit against run(): what each engine command reads (so a
divergence shows at the first engine command after it), every buffer at the
end, and the whole memory: the KV cache in channel order (D-043), every
other byte unchanged. Both start from the same buffers, all NaN
(commands.empty_buffers), so a read of an element no command wrote shows.

The memory map: Layout's addresses are still flat (D-040: channel offsets
come in M7), so channel 0 holds the flat image whole, LOAD reads it at its
flat addresses, and KV_STORE takes a cache's flat address as its offset in
every channel. Channel 0's share of a cache lies inside the flat image's
place for it."""

import commands
import numpy as np
import pytest
import torch
import vector
from commands import Command, Layout, Op
from engine import cache_by_channel
from golden import decoder, tiny
from vector import EngineStep, Step, with_waits

import rtl

SEED = 20261030
CONFIG = tiny.tiny_config(n_kv_heads=2)  # 4 query heads: 2 per KV head
LENGTH = 256  # elements of every buffer: the largest, LOGITS
CHAN_W = 18  # 256 KiB channels: the flat image fits in channel 0


def image(layout: Layout, hbm: dict[int, torch.Tensor], cmds: list[Command]) -> np.ndarray:
    """The memory's words: every tensor at its flat address in channel 0,
    each KV cache (as KV_STORE writes it) by D-043 at its flat address in
    every channel."""
    assert layout.total_bytes <= 1 << CHAN_W, layout.total_bytes
    mem = np.zeros((vector.KV_PORTS << CHAN_W) // 4, dtype=np.uint32)
    caches = {c.addr: c for c in cmds if c.op == Op.KV_STORE}
    for addr, x in hbm.items():
        raw = x.contiguous().view(-1).view(torch.uint8).numpy()
        if addr in caches:
            rows = raw.view(np.uint16).reshape(x.shape)
            vector.cache_view(mem, caches[addr])[...] = cache_by_channel(rows, vector.KV_PORTS)
        else:
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


# (position, cap, stride): position 0, and position 5 after a prompt with a
# cache of 8 rows and S and P's rows at 32.
CASES = [(0, 16, 16), (5, 8, 32)]


@rtl.needs_verilator
@pytest.mark.parametrize(("position", "cap", "stride"), CASES)
def test_decode_step(position: int, cap: int, stride: int) -> None:
    gen = torch.Generator().manual_seed(SEED + position)
    tokens = torch.randint(CONFIG.vocab_size, (position + 1,), generator=gen).tolist()
    model = decoder.from_state_dict(CONFIG, tiny.random_weights(CONFIG, SEED))
    layout = Layout.of(CONFIG, cap, stride)
    cache = decoder.KVCache.empty(model, cap)
    if position:
        decoder.step(model, cache, tokens[:-1])
    hbm = commands.load(model, layout, cache)
    cmds = commands.build(layout)
    mem = image(layout, hbm, cmds)

    start = commands.empty_buffers({b: (vector.dtype(b), LENGTH) for b in vector.BUFS})
    want, calls = {b: x.clone() for b, x in start.items()}, []
    commands.run(cmds, layout, hbm, tokens[-1], position, calls.append, want)
    steps = unit_steps(cmds, calls, tokens[-1], position + 1, stride)
    result = vector.run(steps, start, LENGTH, mem, chan_w=CHAN_W)
    label = f"position {position}, seed {SEED}"

    for call, (got,) in zip(calls, result.engine_reads, strict=True):
        bad = np.flatnonzero(got != vector.bits(call.a))
        assert not len(bad), (
            f"{label}: command {call.index} ({call.cmd.op.name} {call.name}) reads "
            f"{len(bad)} elements that differ, first {bad[0]}"
        )
    vector.assert_same(result.buffers, {b: vector.bits(x) for b, x in want.items()}, label)
    bad = np.flatnonzero(result.mem != image(layout, hbm, cmds))
    assert not len(bad), f"{label}: memory: {len(bad)} words differ, first at byte 0x{4 * bad[0]:X}"
    assert not result.err, f"{label}: err_o"
    print(f"{label}: {len(steps)} steps in {result.cycles} cycles")
