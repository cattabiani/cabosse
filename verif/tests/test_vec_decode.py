# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""A whole decode step of the tiny config (GQA) on rtl/vec_unit.sv: every
vector, LOAD and KV_STORE command of commands.build()'s list in one run,
the engine's commands stood in by the driver's engine steps with
commands.run()'s results (OUTPUT and END are not the unit's).

Checked bit for bit against run(): what each engine command reads (so a
divergence shows at the first engine command after it), every buffer at the
end, and the whole memory: the KV cache in channel order (D-043), every
other byte unchanged. Buffers start as NaN, as run()'s do, so a read of an
element no command wrote shows.

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
from commands import Buf, Command, Layout, Op
from engine import cache_by_channel
from golden import decoder, tiny
from test_vec_unit import with_waits
from vector import EngineStep, Step

import rtl

SEED = 20261030
CONFIG = tiny.tiny_config(n_kv_heads=2)  # 4 query heads: 2 per KV head
LENGTH = 256  # elements of every buffer: the largest, LOGITS
CHAN_W = 18  # 256 KiB channels: the flat image fits in channel 0
BUFS = [b for b in Buf if b != Buf.NONE]


def image(layout: Layout, hbm: dict[int, torch.Tensor]) -> np.ndarray:
    """The memory's words: every tensor at its flat address in channel 0,
    each KV cache by D-043 at its flat address in every channel."""
    assert layout.total_bytes <= 1 << CHAN_W, layout.total_bytes
    channels = np.zeros((vector.KV_PORTS, 1 << CHAN_W), dtype=np.uint8)
    for name, (addr, _, _) in layout.tensors.items():
        raw = hbm[addr].contiguous().view(torch.uint8).numpy()
        if "cache" not in name:
            channels[0, addr : addr + raw.size] = raw.ravel()
            continue
        rows = cache_by_channel(raw.view(np.uint16), vector.KV_PORTS)
        channels[:, addr : addr + raw.size // vector.KV_PORTS] = rows.reshape(
            vector.KV_PORTS, -1
        ).view(np.uint8)
    return channels.view(np.uint32).ravel()


def engine_step(call: commands.EngineCall) -> EngineStep:
    """The engine command's reads of its input and its result's writes: S
    and P as rows of t at the stride."""
    cmd, stride, t = call.cmd, call.stride, call.t
    if cmd.op == Op.MATVEC:
        return EngineStep([(cmd.a, 0, cmd.m)], [(cmd.dst, 0, call.out)])
    if cmd.op == Op.SCORES:
        rows = call.rows(call.out)
        return EngineStep(
            [(cmd.a, 0, cmd.n * cmd.m)], [(cmd.dst, h * stride, rows[h]) for h in range(cmd.n)]
        )
    return EngineStep([(cmd.a, h * stride, t) for h in range(cmd.n)], [(cmd.dst, 0, call.out)])


def engine_input(call: commands.EngineCall) -> np.ndarray:
    """What the engine command reads, as engine_step's reads, raw bits."""
    a = call.rows(call.a) if call.cmd.op == Op.VALUES else call.a
    return vector.bits(a.reshape(-1))


def unit_steps(
    cmds: list[Command], calls: list[commands.EngineCall], token: int, t: int, stride: int
) -> list[Step | EngineStep]:
    """The list as the driver runs it: vector, LOAD and KV_STORE commands
    (a command waits where it reads what one still running writes), engine
    steps for the engine's."""
    by_index = {c.index: c for c in calls}
    out: list[Step | EngineStep] = []
    run: list[Step] = []  # since the last engine step
    for i, cmd in enumerate(cmds):
        if cmd.op in (Op.OUTPUT, Op.END):
            continue
        if cmd.op in commands.ENGINE_OPS:
            out += [*with_waits(run), engine_step(by_index[i])]
            run = []
        else:
            run.append(Step(cmd, t=t, stride=stride, token=token))
    return out + with_waits(run)


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
    mem = image(layout, hbm)

    cmds, calls, want = commands.build(layout), [], {}
    commands.run(cmds, layout, hbm, tokens[-1], position, calls.append, want)
    steps = unit_steps(cmds, calls, tokens[-1], position + 1, stride)
    start = {b: torch.full((LENGTH,), torch.nan, dtype=vector.dtype(b)) for b in BUFS}
    result = vector.run(steps, start, LENGTH, mem, chan_w=CHAN_W)
    label = f"position {position}, seed {SEED}"

    for call, got in zip(calls, result.engine_reads, strict=True):
        bad = np.flatnonzero(np.concatenate(got) != engine_input(call))
        assert not len(bad), (
            f"{label}: command {call.index} ({call.cmd.op.name} {call.name}) reads "
            f"{len(bad)} elements that differ, first {bad[0]}"
        )
    want_bits = {b: vector.bits(start[b]) for b in BUFS}
    for b, x in want.items():
        want_bits[b][: len(x)] = vector.bits(x)
    vector.assert_same(result.buffers, want_bits, label)
    bad = np.flatnonzero(result.mem != image(layout, hbm))
    assert not len(bad), f"{label}: memory: {len(bad)} words differ, first at byte 0x{4 * bad[0]:X}"
    assert not result.err, f"{label}: err_o"
    print(f"{label}: {len(steps)} steps in {result.cycles} cycles")
