# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""A whole decode step on rtl/vec_unit.sv (verif/tests/test_vec_decode.py,
scripts/measure_m6.py): every vector, LOAD and KV_STORE command of
commands.build()'s list in one run, the engine's commands stood in by the
driver's engine steps with commands.run()'s results (OUTPUT and END are
not the unit's), and its check bit for bit against run().

The memory holds only what the unit reads and writes (PLAN.md, M6's
real-data answers; D-040's channel offsets come in M7): its own map puts
each KV cache at a low offset in every channel (D-043), the tensors LOAD
reads in channel 0 after them, and the rest (the engine's) past the
memory."""

import dataclasses

import commands
import engine as eng
import numpy as np
import perf
import torch
import vector
from commands import Command, Layout, Op
from golden import decoder
from vector import EngineStep, Step

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
    (waiting where the controller stops, commands.stops), engine steps for
    the engine's."""
    by_index = {c.index: c for c in calls}
    return [
        engine_step(by_index[i])
        if cmd.op in commands.ENGINE_OPS
        else Step(cmd, t=t, stride=stride, token=token, wait=stop)
        for i, (cmd, stop) in enumerate(zip(cmds, commands.stops(cmds), strict=True))
        if cmd.op not in (Op.OUTPUT, Op.END)
    ]


def unit_cycles(steps: list[Step | EngineStep], result: vector.Result) -> list[int]:
    """Each command's cycles, as perf.vector_cycles counts them: from the
    done before it, or from the cycle before it was taken if it waited.
    Every command after an engine step must wait (the driver holds it until
    the step is done, the controller would not)."""
    cmd_steps = [s for s in steps if isinstance(s, Step)]
    after_engine = [
        isinstance(p, EngineStep)
        for p, s in zip([None, *steps[:-1]], steps, strict=True)
        if isinstance(s, Step)
    ]
    out, done = [], None
    for s, follows, taken, d in zip(
        cmd_steps, after_engine, result.taken, result.done, strict=True
    ):
        assert s.wait or not follows, f"{s.cmd} after an engine step does not wait"
        out.append(int(d) - (int(taken) - 1 if s.wait or done is None else done))
        done = int(d)
    return out


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


def smollm2_step(position: int) -> tuple[list[int], list[float], tuple[perf.Step, ...]]:
    """SmolLM2's step at `position` of the long cache's text (NoLongCache
    without it), at F2's memory latency, checked: each command's cycles
    (unit_cycles), the perf model's (design v0) and its perf.Step."""
    tokens, saved = eng.saved_cache()
    cache = dataclasses.replace(saved, length=position)
    cap = -(-(position + 1) // 16) * 16
    model = eng.smollm2()
    steps, result = decode_step(
        model,
        cache,
        tokens[position],
        cap,
        cap + 16,
        SMOLLM2_LENGTH,
        SMOLLM2_CHAN_W,
        f"SmolLM2 at position {position}",
        **F2_MEMORY,
    )
    design = perf.load_design("v0").values()
    work = perf.Work.of(perf.design_ops(model.config, position, design))
    model_cycles = perf.vector_cycles(work.vector_steps, design)
    return unit_cycles(steps, result), model_cycles, work.vector_steps
