# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/stream_fifo.sv (PLAN.md, M5) against a Python queue, cycle by cycle:
in_ready_o low exactly when full, out_valid_o high exactly when not empty,
the head's data (so it holds while out_ready_i is low), order. Cases: fill
to full and drain to empty; random push and pop patterns that fill, drain,
and push and pop together both when full and when empty; push and pop every
cycle (one item per cycle from depth 2, every other cycle at depth 1, where
a full queue takes no push even while it pops); reset with items inside.
Depths 1 to 5 (the slot index wraps early unless the depth is a power of
two) and the DMA's 64; widths 1, 16 and the DMA's 256."""

import collections
from dataclasses import dataclass, field

import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly

import rtl

SEED = 20261010
# (Width, Depth): small depths, odd and even; the DMA's FIFO; the narrowest.
SHAPES = [(16, 1), (16, 2), (16, 3), (16, 4), (16, 5), (256, 64), (1, 4)]
CYCLES_PER_PHASE = 2000
# (chance of a push, chance of a pop) per cycle: filling, draining, balanced.
PHASES = ((0.9, 0.2), (0.2, 0.9), (0.5, 0.5))
STREAM_CYCLES = 500


@rtl.needs_verilator
@pytest.mark.parametrize(("width", "depth"), SHAPES, ids=[f"w{w}-d{d}" for w, d in SHAPES])
def test_stream_fifo(width: int, depth: int) -> None:
    rtl.simulate("stream_fifo", params=(("Width", width), ("Depth", depth)))


@dataclass
class Bench:
    """The FIFO, its Python twin, and counts of the cases reached."""

    dut: object
    depth: int
    width: int
    seed: int
    rng: np.random.Generator = field(init=False)
    queue: collections.deque = field(default_factory=collections.deque)
    cycle: int = 0
    seen: collections.Counter = field(default_factory=collections.Counter)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def where(self) -> str:
        return (
            f"seed {self.seed}, width {self.width}, depth {self.depth}, cycle {self.cycle}, "
            f"{len(self.queue)} queued"
        )

    async def step(self, push: bool, pop: bool) -> tuple[bool, bool]:
        """One cycle with in_valid_i = push and out_ready_i = pop: check the
        outputs against the queue, then update it. Returns the FIFO's own
        handshakes: whether an item went in and whether one came out."""
        dut = self.dut
        await FallingEdge(dut.clk_i)
        data = int.from_bytes(self.rng.bytes(-(-self.width // 8)), "little") % (1 << self.width)
        dut.in_valid_i.value, dut.in_data_i.value, dut.out_ready_i.value = push, data, pop
        await ReadOnly()
        n = len(self.queue)
        pushed, popped = push and bool(dut.in_ready_o.value), pop and bool(dut.out_valid_o.value)
        assert bool(dut.in_ready_o.value) == (n < self.depth), self.where()
        assert bool(dut.out_valid_o.value) == (n > 0), self.where()
        if n:
            assert int(dut.out_data_o.value) == self.queue[0], self.where()
        state = "full" if n == self.depth else "empty" if n == 0 else "partial"
        self.seen[state] += 1
        if push and pop:
            self.seen[f"both when {state}"] += 1
        if popped:
            self.queue.popleft()
        if pushed:
            self.queue.append(data)
        self.cycle += 1
        return pushed, popped

    async def reset(self) -> None:
        """Three cycles in reset, the outputs checked in each: empty, ready."""
        dut = self.dut
        await FallingEdge(dut.clk_i)  # a step ends in ReadOnly
        dut.rst_ni.value = 0
        dut.in_valid_i.value = 0
        dut.out_ready_i.value = 0
        self.queue.clear()
        for _ in range(3):
            await FallingEdge(dut.clk_i)
            await ReadOnly()
            assert not dut.out_valid_o.value and dut.in_ready_o.value, self.where()
        await FallingEdge(dut.clk_i)
        dut.rst_ni.value = 1


async def start(dut) -> Bench:
    width, depth = int(dut.Width.value), int(dut.Depth.value)
    bench = Bench(dut, depth, width, SEED + 1000 * width + depth)
    dut.rst_ni.value = 0  # before the first edge
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await bench.reset()
    return bench


async def stream(bench: Bench, cycles: int) -> tuple[int, int]:
    """Push and pop every cycle; the items in and out, by the FIFO's
    handshakes."""
    steps = [await bench.step(push=True, pop=True) for _ in range(cycles)]
    return sum(i for i, _ in steps), sum(o for _, o in steps)


@cocotb.test()
async def fill_then_drain(dut) -> None:
    """Push until full, then keep pushing for 3 more cycles with nothing
    taken out (each refused, nothing overwritten), then pop until empty."""
    bench = await start(dut)
    for _ in range(bench.depth + 3):
        await bench.step(push=True, pop=False)
    assert len(bench.queue) == bench.depth, bench.where()
    for _ in range(bench.depth + 1):
        await bench.step(push=False, pop=True)
    assert not bench.queue, bench.where()


@cocotb.test()
async def random_traffic(dut) -> None:
    bench = await start(dut)
    for p_push, p_pop in PHASES:
        for _ in range(CYCLES_PER_PHASE):
            await bench.step(bench.rng.random() < p_push, bench.rng.random() < p_pop)
    for case in ("full", "empty", "both when full", "both when empty"):
        assert bench.seen[case], (bench.where(), case, bench.seen)


@cocotb.test()
async def push_and_pop_every_cycle(dut) -> None:
    """Streaming, counted by the FIFO's handshakes: from empty, one item per
    cycle from depth 2; at depth 1 the queue is full whenever it holds an
    item, so every other cycle. Then from full: the first cycle refuses the
    push (full), after which it streams one per cycle near full."""
    bench = await start(dut)
    n_in, n_out = await stream(bench, STREAM_CYCLES)
    if bench.depth > 1:
        assert (n_in, n_out) == (STREAM_CYCLES, STREAM_CYCLES - 1), (bench.where(), n_in, n_out)
    else:
        assert n_in == STREAM_CYCLES // 2 and n_out in (n_in, n_in - 1), (bench.where(), n_in)
    while len(bench.queue) < bench.depth:
        await bench.step(push=True, pop=False)
    n_in, n_out = await stream(bench, STREAM_CYCLES)
    if bench.depth > 1:
        assert (n_in, n_out) == (STREAM_CYCLES - 1, STREAM_CYCLES), (bench.where(), n_in, n_out)
        assert len(bench.queue) == bench.depth - 1, bench.where()


@cocotb.test()
async def reset_empties(dut) -> None:
    """Reset with items inside: the queue comes back empty and works."""
    bench = await start(dut)
    for _ in range(bench.depth):
        await bench.step(push=True, pop=False)
    await bench.reset()
    for _ in range(3 * bench.depth):
        await bench.step(push=True, pop=bench.rng.random() < 0.5)
