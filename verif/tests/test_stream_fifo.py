# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/stream_fifo.sv (PLAN.md, M5) against a Python queue, cycle by cycle:
in_ready_o low exactly when full, out_valid_o high exactly when not empty,
the head's data, order. Random push and pop patterns that fill it, drain it
and push and pop in the same cycle, at depths 1, a power of two and one that
is not (the slot index wraps early)."""

import collections

import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly

import rtl

SEED = 20261010
WIDTH = 16
DEPTHS = (1, 4, 5)
CYCLES_PER_PHASE = 2000
# (chance of a push, chance of a pop) per cycle: filling, draining, balanced.
PHASES = ((0.9, 0.2), (0.2, 0.9), (0.5, 0.5))


@rtl.needs_verilator
@pytest.mark.parametrize("depth", DEPTHS)
def test_stream_fifo(depth: int) -> None:
    rtl.simulate("stream_fifo", params=(("Width", WIDTH), ("Depth", depth)))


@cocotb.test()
async def random_traffic(dut) -> None:
    depth = int(dut.Depth.value)
    rng = np.random.default_rng(SEED + depth)
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    dut.rst_ni.value = 0
    dut.in_valid_i.value = 0
    dut.out_ready_i.value = 0
    for _ in range(3):
        await FallingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    queue: collections.deque[int] = collections.deque()
    full = empty = both_when_full = 0
    for p_push, p_pop in PHASES:
        for cycle in range(CYCLES_PER_PHASE):
            await FallingEdge(dut.clk_i)
            push, pop = rng.random() < p_push, rng.random() < p_pop
            data = int(rng.integers(0, 1 << WIDTH))
            dut.in_valid_i.value, dut.in_data_i.value, dut.out_ready_i.value = push, data, pop
            await ReadOnly()
            where = f"seed {SEED + depth}, depth {depth}, cycle {cycle}, {len(queue)} queued"
            assert bool(dut.in_ready_o.value) == (len(queue) < depth), where
            assert bool(dut.out_valid_o.value) == bool(queue), where
            if queue:
                assert int(dut.out_data_o.value) == queue[0], where
            full += len(queue) == depth
            empty += not queue
            both_when_full += len(queue) == depth and push and pop
            if pop and queue:
                queue.popleft()
            if push and dut.in_ready_o.value:
                queue.append(data)
    assert full and empty and both_when_full, (full, empty, both_when_full)
