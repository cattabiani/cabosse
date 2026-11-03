# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""platforms/f2/bringup/bringup_pipe.sv against a Python queue, cycle by
cycle: every beat comes out once, in order; the output holds while
out_ready_i is low; busy_o high exactly when a beat is inside; a stage
takes a beat while its output waits (the skid register), and the pipe
streams one beat a cycle with out_ready_i high. Random valid and ready
(the source keeps a beat offered until taken, as AXI asks); reset with
beats inside. 1 to 3 stages."""

import collections
from dataclasses import dataclass, field

import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly

import rtl

SEED = 20261103
WIDTH = 16
STAGES = [1, 2, 3]
CYCLES = 4000
STREAM_CYCLES = 200


@rtl.needs_verilator
@pytest.mark.parametrize("stages", STAGES)
def test_bringup_pipe(stages: int) -> None:
    rtl.simulate("bringup_pipe", params=(("Width", WIDTH), ("Stages", stages)))


@dataclass
class Bench:
    """The pipe, the beats inside it, and the source's offered beat."""

    dut: object
    stages: int
    seed: int
    rng: np.random.Generator = field(init=False)
    inside: collections.deque = field(default_factory=collections.deque)
    offer: int | None = None  # the beat offered, kept until taken
    held: int | None = None  # the output's beat last cycle, if not taken
    cycle: int = 0
    out: list[int] = field(default_factory=list)
    skids: int = 0  # cycles a beat went in while the output waited

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def where(self) -> str:
        return f"seed {self.seed}, stages {self.stages}, cycle {self.cycle}"

    async def step(self, p_valid: float, p_ready: float) -> None:
        dut = self.dut
        await FallingEdge(dut.clk_i)
        if self.offer is None and self.rng.random() < p_valid:
            self.offer = int(self.rng.integers(0, 1 << WIDTH))
        ready = bool(self.rng.random() < p_ready)
        dut.in_valid_i.value = self.offer is not None
        dut.in_data_i.value = self.offer or 0
        dut.out_ready_i.value = ready
        await ReadOnly()
        assert bool(dut.busy_o.value) == bool(self.inside), self.where()
        valid = bool(dut.out_valid_o.value)
        if valid:
            data = int(dut.out_data_o.value)
            assert self.inside and data == self.inside[0], self.where()
            if self.held is not None:
                assert data == self.held, self.where()
        else:
            assert self.held is None, self.where()  # a beat offered stays offered
        if self.offer is not None and dut.in_ready_o.value:
            if valid and not ready:
                self.skids += 1
            self.inside.append(self.offer)
            self.offer = None
        self.held = None
        if valid and ready:
            self.out.append(self.inside.popleft())
        elif valid:
            self.held = data
        self.cycle += 1


async def start(dut) -> Bench:
    stages = int(dut.Stages.value)
    bench = Bench(dut, stages, SEED + stages)
    dut.rst_ni.value = 0
    dut.in_valid_i.value = 0
    dut.out_ready_i.value = 0
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    for _ in range(3):
        await FallingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    return bench


@cocotb.test()
async def random_traffic(dut) -> None:
    """Random offers and readiness: in order, none lost, the skid used."""
    bench = await start(dut)
    for p_valid, p_ready in ((0.9, 0.3), (0.3, 0.9), (0.7, 0.7)):
        for _ in range(CYCLES):
            await bench.step(p_valid, p_ready)
    for _ in range(4 * bench.stages + 4):
        await bench.step(0.0, 1.0)
    sent = len(bench.out)
    assert not bench.inside and bench.offer is None, bench.where()
    assert sent > CYCLES and bench.skids > 0, (bench.where(), sent, bench.skids)


@cocotb.test()
async def streams_one_a_cycle(dut) -> None:
    """Offered and taken every cycle: one beat a cycle, Stages cycles late."""
    bench = await start(dut)
    for _ in range(STREAM_CYCLES):
        await bench.step(1.0, 1.0)
    assert len(bench.out) == STREAM_CYCLES - bench.stages, (bench.where(), len(bench.out))


@cocotb.test()
async def reset_empties(dut) -> None:
    """Reset with beats inside: empty, not busy, then works."""
    bench = await start(dut)
    for _ in range(2 * bench.stages):
        await bench.step(1.0, 0.0)
    assert bench.inside, bench.where()
    await FallingEdge(dut.clk_i)
    dut.rst_ni.value = 0
    dut.in_valid_i.value = 0
    await FallingEdge(dut.clk_i)
    await ReadOnly()
    assert not dut.out_valid_o.value and not dut.busy_o.value, bench.where()
    await FallingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    bench.inside.clear()
    bench.offer = bench.held = None
    for _ in range(500):
        await bench.step(0.6, 0.6)
