# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/controller.sv's reset window (M10): the units' reset reaches them
UnitsResetDelay cycles after units_reset_o (core.sv's registers), so after
rst_ni and after a CONTROL reset the controller takes no AXI-Lite write for
SettleCycles = UnitsResetDelay + 4 cycles; reads go on. units_reset_o pulses
for one cycle on a CONTROL reset."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

import rtl

SETTLE_CYCLES = 2 + 4  # controller.sv's SettleCycles at its default UnitsResetDelay = 2
REG_CONTROL, REG_STATUS, REG_TOKEN = 0x08, 0x0C, 0x10
CONTROL_RESET = 2
# The inputs at rest: nothing offered, the units always ready and never done.
IDLE_LOW = (
    "s_aw_valid_i",
    "s_w_valid_i",
    "s_ar_valid_i",
    "s_aw_addr_i",
    "s_w_data_i",
    "s_ar_addr_i",
    "eng_done_i",
    "eng_err_i",
    "vec_done_i",
    "vec_err_i",
)
IDLE_HIGH = ("s_b_ready_i", "s_r_ready_i", "eng_ready_i", "vec_ready_i")


@rtl.needs_verilator
def test_controller_reset() -> None:
    rtl.simulate("controller", test_module="test_controller_reset")


async def start(dut) -> None:
    """Clock, the inputs at rest and 3 cycles of rst_ni."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    for name in IDLE_LOW:
        getattr(dut, name).value = 0
    for name in IDLE_HIGH:
        getattr(dut, name).value = 1
    dut.rst_ni.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.rst_ni.value = 1


def offer_write(dut, addr: int, value: int) -> None:
    dut.s_aw_valid_i.value = dut.s_w_valid_i.value = 1
    dut.s_aw_addr_i.value = addr
    dut.s_w_data_i.value = value


async def wait_write(dut) -> int:
    """The offered write taken: the cycles it waited for AWREADY. Ends on
    the falling edge after it was taken, nothing offered (s_b_ready_i
    takes the response then)."""
    waited = 0
    while True:
        await ReadOnly()
        if dut.s_aw_ready_o.value:
            break
        await FallingEdge(dut.clk_i)
        waited += 1
    await FallingEdge(dut.clk_i)
    dut.s_aw_valid_i.value = dut.s_w_valid_i.value = 0
    return waited


@cocotb.test()
async def writes_wait_out_the_reset(dut) -> None:
    await start(dut)
    offer_write(dut, REG_TOKEN, 5)
    waited = await wait_write(dut)
    assert waited == SETTLE_CYCLES, ("after rst_ni", waited)
    await FallingEdge(dut.clk_i)  # the response
    offer_write(dut, REG_TOKEN, 6)
    waited = await wait_write(dut)
    assert waited == 0, ("settled: a write goes at once", waited)
    await FallingEdge(dut.clk_i)

    offer_write(dut, REG_CONTROL, CONTROL_RESET)
    await wait_write(dut)
    # From the cycle after the reset was taken: a read goes at once, a
    # write waits SettleCycles.
    dut.s_ar_valid_i.value = 1
    dut.s_ar_addr_i.value = REG_STATUS
    offer_write(dut, REG_TOKEN, 7)
    await ReadOnly()
    assert dut.s_ar_ready_o.value, "a read during the window"
    await FallingEdge(dut.clk_i)
    dut.s_ar_valid_i.value = 0
    waited = await wait_write(dut)
    assert waited + 1 == SETTLE_CYCLES, ("after a CONTROL reset", waited + 1)


@cocotb.test()
async def units_reset_pulses_once(dut) -> None:
    await start(dut)
    for _ in range(SETTLE_CYCLES):
        await FallingEdge(dut.clk_i)
    offer_write(dut, REG_CONTROL, CONTROL_RESET)
    await wait_write(dut)
    pulses = 0
    for _ in range(SETTLE_CYCLES + 2):
        await ReadOnly()
        pulses += int(dut.units_reset_o.value)
        await FallingEdge(dut.clk_i)
    assert pulses == 1, ("units_reset_o's cycles", pulses)
