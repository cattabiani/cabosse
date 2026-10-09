# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Every RTL module passes the Verilator lint with no warnings and
synthesizes in Yosys (AGENTS.md, SystemVerilog conventions), as the top of
all of rtl/. One module per file, named after it."""

import pytest

import rtl

MODULES = rtl.modules()
FTZ = (("Ftz", 1),)  # the flush-to-zero variant (D-016's cost)
# Logic depth (rtl.logic_depth) the lane may reach: its value when written.
# mac_loop, the accumulate loop that met 250 MHz on F2's part (Vivado, M3,
# +0.877 ns), has 17 by the same count.
MAX_LANE_DEPTH = 19
# (top, extra files, parameters): our modules, the harnesses in platforms/,
# then the units' flush-to-zero variants.
TOPS = (
    [(m, (), ()) for m in MODULES]
    + [(p.stem, (p,), ()) for p in rtl.harnesses()]
    + [(t, (), FTZ) for t in ("bf16_mac", "fp32_fma", "dot_lane")]
)
IDS = [top + ("-ftz" if params else "") for top, _, params in TOPS]


def test_there_is_rtl() -> None:
    assert MODULES


def test_sources_lists_every_rtl_file() -> None:
    """rtl/sources.f names every .sv file under rtl/, and nothing else."""
    lines = [ln.strip() for ln in rtl.SOURCES.read_text().splitlines()]
    listed = {rtl.RTL / ln for ln in lines if ln.endswith(".sv")}
    assert listed == set(rtl.RTL.rglob("*.sv"))


@rtl.needs_verilator
@pytest.mark.parametrize(("top", "extra", "params"), TOPS, ids=IDS)
def test_lint_is_clean(top: str, extra: tuple, params: tuple) -> None:
    result = rtl.lint(top, extra, params)
    assert result.returncode == 0 and not result.stderr.strip(), result.stderr


@rtl.needs_yosys
@pytest.mark.parametrize(("top", "extra", "params"), TOPS, ids=IDS)
def test_synthesizes(top: str, extra: tuple, params: tuple) -> None:
    result = rtl.synthesize(top, extra, params)
    assert result.returncode == 0, result.stdout + result.stderr


@rtl.needs_verilator
def test_fp_product_rejects_more_than_two_registers() -> None:
    """Its callers delay the addend by Regs cycles; a third register would not
    be placed, so Regs = 3 must not elaborate."""
    result = rtl.lint("fp_product", params=[("Regs", 3)])
    assert result.returncode != 0 and "Regs must be 0, 1 or 2" in result.stderr, result.stderr


@rtl.needs_yosys
def test_lane_logic_depth() -> None:
    """A proxy for the lane's timing at 250 MHz (PLAN.md, M4): its deepest
    register-to-register path, the accumulate loop's stages included, may
    not grow past MAX_LANE_DEPTH cells. Vivado has the last word."""
    depth = rtl.logic_depth("dot_lane")
    assert depth <= MAX_LANE_DEPTH, f"dot_lane: {depth} cells deep, limit {MAX_LANE_DEPTH}"


@rtl.needs_verilator
def test_dot_lane_rejects_a_short_loop() -> None:
    """E = 8 leaves each unit A / E = 2 partial sums for a 4-cycle loop: a
    group would read a sum still in the adder, so it must not elaborate."""
    result = rtl.lint("dot_lane", params=[("E", 8)])
    assert result.returncode != 0 and "A / E >= 4" in result.stderr, result.stderr


@rtl.needs_verilator
@pytest.mark.parametrize(
    ("params", "message"),
    [
        ((("MaxBurst", 17),), "MaxBurst"),
        ((("Depth", 8),), "Depth"),
        ((("DataWidth", 96),), "DataWidth"),
        ((("CountWidth", 6),), "CountWidth"),
    ],
)
def test_port_dma_rejects_a_bad_shape(params: tuple, message: str) -> None:
    """Longer bursts than AXI3 allows, a FIFO smaller than a burst, a data
    width that is not a power of two, a count too narrow for a page's
    beats: elaboration stops with a message."""
    result = rtl.lint("port_dma", params=list(params))
    assert result.returncode != 0 and message in result.stderr, result.stderr
