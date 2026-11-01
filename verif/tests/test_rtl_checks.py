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
# The engine's own control path (8 ports, the scale's FMA stubbed: the FMA
# alone is 27, and met 250 MHz in M3), its value when written (M5's walker
# PR): next-cycle unit state into the RAMs' addresses.
MAX_ENGINE_CONTROL_DEPTH = 22
# (top, extra files, parameters): our modules, the harnesses in platforms/,
# then the units' flush-to-zero variants.
TOPS = (
    [(m, (), ()) for m in MODULES]
    + [(p.stem, (p,), ()) for p in rtl.harnesses()]
    + [(t, (), FTZ) for t in ("bf16_mac", "fp32_fma", "dot_lane")]
)
IDS = [top + ("-ftz" if params else "") for top, _, params in TOPS]
# Synthesized smaller than their default, which is linted: the check is that
# Yosys takes the code, and the 32-port engine needs more than 8 GB (killed
# at that cap after 3 minutes, M5). 4 ports (16 lanes, with the default
# Writes = 8) still build a drain of several write groups. The vector unit
# at 16 lanes (112 FMAs) needs more than 8 GB too (M6); 8 lanes, the
# fewest its stores take (2 * E), still bank every buffer. engine_path, the
# engine's timing harness, as the engine.
SYNTH_PARAMS = {
    "engine": (("Ports", 4),),
    "engine_path": (("Ports", 4),),
    "vec_unit": (("Lanes", 8), ("Len", 64)),
}
# (top, parameters, message): shapes that must not elaborate.
BAD_SHAPES = [
    # fp_product: its callers delay the addend by Regs cycles; a fourth
    # register would not be placed.
    ("fp_product", (("Regs", 4),), "Regs must be 0 to 3"),
    # dot_lane: E = 8 leaves each unit A / E = 2 partial sums for a 4-cycle
    # loop; a group would read a sum still in the adder.
    ("dot_lane", (("E", 8),), "A / E >= 4"),
    # dot_lane: one level-1 buffer leaves no room for a row's final sum
    # while the next row ends (and a 0-bit buffer pointer).
    ("dot_lane", (("Banks", 1),), "Banks must be at least 2"),
    # port_dma: longer bursts than AXI3 allows, a FIFO smaller than a burst,
    # a data width that is not a power of two, a count narrower than a burst,
    # slices that do not divide the beat.
    ("port_dma", (("MaxBurst", 17),), "MaxBurst"),
    ("port_dma", (("Depth", 8),), "Depth"),
    ("port_dma", (("DataWidth", 96),), "DataWidth"),
    ("port_dma", (("CountWidth", 4),), "CountWidth"),
    ("port_dma", (("Slices", 3),), "Slices"),
    # engine: a port count that is not a multiple of 4 (its groups of 4
    # ports), a write port wider than a pass, a head dimension unit_t.dlog
    # cannot hold or not a power of two, a row's beats wider than step's
    # cycle count.
    ("engine", (("Ports", 5),), "multiple of 4"),
    ("engine", (("Ports", 4), ("Writes", 32)), "dividing L"),
    ("engine", (("Ports", 4), ("MaxD", 256)), "MaxD"),
    ("engine", (("Ports", 4), ("MaxD", 96)), "MaxD"),
    ("engine", (("Ports", 4), ("ColWidth", 24)), "ColWidth"),
    # port_wdma: longer bursts than AXI3 allows, a data width that is not a
    # power of two.
    ("port_wdma", (("MaxBurst", 17),), "MaxBurst"),
    ("port_wdma", (("DataWidth", 96),), "DataWidth"),
    # vec_unit: 32 lanes leave 4 partials a slot, fewer than its 7-cycle
    # loop; the KV cache over a channel count that is not a power of two; a
    # beat of one chunk (E values, a K slot's).
    ("vec_unit", (("Lanes", 32),), "Partials"),
    ("vec_unit", (("KvPorts", 24),), "KvPorts"),
    ("vec_unit", (("Lanes", 4), ("Len", 64), ("Partials", 64)), "Lanes must be 2 * E"),
]


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
    result = rtl.synthesize(top, extra, (*SYNTH_PARAMS.get(top, ()), *params))
    assert result.returncode == 0, result.stdout + result.stderr


@rtl.needs_yosys
def test_lane_logic_depth() -> None:
    """A proxy for the lane's timing at 250 MHz (PLAN.md, M4): its deepest
    register-to-register path, the accumulate loop's stages included, may
    not grow past MAX_LANE_DEPTH cells. Vivado has the last word."""
    depth = rtl.logic_depth("dot_lane")
    assert depth <= MAX_LANE_DEPTH, f"dot_lane: {depth} cells deep, limit {MAX_LANE_DEPTH}"


@pytest.mark.slow
@rtl.needs_yosys
def test_engine_control_depth() -> None:
    """The engine's deepest path without the FMA may not grow past
    MAX_ENGINE_CONTROL_DEPTH cells (about 5 minutes and 4 GB). Vivado has
    the last word (M8)."""
    depth = rtl.logic_depth("engine", params=[("Ports", 8)], stubs=[rtl.STUBS / "fp32_fma.sv"])
    assert depth <= MAX_ENGINE_CONTROL_DEPTH, f"engine: {depth} cells deep"


@rtl.needs_verilator
@pytest.mark.parametrize(("top", "params", "message"), BAD_SHAPES)
def test_rejects_a_bad_shape(top: str, params: tuple, message: str) -> None:
    """Elaboration stops with a message."""
    result = rtl.lint(top, params=list(params))
    assert result.returncode != 0 and message in result.stderr, result.stderr
