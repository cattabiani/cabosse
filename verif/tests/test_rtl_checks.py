# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Every RTL module passes the Verilator lint with no warnings and
synthesizes in Yosys (AGENTS.md, SystemVerilog conventions), as the top of
all of rtl/. One module per file, named after it."""

import re

import commands
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
# The controller's (a 16-command buffer), its value when written (M7): 31
# with a divider for the group, 17 without.
MAX_CONTROLLER_DEPTH = 17
# (top, extra files, parameters): our modules, the harnesses, the bring-up
# design and the core's wrapper in platforms/, then the units' flush-to-zero variants.
TOPS = (
    [(m, (), ()) for m in MODULES]
    + [(p.stem, rtl.extra_sources(p.stem), ()) for p in (*rtl.harnesses(), *rtl.BRINGUP)]
    + [(p.stem, rtl.F2_CORE, ()) for p in rtl.F2_CORE if p.parent == rtl.F2_CORE_TOP.parent]
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
    "bringup_core": (("Ports", 2),),
    "controller": (("Commands", 16),),  # Yosys keeps a 64 KiB buffer as flip-flops
}
# Not synthesized in Yosys, with why: each of their parts is.
NOT_SYNTHESIZED = {
    "core": "its vector unit takes 16 lanes (an HBM beat), more than 8 GB in Yosys (M6);"
    " Vivado builds it in M9",
    "f2_core": "the core inside, as core",
}
SYNTH_TOPS = [t for t in TOPS if t[0] not in NOT_SYNTHESIZED]
# (top, parameters, message): shapes that must not elaborate.
BAD_SHAPES = [
    # fp_product: its callers delay the addend by Regs cycles; a sixth
    # register would not be placed.
    ("fp_product", (("Regs", 6),), "Regs must be 0 to 5"),
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
    # bringup_core: more ports than a 32-bit PORT_EN and the counters' map hold.
    ("bringup_core", (("Ports", 33),), "Ports must be 1 to 32"),
]


def test_there_is_rtl() -> None:
    assert MODULES


def test_sources_lists_every_rtl_file() -> None:
    """rtl/sources.f names every .sv file under rtl/, and nothing else; the
    bring-up design's sources.f its folder's vendor-free modules (all but
    the CL top, cabosse_bringup.sv), and the core wrapper's its own folder's
    (all but cabosse_core.sv)."""
    assert set(rtl.file_list(rtl.SOURCES)) == set(rtl.RTL.rglob("*.sv"))
    folder = rtl.BRINGUP[0].parent
    assert set(rtl.BRINGUP) == set(folder.glob("*.sv")) - {rtl.BRINGUP_TOP}
    folder = rtl.F2_CORE_TOP.parent
    own = {p for p in rtl.F2_CORE if p.parent == folder}
    assert own == set(folder.glob("*.sv")) - {rtl.F2_CORE_TOP}


@rtl.needs_verilator
@pytest.mark.parametrize(("top", "extra", "params"), TOPS, ids=IDS)
def test_lint_is_clean(top: str, extra: tuple, params: tuple) -> None:
    result = rtl.lint(top, extra, params)
    assert result.returncode == 0 and not result.stderr.strip(), result.stderr


def test_not_synthesized_are_tops() -> None:
    """NOT_SYNTHESIZED names modules that exist (left out of test_synthesizes,
    not skipped: a skip reads as a missing tool)."""
    assert set(NOT_SYNTHESIZED) <= {top for top, _, _ in TOPS}


@rtl.needs_yosys
@pytest.mark.parametrize(
    ("top", "extra", "params"), SYNTH_TOPS, ids=[IDS[TOPS.index(t)] for t in SYNTH_TOPS]
)
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


@rtl.needs_yosys
def test_controller_logic_depth() -> None:
    """The controller's deepest path may not grow past MAX_CONTROLLER_DEPTH
    cells. Vivado has the last word (M9)."""
    depth = rtl.logic_depth("controller", params=SYNTH_PARAMS["controller"])
    assert depth <= MAX_CONTROLLER_DEPTH, f"controller: {depth} cells deep"


@rtl.needs_yosys
def test_memories_found() -> None:
    """rtl.memories finds a FIFO's storage with its width and depth (M9's
    on-chip memory total rests on it)."""
    found = rtl.memories("stream_fifo", params=[("Width", 40), ("Depth", 8)])
    assert found == {("mem_q", 40, 8): 1}, found


@rtl.needs_verilator
@pytest.mark.parametrize(("top", "params", "message"), BAD_SHAPES)
def test_rejects_a_bad_shape(top: str, params: tuple, message: str) -> None:
    """Elaboration stops with a message."""
    result = rtl.lint(top, rtl.extra_sources(top), list(params))
    assert result.returncode != 0 and message in result.stderr, result.stderr


def test_controllers_opcodes_are_the_command_sets() -> None:
    """rtl/controller.sv's opcode table (its Op... localparams) matches
    commands.Op, the one the runtime encodes with."""
    text = (rtl.RTL / "controller.sv").read_text()
    found = {
        m[1]: int(m[2]) for m in re.finditer(r"localparam logic \[7:0\] Op(\w+)\s*= 8'd(\d+)", text)
    }
    names = {op.name.replace("_", "").lower(): op.value for op in commands.Op}
    assert found and all(names[n.lower()] == v for n, v in found.items()), found
    assert max(found.values()) == max(commands.Op)
