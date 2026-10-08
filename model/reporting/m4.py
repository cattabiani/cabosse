# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The generated tables of the M4 checkpoint report (reports/M4.md), from
the lane's Vivado logs in reports/data/f2/ (format: reporting/m3.py), the
design file (model/designs/v0.json) and the test runs in
reports/data/M4.json. Filled by scripts/report_m4.py; needs no weights and
no RTL tools.

Besides m3's per-run line, each lane run has Vivado's utilization by
hierarchy, read here for the lane's own resources (the harness adds
registers around it).
"""

import json
import re
from dataclasses import dataclass

import paths
import perf
from golden import dot

from reporting import blocks, m1, m3

PATH = paths.REPORTS / "M4.md"
DATA = paths.REPORTS / "data" / "M4.json"
# The lane's Vivado logs in the order they ran, with the final sum each had.
LOGS = {
    "2026-10-08-lane.txt": "3 contexts and a scheduler",
    "2026-10-08-lane-belt.txt": "the conveyor belt",
}
ENGINE_SETTING = ("MulRegs = 1", False)  # the run the engine estimate uses: setting, retimed

BEGIN = re.compile(r"^CABOSSE-BEGIN run\d+ \S+$", re.M)
HIER_ROW = re.compile(
    r"^\|\s+(\S+(?: \S+)?)\s+\|\s+\S+\s+\|\s+(\d+)\s+\|(?:\s+\d+\s+\|){3}\s+(\d+)\s+\|", re.M
)
AVAILABLE = re.compile(
    r"^\| (CLB LUTs|CLB Registers)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+(\d+)\s+\|", re.M
)


@dataclass
class Lane:
    """One Vivado run of the lane: m3's run, and the lane's own resources
    from the utilization by hierarchy."""

    run: m3.Run
    luts: int
    registers: int
    unit_luts: int  # the four multiply-add units: products and adders
    tree_luts: int  # the final sum's adder
    own_luts: int  # the rest: accumulator files, buffers, queues, control


def sections(text: str) -> list[str]:
    """Each run's part of a log, from its CABOSSE-BEGIN to the next."""
    starts = [m.start() for m in BEGIN.finditer(text)]
    return [text[a:b] for a, b in zip(starts, [*starts[1:], len(text)], strict=True)]


def hierarchy(section: str) -> dict[str, tuple[int, int]]:
    """Instance -> (total LUTs, flip-flops) from the utilization by hierarchy."""
    table = section[section.index("Utilization by Hierarchy") :]
    return {m[1].strip(): (int(m[2]), int(m[3])) for m in HIER_ROW.finditer(table)}


def lanes(texts: dict[str, str], parsed: list[tuple[str, str, list[m3.Run]]]) -> list[Lane]:
    out = []
    for name, _, runs in parsed:
        for run, section in zip(runs, sections(texts[name]), strict=True):
            h = hierarchy(section)
            units = sum(v[0] for k, v in h.items() if k.startswith("gen_unit"))
            out.append(Lane(run, *h["u_lane"], units, h["u_tree"][0], h["(u_lane)"][0]))
    return out


def available(text: str) -> dict[str, int]:
    """The part's LUTs and registers, from a log's utilization report."""
    return {m[1]: int(m[2]) for m in AVAILABLE.finditer(text)}


def runs_table(runs: list[Lane]) -> str:
    names = list(LOGS)
    rows = [
        [
            r.run.log,
            LOGS[names[r.run.log - 1]],
            r.run.setting,
            "yes" if r.run.retime else "no",
            f"{r.run.slack_ns:+.3f}",
            f"{r.run.path_ns:.3f} ({r.run.levels})",
            f"{r.luts:,}",
            f"{r.unit_luts:,}",
            f"{r.tree_luts:,}",
            f"{r.own_luts:,}",
            f"{r.registers:,}",
            r.run.dsps,
        ]
        for r in runs
    ]
    header = [
        "log",
        "final sum",
        "setting",
        "retimed",
        "slack (ns)",
        "worst path (ns, logic levels)",
        "LUTs",
        "units",
        "final sum's adder",
        "the lane's own",
        "flip-flops",
        "DSPs",
    ]
    return blocks.table(header, rows)


def engine_table(runs: list[Lane], n_lanes: int, part: dict[str, int]) -> str:
    """n_lanes lanes of each log's ENGINE_SETTING run."""
    names = list(LOGS)
    rows = []
    for r in runs:
        if (r.run.setting, r.run.retime) == ENGINE_SETTING:
            luts, regs = n_lanes * r.luts, n_lanes * r.registers
            rows.append(
                [
                    LOGS[names[r.run.log - 1]],
                    f"{luts:,} ({luts / part['CLB LUTs']:.1%})",
                    f"{regs:,} ({regs / part['CLB Registers']:.1%})",
                ]
            )
    assert len(rows) == len(LOGS), f"a log has no {ENGINE_SETTING} run"
    header = [f"{n_lanes} lanes", "LUTs (of the part)", "flip-flops (of the part)"]
    return blocks.table(header, rows)


def generated() -> dict[str, str]:
    data = json.loads(DATA.read_text())
    tests, inputs = data["tests"], data["inputs"]
    design = perf.load_design("v0").values()
    n_lanes, per_cycle = int(design["lanes"]), int(design["macs_per_lane"])
    texts = {name: (m3.F2 / name).read_text() for name in LOGS}
    parsed = m3.logs(LOGS)
    runs = lanes(texts, parsed)
    worst = min(r.run.slack_ns for r in runs if r.run.log == len(LOGS))

    slow_ok, fast_ok = m1.tests_passed(tests["slow"]), m1.tests_passed(tests["fast"])
    exit_rows = [
        [
            "Bit-exact on random lengths, SmolLM2 row lengths and adversarial inputs",
            f"{inputs['bulk_pairs']:,} pairs in {inputs['bulk_chunks']} chunks of random "
            "lengths, SmolLM2's lengths and adversarial families; one SmolLM2 decode "
            f"step's rows, {inputs['smollm2_rows']:,} rows and {inputs['smollm2_pairs']:,} "
            "pairs of real weights and activations (slow suite); the handshake's corner "
            "cases (cocotb, fast suite)",
            m1.met(slow_ok and fast_ok),
        ],
        [
            f"{per_cycle} pairs per cycle sustained under randomized backpressure "
            "(D-027; was 1 MAC/cycle)",
            f"rows of {inputs['full_rate_len']} elements or more: no stall with input "
            "gaps, cycles = beats + gaps + latency; random output ready loses nothing "
            "(fast suite)",
            m1.met(fast_ok),
        ],
    ]
    timing_row = [
        "The lane meets 250 MHz on F2's part (Q6; not an exit criterion)",
        f"every run of the final RTL, worst {worst:+.3f} ns (log {len(LOGS)})",
        m1.met(worst >= 0),
    ]
    return {
        "verdict": m1.verdict(exit_rows, "M4"),
        "exit-criteria": blocks.table(["criterion", "measured", "met"], [*exit_rows, timing_row]),
        "shape": f"`E` = {per_cycle} pairs per cycle, `A` = {dot.ACCUMULATORS} "
        f"partial sums, {n_lanes} lanes in M5 (D-027).",
        "tools": m3.tools_text(texts[list(LOGS)[-1]]),
        "logs": m3.logs_table(parsed, LOGS, "final sum"),
        "runs": runs_table(runs),
        "engine": engine_table(runs, n_lanes, available(texts[list(LOGS)[-1]])),
        "tests-provenance": m1.provenance_text(data["provenance"]),
        "tests": m3.tests_table(tests),
    }


def render() -> str:
    return blocks.fill(PATH.read_text(), generated())
