# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The generated tables of the M4 checkpoint report (reports/M4.md), from
the lane's Vivado logs in reports/data/f2/ (format: reporting/m3.py) and the
test runs in reports/data/M4.json. Filled by scripts/report_m4.py; needs no
weights and no RTL tools.

Besides m3's per-run line, each lane run has Vivado's utilization by
hierarchy, read here for the lane's own resources (the harness adds
registers around it).
"""

import json
import re
from dataclasses import dataclass

import paths
from golden import dot

from reporting import blocks, m1, m3

PATH = paths.REPORTS / "M4.md"
DATA = paths.REPORTS / "data" / "M4.json"
# The lane's Vivado logs in the order they ran, with the final sum each had.
LOGS = {
    "2026-10-08-lane.txt": "3 contexts and a scheduler",
    "2026-10-08-lane-belt.txt": "the conveyor belt",
}
LANES = 128  # D-027
PAIRS_PER_CYCLE = 4  # E, D-027

BEGIN = re.compile(r"^CABOSSE-BEGIN (run\d+) \S+$", re.M)
HIER_ROW = re.compile(
    r"^\|\s+(\S+(?: \S+)?)\s+\|\s+\S+\s+\|\s+(\d+)\s+\|(?:\s+\d+\s+\|){3}\s+(\d+)\s+\|", re.M
)
AVAILABLE = re.compile(
    r"^\| (CLB LUTs|CLB Registers)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+(\d+)\s+\|", re.M
)


@dataclass
class Lane:
    """One Vivado run of the lane: m3's run, and the lane's own resources."""

    run: m3.Run
    log_name: str
    luts: int  # the lane's, from the hierarchy
    registers: int
    unit_luts: int  # the four multiply-add units
    final_luts: int  # the final sum's adder and the lane's own logic


def sections(text: str) -> list[str]:
    """Each run's part of a log, from its CABOSSE-BEGIN to the next."""
    starts = [m.start() for m in BEGIN.finditer(text)]
    return [text[a:b] for a, b in zip(starts, [*starts[1:], len(text)], strict=True)]


def hierarchy(section: str) -> dict[str, tuple[int, int]]:
    """Instance -> (total LUTs, flip-flops) from the utilization by hierarchy."""
    table = section[section.index("Utilization by Hierarchy") :]
    return {m[1].strip(): (int(m[2]), int(m[3])) for m in HIER_ROW.finditer(table)}


def lanes() -> list[Lane]:
    out = []
    for i, name in enumerate(LOGS, 1):
        text = (m3.F2 / name).read_text()
        _, runs = m3.parse_log(i, text)
        for run, section in zip(runs, sections(text), strict=True):
            h = hierarchy(section)
            units = sum(v[0] for k, v in h.items() if k.startswith("gen_unit"))
            out.append(Lane(run, name, *h["u_lane"], units, h["u_tree"][0] + h["(u_lane)"][0]))
    return out


def available(text: str) -> dict[str, int]:
    """The part's LUTs and registers, from a log's utilization report."""
    return {m[1]: int(m[2]) for m in AVAILABLE.finditer(text)}


def runs_table(runs: list[Lane]) -> str:
    rows = [
        [
            r.run.log,
            LOGS[r.log_name],
            r.run.setting,
            "yes" if r.run.retime else "no",
            f"{r.run.slack_ns:+.3f}",
            f"{r.run.path_ns:.3f} ({r.run.levels})",
            f"{r.luts:,}",
            f"{r.unit_luts:,}",
            f"{r.final_luts:,}",
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
        "of which units",
        "of which final sum",
        "flip-flops",
        "DSPs",
    ]
    return blocks.table(header, rows)


def engine_table(runs: list[Lane], part: dict[str, int]) -> str:
    """128 lanes of each design's MulRegs = 1 run, without retiming."""
    rows = []
    for r in runs:
        if r.run.setting == "MulRegs = 1" and not r.run.retime:
            luts, regs = LANES * r.luts, LANES * r.registers
            rows.append(
                [
                    LOGS[r.log_name],
                    f"{luts:,} ({luts / part['CLB LUTs']:.1%})",
                    f"{regs:,} ({regs / part['CLB Registers']:.1%})",
                ]
            )
    return blocks.table([f"{LANES} lanes", "LUTs (of the part)", "flip-flops (of the part)"], rows)


def logs_table() -> str:
    rows = []
    for i, name in enumerate(LOGS, 1):
        commit = re.search(r"^cabosse ([0-9a-f]{7})", (m3.F2 / name).read_text(), re.M)[1]
        rows.append([i, f"[{name}](data/f2/{name})", LOGS[name], f"`{commit}`"])
    return blocks.table(["log", "file", "final sum", "commit"], rows)


def generated() -> dict[str, str]:
    data = json.loads(DATA.read_text())
    tests, inputs = data["tests"], data["inputs"]
    runs = lanes()
    last_text = (m3.F2 / list(LOGS)[-1]).read_text()
    final = [r for r in runs if r.log_name == list(LOGS)[-1]]
    worst = min(r.run.slack_ns for r in final)

    def met(ok: bool) -> str:
        return "yes" if ok else "**no**"

    slow_ok, fast_ok = m1.tests_passed(tests["slow"]), m1.tests_passed(tests["fast"])
    exit_rows = [
        [
            "Bit-exact on random lengths, SmolLM2 row lengths and adversarial inputs",
            f"{inputs['bulk_pairs']:,} pairs of random lengths, SmolLM2's lengths "
            f"and adversarial families; one SmolLM2 decode step's rows, "
            f"{inputs['smollm2_rows']:,} rows and {inputs['smollm2_pairs']:,} pairs of "
            "real weights and activations (slow suite); the handshake's corner cases "
            "(cocotb, fast suite)",
            met(slow_ok and fast_ok),
        ],
        [
            f"{PAIRS_PER_CYCLE} pairs per cycle sustained under randomized backpressure "
            "(D-027; was 1 MAC/cycle)",
            f"rows of {inputs['full_rate_len']} elements or more: no stall with input "
            "gaps, cycles = beats + gaps + latency; random output ready loses nothing "
            "(fast suite)",
            met(fast_ok),
        ],
        [
            "The lane meets 250 MHz on F2's part (Q6, not an exit criterion)",
            f"every run of the final RTL, worst {worst:+.3f} ns (log {len(LOGS)})",
            met(worst >= 0),
        ],
    ]
    unmet = [row[0] for row in exit_rows if row[2] != "yes"]
    verdict = "**Not met:** " + "; ".join(unmet) + "." if unmet else "All M4 exit criteria are met."
    provenance = data["provenance"]
    dirty = " with uncommitted changes" if provenance["dirty"] else ""
    return {
        "verdict": verdict,
        "exit-criteria": blocks.table(["criterion", "measured", "met"], exit_rows),
        "shape": f"`E` = {PAIRS_PER_CYCLE} pairs per cycle, `A` = {dot.ACCUMULATORS} "
        f"partial sums, {LANES} lanes in M5 (D-027).",
        "tools": m3.tools_text(last_text),
        "logs": logs_table(),
        "runs": runs_table(runs),
        "engine": engine_table(runs, available(last_text)),
        "tests-provenance": f"Run on {provenance['date']}, commit "
        f"`{provenance['commit'][:7]}`{dirty}, {provenance['machine']}.",
        "tests": m3.tests_table(tests),
    }


def render() -> str:
    return blocks.fill(PATH.read_text(), generated())
