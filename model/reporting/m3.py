# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The generated tables of the M3 checkpoint report (reports/M3.md), from
the Vivado logs in reports/data/f2/, the Yosys log of the subnormal cost
(reporting/m3_subnormal.py) and the test runs in reports/data/M3.json.
Filled by scripts/report_m3.py; needs no weights and no RTL tools.

A Vivado log (scripts/f2/time_loops.sh) has, per run, a line
    CABOSSE-RESULT [top=<top>] loop_cycles=<n> [retime=<0|1>] wns_ns=<slack>
then Vivado's timing and utilization reports. The first logs predate
`top` (CVFPU's loop) and `retime` (off), and run lengths named
`CABOSSE-BEGIN loop <n>` instead of `run<k> <top>:<param>:<value>:...`.
"""

import json
import re
from dataclasses import dataclass

import paths
from golden import dot

from reporting import blocks, m1, m3_subnormal

PATH = paths.REPORTS / "M3.md"
DATA = paths.REPORTS / "data" / "M3.json"
F2 = paths.REPORTS / "data" / "f2"
# The Vivado logs in the order they ran, with what each tried.
LOGS = {
    "2026-10-07-acc-loop.txt": "CVFPU's FMA in an accumulate loop",
    "2026-10-07-acc-loop-retime.txt": "the same, with Vivado's retiming",
    "2026-10-07-mac-loop.txt": "our own BF16 multiply-add (D-038)",
    "2026-10-07-fma-path.txt": "our own `fp32_fma`, product in one cycle",
    "2026-10-07-fma-pipelined.txt": "`fp32_fma` with registers inside the product",
    "2026-10-07-mac-loop-pipelined.txt": "the lanes' unit with a register inside the product",
    "2026-10-08-review-recheck.txt": "the same runs on the reviewed RTL",
}
UNITS = {"acc_loop": "CVFPU FMA, loop", "mac_loop": "`bf16_mac`, loop", "fma_path": "`fp32_fma`"}
DEFAULT_TOP = "acc_loop"  # the logs before `top` was printed
CLOCK_NS = 4.0  # 250 MHz (D-027), the harnesses' clock

RESULT = re.compile(r"^CABOSSE-RESULT (.*)$", re.M)
BEGIN_RUN = re.compile(r"^CABOSSE-BEGIN run\d+ \w+:(\w+):(\d+):\d+:\d$", re.M)
KEY_VALUES = re.compile(r"(\w+)=(\S+)")


@dataclass
class Run:
    """One Vivado run of a timing harness."""

    log: int  # 1-based position in LOGS
    top: str
    setting: str  # the harness parameter, e.g. "MulRegs = 2", or ""
    cycles: int  # the loop's length; for fma_path, the latency
    retime: bool
    slack_ns: float
    luts: int
    registers: int
    dsps: int
    path_ns: float  # the worst path's data path delay
    levels: int  # and its logic levels


def _first(pattern: str, text: str) -> str:
    return re.search(pattern, text, re.M)[1]


def parse_log(log: int, text: str) -> tuple[str, list[Run]]:
    """(the commit it ran, its runs)."""
    commit = _first(r"^cabosse ([0-9a-f]{40}) ", text)
    results = list(RESULT.finditer(text))
    runs = []
    for m, nxt in zip(results, [*results[1:], None], strict=True):
        fields = dict(KEY_VALUES.findall(m[1]))
        before = text[: m.start()]
        begins = list(BEGIN_RUN.finditer(before))
        setting = f"{begins[-1][1]} = {begins[-1][2]}" if begins else ""
        after = text[m.end() : nxt.start() if nxt else len(text)]
        runs.append(
            Run(
                log=log,
                top=fields.get("top", DEFAULT_TOP),
                setting=setting,
                cycles=int(fields["loop_cycles"]),
                retime=fields.get("retime", "0") == "1",
                slack_ns=float(fields["wns_ns"]),
                luts=int(_first(r"^\| CLB LUTs\s+\|\s+(\d+)", after)),
                registers=int(_first(r"^\| CLB Registers\s+\|\s+(\d+)", after)),
                dsps=int(_first(r"^\| DSPs\s+\|\s+(\d+)", after)),
                path_ns=float(_first(r"Data Path Delay:\s+([\d.]+)ns", after)),
                levels=int(_first(r"Logic Levels:\s+(\d+)", after)),
            )
        )
    return commit, runs


def logs() -> list[tuple[str, str, list[Run]]]:
    """(file, commit, runs) of every Vivado log, in LOGS's order."""
    out = []
    for i, name in enumerate(LOGS, 1):
        commit, runs = parse_log(i, (F2 / name).read_text())
        out.append((name, commit, runs))
    return out


def logs_table(parsed: list[tuple[str, str, list[Run]]]) -> str:
    rows = []
    for name, commit, runs in parsed:
        results = "; ".join(
            f"{UNITS[r.top]}{f' ({r.setting})' if r.setting else ''}, {r.cycles} cycles"
            f"{', retimed' if r.retime else ''}: {r.slack_ns:+.3f} ns"
            for r in runs
        )
        rows.append(
            [f"{runs[0].log} [{name}](data/f2/{name})", LOGS[name], f"`{commit[:7]}`", results]
        )
    return blocks.table(["log", "what was tried", "commit", "slack per run"], rows)


def runs_table(runs: list[Run]) -> str:
    rows = [
        [
            UNITS[r.top],
            r.setting or "–",
            r.cycles,
            "yes" if r.retime else "no",
            f"{r.slack_ns:+.3f}",
            f"{r.path_ns:.3f} ({r.levels})",
            f"{r.luts:,}",
            f"{r.registers:,}",
            r.dsps,
        ]
        for r in runs
    ]
    header = [
        "unit",
        "setting",
        "cycles",
        "retimed",
        "slack (ns)",
        "worst path (ns, logic levels)",
        "LUTs",
        "registers",
        "DSPs",
    ]
    return blocks.table(header, rows)


def final_run(parsed: list, top: str, cycles: int, retime: bool) -> Run:
    """The run of `top` with that length and retiming in the last log."""
    return next(r for r in parsed[-1][2] if (r.top, r.cycles, r.retime) == (top, cycles, retime))


def tools_text(text: str) -> str:
    """The part and the Vivado version a log ran with."""
    part = _first(r"^part (\S+)$", text)
    version = _first(r"^vivado (v\S+) ", text)
    return f"Part `{part}`, Vivado {version}, out of context (last log)."


def tests_table(tests: dict[str, str]) -> str:
    return blocks.table(["suite", "pytest summary"], [[k, v] for k, v in tests.items()])


def generated() -> dict[str, str]:
    data = json.loads(DATA.read_text())
    tests, inputs = data["tests"], data["inputs"]
    parsed = logs()
    cells, cells_provenance, hits = m3_subnormal.parse(m3_subnormal.LOG.read_text())
    loop = final_run(parsed, "mac_loop", 4, False)
    lanes, vector = m3_subnormal.costs(cells)

    def met(ok: bool) -> str:
        return "yes" if ok else "**no**"

    slow_ok, fast_ok = m1.tests_passed(tests["slow"]), m1.tests_passed(tests["fast"])
    exit_rows = [
        [
            "Each FP operation bit-exact against the golden model on ≥ 10⁸ random "
            "inputs and every special-value class (the BF16 multiply exhaustively)",
            f"fma, add, mul: {inputs['fma_add_mul']:.0e} inputs each; max: "
            f"{inputs['max']:.0e}; the lanes' multiply-add: all {inputs['bf16_pairs']:,} "
            "BF16 pairs; every combination of the special values (slow suite)",
            met(slow_ok),
        ],
        [
            "Every mismatch reported and settled by the owner",
            "one, `max` with a NaN operand: settled by D-037",
            "yes",
        ],
        [
            "The accumulate loop closes in 4 cycles at 250 MHz, or `A` changes",
            f"{loop.slack_ns:+.3f} ns slack in {loop.cycles} cycles of {CLOCK_NS} ns "
            f"(log {loop.log}); `A` = {dot.ACCUMULATORS} unchanged",
            met(loop.slack_ns >= 0),
        ],
        [
            "Lint clean, the units synthesize in Yosys",
            "Verilator `-Wall` and Yosys on every module and harness (fast suite)",
            met(fast_ok),
        ],
        [
            "Cost of subnormal support reported in Yosys cells (D-016)",
            f"`bf16_mac` {lanes[0]:+,} cells ({lanes[1]:+.1f} %), "
            f"`fp32_fma` {vector[0]:+,} ({vector[1]:+.1f} %)",
            "yes",
        ],
    ]
    unmet = [row[0] for row in exit_rows if row[2] != "yes"]
    verdict = "**Not met:** " + "; ".join(unmet) + "." if unmet else "All M3 exit criteria are met."
    provenance = data["provenance"]
    dirty = " with uncommitted changes" if provenance["dirty"] else ""
    return {
        "verdict": verdict,
        "exit-criteria": blocks.table(["criterion", "measured", "met"], exit_rows),
        "tools": tools_text((F2 / list(LOGS)[-1]).read_text()),
        "logs": logs_table(parsed),
        "final-runs": runs_table(parsed[-1][2]),
        "all-runs": runs_table([r for _, _, runs in parsed for r in runs]),
        "cells-provenance": m3_subnormal.provenance_text(cells_provenance),
        "cost": m3_subnormal.cost_table(cells),
        "parts": m3_subnormal.parts_table(cells),
        "baseline": m3_subnormal.baseline_table(cells),
        "hits": m3_subnormal.hits_table(hits),
        "tests-provenance": f"Run on {provenance['date']}, commit "
        f"`{provenance['commit'][:7]}`{dirty}, {provenance['machine']}.",
        "tests": tests_table(tests),
    }


def render() -> str:
    return blocks.fill(PATH.read_text(), generated())
