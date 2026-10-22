# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The generated tables of the M5 checkpoint report (reports/M5.md), from
the engine's measurements in reports/data/M5-engine.json
(scripts/measure_m5_engine.py), the test runs in reports/data/M5.json and
the limits the RTL tests hold (verif/tests/test_rtl_checks.py). Filled by
scripts/report_m5.py; needs no weights and no RTL tools."""

import json
import re

import paths

from reporting import blocks, m3

PATH = paths.REPORTS / "M5.md"
DATA = paths.REPORTS / "data" / "M5.json"
ENGINE = paths.REPORTS / "data" / "M5-engine.json"
RTL_CHECKS = paths.REPO / "verif" / "tests" / "test_rtl_checks.py"
ENGINE_RTL = paths.REPO / "rtl" / "engine.sv"
CRITERION = 0.9  # PLAN.md, M5: 90% of the bandwidth, and of the perf model on attention
LONG = ("1023", "8191")  # the long positions of the attention criterion


def span(values: list[float], fmt: str = ".3f") -> str:
    lo, hi = min(values), max(values)
    return f"{lo:{fmt}}" if f"{lo:{fmt}}" == f"{hi:{fmt}}" else f"{lo:{fmt}}-{hi:{fmt}}"


def constant(path, name: str) -> int:
    """An integer constant or a SystemVerilog parameter's default, read from
    its file."""
    m = re.search(
        rf"^\s*(?:{name} = |parameter\s+int unsigned {name}\s*=\s*)(\d+)", path.read_text(), re.M
    )
    assert m, (path, name)
    return int(m[1])


def off(c: dict) -> float:
    """A command's measured cycles against the perf model's, as a fraction."""
    return (c["cycles"] - c["model"]) / c["model"]


def real_table(real: dict) -> str:
    """Per position: how far the measured cycles are from the perf model's."""
    rows = []
    for position, cmds in real["positions"].items():
        others = [off(c) for c in cmds[1:] if c["op"] == "matvec"]
        worst = [
            max((off(c) for c in cmds if c["op"] == op), key=abs) for op in ("scores", "values")
        ]
        rows.append(
            [
                position,
                len(cmds),
                f"{off(cmds[0]):+.1%}",
                f"{min(others):+.1%} to {max(others):+.1%}",
                *(f"{w:+.1%}" for w in worst),
            ]
        )
    header = ["position", "commands", "first q", "other MATVECs", "SCORES", "VALUES"]
    return blocks.table(header, rows)


def attention_use(real: dict) -> float:
    """The least share of the perf model's speed SCORES and VALUES reach at
    the long positions: model cycles over measured cycles."""
    att = [c for p in LONG for c in real["positions"][p] if c["op"] != "matvec"]
    return min(c["model"] / c["cycles"] for c in att)


def bandwidth_table(bandwidth: dict) -> str:
    rows = [
        [
            f"{m['shape'][0]} × {m['shape'][1]}",
            f"{m['rate'] / 1000:.0%}",
            f"{m['share']:.3f}",
            f"{m['engine_use']:.3f}",
            f"{m['share'] / m['engine_use']:.3f}",
        ]
        for m in bandwidth["matrices"]
    ]
    header = ["rows × cols", "memory rate", "share", "engine use", "share ÷ engine use"]
    return blocks.table(header, rows)


def depth_table(depth: dict) -> str:
    """Rate by FIFO depth, as a share of the deepest FIFO's (deep enough that
    only the pauses limit), per pause setting (a range over the memory
    seeds)."""
    rates = depth["rates"]
    pauses = list(rates)
    *depths, reference = sorted(rates[pauses[0]], key=int)
    rows = [
        [d]
        + [
            span([r / ref for r, ref in zip(rates[p][d], rates[p][reference], strict=True)])
            for p in pauses
        ]
        for d in depths
    ]
    return blocks.table(["depth (beats)", *(f"pauses: {p}" for p in pauses)], rows)


def cycles_per_s(speed: dict) -> list[float]:
    return [c / t for c, t in zip(speed["run_cycles"], speed["run_s"], strict=True)]


def speed_measured(speed: dict) -> str:
    return (
        f"build {span(speed['build_s'], '.0f')} s; "
        f"{span(cycles_per_s(speed), ',.0f')} simulated cycles/s"
    )


def speed_text(speed: dict) -> str:
    per_s = min(cycles_per_s(speed))
    tokens = [
        f"position {p}: {sum(terms.values()):,.0f} cycles per token (perf model, serial), "
        f"about {sum(terms.values()) / per_s:.0f} s"
        for p, terms in speed["token_cycles"].items()
    ]
    return (
        f"{speed_measured(speed).capitalize()} "
        f"({speed['ports']} ports, three runs). In M7 at that speed (estimate; the full "
        f"design will simulate slower): {'; '.join(tokens)}."
    )


def generated() -> dict[str, str]:
    """Every block of the report, by name."""
    data, engine = json.loads(DATA.read_text()), json.loads(ENGINE.read_text())
    tests, real, bandwidth = data["tests"], engine["real"], engine["bandwidth"]
    fast_ok, slow_ok = blocks.tests_passed(tests["fast"]), blocks.tests_passed(tests["slow"])
    full = [m["share"] for m in bandwidth["matrices"]]
    short = [
        f"{m['shape'][0]} × {m['shape'][1]} at {m['rate'] / 1000:.0%}: {m['share']:.3f}"
        for m in bandwidth["matrices"]
        if m["share"] < CRITERION
    ]
    use = attention_use(real)
    worst_p, worst_c = max(
        ((p, c) for p, cmds in real["positions"].items() for c in cmds),
        key=lambda pc: abs(off(pc[1])),
    )
    worst = abs(off(worst_c))
    speed = engine["speed"]
    exit_rows = [
        [
            "Bit-exact on one full SmolLM2 layer (MATVEC, SCORES, VALUES) and the "
            "classifier with real data",
            f"the first and last layers' engine commands and the classifier at positions "
            f"{', '.join(real['positions'])}, real text from the long cache (slow suite)",
            blocks.met(slow_ok),
        ],
        [
            f"Bytes/cycle ≥ {CRITERION:.0%} of min(BW, 2·E·L) on large matrices",
            f"{span(full)} of it; below: {'; '.join(short) or 'none'} (a partial last "
            "pass leaves ports idle: share ÷ engine use is at least "
            f"{min(m['share'] / m['engine_use'] for m in bandwidth['matrices']):.3f})",
            blocks.met(not short),
        ],
        [
            f"SCORES and VALUES at long positions: engine use ≥ {CRITERION:.0%} of the "
            "perf model's",
            f"{use:.1%} of the perf model's speed or more, at positions "
            f"{' and '.join(LONG)} (slow suite)",
            blocks.met(use >= CRITERION and slow_ok),
        ],
        [
            "Verilator speed measured, to plan M7",
            speed_measured(speed),
            "yes",
        ],
        [
            "Row edge cases: fewer rows than L, rows not a multiple of L (192 rows at 128 "
            "lanes), one row; idle lanes write nothing",
            "every row count around a group and a pass at 4, 8, 20 and 32 ports; SmolLM2's "
            "192-row k and v at 128 lanes; the driver checks every write (a row below the "
            "command's count, once) (both suites)",
            blocks.met(fast_ok and slow_ok),
        ],
    ]
    answer_row = [
        "Every engine command of the layer test within 10% of the perf model (M5 "
        "answers; not an exit criterion)",
        f"at most {worst:.1%} off ({worst_c['name']} at {worst_p})",
        blocks.met(worst <= 0.10),
    ]
    shape = (
        f"{real['ports']} ports, {4 * real['ports']} lanes, a "
        f"{constant(ENGINE_RTL, 'Depth')}-beat FIFO per port, "
        f"{constant(ENGINE_RTL, 'Writes')} results per cycle on the write port; the "
        f"memory model at F2's read latency, {real['memory']['latency']} cycles, with "
        f"{real['memory']['outstanding']} bursts in flight per port."
    )
    depth_limits = (
        f"`dot_lane` at most {constant(RTL_CHECKS, 'MAX_LANE_DEPTH')} cells; the engine "
        f"with the FMA stubbed at most {constant(RTL_CHECKS, 'MAX_ENGINE_CONTROL_DEPTH')} "
        "(8 ports)."
    )
    return {
        "verdict": blocks.verdict(exit_rows, "M5"),
        "exit-criteria": blocks.table(["criterion", "measured", "met"], [*exit_rows, answer_row]),
        "shape": shape,
        "real": real_table(real),
        "real-provenance": blocks.provenance_text(real["provenance"]),
        "bandwidth": bandwidth_table(bandwidth),
        "bandwidth-provenance": blocks.provenance_text(bandwidth["provenance"]),
        "depth": depth_table(engine["depth"]),
        "depth-provenance": blocks.provenance_text(engine["depth"]["provenance"]),
        "speed": speed_text(speed),
        "speed-provenance": blocks.provenance_text(speed["provenance"]),
        "logic-depth": depth_limits,
        "tests-provenance": blocks.provenance_text(data["provenance"]),
        "tests": m3.tests_table(tests),
    }


def render() -> str:
    return blocks.fill(PATH.read_text(), generated())
