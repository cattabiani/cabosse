# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The generated tables of the M6 checkpoint report (reports/M6.md), from
the vector unit's measurements and test runs in reports/data/M6.json
(scripts/measure_m6.py, scripts/report_m6.py), the perf model, and the
engine's placed Vivado builds in reports/data/f2/. Filled by
scripts/report_m6.py; needs no weights and no RTL tools.

A build log (scripts/f2/time_loops.sh) has, per run, a line
    CABOSSE-RESULT top=<top> loop_cycles=<n> retime=<0|1> [flow=<f>] [place=<p>] wns_ns=<slack>
which may appear twice (the console's echo, then the summary); Vivado's
timing summary follows the last one."""

import json
import re

import paths
import perf

from reporting import blocks, m3

PATH = paths.REPORTS / "M6.md"
DATA = paths.REPORTS / "data" / "M6.json"
CRITERION = 0.10  # PLAN.md, M6: every vector command within 10% of the perf model
PLATFORM, DESIGN = "f2", "v0"
MODELS = ("SmolLM2-135M-Instruct", "Qwen2.5-0.5B-Instruct")
SHARE_POSITION = 1023  # PLAN.md, M6 answers: the share at 0, 1023 and the full window
SLOTS = 16  # the vector unit's width (D-030)
# One fp32_fma in Vivado on F2's part, MulRegs = 2 (PLAN.md, M3: the fifth
# timing log): LUTs, registers, DSPs.
VIVADO_FMA = {"LUTs": 1262, "registers": 535, "DSPs": 2}
# The engine's placed builds (D-059, D-060), in order, with the RTL each ran.
BUILDS = {
    "2026-10-30-engine-placed.txt": "0: the engine as M5 left it",
    "2026-10-31-engine-placed.txt": "1: pipelined step control (D-051)",
    "2026-10-31-engine-placed-build2.txt": "2: KV cache in the lanes' order (D-062)",
    "2026-11-01-engine-placed-build3.txt": "3: round 3's registers",
    "2026-11-01-engine-placed-build4.txt": "4: round 4's registers, the shell's flow",
}

RESULT = re.compile(r"^CABOSSE-RESULT (.*)$", re.M)
KEY_VALUES = re.compile(r"(\w+)=(\S+)")
SUMMARY = re.compile(r"^\s+WNS\(ns\)\s+TNS\(ns\).*\n.*\n\s*(\S+)\s+\S+\s+(\d+)\s+(\d+)", re.M)


def build_runs(text: str) -> list[dict]:
    """Each run of a build log: its fields, and its failing and total
    endpoints from the timing summary after the run's last result line
    (None if the log lost them)."""
    last = {}
    for m in RESULT.finditer(text):
        last[m[1]] = m  # the last line of each run
    runs = []
    for line, m in last.items():
        fields = dict(KEY_VALUES.findall(line))
        summary = SUMMARY.search(text, m.end())
        assert summary is None or float(summary[1]) == float(fields["wns_ns"]), line
        fields["endpoints"] = (int(summary[2]), int(summary[3])) if summary else None
        runs.append(fields)
    return runs


def builds_table() -> str:
    rows = []
    for name, what in BUILDS.items():
        runs = build_runs((m3.F2 / name).read_text())
        link = f"[{name[:10]}](data/f2/{name})"
        if not runs:
            rows.append([what, link, "-", "not kept", "not kept"])
        for r in runs:
            flow = r.get("flow", "default")
            setting = f"{flow}, {r['place']}" if "place" in r else flow
            failing = f"{r['endpoints'][0]:,} of {r['endpoints'][1]:,}" if r["endpoints"] else "-"
            rows.append([what, link, setting, f"{float(r['wns_ns']):+.3f}", failing])
    header = ["build", "log", "flow", "worst slack (ns)", "failing endpoints"]
    return blocks.table(header, rows)


def timing_text() -> str:
    runs = build_runs((m3.F2 / list(BUILDS)[-1]).read_text())
    slacks = " and ".join(f"{float(r['wns_ns']):+.3f}" for r in runs)
    failing = sum(r["endpoints"][0] for r in runs)
    met = all(float(r["wns_ns"]) >= 0 for r in runs)
    return (
        f"Build 4 {'meets' if met else 'misses'} 250 MHz: worst slack {slacks} ns in its "
        f"{len(runs)} placements, {failing:,} failing endpoints."
    )


def positions(latency: dict) -> list[str]:
    return [k for k in latency if k != "provenance"]


def worst_command(latency: dict) -> tuple[str, dict]:
    """The position and command furthest from the perf model's cycles."""
    return max(
        ((p, latency[p]["worst"]) for p in positions(latency)), key=lambda pw: abs(pw[1]["off"])
    )


def cycles_table(latency: dict) -> str:
    """Per position: the unit's cycles, the perf model's, the worst command."""
    rows = []
    for p in positions(latency):
        d, w = latency[p], latency[p]["worst"]
        worst = f"`{w['kind']}`: {w['cycles']} against {w['model_cycles']:.0f} ({w['off']:+.1%})"
        rows.append([p, f"{d['cycles']:,}", f"{d['model_cycles']:,.0f}", worst])
    return blocks.table(["position", "unit cycles", "perf model", "worst command"], rows)


def kinds_table(latency: dict) -> str:
    """The unit's cycles by command kind, per position, largest first at the
    last position."""
    ps = positions(latency)
    kinds = sorted(latency[ps[-1]]["cycles_by_kind"].items(), key=lambda kv: -kv[1])
    rows = [
        [f"`{k.upper()}`", *(f"{latency[p]['cycles_by_kind'].get(k, 0):,}" for p in ps)]
        for k, _ in kinds
    ]
    return blocks.table(["command", *(f"position {p}" for p in ps)], rows)


def latencies_table(latency: dict) -> str:
    """Each latency of the perf model's vector term: the values found at each
    position (with how many commands showed each), and the value design v0
    takes."""
    design = perf.load_design(DESIGN).values()
    ps = positions(latency)
    names = sorted({n for p in ps for n in latency[p]["latencies"]})
    rows = [
        [
            f"`{n}`",
            *(
                "; ".join(f"{v} ({c}x)" for v, c in latency[p]["latencies"].get(n, {}).items())
                or "-"
                for p in ps
            ),
            f"{design[n]:g}",
        ]
        for n in names
    ]
    return blocks.table(["latency", *(f"position {p}" for p in ps), "design v0"], rows)


def share_why(latency: dict) -> str:
    """Where SmolLM2's vector time goes at position 0: the reductions' rows
    queueing on the tree, and the start-ups of the commands that wait."""
    at0 = latency["0"]
    lat = {n: next(iter(c.items())) for n, c in at0["latencies"].items()}
    (row, _), (startup, n_waiting) = (
        lat["reduction_row_cycles"],
        lat["vector_startup_cycles"],
    )
    by_kind = at0["cycles_by_kind"]
    heads = perf.load_config(MODELS[0]).num_attention_heads
    return (
        f"At position 0 most of SmolLM2's vector time is latency, not work: `SUM` "
        f"and `MAX` over {heads} rows of one element each wait for the tree ({row} "
        f"cycles a row, Q-32), {by_kind['sum'] + by_kind['max']:,} of the unit's "
        f"{at0['cycles']:,} cycles; and {n_waiting} commands wait for the one before "
        f"({startup} cycles each, Q-30)."
    )


def share(config, position: int) -> tuple[float, float]:
    """The vector unit's seconds of a token and the token's, in order
    (Prediction.tokens_per_s_in_order)."""
    t = perf.predict(
        config, perf.load_platform(PLATFORM).values(), perf.load_design(DESIGN).values(), position
    ).terms
    return t["vector"], max(t["memory"], t["engine"]) + t["vector"] + t["commands"]


def share_table() -> str:
    rows = []
    for name in MODELS:
        config = perf.load_config(name)
        for p in (0, SHARE_POSITION, config.max_position_embeddings - 1):
            vec, token = share(config, p)
            rows.append([name, p, f"{vec * 1e3:.3f}", f"{token * 1e3:.3f}", f"{vec / token:.1%}"])
    header = ["model", "position", "vector unit (ms)", "token (ms)", "share"]
    return blocks.table(header, rows)


def shares_at_0() -> str:
    return " and ".join(
        f"{name.split('-')[0]} {s[0] / s[1]:.1%}"
        for name in MODELS
        for s in [share(perf.load_config(name), 0)]
    )


def area_table(area: dict) -> str:
    """Per kind: Yosys's cells for one FMA and one slot's chain, and the
    Vivado estimate (M3's FMA times the ratio) for a slot and 16 slots."""
    fma, chain = area["fp32_fma"], area["fp32_chain"]
    rows = []
    for kind in fma:
        row = [kind, f"{fma[kind]:,}", f"{chain[kind]:,}", "-", "-"]
        if kind in VIVADO_FMA:
            est = VIVADO_FMA[kind] * chain[kind] / fma[kind]
            row[3:] = [f"{est:,.0f}", f"{est * SLOTS:,.0f}"]
        rows.append(row)
    head = ["", "fp32_fma (Yosys)", "fp32_chain (Yosys)", "a slot (Vivado, estimate)", "16 slots"]
    return blocks.table(head, rows)


def generated() -> dict[str, str]:
    """Every block of the report, by name."""
    data = json.loads(DATA.read_text())
    tests, latency, area = data["tests"], data["latency"], data["area"]
    fast_ok, slow_ok = blocks.tests_passed(tests["fast"]), blocks.tests_passed(tests["slow"])
    worst_p, worst = worst_command(latency)
    real = ", ".join(positions(latency))
    exit_rows = [
        [
            "Each operation bit-exact on random and real activations",
            "random: one slot's chain per operation, whole commands, random sequences, "
            "LOAD and the stores, the tiny config's decode step (fast suite); real: "
            f"SmolLM2's whole decode step at positions {real}, every buffer, every "
            "engine input and the memory (slow suite)",
            blocks.met(fast_ok and slow_ok),
        ],
        [
            f"Every vector-unit command of the real-data test within {CRITERION:.0%} of "
            "the perf model's cycles",
            (
                f"at most {abs(worst['off']):.1%} off (`{worst['kind']}` at position {worst_p})"
                if worst["off"]
                else "every command takes the perf model's cycles exactly, at every position"
            ),
            blocks.met(abs(worst["off"]) <= CRITERION and slow_ok),
        ],
        [
            "The perf model's share of token time at position 0 recorded (not a criterion, D-058)",
            f"{shares_at_0()} (in order, estimate; below)",
            "yes",
        ],
    ]
    return {
        "verdict": blocks.verdict(exit_rows, "M6"),
        "exit-criteria": blocks.table(["criterion", "measured", "met"], exit_rows),
        "share": share_table(),
        "cycles": cycles_table(latency),
        "kinds": kinds_table(latency),
        "share-why": share_why(latency),
        "latencies": latencies_table(latency),
        "latency-provenance": blocks.provenance_text(latency["provenance"]),
        "area": area_table(area),
        "area-provenance": blocks.provenance_text(area["provenance"]),
        "builds": builds_table(),
        "timing": timing_text(),
        "tests-provenance": blocks.provenance_text(tests["provenance"]),
        "tests": m3.tests_table({k: v for k, v in tests.items() if k != "provenance"}),
    }


def render() -> str:
    return blocks.fill(PATH.read_text(), generated())
