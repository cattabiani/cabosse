# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The generated tables of the M6 checkpoint report (reports/M6.md), from
the vector unit's measurements and test runs in reports/data/M6.json
(scripts/measure_m6.py, scripts/report_m6.py), the perf model, M3's Vivado
logs (one FMA) and the engine's placed Vivado builds in reports/data/f2/.
Filled by scripts/report_m6.py; needs no weights and no RTL tools.

A build log (scripts/f2/time_loops.sh) has m3's CABOSSE-RESULT line per run,
here with `flow` and `place`, which may appear twice (the console's echo,
then the summary); Vivado's timing summary follows the last one."""

import functools
import json
import re

import paths
import perf

from reporting import blocks, m3, perf_doc

PATH = paths.REPORTS / "M6.md"
DATA = paths.REPORTS / "data" / "M6.json"
CRITERION = 0.10  # PLAN.md, M6: every vector command within 10% of the perf model
PLATFORM, DESIGN = "f2", "v0"
MODELS = ("SmolLM2-135M-Instruct", "Qwen2.5-0.5B-Instruct")
VIVADO_KINDS = {"LUTs": "luts", "registers": "registers", "DSPs": "dsps"}  # m3.Run's fields
# The engine's placed builds (D-059, D-060), in order, with the RTL each ran.
BUILDS = {
    "2026-10-30-engine-placed.txt": "0: the engine as M5 left it",
    "2026-10-31-engine-placed.txt": "1: pipelined step control (D-051)",
    "2026-10-31-engine-placed-build2.txt": "2: KV cache in the lanes' order (D-062)",
    "2026-11-01-engine-placed-build3.txt": "3: round 3's registers",
    "2026-11-01-engine-placed-build4.txt": "4: round 4's registers, the shell's flow",
}

SUMMARY = re.compile(r"^\s+WNS\(ns\)\s+TNS\(ns\).*\n.*\n\s*(\S+)\s+\S+\s+(\d+)\s+(\d+)", re.M)


@functools.cache
def design() -> dict[str, float]:
    return perf.load_design(DESIGN).values()


@functools.cache
def platform() -> dict[str, float]:
    return perf.load_platform(PLATFORM).values()


def without_provenance(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "provenance"}


def build_runs(text: str) -> list[dict]:
    """Each run of a build log: its fields, and its failing and total
    endpoints from the timing summary after the run's last result line
    (None if the log lost them)."""
    last = {m[1]: m for m in m3.RESULT.finditer(text)}  # the last line of each run
    runs = []
    for line, m in last.items():
        fields = dict(m3.KEY_VALUES.findall(line))
        summary = SUMMARY.search(text, m.end())
        assert summary is None or float(summary[1]) == float(fields["wns_ns"]), line
        fields["endpoints"] = (int(summary[2]), int(summary[3])) if summary else None
        runs.append(fields)
    return runs


def builds_table(builds: dict[str, list[dict]]) -> str:
    rows = []
    for name, runs in builds.items():
        what, link = BUILDS[name], f"[{name[:10]}](data/f2/{name})"
        if not runs:
            rows.append([what, link, "-", "not kept", "not kept"])
        for r in runs:
            flow = r.get("flow", "default")
            setting = f"{flow}, {r['place']}" if "place" in r else flow
            failing = f"{r['endpoints'][0]:,} of {r['endpoints'][1]:,}" if r["endpoints"] else "-"
            rows.append([what, link, setting, f"{float(r['wns_ns']):+.3f}", failing])
    header = ["build", "log", "flow", "worst slack (ns)", "failing endpoints"]
    return blocks.table(header, rows)


def timing_text(runs: list[dict]) -> str:
    """The last build's verdict at 250 MHz."""
    slacks = " and ".join(f"{float(r['wns_ns']):+.3f}" for r in runs)
    failing = sum(r["endpoints"][0] for r in runs)
    met = all(float(r["wns_ns"]) >= 0 for r in runs)
    return (
        f"Build 4 {'meets' if met else 'misses'} 250 MHz: worst slack {slacks} ns in its "
        f"{len(runs)} placements, {failing:,} failing endpoints."
    )


def worst_command(latency: dict) -> tuple[str, dict]:
    """The position and command furthest from the perf model's cycles."""
    return max(latency.items(), key=lambda pw: abs(pw[1]["worst"]["off"]))


def cycles_table(latency: dict) -> str:
    """Per position: the unit's cycles, the perf model's, the worst command."""
    rows = []
    for p, d in latency.items():
        w = d["worst"]
        worst = (
            f"`{w['kind']}`: {w['cycles']} against {w['model_cycles']:.0f} ({w['off']:+.1%})"
            if w["off"]
            else "all exact"
        )
        rows.append([p, f"{d['cycles']:,}", f"{d['model_cycles']:,.0f}", worst])
    return blocks.table(["position", "unit cycles", "perf model", "worst command"], rows)


def kinds_table(latency: dict) -> str:
    """The unit's cycles by command kind, per position, largest first at the
    last position."""
    last = list(latency.values())[-1]["cycles_by_kind"]
    rows = [
        [f"`{k.upper()}`", *(f"{d['cycles_by_kind'].get(k, 0):,}" for d in latency.values())]
        for k in sorted(last, key=last.get, reverse=True)
    ]
    return blocks.table(["command", *(f"position {p}" for p in latency)], rows)


def latencies_table(latency: dict) -> str:
    """Each latency of the perf model's vector term: the values found at each
    position (with how many commands showed each), and the value design v0
    takes."""
    names = sorted({n for d in latency.values() for n in d["latencies"]})
    rows = [
        [
            f"`{n}`",
            *(
                "; ".join(f"{v} ({c}x)" for v, c in d["latencies"].get(n, {}).items()) or "-"
                for d in latency.values()
            ),
            f"{design()[n]:g}",
        ]
        for n in names
    ]
    return blocks.table(["latency", *(f"position {p}" for p in latency), "design v0"], rows)


def share_why(latency: dict) -> str:
    """Where SmolLM2's vector time goes at position 0: the reductions' rows
    queueing on the tree, and the start-ups of the commands that wait."""
    at0 = latency["0"]
    row = next(iter(at0["latencies"]["reduction_row_cycles"]))
    startup, n_waiting = next(iter(at0["latencies"]["vector_startup_cycles"].items()))
    by_kind = at0["cycles_by_kind"]
    heads = perf.load_config(MODELS[0]).num_attention_heads
    return (
        f"At position 0 most of SmolLM2's vector time is latency, not work: `SUM` "
        f"and `MAX` over {heads} rows of one element each wait for the tree ({row} "
        f"cycles a row, Q-32), {by_kind['sum'] + by_kind['max']:,} of the unit's "
        f"{at0['cycles']:,} cycles; and {n_waiting} commands wait for the one before "
        f"({startup} cycles each, Q-30)."
    )


def shares() -> dict[str, dict[int, tuple[float, float]]]:
    """Per model and position (perf_doc's): the vector unit's seconds of a
    token and the token's, in order (Prediction.tokens_per_s_in_order)."""
    out = {}
    for name in MODELS:
        config = perf.load_config(name)
        out[name] = {}
        for p in perf_doc.positions(config):
            pred = perf.predict(config, platform(), design(), p)
            out[name][p] = pred.terms["vector"], 1 / pred.tokens_per_s_in_order
    return out


def share_table(shares: dict) -> str:
    rows = [
        [name, p, f"{vec * 1e3:.3f}", f"{token * 1e3:.3f}", f"{vec / token:.1%}"]
        for name, by_position in shares.items()
        for p, (vec, token) in by_position.items()
    ]
    header = ["model", "position", "vector unit (ms)", "token (ms)", "share"]
    return blocks.table(header, rows)


def area_table(area: dict) -> str:
    """Per kind: Yosys's cells for one FMA and one slot's chain, and the
    Vivado estimate (M3's FMA, MulRegs = 2, times the ratio) for a slot and
    the unit's slots."""
    vivado = m3.final_run(m3.logs(), "fma_path", 6, False)  # MulRegs = 2: 6 cycles
    slots = int(design()["vector_elems_per_cycle"])
    fma, chain = area["fp32_fma"], area["fp32_chain"]
    rows = []
    for kind in fma:
        row = [kind, f"{fma[kind]:,}", f"{chain[kind]:,}", "-", "-"]
        if kind in VIVADO_KINDS:
            est = getattr(vivado, VIVADO_KINDS[kind]) * chain[kind] / fma[kind]
            row[3:] = [f"{est:,.0f}", f"{est * slots:,.0f}"]
        rows.append(row)
    head = [
        "",
        "fp32_fma (Yosys)",
        "fp32_chain (Yosys)",
        "a slot (Vivado, estimate)",
        f"{slots} slots",
    ]
    return blocks.table(head, rows)


def generated() -> dict[str, str]:
    """Every block of the report, by name."""
    data = json.loads(DATA.read_text())
    tests, area = data["tests"], data["area"]
    latency = without_provenance(data["latency"])
    fast_ok, slow_ok = blocks.tests_passed(tests["fast"]), blocks.tests_passed(tests["slow"])
    worst_p, worst = worst_command(latency)
    off = worst["worst"]["off"]
    builds = {name: build_runs((m3.F2 / name).read_text()) for name in BUILDS}
    by_model = shares()
    at_0 = " and ".join(
        f"{name.split('-')[0]} {at[0][0] / at[0][1]:.1%}" for name, at in by_model.items()
    )
    exit_rows = [
        [
            "Each operation bit-exact on random and real activations",
            "random: one slot's chain per operation, whole commands, random sequences, "
            "LOAD and the stores, the tiny config's decode step (fast suite); real: "
            f"SmolLM2's whole decode step at positions {', '.join(latency)}, every "
            "buffer, every engine input and the memory (slow suite)",
            blocks.met(fast_ok and slow_ok),
        ],
        [
            f"Every vector-unit command of the real-data test within {CRITERION:.0%} of "
            "the perf model's cycles",
            (
                f"at most {abs(off):.1%} off (`{worst['worst']['kind']}` at position {worst_p})"
                if off
                else "every command takes the perf model's cycles exactly, at every position"
            ),
            blocks.met(abs(off) <= CRITERION and slow_ok),
        ],
        [
            "The perf model's share of token time at position 0 recorded (not a criterion, D-058)",
            f"{at_0} (in order, estimate; below)",
            "yes",
        ],
    ]
    return {
        "verdict": blocks.verdict(exit_rows, "M6"),
        "exit-criteria": blocks.table(["criterion", "measured", "met"], exit_rows),
        "share": share_table(by_model),
        "share-why": share_why(latency),
        "cycles": cycles_table(latency),
        "kinds": kinds_table(latency),
        "latencies": latencies_table(latency),
        "latency-provenance": blocks.provenance_text(data["latency"]["provenance"]),
        "area": area_table(area),
        "area-provenance": blocks.provenance_text(area["provenance"]),
        "builds": builds_table(builds),
        "timing": timing_text(builds[list(BUILDS)[-1]]),
        "tests-provenance": blocks.provenance_text(tests["provenance"]),
        "tests": m3.tests_table(without_provenance(tests)),
    }


def render() -> str:
    return blocks.fill(PATH.read_text(), generated())
