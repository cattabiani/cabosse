# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The generated tables of the M7 checkpoint report (reports/M7.md), from
reports/data/M7.json: SmolLM2 on the simulated core with each token's
cycles by command against the perf model (scripts/measure_m7.py smollm2),
D-067's CPU and GPU baseline (cpu, cuda) and the test runs
(scripts/report_m7.py measure-tests). Filled by scripts/report_m7.py; needs
no weights and no RTL tools."""

import collections
import functools
import json
import statistics

import paths
import perf

from reporting import blocks, m3
from reporting.measure import QUIET_LOAD

PATH = paths.REPORTS / "M7.md"
DATA = paths.REPORTS / "data" / "M7.json"
DEVICES = ("cpu", "cuda")  # D-067's parts of the data
DEVICE_NAMES = {"cpu": "CPU", "cuda": "GPU"}
MODEL = "SmolLM2-135M-Instruct"
PLATFORM, DESIGN = "f2", "v0"
MIN_NEW = 8  # PLAN.md, M7: SmolLM2 bit-exact for at least 8 tokens


@functools.cache
def design() -> dict[str, float]:
    return perf.load_design(DESIGN).values()


def off(measured: float, model: float) -> str:
    """measured against the model, as a signed share of the model."""
    return f"{measured / model - 1:+.1%}"


def spread(values: list[float]) -> str:
    """The range of shares, one value if they round the same."""
    low, high = f"{min(values):+.1%}", f"{max(values):+.1%}"
    return low if low == high else f"{low} to {high}"


def baseline_text(data: dict) -> str:
    """D-067's rows: device, median decode tokens/s, what it ran on; a row
    timed on a busy machine (QUIET_LOAD) says so."""
    lines = ["D-067 baseline, SmolLM2 BF16, batch 1, greedy, decode tokens/s (median):"]
    for device in DEVICES:
        if device in data:
            b = data[device]
            on = f"{b['gpu']}, host " if "gpu" in b else ""
            load = b["load_average"][0]
            busy = " (busy: timing suspect)" if load > QUIET_LOAD else ""
            lines.append(
                f"{device:4}  {statistics.median(b['tokens_per_s']):8.1f}  {on}{b['cpu']},"
                f" {b['threads']} threads, load {load:.2f}{busy}; torch {b['torch']},"
                f" transformers {b['transformers']}, {b['attention']} attention;"
                f" {b['provenance']['date']}"
            )
    return "\n".join(lines)


def tokens_table(run: dict) -> str:
    """Each run's cycles against the perf model's in order."""
    rows = [
        [
            t["position"],
            f"{t['cycles']:,}",
            f"{t['model']['in order']:,.0f}",
            off(t["cycles"], t["model"]["in order"]),
        ]
        for t in run["tokens"]
    ]
    return blocks.table(["position", "cycles", "perf model, in order", "off"], rows)


def units_text(run: dict) -> str:
    """The last run's cycles by which unit works."""
    t = run["tokens"][-1]
    share = {k: f"{v:,} ({v / t['cycles']:.1%})" for k, v in t["units"].items()}
    return (
        f"At position {t['position']}, of {t['cycles']:,} cycles: the engine alone"
        f" {share['engine only']}, the vector unit alone {share['vector only']}, both"
        f" {share['both']}, neither (the controller between them) {share['neither']}."
    )


def kinds_table(run: dict) -> str:
    """The last run's cycles by unit and command against the perf model."""
    cycles, model, n = collections.Counter(), collections.Counter(), collections.Counter()
    for r in run["tokens"][-1]["commands"]:
        key = (r["unit"], r["kind"])
        cycles[key] += r["cycles"]
        model[key] += r["model_cycles"]
        n[key] += 1
    rows = []
    for unit, kind in sorted(n, key=lambda k: -cycles[k]):
        k = (unit, kind)
        rows.append(
            [
                unit,
                f"`{kind.upper()}`",
                n[k],
                f"{cycles[k]:,}",
                f"{model[k]:,.0f}",
                off(cycles[k], model[k]),
            ]
        )
    return blocks.table(["unit", "command", "commands", "cycles", "perf model", "off"], rows)


def startup_text(run: dict) -> str:
    """The waiting engine commands' start-up against engine_startup_cycles."""
    model = run["engine_startup_cycles"]
    waits = [r for r in run["tokens"][-1]["commands"] if r["waits"]]
    found = [round(r["cycles"] - r["model_cycles"] + model) for r in waits]
    return (
        f"The {len(waits)} engine commands that wait for an earlier result start in"
        f" {min(found)} to {max(found)} cycles (median {statistics.median(found):.0f});"
        f" the perf model's `engine_startup_cycles` is {model:.0f} (D-042)."
    )


def f2_tokens_per_s(position: int) -> float:
    """The perf model's in-order decode speed of SmolLM2 on F2 (estimate)."""
    config = perf.load_config(MODEL)
    platform = perf.load_platform(PLATFORM).values()
    return perf.predict(config, platform, design(), position).tokens_per_s_in_order


def speed_rows(data: dict) -> list[list]:
    """Decode tokens/s: the CPU and GPU (measured), the simulated core at
    the design's clock, the perf model on F2 (estimate)."""
    rows = []
    for device in DEVICES:
        b = data[device]
        on = (
            f"{b['gpu']} (host {b['cpu']})" if "gpu" in b else f"{b['cpu']}, {b['threads']} threads"
        )
        rows.append(
            [
                DEVICE_NAMES[device],
                f"{statistics.median(b['tokens_per_s']):.1f}",
                f"{on}; `transformers` {b['transformers']}, torch {b['torch']}",
                "measured",
            ]
        )
    run = data["smollm2"]
    last = run["tokens"][-1]
    clock = design()["clock_hz"]
    rows.append(
        [
            "Cabosse, simulated",
            f"{clock / last['cycles']:.1f}",
            f"rtl/core.sv at {clock / 1e6:.0f} MHz, position {last['position']}, a beat a cycle"
            " on each of 32 ports at F2's latency; no host",
            "simulated",
        ]
    )
    rows.append(
        [
            "Cabosse on F2",
            f"{f2_tokens_per_s(last['position']):.1f}",
            f"perf model, design {DESIGN}, F2's measured HBM rate, in order; no host",
            "estimate",
        ]
    )
    return rows


def generated() -> dict[str, str]:
    """Every block of the report, by name."""
    data = json.loads(DATA.read_text())
    tests, run = data["tests"], data["smollm2"]
    fast_ok, slow_ok = blocks.tests_passed(tests["fast"]), blocks.tests_passed(tests["slow"])
    in_order = [t["cycles"] / t["model"]["in order"] - 1 for t in run["tokens"]]
    prompt = len(run["tokens"]) - run["n_new"] + 1  # the last token is not run
    speeds = speed_rows(data)
    exit_rows = [
        [
            "Tiny configs bit-exact for 3 prompts × 64 tokens",
            "tokens, logits and KV cache bit for bit, 3 prompts at 4 ports and one at 32"
            " (fast suite)",
            blocks.met(fast_ok),
        ],
        [
            f"SmolLM2 bit-exact for ≥ {MIN_NEW} tokens",
            f"{run['n_new']} new tokens after a {prompt}-token prompt, tokens, logits and KV"
            " cache bit for bit, 32 ports (slow suite; below)",
            blocks.met(run["bit_exact"] and run["n_new"] >= MIN_NEW and slow_ok),
        ],
        [
            "The cycle breakdown per token compared with the perf model",
            f"every token {spread(in_order)} off the model in order; each command below",
            "yes",
        ],
        [
            "Decode speed compared with a CPU and a GPU on the same model (D-067)",
            "; ".join(f"{r[0]} {r[1]}" for r in speeds) + " tokens/s (below)",
            "yes",
        ],
    ]
    return {
        "verdict": blocks.verdict(exit_rows, "M7"),
        "exit-criteria": blocks.table(["criterion", "measured", "met"], exit_rows),
        "smollm2-provenance": blocks.provenance_text(run["provenance"]),
        "tokens": tokens_table(run),
        "units": units_text(run),
        "kinds": kinds_table(run),
        "startup": startup_text(run),
        "speed": blocks.table(["", "decode tokens/s", "on", "kind"], speeds),
        "speed-provenance": " ".join(
            f"{DEVICE_NAMES[d]}: {blocks.provenance_text(data[d]['provenance'])}" for d in DEVICES
        ),
        "tests-provenance": blocks.provenance_text(tests["provenance"]),
        "tests": m3.tests_table({k: v for k, v in tests.items() if k != "provenance"}),
    }


def render() -> str:
    return blocks.fill(PATH.read_text(), generated())
