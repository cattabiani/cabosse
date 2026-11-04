# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The generated tables of the M7 checkpoint report (reports/M7.md), from
reports/data/M7.json: SmolLM2 on the simulated core with each token's
cycles by command against the perf model (scripts/measure_m7.py smollm2),
D-067's CPU and GPU baseline (cpu, cuda) and the test runs
(scripts/report_m7.py measure-tests). Filled by scripts/report_m7.py; needs
no weights and no RTL tools. scripts/measure_m7.py prints the same tables."""

import collections
import json
import statistics

import paths
import perf

from reporting import blocks, m3, m6
from reporting.measure import QUIET_LOAD

PATH = paths.REPORTS / "M7.md"
DATA = paths.REPORTS / "data" / "M7.json"
DEVICES = ("cpu", "cuda")  # D-067's parts of the data
DEVICE_NAMES = {"cpu": "CPU", "cuda": "GPU"}
MODEL = "SmolLM2-135M-Instruct"
MIN_NEW = 8  # PLAN.md, M7: SmolLM2 bit-exact for at least 8 tokens


def share(measured: float, model: float) -> float:
    return measured / model - 1


def spread(values: list[float]) -> str:
    """The range of shares, one value if they round the same."""
    low, high = f"{min(values):+.1%}", f"{max(values):+.1%}"
    return low if low == high else f"{low} to {high}"


def busy(b: dict) -> bool:
    """A baseline row timed on a busy machine (QUIET_LOAD)."""
    return b["load_average"][0] > QUIET_LOAD


def ran_on(b: dict) -> str:
    """What a baseline row ran on, and whether the machine was busy."""
    host = f"{b['cpu']}, {b['threads']} threads"
    on = f"{b['gpu']} (host {host})" if "gpu" in b else host
    flag = " (busy: timing suspect)" if busy(b) else ""
    return (
        f"{on}; `transformers` {b['transformers']} ({b['attention']} attention),"
        f" torch {b['torch']}; load {b['load_average'][0]:.2f}{flag}"
    )


def median(b: dict) -> float:
    return statistics.median(b["tokens_per_s"])


def baseline_text(data: dict) -> str:
    """D-067's rows, for scripts/measure_m7.py show."""
    lines = ["D-067 baseline, SmolLM2 BF16, batch 1, greedy, decode tokens/s (median):"]
    for d in DEVICES:
        if d in data:
            lines.append(f"{d:4}  {median(data[d]):8.1f}  {ran_on(data[d])}")
    return "\n".join(lines)


def tokens_table(run: dict) -> str:
    """Each run's cycles against the perf model's in order."""
    rows = [
        [
            t["position"],
            f"{t['cycles']:,}",
            f"{t['model']['in order']:,.0f}",
            f"{share(t['cycles'], t['model']['in order']):+.1%}",
        ]
        for t in run["tokens"]
    ]
    return blocks.table(["position", "cycles", "perf model, in order", "off"], rows)


def units_text(run: dict) -> str:
    """The last run's cycles by which unit works."""
    t = run["tokens"][-1]
    part = {k: f"{v:,} ({v / t['cycles']:.1%})" for k, v in t["units"].items()}
    return (
        f"At position {t['position']}, of {t['cycles']:,} cycles: the engine alone"
        f" {part['engine only']}, the vector unit alone {part['vector only']}, both"
        f" {part['both']}, neither (the controller between them) {part['neither']}."
    )


def by_kind(run: dict) -> dict[tuple[str, str], tuple[int, int, float]]:
    """The last run's (commands, cycles, model cycles) by (unit, kind),
    most cycles first."""
    out = collections.defaultdict(lambda: (0, 0, 0.0))
    for r in run["tokens"][-1]["commands"]:
        n, c, m = out[r["unit"], r["kind"]]
        out[r["unit"], r["kind"]] = n + 1, c + r["cycles"], m + r["model_cycles"]
    return dict(sorted(out.items(), key=lambda kv: -kv[1][1]))


def kinds_table(run: dict) -> str:
    rows = [
        [unit, f"`{kind.upper()}`", n, f"{c:,}", f"{m:,.0f}", f"{share(c, m):+.1%}"]
        for (unit, kind), (n, c, m) in by_kind(run).items()
    ]
    return blocks.table(["unit", "command", "commands", "cycles", "perf model", "off"], rows)


def startups(run: dict) -> list[int]:
    """The last run's waiting engine commands' start-up: their cycles past
    the perf model's engine cycles."""
    model = run["engine_startup_cycles"]
    return [
        round(r["cycles"] - r["model_cycles"] + model)
        for r in run["tokens"][-1]["commands"]
        if r["waits"]
    ]


def startup_text(run: dict) -> str:
    found = startups(run)
    return (
        f"The {len(found)} engine commands that wait for an earlier result start in"
        f" {min(found)} to {max(found)} cycles (median {statistics.median(found):.0f});"
        f" the perf model's `engine_startup_cycles` is"
        f" {run['engine_startup_cycles']:.0f} (D-042)."
    )


def memory_text(run: dict) -> str:
    """The simulated memory the SmolLM2 run used."""
    mem = run["memory"]
    return (
        f"The simulated memory answers a read {mem['latency']} cycles and a write"
        f" {mem['wlatency']} cycles after its request, a beat a cycle on each of the"
        f" {run['ports']} ports."
    )


def smollm2_text(run: dict) -> str:
    """The SmolLM2 run's tables, for scripts/measure_m7.py show."""
    return "\n\n".join(
        [
            blocks.provenance_text(run["provenance"]),
            f"{run['n_new']} new tokens, bit-exact: {run['bit_exact']},"
            f" {run['seconds']} s with the build. {memory_text(run)}",
            tokens_table(run),
            units_text(run),
            kinds_table(run),
            startup_text(run),
        ]
    )


def f2_tokens_per_s(position: int) -> float:
    """The perf model's in-order decode speed of SmolLM2 on F2 (estimate)."""
    config = perf.load_config(MODEL)
    return perf.predict(config, m6.platform(), m6.design(), position).tokens_per_s_in_order


def speed_rows(data: dict) -> list[list]:
    """Decode tokens/s: the CPU and GPU (measured), the simulated core at
    the design's clock, the perf model on F2 (estimate)."""
    rows = [
        [DEVICE_NAMES[d], f"{median(data[d]):.1f}", ran_on(data[d]), "measured"] for d in DEVICES
    ]
    run = data["smollm2"]
    last = run["tokens"][-1]
    clock = m6.design()["clock_hz"]
    rows.append(
        [
            "Cabosse, simulated",
            f"{clock / last['cycles']:.1f}",
            f"rtl/core.sv at {clock / 1e6:.0f} MHz, position {last['position']}"
            " (the memory above); no host",
            "simulated",
        ]
    )
    rows.append(
        [
            "Cabosse on F2",
            f"{f2_tokens_per_s(last['position']):.1f}",
            f"perf model, design {m6.DESIGN}, F2's measured HBM rate, in order; no host",
            "estimate",
        ]
    )
    return rows


def timing_text(data: dict) -> str:
    """How the CPU and GPU rows were timed, from their data."""
    rows = [data[d] for d in DEVICES]
    assert len({(b["n_decode"], len(b["tokens_per_s"]), b["warmup"]) for b in rows}) == 1
    b = rows[0]
    return (
        f"The CPU and GPU decode {b['n_decode']} tokens after the prompt (prefill excluded),"
        f" the median of {len(b['tokens_per_s'])} runs after {b['warmup']} warmup."
        + "".join(
            f" {DEVICE_NAMES[d]}: {blocks.provenance_text(data[d]['provenance'])}" for d in DEVICES
        )
    )


def generated() -> dict[str, str]:
    """Every block of the report, by name."""
    data = json.loads(DATA.read_text())
    tests, run = data["tests"], data["smollm2"]
    fast_ok, slow_ok = blocks.tests_passed(tests["fast"]), blocks.tests_passed(tests["slow"])
    in_order = [share(t["cycles"], t["model"]["in order"]) for t in run["tokens"]]
    prompt = len(run["tokens"]) - run["n_new"] + 1  # the last token is not run
    speeds = speed_rows(data)
    exit_rows = [
        [
            "Tiny configs bit-exact for 3 prompts × 64 tokens",
            "tokens, logits and KV cache bit for bit (fast suite)",
            blocks.met(fast_ok),
        ],
        [
            f"SmolLM2 bit-exact for ≥ {MIN_NEW} tokens",
            f"{run['n_new']} new tokens after a {prompt}-token prompt, tokens, logits and KV"
            f" cache bit for bit, {run['ports']} ports (slow suite; below)",
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
            blocks.met(not any(busy(data[d]) for d in DEVICES)),
        ],
    ]
    return {
        "verdict": blocks.verdict(exit_rows, "M7"),
        "exit-criteria": blocks.table(["criterion", "measured", "met"], exit_rows),
        "smollm2-provenance": blocks.provenance_text(run["provenance"]),
        "memory": memory_text(run),
        "tokens": tokens_table(run),
        "units": units_text(run),
        "kinds": kinds_table(run),
        "startup": startup_text(run),
        "speed": blocks.table(["", "decode tokens/s", "on", "kind"], speeds),
        "speed-provenance": timing_text(data),
        "tests-provenance": blocks.provenance_text(tests["provenance"]),
        "tests": m3.tests_table(m6.without_provenance(tests)),
    }


def render() -> str:
    return blocks.fill(PATH.read_text(), generated())
