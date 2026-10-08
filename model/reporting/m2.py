# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The generated tables of the M2 checkpoint report (reports/M2.md): the exit
criteria, the predictions and the summation-parameter comparisons, from the
perf model, the command list and the comparison runs in reports/data/.
Filled by scripts/report_m2.py; needs no weights."""

import json

import commands
import compare
import paths
import perf
from golden import dot, vector

from reporting import arch_doc, blocks, perf_doc

PATH = paths.REPORTS / "M2.md"
# The comparison runs on the M1 set, in order: summation parameters -> data.
RUNS = {
    "M1: `A` = 8, `S` = 8": ("M1.json", "comparison"),
    "`A` = 16, `S` = 8": ("M2-accumulators-16-compare.json", None),
    "`A` = 16, `S` = 128": ("M2-reduce-width-128-compare.json", None),
}
PREDICTION_SCENARIOS = ("as planned", "256 lanes, HBM at 450 MHz")  # v0 and M9's step up


def runs() -> dict[str, dict]:
    out = {}
    for name, (file, key) in RUNS.items():
        data = json.loads((paths.REPORTS / "data" / file).read_text())
        out[name] = data[key] if key else data
    return out


def comparison_table(results: dict[str, dict]) -> str:
    last = list(results.values())[-1]
    header = ["", *(f"golden, {name}" for name in results), "transformers BF16"]
    rows = [
        [label, *(fmt.format(r["golden"][k]) for r in results.values())]
        + [fmt.format(last["transformers_bf16"][k])]
        for label, k, fmt in compare.REPORT_ROWS
    ]
    positions = last["golden"]["positions"]
    return (
        f"Against FP32 `transformers`, {positions} positions "
        f"({len(last['per_sequence'])} sequences).\n\n" + blocks.table(header, rows)
    )


def predictions_table() -> str:
    config = perf.load_config(perf_doc.MODEL)
    platform, design = perf.load_platform(perf_doc.PLATFORM), perf.load_design(perf_doc.DESIGN)
    rows = []
    for scenario in PREDICTION_SCENARIOS:
        for p in perf_doc.positions(config):
            pred = perf.predict(config, platform.values(), design.values(scenario), p)
            rate = f"{pred.tokens_per_s_serial:.0f}–{pred.tokens_per_s_overlapped:.0f}"
            rows.append([scenario, p, rate, pred.bound])
    return blocks.table(["scenario", "position", "tokens/s", "limited by"], rows)


def generated() -> dict[str, str]:
    results = runs()
    final = list(results.values())[-1]
    golden, bf16 = final["golden"], final["transformers_bf16"]
    config = perf.load_config(perf_doc.MODEL)
    layout = commands.Layout.of(config, config.max_position_embeddings)
    n_commands = len(commands.build(layout))
    per_layer = len(commands.layer_commands(layout, 0))
    platform, design = perf.load_platform(perf_doc.PLATFORM), perf.load_design(perf_doc.DESIGN)
    pred = perf.predict(config, platform.values(), design.values(), 0)

    met = blocks.met
    still_passes = (
        golden["top1"] >= bf16["top1"] and golden["logit_err_mean"] <= bf16["logit_err_mean"]
    )
    exit_rows = [
        [
            "Every golden-model operation maps to a command, no host work mid-token",
            f"{n_commands} commands per {perf_doc.MODEL} token ({per_layer} per layer); "
            "the reference controller gives the golden decode step bit for bit "
            "(`test_commands.py`: tiny configs and SmolLM2)",
            "yes",
        ],
        [
            f"Predicted tokens/s for {perf_doc.MODEL} on F2, dominant term identified",
            f"{pred.tokens_per_s_serial:.0f}–{pred.tokens_per_s_overlapped:.0f} tokens/s "
            f"at position 0, limited by {pred.bound} (table below)",
            "yes",
        ],
        [
            "Summation parameters fixed",
            f"`A` = {dot.ACCUMULATORS} (D-027), `S` = {vector.REDUCE_WIDTH} (D-030)",
            "yes",
        ],
        [
            "Golden model still passes M1 (no worse than BF16 `transformers`)",
            f"top-1 {golden['top1']:.4f} vs {bf16['top1']:.4f}; mean logit error "
            f"{golden['logit_err_mean']:.2e} vs {bf16['logit_err_mean']:.2e}",
            met(still_passes),
        ],
    ]
    unmet = [row[0] for row in exit_rows if row[2] != "yes"]
    verdict = "**Not met:** " + "; ".join(unmet) + "." if unmet else "All M2 exit criteria are met."
    return {
        "verdict": verdict,
        "exit-criteria": blocks.table(["criterion", "measured or predicted", "met"], exit_rows),
        "predictions": predictions_table(),
        "comparison": comparison_table(results),
        "ladder": arch_doc.ladder_table(),
    }


def render() -> str:
    return blocks.fill(PATH.read_text(), generated())
