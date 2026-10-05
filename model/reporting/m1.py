# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The generated tables of the M1 checkpoint report (reports/M1.md), filled
from the measurement data in reports/data/M1.json (made by
scripts/report_m1.py measure)."""

import json
import re
import statistics

import compare
import paths

from reporting import blocks

DATA = paths.REPORTS / "data" / "M1.json"
REPORT = paths.REPORTS / "M1.md"
FIXTURES = ("tiny_greedy.json", "smollm2_greedy.json")
MAX_SECONDS_PER_TOKEN = 10  # exit criterion (PLAN.md, M1)
MIN_SEQUENCES, MIN_POSITIONS = 10, 256  # exit criterion (PLAN.md, M1)


def tests_passed(summary: str) -> bool:
    """A pytest summary line such as '166 passed, 5 deselected in 38.8s'."""
    return " passed" in summary and not re.search(r"\b(failed|error|errors)\b", summary)


def generated(data: dict, fixtures: dict[str, dict]) -> dict[str, str]:
    comparison, tests = data["comparison"], data["tests"]
    golden, bf16 = comparison["golden"], comparison["transformers_bf16"]
    n_sequences = len(comparison["per_sequence"])
    n_positions = {s["golden"]["positions"] for s in comparison["per_sequence"]}
    seconds = data["decode"]["seconds_per_token"]
    p90 = statistics.quantiles(seconds, n=10)[-1]
    all_passed = all(tests_passed(s) for s in tests.values())

    def met(ok: bool) -> str:
        return "yes" if ok else "**no**"

    no_worse = golden["top1"] >= bf16["top1"] and golden["logit_err_mean"] <= bf16["logit_err_mean"]
    enough = n_sequences >= MIN_SEQUENCES and min(n_positions) >= MIN_POSITIONS
    fixture_runs = {
        name: f"{len(f['runs'])} prompts × {len(f['runs'][0]['tokens'])} tokens"
        for name, f in fixtures.items()
    }
    exit_rows = [
        [
            "Teacher-forced on ≥ 10 prompts × 256 positions",
            f"{n_sequences} prompts × {', '.join(map(str, sorted(n_positions)))} positions",
            met(enough),
        ],
        [
            "Golden no worse than BF16 `transformers` (top-1, logit error)",
            f"top-1 {golden['top1']:.4f} vs {bf16['top1']:.4f}; mean logit error "
            f"{golden['logit_err_mean']:.2e} vs {bf16['logit_err_mean']:.2e}",
            met(no_worse),
        ],
        [
            "Greedy decode saved as a regression fixture",
            "; ".join(f"`{name}`: {runs}" for name, runs in fixture_runs.items()),
            met(all_passed),
        ],
        [
            "Deterministic",
            "fixture tests reproduce every stored token and logit hash",
            met(all_passed),
        ],
        [
            f"One SmolLM2 token ≤ {MAX_SECONDS_PER_TOKEN} s",
            f"median {statistics.median(seconds):.2f} s, min {min(seconds):.2f} s, "
            f"p90 {p90:.2f} s, max {max(seconds):.2f} s ({len(seconds)} timed steps "
            f"after {data['decode']['warmup']} warmup)",
            met(max(seconds) <= MAX_SECONDS_PER_TOKEN),
        ],
    ]

    prov = data["provenance"]
    dirty = " (with uncommitted changes)" if prov["dirty"] else ""
    provenance = "\n".join(
        [
            f"- commit `{prov['commit'][:12]}`{dirty}, {prov['date']}",
            f"- {prov['cpu']}, {prov['machine']}, {prov['threads']} torch threads",
            "- load average before measuring (1, 5, 15 min): "
            + ", ".join(f"{x:.2f}" for x in prov["load_average"]),
            f"- Python {prov['python']}, torch {prov['torch']}, "
            f"transformers {prov['transformers']}",
        ]
    )
    unmet = [row[0] for row in exit_rows if row[2] != "yes"]
    verdict = "**Not met:** " + "; ".join(unmet) + "." if unmet else "All M1 exit criteria are met."

    per_sequence = blocks.table(
        ["prompt", "top-1 golden", "top-1 BF16", "logit error golden", "logit error BF16"],
        [
            [
                f"{prompt[:40]}…",
                f"{s['golden']['top1']:.4f}",
                f"{s['transformers_bf16']['top1']:.4f}",
                f"{s['golden']['logit_err_mean']:.2e}",
                f"{s['transformers_bf16']['logit_err_mean']:.2e}",
            ]
            for prompt, s in zip(comparison["prompts"], comparison["per_sequence"], strict=True)
        ],
    )

    def logits(choice: dict) -> str:
        return ", ".join(f"{tok!r} {value:.4f}" for tok, value in choice)

    divergences = blocks.table(
        ["prompt", "first difference", "golden logits there", "FP32 `transformers` logits there"],
        [
            [
                f"{d['prompt'][:40]}…",
                "none" if d["first_difference"] is None else f"token {d['first_difference']}",
                logits(d["golden"]) if d["golden"] else "",
                logits(d["transformers_fp32"]) if d["transformers_fp32"] else "",
            ]
            for d in data["divergences"]
        ],
    )

    return {
        "verdict": verdict,
        "provenance": provenance,
        "exit-criteria": blocks.table(["criterion", "measured", "met"], exit_rows),
        "comparison": compare.report(comparison)
        + f"\n\nFull run: {comparison['minutes']:.1f} min.",
        "per-sequence": per_sequence,
        "divergences": divergences,
        "tests": blocks.table(["suite", "result"], [[name, f"`{s}`"] for name, s in tests.items()]),
    }


def render() -> str:
    data = json.loads(DATA.read_text())
    fixtures = {name: json.loads((paths.FIXTURES / name).read_text()) for name in FIXTURES}
    return blocks.fill(REPORT.read_text(), generated(data, fixtures))
