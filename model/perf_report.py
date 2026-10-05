# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The generated tables of docs/perf.md (model/perf.py's predictions), filled
by scripts/perf_report.py. Every number comes from the model config, the
platform and the design files."""

import paths
import perf
import report

PERF_DOC = paths.REPO / "docs" / "perf.md"
# The inputs docs/perf.md shows: files in paths.CONFIGS, PLATFORMS, DESIGNS.
MODEL, PLATFORM, DESIGN = "SmolLM2-135M-Instruct", "f2", "v0"
SENSITIVITY_POSITION = 1023
SENSITIVITY_FACTORS = (0.5, 1.0, 2.0)
UNITS = {"Hz": (1e6, "MHz"), "B/s": (1e9, "GB/s"), "B": (2**30, "GiB")}
# Rows of the work table: label, Work attribute, scale, format.
WORK_ROWS = [
    ("weights read (MB)", "weight_bytes", 1e6, "{:.2f}"),
    ("KV cache read (MB)", "kv_read_bytes", 1e6, "{:.2f}"),
    ("KV cache written (MB)", "kv_write_bytes", 1e6, "{:.2f}"),
    ("memory traffic (MB)", "memory_bytes", 1e6, "{:.2f}"),
    ("multiply-adds, matrix-vector (M)", "matvec_macs", 1e6, "{:.1f}"),
    ("multiply-adds, attention (M)", "attention_macs", 1e6, "{:.1f}"),
    ("vector-unit element passes (M)", "vector_elems", 1e6, "{:.1f}"),
    ("commands", "commands", 1, "{:.0f}"),
]


def show(value: float, unit: str) -> str:
    scale, name = UNITS.get(unit, (1, unit))
    return f"{value / scale:g} {name}".strip()


def params_table(inputs: perf.Inputs) -> str:
    return report.table(
        ["parameter", "value", "status", "source"],
        [[f"`{k}`", show(p.value, p.unit), p.status, p.source] for k, p in inputs.params.items()],
    )


def platform_table(platform: perf.Platform, documented: perf.Platform) -> str:
    """The values in use, and the documented value next to each of our
    measurements."""
    rows = []
    for k, p in platform.params.items():
        doc = documented.params[k]
        vs_doc = "" if p == doc else f"{show(doc.value, doc.unit)} ({p.value / doc.value - 1:+.1%})"
        rows.append([f"`{k}`", show(p.value, p.unit), p.status, p.source, vs_doc])
    return report.table(["parameter", "value", "status", "source", "documented"], rows)


def cleared(platform: perf.Platform) -> str:
    missing = platform.unmeasured()
    if not missing:
        return "All platform values are measured by us: the predictions rest on measurements."
    names = ", ".join(f"`{k}`" for k in missing)
    n = len(platform.params)
    return (
        f"**Provisional:** {len(missing)} of {n} platform values not yet measured by us: {names}."
    )


def scenarios_table(design: perf.Inputs) -> str:
    def changes(overrides: dict[str, float]) -> str:
        shown = [f"`{k}` = {show(v, design.params[k].unit)}" for k, v in overrides.items()]
        return ", ".join(shown) or "none"

    return report.table(
        ["scenario", "changes"], [[name, changes(o)] for name, o in design.scenarios.items()]
    )


def positions(config) -> list[int]:
    return [0, SENSITIVITY_POSITION, config.max_position_embeddings - 1]


def work_table(config) -> str:
    cols = positions(config)
    work = [perf.Work.of(perf.ops(config, p)) for p in cols]
    return report.table(
        ["per token, at position"] + [str(p) for p in cols],
        [
            [label] + [fmt.format(getattr(w, attr) / scale) for w in work]
            for label, attr, scale, fmt in WORK_ROWS
        ],
    )


def predictions_table(config, platform: perf.Inputs, design: perf.Inputs) -> str:
    rows = []
    for scenario in design.scenarios:
        for p in positions(config):
            pred = perf.predict(config, platform.values(), design.values(scenario), p)
            rate = f"{pred.tokens_per_s_serial:.0f}–{pred.tokens_per_s_overlapped:.0f}"
            ms = [f"{pred.terms[t] * 1e3:.3f}" for t in perf.TERMS]
            rows.append([scenario, str(p), rate, pred.bound, *ms])
    return report.table(
        ["scenario", "position", "tokens/s", "limited by"] + [f"{t} (ms)" for t in perf.TERMS], rows
    )


def sensitivity_table(config, platform: perf.Inputs, design: perf.Inputs) -> str:
    """Overlapped tokens/s with each unsettled input scaled, per scenario."""
    assert not set(platform.params) & set(design.params)
    params = {**platform.params, **design.params}
    unsettled = [k for k, p in params.items() if not perf.STATUSES[p.status]]
    cells = {k: [] for k in unsettled}
    for scenario in design.scenarios:
        values = platform.values() | design.values(scenario)  # names do not overlap
        for k in unsettled:
            scaled = [values | {k: values[k] * f} for f in SENSITIVITY_FACTORS]
            rates = [perf.predict(config, v, v, SENSITIVITY_POSITION) for v in scaled]
            cells[k].append(" / ".join(f"{r.tokens_per_s_overlapped:.0f}" for r in rates))
    rows = [[f"`{k}` ({params[k].status})", *cells[k]] for k in unsettled]
    return report.table(["input", *design.scenarios], rows)


def blocks(model: str, platform_name: str, design_name: str) -> dict[str, str]:
    config = perf.load_config(model)
    platform, design = perf.load_platform(platform_name), perf.load_design(design_name)
    documented = perf.load_platform(platform_name, measured=False)
    factors = " / ".join(f"×{f:g}" for f in SENSITIVITY_FACTORS)
    return {
        "platform": f"{platform.name}, from `model/platforms/{platform_name}.json` and "
        f"`{platform_name}.measured.json`:\n\n"
        + platform_table(platform, documented)
        + "\n\n"
        + cleared(platform),
        "design": f"{design.name}, from `model/designs/{design_name}.json`:\n\n"
        + params_table(design)
        + "\n\n"
        + scenarios_table(design),
        "work": f"{model}:\n\n" + work_table(config),
        "predictions": predictions_table(config, platform, design),
        "sensitivity": f"Tokens/s (overlapped) at position {SENSITIVITY_POSITION}, with the input "
        f"scaled {factors}:\n\n" + sensitivity_table(config, platform, design),
    }


def render() -> str:
    return report.fill(PERF_DOC.read_text(), blocks(MODEL, PLATFORM, DESIGN))
