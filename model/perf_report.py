# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The generated tables of docs/perf.md (model/perf.py's predictions), filled
by scripts/perf_report.py. Every number comes from the model config, the
platform and the design files."""

import paths
import perf
import report

PERF_DOC = paths.REPO / "docs" / "perf.md"
MODEL = "SmolLM2-135M-Instruct"
PLATFORM = perf.PLATFORMS / "f2.json"
DESIGN = perf.DESIGNS / "v0.json"
SENSITIVITY_POSITION = 1023
SENSITIVITY_FACTORS = (0.5, 2.0)
SETTLED = {"quoted", "computed", "choice", "measured by us"}  # no sensitivity row
UNITS = {"Hz": (1e6, "MHz"), "B/s": (1e9, "GB/s"), "B": (2**30, "GiB")}


def show(value: float, unit: str) -> str:
    scale, name = UNITS.get(unit, (1, unit))
    return f"{value / scale:g} {name}".strip()


def params_table(inputs: perf.Inputs) -> str:
    return report.table(
        ["parameter", "value", "status", "source"],
        [[f"`{k}`", show(p.value, p.unit), p.status, p.source] for k, p in inputs.params.items()],
    )


def positions(config) -> list[int]:
    return [0, SENSITIVITY_POSITION, config.max_position_embeddings - 1]


def work_table(config) -> str:
    cols = positions(config)
    ops = {p: perf.ops(config, p) for p in cols}

    def row(label: str, f) -> list[str]:
        return [label] + [f(ops[p]) for p in cols]

    def total(attr: str):
        return lambda op_list: sum(getattr(o, attr) for o in op_list)

    def mb(attr: str):
        return lambda op_list: f"{total(attr)(op_list) / 1e6:.2f}"

    def mega(f):
        return lambda op_list: f"{f(op_list) / 1e6:.1f}"

    def macs(names: set[str], inside: bool):
        return lambda op_list: sum(o.macs for o in op_list if (o.name in names) == inside)

    attention = {"scores", "pv"}
    return report.table(
        ["per token, at position"] + [str(p) for p in cols],
        [
            row("weights read (MB)", mb("weight_bytes")),
            row("KV cache read (MB)", mb("kv_read_bytes")),
            row("KV cache written (MB)", mb("kv_write_bytes")),
            row("memory traffic (MB)", lambda o: f"{perf.Work.of(o).memory_bytes / 1e6:.2f}"),
            row("multiply-adds, matrix-vector (M)", mega(macs(attention, False))),
            row("multiply-adds, attention (M)", mega(macs(attention, True))),
            row("vector-unit element passes (M)", mega(total("vector_elems"))),
            row("commands", lambda o: str(len(o))),
        ],
    )


def predictions_table(config, platform: perf.Inputs, design: perf.Inputs) -> str:
    rows = []
    for scenario in design.scenarios:
        for p in positions(config):
            pred = perf.predict(config, platform.values(), design.values(scenario), p)
            ms = [f"{pred.terms[t] * 1e3:.3f}" for t in perf.TERMS]
            rows.append(
                [
                    scenario,
                    str(p),
                    f"{pred.tokens_per_s_serial:.0f}–{pred.tokens_per_s_overlapped:.0f}",
                    pred.bound,
                    *ms,
                ]
            )
    return report.table(
        ["scenario", "position", "tokens/s", "limited by"] + [f"{t} (ms)" for t in perf.TERMS], rows
    )


def scaled_rate(config, plat: dict, des: dict, name: str, factor: float) -> float:
    """Overlapped tokens/s with input `name` (in plat or des) scaled."""
    if name in plat:
        plat = plat | {name: plat[name] * factor}
    else:
        des = des | {name: des[name] * factor}
    return perf.predict(config, plat, des, SENSITIVITY_POSITION).tokens_per_s_overlapped


def sensitivity_table(config, platform: perf.Inputs, design: perf.Inputs) -> str:
    """Overlapped tokens/s with each unsettled input scaled, per scenario."""
    assert not set(platform.params) & set(design.params)
    params = {**platform.params, **design.params}
    rows = []
    for name, param in params.items():
        if param.status in SETTLED:
            continue
        cells = []
        for scenario in design.scenarios:
            plat, des = platform.values(), design.values(scenario)
            rates = [
                scaled_rate(config, plat, des, name, f)
                for f in (SENSITIVITY_FACTORS[0], 1, SENSITIVITY_FACTORS[1])
            ]
            cells.append(" / ".join(f"{r:.0f}" for r in rates))
        rows.append([f"`{name}` ({param.status})", *cells])
    return report.table(["input", *design.scenarios], rows)


def blocks() -> dict[str, str]:
    config = perf.load_config(MODEL)
    platform, design = perf.load_inputs(PLATFORM), perf.load_inputs(DESIGN)
    scenarios = report.table(
        ["scenario", "changes"],
        [
            [
                name,
                ", ".join(f"`{k}` = {show(v, design.params[k].unit)}" for k, v in o.items())
                or "none",
            ]
            for name, o in design.scenarios.items()
        ],
    )
    factors = " / ".join(f"×{f:g}" for f in (SENSITIVITY_FACTORS[0], 1, SENSITIVITY_FACTORS[1]))
    return {
        "platform": f"{platform.name}, from `{PLATFORM.relative_to(paths.REPO)}`:\n\n"
        + params_table(platform),
        "design": f"{design.name}, from `{DESIGN.relative_to(paths.REPO)}`:\n\n"
        + params_table(design)
        + "\n\n"
        + scenarios,
        "work": f"{MODEL}:\n\n" + work_table(config),
        "predictions": predictions_table(config, platform, design),
        "sensitivity": f"Tokens/s (overlapped) at position {SENSITIVITY_POSITION}, with the input "
        f"scaled {factors}:\n\n" + sensitivity_table(config, platform, design),
    }


def render() -> str:
    return report.fill(PERF_DOC.read_text(), blocks())
