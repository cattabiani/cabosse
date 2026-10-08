# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The cost of subnormal support (D-016) for the M3 report (reporting/m3.py),
parsed from the log that scripts/report_m3.py measure-cells writes to
reports/data/.

The log is Yosys's own output. Each run starts with a line
    CABOSSE-YOSYS label=<label> rev=<rev> top=<top> params=<k=v,...>
and its cell count is the `<n> cells` line of the top's `stat` block.
Other lines: CABOSSE-PROVENANCE (commit, date, tool versions) and
CABOSSE-FLUSH-HITS (per flush, the test rows it changes).
"""

import re

import paths

from reporting import blocks

LOG = paths.REPORTS / "data" / "M3-subnormal-yosys.txt"
# label -> (top, parameters); each is synthesized with Ftz = 0 and Ftz = 1.
UNITS = {
    "`bf16_mac` (lanes)": ("bf16_mac", ()),
    "`fp32_fma` (vector unit)": ("fp32_fma", ()),
    "`fp_product`, M = 8": ("fp_product", (("M", 8),)),
    "`fp_product`, M = 24": ("fp_product", (("M", 24),)),
    "`fp_add`, W = 27": ("fp_add", (("W", 27),)),
    "`fp_add`, W = 51": ("fp_add", (("W", 51),)),
}
# label -> (top, parameters): blocks for scale, synthesized once.
PARTS = {
    "`sticky_shift`, 27 bits (shift below exponent 1, M = 8)": (
        "sticky_shift",
        (("Width", 27),),
    ),
    "`sticky_shift`, 51 bits (shift below exponent 1, M = 24)": (
        "sticky_shift",
        (("Width", 51),),
    ),
    "`leading_zeros`, 16 bits (product, M = 8)": ("leading_zeros", (("Width", 16),)),
    "`leading_zeros`, 48 bits (product, M = 24)": ("leading_zeros", (("Width", 48),)),
}
# The same logic as Ftz = 0, written before the Ftz parameter: main when it
# was added. Yosys's count moves with the source text, not only the logic.
BASELINE_REV = "94a894f"
BASELINE_TOPS = ("bf16_mac", "fp32_fma")
HEAD = "HEAD"

RUN = re.compile(r"^CABOSSE-YOSYS label=(\S+) rev=(\S+) top=(\S+) params=(\S*)$", re.M)
KEY_VALUES = re.compile(r"(\w+)=(\S+)")


def run_label(top: str, params: tuple, rev: str = HEAD) -> str:
    """The log's key for one run: top, parameters and source revision."""
    return "/".join([rev, top, *(f"{k}={v}" for k, v in params)])


def parse(text: str) -> tuple[dict[str, int], dict[str, str], dict[str, dict[str, int]]]:
    """(cells per run label, provenance, flush hits per unit and operation)."""
    cells = {}
    starts = list(RUN.finditer(text))
    for m, nxt in zip(starts, [*starts[1:], None], strict=True):
        body = text[m.end() : nxt.start() if nxt else len(text)]
        top = m[3]
        block = body.split(f"=== {top} ===", 1)[1]
        cells[m[1]] = int(re.search(r"^\s+(\d+) cells$", block, re.M)[1])
    provenance = dict(KEY_VALUES.findall(_line(text, "CABOSSE-PROVENANCE")))
    hits = {}
    for line in re.findall(r"^CABOSSE-FLUSH-HITS (.*)$", text, re.M):
        fields = dict(KEY_VALUES.findall(line))
        name = f"{fields.pop('unit')} {fields.pop('op')}"
        hits[name] = {k: int(v) for k, v in fields.items()}
    return cells, provenance, hits


def _line(text: str, tag: str) -> str:
    return re.search(rf"^{tag} (.*)$", text, re.M)[1]


def cost(cells: dict[str, int], top: str, params: tuple = ()) -> tuple[int, float]:
    """(cells, percent of the unit with subnormals) that subnormals cost."""
    keep = cells[run_label(top, (*params, ("Ftz", 0)))]
    n = keep - cells[run_label(top, (*params, ("Ftz", 1)))]
    return n, 100 * n / keep


def costs(cells: dict[str, int]) -> tuple[tuple[int, float], tuple[int, float]]:
    """cost() of the lanes' bf16_mac and the vector unit's fp32_fma."""
    return cost(cells, "bf16_mac"), cost(cells, "fp32_fma")


def cost_table(cells: dict[str, int]) -> str:
    rows = []
    for label, (top, params) in UNITS.items():
        keep = cells[run_label(top, (*params, ("Ftz", 0)))]
        n, percent = cost(cells, top, params)
        rows.append([label, f"{keep:,}", f"{keep - n:,}", f"{n:+,}", f"{percent:+.1f} %"])
    header = ["unit", "with subnormals", "flush to zero", "cost (cells)", "share of the unit"]
    return blocks.table(header, rows)


def parts_table(cells: dict[str, int]) -> str:
    rows = [[label, f"{cells[run_label(top, params)]:,}"] for label, (top, params) in PARTS.items()]
    return blocks.table(["block", "cells"], rows)


def baseline_table(cells: dict[str, int]) -> str:
    rows = []
    for top in BASELINE_TOPS:
        before = cells[run_label(top, (), BASELINE_REV)]
        now = cells[run_label(top, (("Ftz", 0),))]
        rows.append([f"`{top}`", f"{before:,}", f"{now:,}", f"{now - before:+,}"])
    header = ["unit", f"`{BASELINE_REV}` (no `Ftz`)", "this commit, `Ftz` = 0", "difference"]
    return blocks.table(header, rows)


def hits_table(hits: dict[str, dict[str, int]]) -> str:
    sites = ["product input", "addend", "result"]
    keys = [s.replace(" ", "_") for s in sites]
    rows = [
        [f"`{name}`", *(f"{h[k]:,}" if k in h else "–" for k in keys)] for name, h in hits.items()
    ]
    return blocks.table(["unit, operation", *sites], rows)


def provenance_text(p: dict[str, str]) -> str:
    dirty = " with uncommitted changes" if p["dirty"] == "True" else ""
    return (
        f"Measured on {p['date']}, commit `{p['commit'][:7]}`{dirty}, "
        f"Yosys {p['yosys']} with yosys-slang, generic `synth -flatten`."
    )
