# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Measurements of the vector unit (PLAN.md, M6) into reports/data/M6.json,
each part replacing its own key:

    python scripts/measure_m6.py area   # Yosys for UltraScale+: one FMA, one slot's chain
    python scripts/measure_m6.py show   # print the table from the saved data

`area` needs Yosys with the slang plugin (.tools/, scripts/get_rtl_tools.sh)
and takes a few minutes. Yosys's cells are a proxy, not Vivado's
utilization: the table scales M3's Vivado numbers for one fp32_fma by the
chain's ratio to it (an estimate until M8's build). Commit first: the data
records the commit.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for sub in ("model", "verif"):
    sys.path.insert(0, str(REPO / sub))

from reporting import blocks, measure  # noqa: E402

import rtl  # noqa: E402  (puts .tools/ on PATH)

DATA = REPO / "reports" / "data" / "M6.json"
TOPS = ("fp32_fma", "fp32_chain")
SLOTS = 16  # the vector unit's width (D-030)
# One fp32_fma in Vivado on F2's part, MulRegs = 2 (PLAN.md, M3: the fifth
# timing log): LUTs, registers, DSPs.
VIVADO_FMA = {"LUTs": 1262, "registers": 535, "DSPs": 2}
# Yosys's UltraScale+ cells, by what they count.
KINDS = {
    "LUTs": re.compile(r"LUT\d|SRL\w*"),  # shift registers are LUTs too
    "registers": re.compile(r"FD\w+"),
    "DSPs": re.compile(r"DSP48E2"),
    "carry blocks": re.compile(r"CARRY\d"),
}


def yosys_cells(top: str) -> dict[str, int]:
    """Cells of `top` after Yosys's synthesis for UltraScale+, by kind."""
    script = (
        f"read_slang -F {rtl.SOURCES} --top {top}; "
        f"synth_xilinx -family xcup -flatten -noiopad -noclkbuf -top {top}; stat"
    )
    run = subprocess.run(
        ["yosys", "-m", "slang", "-p", script], capture_output=True, text=True, check=True
    )
    stats = run.stdout[run.stdout.rindex("Printing statistics.") :]
    stats = stats[: stats.index("=== design hierarchy")]  # the flattened top only
    cells = Counter()
    for count, name in re.findall(r"^\s+(\d+)\s+([A-Z]\w*)\s*$", stats, re.M):
        cells[name] += int(count)
    return {
        kind: sum(n for c, n in cells.items() if pat.fullmatch(c)) for kind, pat in KINDS.items()
    }


def area() -> dict:
    return {top: yosys_cells(top) for top in TOPS}


def table(data: dict) -> str:
    """Per top: Yosys's cells, and the Vivado estimate (M3's FMA times the
    ratio); the vector unit's 16 slots."""
    fma, chain = data["fp32_fma"], data["fp32_chain"]
    rows = []
    for kind in KINDS:
        row = [kind, f"{fma[kind]:,}", f"{chain[kind]:,}", "-", "-"]
        if kind in VIVADO_FMA:
            est = VIVADO_FMA[kind] * chain[kind] / fma[kind]
            row[3:] = [f"{est:,.0f}", f"{est * SLOTS:,.0f}"]
        rows.append(row)
    head = ["", "fp32_fma (Yosys)", "fp32_chain (Yosys)", "chain (Vivado, estimate)", "16 slots"]
    return blocks.table(head, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=("area", "show"))
    args = parser.parse_args()
    data = json.loads(DATA.read_text()) if DATA.exists() else {}
    if args.part != "show":
        data[args.part] = {"provenance": measure.provenance()} | globals()[args.part]()
        DATA.write_text(json.dumps(data, indent=1) + "\n")
    print(table(data["area"]))


if __name__ == "__main__":
    main()
