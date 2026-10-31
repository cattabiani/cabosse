# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Measurements of the vector unit (PLAN.md, M6) into reports/data/M6.json,
each part replacing its own key:

    python scripts/measure_m6.py area     # Yosys for UltraScale+: one FMA, one slot's chain
    python scripts/measure_m6.py latency  # SmolLM2's whole step: each command's cycles
    python scripts/measure_m6.py show     # print the tables from the saved data

`area` needs Yosys with the slang plugin (.tools/, scripts/get_rtl_tools.sh)
and takes a few minutes. Yosys's cells are a proxy, not Vivado's
utilization: the table scales M3's Vivado numbers for one fp32_fma by the
chain's ratio to it (an estimate until M8's build).

`latency` runs SmolLM2's whole decode step on the vector unit
(verif/decode.py) at the real-data positions, about 2 minutes and 3 GB:
run it under the memory cap. It needs the checkpoint and the long cache
(scripts/fill_long_cache.py). Each latency of the perf model's
vector_cycles (D-053) comes from the commands that show it alone, every
value found with its count: design v0 takes them when each has one value.
Commit first: the data records the commit.
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for sub in ("model", "model/tests", "verif"):
    sys.path.insert(0, str(REPO / sub))

import perf  # noqa: E402
from reporting import blocks, measure  # noqa: E402

import rtl  # noqa: E402  (puts .tools/ on PATH)

DATA = REPO / "reports" / "data" / "M6.json"
TOPS = ("fp32_fma", "fp32_chain")
SLOTS = 16  # the vector unit's width (D-030)
# One fp32_fma in Vivado on F2's part, MulRegs = 2 (PLAN.md, M3: the fifth
# timing log): LUTs, registers, DSPs.
VIVADO_FMA = {"LUTs": 1262, "registers": 535, "DSPs": 2}
DESIGN = perf.load_design("v0").values()
# Yosys's UltraScale+ cells, by what they count.
KINDS = {
    "LUTs": re.compile(r"LUT\d|SRL\w*"),  # shift registers are LUTs too
    "registers": re.compile(r"FD\w+"),
    "DSPs": re.compile(r"DSP48E2"),
    "carry blocks": re.compile(r"CARRY\d"),
}


def yosys_cells(top: str) -> dict[str, int]:
    """Cells of `top` after Yosys's synthesis for UltraScale+, by kind."""
    cells = rtl.cells(top)
    return {
        kind: sum(n for c, n in cells.items() if pat.fullmatch(c)) for kind, pat in KINDS.items()
    }


def area() -> dict:
    return {top: yosys_cells(top) for top in TOPS}


def mode(found: collections.Counter) -> float:
    """The value found most often (the tables show every value)."""
    return found.most_common(1)[0][0]


def latencies(steps: tuple[perf.Step, ...], got: list[int]) -> dict[str, collections.Counter]:
    """Each latency of perf.vector_cycles as the commands that show it
    alone give it, value -> count, from each command's cycles past what it
    issues (perf.vector_issue): vector_startup_cycles for an elementwise
    command that stops; reduction_tail_cycles past the start-up for a
    reduction of one row that stops; reduction_row_cycles for one of more
    rows, each shorter than that; kv_store_latency_cycles past the start-up
    and its beats past its groups (a beat a cycle) for a store
    that stops; load_latency_cycles for a LOAD right behind a LOAD or an
    elementwise command done on time (the latency the two share is the
    start-up; after a reduction or a store it starts later)."""
    lanes = DESIGN["vector_elems_per_cycle"]
    issue = [perf.vector_issue(s, lanes) for s in steps]
    past = [g - n for g, (_, n) in zip(got, issue, strict=True)]
    plain = [s.kind not in ("load", *perf.STORE_KINDS, *perf.REDUCTION_KINDS) for s in steps]
    found = collections.defaultdict(collections.Counter)
    for s, x, is_plain in zip(steps, past, plain, strict=True):
        if s.stops and is_plain:
            found["vector_startup_cycles"][x] += 1
    startup = mode(found["vector_startup_cycles"])
    # Elementwise commands done when their own groups and start-up say, not
    # pushed back behind an earlier done.
    on_time = [
        is_plain and x == (startup if s.stops else 0)
        for s, x, is_plain in zip(steps, past, plain, strict=True)
    ]
    for i in range(1, len(steps)):
        p, s, x = steps[i - 1], steps[i], past[i]
        if s.kind in perf.REDUCTION_KINDS and s.stops and s.walk[0] == 1:
            found["reduction_tail_cycles"][x - startup] += 1
        elif s.kind in perf.STORE_KINDS and s.stops:  # the writer free
            beats = s.kv_write_bytes / perf.commands.BEAT_BYTES
            found["kv_store_latency_cycles"][x + issue[i][1] - startup - beats] += 1
        elif s.kind == "load" and not s.stops and (on_time[i - 1] or p.kind == "load"):
            found["load_latency_cycles"][x] += 1
    tail = mode(found["reduction_tail_cycles"])
    for s, (row_groups, _), g in zip(steps, issue, got, strict=True):
        rows = s.walk[0]
        if s.kind in perf.REDUCTION_KINDS and s.stops and rows > 1:
            row = (g - startup - tail - row_groups) / (rows - 1)
            if row > row_groups:
                found["reduction_row_cycles"][int(row) if row.is_integer() else row] += 1
    return found


def latency() -> dict:
    """Per position: the unit's cycles and the perf model's in total, the
    worst command against the model, the unit's cycles by kind, and the
    latencies found (latencies())."""
    out = {}
    import decode  # the verification stack, only here

    for position in decode.SMOLLM2_POSITIONS:
        got, model, steps = decode.smollm2_step(position)
        worst, off = decode.worst(got, model)
        by_kind = collections.Counter()
        for s, g in zip(steps, got, strict=True):
            by_kind[s.kind] += g
        out[str(position)] = {
            "cycles": sum(got),
            "model_cycles": sum(model),
            "worst": {
                "index": worst,
                "kind": steps[worst].kind,
                "cycles": got[worst],
                "model_cycles": model[worst],
                "off": off,
            },
            "cycles_by_kind": dict(by_kind),
            "latencies": {
                k: {str(v): n for v, n in c.items()} for k, c in latencies(steps, got).items()
            },
        }
    return out


def latency_tables(data: dict) -> str:
    """The latencies found at each position, and each position's cycles."""
    positions = [k for k in data if k != "provenance"]
    names = sorted({n for p in positions for n in data[p]["latencies"]})
    rows = [
        [
            n,
            *(
                "; ".join(f"{v} ({c}x)" for v, c in data[p]["latencies"].get(n, {}).items()) or "-"
                for p in positions
            ),
        ]
        for n in names
    ]
    head = ["latency (cycles found, count)", *(f"position {p}" for p in positions)]
    cycles = []
    for p in positions:
        d, w = data[p], data[p]["worst"]
        worst = f"{w['kind']}: {w['cycles']} against {w['model_cycles']:.0f} ({w['off']:+.1%})"
        cycles.append([p, f"{d['cycles']:,}", f"{d['model_cycles']:,.0f}", worst])
    head2 = ["position", "unit cycles", "perf model", "worst command"]
    return blocks.table(head, rows) + "\n\n" + blocks.table(head2, cycles)


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
    parser.add_argument("part", choices=("area", "latency", "show"))
    args = parser.parse_args()
    data = json.loads(DATA.read_text()) if DATA.exists() else {}
    if args.part != "show":
        data[args.part] = {"provenance": measure.provenance()} | globals()[args.part]()
        DATA.write_text(json.dumps(data, indent=1) + "\n")
    if "area" in data:
        print(table(data["area"]))
    if "latency" in data:
        print(latency_tables(data["latency"]))
    if not data:
        sys.exit(f"no data in {DATA}: run the `area` or `latency` part first")


if __name__ == "__main__":
    main()
