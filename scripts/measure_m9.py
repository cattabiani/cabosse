# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Measurements for M9 (PLAN.md) into reports/data/M9.json, each part
replacing its own key:

    python scripts/measure_m9.py memory  # the core's on-chip memories (D-069)
    python scripts/measure_m9.py show    # print the tables from the saved data

`memory` elaborates rtl/core.sv in Yosys at SmolLM2's sizing (front end
only, under a minute, about 2 GB) and lists every memory it infers. The
block RAM and UltraRAM counts are estimates: each memory alone, in the
primitive's best shape; Vivado chooses the mapping, and the build's
utilization report has the real counts. Commit first: the data records the
commit.
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for sub in ("model", "verif"):
    sys.path.insert(0, str(REPO / sub))

from reporting import measure  # noqa: E402

import rtl  # noqa: E402  (puts .tools/ on PATH)

DATA = REPO / "reports" / "data" / "M9.json"
# M7's SmolLM2 runs (verif/generation.py): one Len for every buffer, set by
# LOGITS (the vocabulary, 49,152); 64 MiB channels.
CORE_PARAMS = (("Ports", 32), ("Len", 49152), ("ChanW", 26))
SMALL_BITS = 16384  # below a RAMB18: LUT RAM or registers, left out of the estimates
RAMB36_SHAPES = ((72, 512), (36, 1024), (18, 2048), (9, 4096))  # width, depth
URAM_SHAPE = (72, 4096)
DEVICE = {
    "RAMB36": {
        "count": 2016,
        "source": "Vivado, bring-up build 4's utilization"
        " (reports/data/f2/2026-11-03-bringup-build4.txt)",
    },
    "URAM288": {"count": 960, "source": "VU47P datasheet, to verify in M9's build report"},
}


def ramb36(width: int, depth: int) -> int:
    return min(math.ceil(width / w) * math.ceil(depth / d) for w, d in RAMB36_SHAPES)


def uram(width: int, depth: int) -> int:
    return math.ceil(width / URAM_SHAPE[0]) * math.ceil(depth / URAM_SHAPE[1])


def memory() -> dict:
    rows = [
        {"name": name, "width_bits": w, "depth": d, "count": n}
        for (name, w, d), n in rtl.memories("core", CORE_PARAMS).items()
    ]
    rows.sort(key=lambda r: -r["width_bits"] * r["depth"] * r["count"])
    big = [r for r in rows if r["width_bits"] * r["depth"] >= SMALL_BITS]
    return {
        "params": dict(CORE_PARAMS),
        "memories": rows,
        "total_bytes": sum(r["width_bits"] * r["depth"] * r["count"] for r in rows) // 8,
        "large_bytes": sum(r["width_bits"] * r["depth"] * r["count"] for r in big) // 8,
        "estimate_ramb36_only": sum(r["count"] * ramb36(r["width_bits"], r["depth"]) for r in big),
        "estimate_uram_only": sum(r["count"] * uram(r["width_bits"], r["depth"]) for r in big),
        "device": DEVICE,
    }


def show(data: dict) -> None:
    m = data["memory"]
    print(f"core {m['params']}, commit {m['provenance']['commit']}")
    print("| memory | width (bits) | depth | count | KiB |")
    print("|---|---:|---:|---:|---:|")
    for r in m["memories"]:
        kib = r["width_bits"] * r["depth"] * r["count"] / 8 / 1024
        if kib >= 1:
            print(
                f"| `{r['name']}` | {r['width_bits']} | {r['depth']} | {r['count']} | {kib:,.1f} |"
            )
    mib = 2**20
    print(
        f"\nall memories {m['total_bytes'] / mib:.2f} MiB; those of a RAMB18 or more "
        f"{m['large_bytes'] / mib:.2f} MiB"
    )
    for kind, key in (("RAMB36", "estimate_ramb36_only"), ("URAM288", "estimate_uram_only")):
        have = m["device"][kind]["count"]
        print(f"all in {kind}: {m[key]} of {have} ({m[key] / have:.0%}; estimate)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=("memory", "show"))
    args = parser.parse_args()
    data = json.loads(DATA.read_text()) if DATA.exists() else {}
    if args.part != "show":
        data[args.part] = {"provenance": measure.provenance()} | globals()[args.part]()
        DATA.write_text(json.dumps(data, indent=1) + "\n")
    show(data)


if __name__ == "__main__":
    main()
