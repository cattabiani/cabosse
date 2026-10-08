# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The M4 checkpoint report, reports/M4.md: its numbers are generated, never
typed (model/reporting/m4.py). The Vivado logs come from scripts/f2/.

    python scripts/report_m4.py measure-tests   # the test suites -> reports/data/M4.json
    python scripts/report_m4.py render          # fills reports/M4.md from the data
    python scripts/report_m4.py render --check  # fails if M4.md is stale

`measure-tests` runs the fast and slow suites (the slow one needs the
SmolLM2 checkpoint and the RTL tools) and takes about 20 minutes; run it
under the memory cap. Commit first: the data records the commit it was
measured on.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for sub in ("model", "model/tests", "verif", "verif/tests"):
    sys.path.insert(0, str(REPO / sub))

import lane  # noqa: E402
import paths  # noqa: E402
import test_dot_lane_bulk  # noqa: E402
from golden import decoder  # noqa: E402
from reporting import (  # noqa: E402
    blocks,
    m4,
    measure,  # noqa: E402
)
from reporting.measure import pytest_summary  # noqa: E402

import rtl  # noqa: E402, F401  (puts .tools/ on PATH for the suites)

UP_TO_DATE = "model/tests/test_reporting.py::test_m4_report_is_up_to_date"
SUITES = {"fast": ["--deselect", UP_TO_DATE], "slow": ["-m", "slow"]}


def smollm2_counts() -> tuple[int, int]:
    """Rows and pairs of the slow suite's SmolLM2 test, counted on its rows."""
    found = test_dot_lane_bulk.smollm2_rows(decoder.load(paths.SMOLLM2))
    groups = [g for gs in found.values() for g in gs]
    return sum(g.w.shape[0] for g in groups), sum(g.w.size for g in groups)


def volume_counts() -> tuple[int, int]:
    """Chunks and pairs of the slow suite's volume test, counted on its rows."""
    seeds = test_dot_lane_bulk.volume_seeds()
    chunks = (test_dot_lane_bulk.volume_chunk(seed) for seed in seeds)
    return len(seeds), sum(g.w.size for groups in chunks for g in groups)


def measure_tests() -> None:
    rows, pairs = smollm2_counts()
    chunks, bulk_pairs = volume_counts()
    data = {
        "provenance": measure.provenance(),
        # What the tests run, from their own constants and rows.
        "inputs": {
            "bulk_pairs": bulk_pairs,
            "bulk_chunks": chunks,
            "smollm2_rows": rows,
            "smollm2_pairs": pairs,
            "full_rate_len": lane.FULL_RATE_LEN,
        },
        "tests": {},
    }
    for name, extra in SUITES.items():
        data["tests"][name] = pytest_summary(extra)
        print(name, data["tests"][name], flush=True)
    m4.DATA.write_text(json.dumps(data, indent=1) + "\n")
    print(f"written: {m4.DATA}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("measure-tests")
    sub.add_parser("render").add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.command == "measure-tests":
        measure_tests()
    else:
        blocks.write_or_check(m4.PATH, m4.render(), args.check)


if __name__ == "__main__":
    main()
