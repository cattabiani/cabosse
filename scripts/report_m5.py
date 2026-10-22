# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The M5 checkpoint report, reports/M5.md: its numbers are generated, never
typed (model/reporting/m5.py). The engine's measurements come from
scripts/measure_m5_engine.py.

    python scripts/report_m5.py measure-tests   # the test suites -> reports/data/M5.json
    python scripts/report_m5.py render          # fills reports/M5.md from the data
    python scripts/report_m5.py render --check  # fails if M5.md is stale

`measure-tests` runs the fast suite (the 128-lane tests included) and the
slow one (it needs the SmolLM2 checkpoint, the long cache and the RTL
tools); about 35 minutes on 16 cores. Run it under the memory cap. Commit
first: the data records the commit it was measured on.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for sub in ("model", "verif"):
    sys.path.insert(0, str(REPO / sub))

from reporting import blocks, m5, measure  # noqa: E402
from reporting.measure import pytest_summary  # noqa: E402

import rtl  # noqa: E402, F401  (puts .tools/ on PATH for the suites)

UP_TO_DATE = "model/tests/test_reporting.py::test_m5_report_is_up_to_date"
SUITES = {"fast": ["--deselect", UP_TO_DATE], "slow": ["-m", "slow"]}


def measure_tests() -> None:
    data = {"provenance": measure.provenance(), "tests": {}}
    for name, extra in SUITES.items():
        data["tests"][name] = pytest_summary(extra)
        print(name, data["tests"][name], flush=True)
    m5.DATA.write_text(json.dumps(data, indent=1) + "\n")
    print(f"written: {m5.DATA}")


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
        blocks.write_or_check(m5.PATH, m5.render(), args.check)


if __name__ == "__main__":
    main()
