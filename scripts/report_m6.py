# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The M6 checkpoint report, reports/M6.md: its numbers are generated, never
typed (model/reporting/m6.py). The vector unit's area and latencies come
from scripts/measure_m6.py.

    python scripts/report_m6.py measure-tests   # the test suites -> reports/data/M6.json
    python scripts/report_m6.py render          # fills reports/M6.md from the data
    python scripts/report_m6.py render --check  # fails if M6.md is stale

`measure-tests` runs the fast suite and the slow one (it needs the SmolLM2
checkpoint, the long cache and the RTL tools), and replaces the data's
`tests` key. Run it under the memory cap. Commit first: the data records the
commit it was measured on.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for sub in ("model", "verif"):
    sys.path.insert(0, str(REPO / sub))

from reporting import blocks, m6, measure  # noqa: E402

import rtl  # noqa: E402, F401  (puts .tools/ on PATH for the suites)

UP_TO_DATE = "model/tests/test_reporting.py::test_m6_report_is_up_to_date"
SUITES = {"fast": ["--deselect", UP_TO_DATE], "slow": ["-m", "slow"]}


def measure_tests() -> None:
    tests = {"provenance": measure.provenance()}
    for name, extra in SUITES.items():
        tests[name] = measure.pytest_summary(extra)
        print(name, tests[name], flush=True)
    data = json.loads(m6.DATA.read_text())
    data["tests"] = tests
    m6.DATA.write_text(json.dumps(data, indent=1) + "\n")
    print(f"written: {m6.DATA}")


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
        blocks.write_or_check(m6.PATH, m6.render(), args.check)


if __name__ == "__main__":
    main()
