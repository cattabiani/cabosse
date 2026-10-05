# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Fill docs/perf.md's tables from the perf model (model/perf.py) and its
input files (model/configs, model/platforms, model/designs). Runs in about a
second and needs no weights. After changing an input file:
    python scripts/perf_report.py            # rewrite docs/perf.md
    python scripts/perf_report.py --check    # fail if docs/perf.md is stale
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

import perf_report  # noqa: E402
import report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report.write_or_check(perf_report.PERF_DOC, perf_report.render(), args.check)


if __name__ == "__main__":
    main()
