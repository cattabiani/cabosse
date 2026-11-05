# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The M9 checkpoint report, reports/M9.md: its numbers are generated, never
typed (model/reporting/m9.py), from the F2 runs' logs in reports/data/f2/,
reports/data/M9.json and the perf model.

    python scripts/report_m9.py          # fills reports/M9.md from the data
    python scripts/report_m9.py --check  # fails if M9.md is stale
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "model"))

from reporting import blocks, m9  # noqa: E402

if __name__ == "__main__":
    blocks.main(__doc__, m9.PATH, m9.render)
