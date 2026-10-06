# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Fill reports/M2.md's tables from the perf model, the command list and the
comparison runs in reports/data/. Needs no weights:
    python scripts/report_m2.py            # rewrite reports/M2.md
    python scripts/report_m2.py --check    # fail if reports/M2.md is stale
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from reporting import blocks, m2  # noqa: E402

if __name__ == "__main__":
    blocks.main(__doc__, m2.PATH, m2.render)
