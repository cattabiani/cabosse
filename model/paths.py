# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Where things live on disk, and which platform pins values. Weights are
downloaded into the git-ignored weights/ directory, or wherever
CABOSSE_WEIGHTS points."""

import os
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = Path(os.environ.get("CABOSSE_WEIGHTS", REPO / "weights"))
SMOLLM2 = WEIGHTS / "SmolLM2-135M-Instruct"
SMOLLM2_BASE = WEIGHTS / "SmolLM2-135M"
RUNS = REPO / "runs"
REPORTS = REPO / "reports"
FIXTURES = REPO / "model" / "tests" / "fixtures"
# Perf model inputs (model/perf.py): model configs, platforms, designs.
CONFIGS = REPO / "model" / "configs"
PLATFORMS = REPO / "model" / "platforms"
DESIGNS = REPO / "model" / "designs"

# Values pinned to exact bits (table hashes, fixtures) come from here (D-023).
REFERENCE_PLATFORM = sys.platform == "linux" and platform.machine() == "x86_64"
REFERENCE_PLATFORM_NOTE = "pinned values come from x86-64 Linux only (D-023)"
