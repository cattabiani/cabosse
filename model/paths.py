# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Where things live on disk. Weights are downloaded into the git-ignored
weights/ directory, or wherever CABOSSE_WEIGHTS points."""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = Path(os.environ.get("CABOSSE_WEIGHTS", REPO / "weights"))
SMOLLM2 = WEIGHTS / "SmolLM2-135M-Instruct"
SMOLLM2_BASE = WEIGHTS / "SmolLM2-135M"
RUNS = REPO / "runs"
