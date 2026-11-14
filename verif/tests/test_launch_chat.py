# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""scripts/f2/launch_chat.sh's user data (DRY_RUN=1, no AWS call):
run_core.sh with each launch setting replaced in its own line only, still
valid bash."""

import os
import shutil
import subprocess

import pytest

import rtl

SCRIPT = rtl.REPO / "scripts" / "f2" / "launch_chat.sh"
SETTINGS = {"MHZ": "190", "DEADLINE_MIN": "45", "PCIS_READ": "stream32:4", "REF": "m11/x"}


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_user_data() -> None:
    env = os.environ | SETTINGS | {"DRY_RUN": "1"}
    run = subprocess.run([SCRIPT, "t"], env=env, capture_output=True, text=True, check=True)
    lines = run.stdout.splitlines()
    want = SETTINGS | {"INTERACTIVE": "1", "BENCH": "1", "AGFI": "agfi-00e8542765d44a754"}
    for var, value in want.items():
        found = [line.split()[0] for line in lines if line.startswith(f"{var}=")]
        assert found == [f"{var}={value}"], var
    s3 = [line for line in lines if line.startswith("S3=")]
    assert len(s3) == 1 and s3[0].split()[0].endswith("-t")
    assert "env S3=$S3/chat-manual/" in run.stdout  # chat.sh's own S3, untouched
    subprocess.run(["bash", "-n"], input=run.stdout, text=True, check=True)
