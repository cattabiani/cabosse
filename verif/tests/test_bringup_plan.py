# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""scripts/f2/bringup_plan.py, M8's measurements on F2, against a stand-in
for the host tool: every step runs and is recorded with its provenance, the
clock is set before any register access, and a failing command is recorded,
not fatal. The plan's numbers come only from F2."""

import json
import subprocess
import sys
from pathlib import Path

from paths import REPO

PLAN = REPO / "scripts" / "f2" / "bringup_plan.py"
# The tool's stand-in: a command and its answer; FAIL_ON makes one command fail.
FAKE_TOOL = """#!{python}
import json, os, sys
cmd = sys.argv[1]
with open(os.environ["CALLS"], "a") as f:
    f.write(cmd + "\\n")
if cmd == os.environ.get("FAIL_ON"):
    print(json.dumps({{"error": "stand-in failure"}}))
    sys.exit(1)
if cmd == "run":
    print(json.dumps({{"ports": [{{"rd_first": 70}}]}}))
else:
    print(json.dumps({{"ok": cmd}}))
"""


def run_plan(tmp_path: Path, fail_on: str = "") -> tuple[list[dict], list[str]]:
    tool = tmp_path / "bringup"
    tool.write_text(FAKE_TOOL.format(python=sys.executable))
    tool.chmod(0o755)
    out, calls = tmp_path / "results.jsonl", tmp_path / "calls.txt"
    env = {"CALLS": str(calls), "FAIL_ON": fail_on, "PROVENANCE": json.dumps({"commit": "abc"})}
    result = subprocess.run(
        [sys.executable, str(PLAN), str(tool), str(out), "--quick"],
        capture_output=True, text=True, env=env, timeout=600,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    lines = [json.loads(ln) for ln in out.read_text().splitlines()]
    return lines, calls.read_text().split()


def test_plan_runs_every_step(tmp_path: Path):
    lines, calls = run_plan(tmp_path)
    assert calls[0] == "clock"  # before any register access
    steps = {ln["step"] for ln in lines}
    assert {
        "info",
        "ocl",
        "pcis_write",
        "pcis_check",
        "pcis_read",
        "fill",
        "read_check_all",
    } <= steps
    assert {"read_one", "read_burst", "read_limit", "write_limit", "switch_other_stack"} <= steps
    assert {"write_partial_1beat", "read_latency", "done"} <= steps
    assert all(ln["commit"] == "abc" and "time" in ln for ln in lines)
    assert sum(ln["step"] == "read_one" for ln in lines) == 32 * 3  # every port, every clock
    assert [ln["params"]["mhz"] for ln in lines if ln["step"] == "clock"] == [
        250,
        250,
        200,
        125,
        250,
    ]


def test_plan_records_failures(tmp_path: Path):
    lines, _ = run_plan(tmp_path, fail_on="pcis-read")
    reads = [ln for ln in lines if ln["step"] == "pcis_read"]
    assert reads and all(ln["result"] == {"error": "stand-in failure"} for ln in reads)
    assert lines[-1]["step"] == "done"
