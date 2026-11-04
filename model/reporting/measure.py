# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""What the report scripts (scripts/report_m*.py) share when they measure:
where the data comes from, and pytest's summary of a suite."""

import datetime
import os
import platform
import subprocess
import sys
from pathlib import Path

import paths


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=paths.REPO, capture_output=True, text=True, check=True
    ).stdout


def commit_and_dirty() -> tuple[str, bool]:
    """The checked-out commit, and whether there are uncommitted changes
    outside reports/data/ (a script that measures in parts writes its data
    file after each one)."""
    status = git("status", "--porcelain", "--", ".", ":!reports/data")
    commit, dirty = git("rev-parse", "HEAD").strip(), bool(status.strip())
    if dirty:
        print("warning: uncommitted changes; the report will say so")
    return commit, dirty


def provenance() -> dict:
    """Where the data comes from: commit, uncommitted changes, date, machine
    (the keys blocks.provenance_text reads)."""
    commit, dirty = commit_and_dirty()
    return {
        "commit": commit,
        "dirty": dirty,
        "date": datetime.date.today().isoformat(),
        "machine": f"{platform.system()} {platform.machine()}",
    }


def host() -> dict:
    """What a timing on this machine ran on: CPU, torch's threads, the load
    average before measuring (1, 5, 15 min) and the software's versions."""
    import torch  # only for the timings
    import transformers

    cpuinfo = Path("/proc/cpuinfo").read_text().splitlines()  # Linux: the reference platform
    cpu = next(line.split(":", 1)[1].strip() for line in cpuinfo if line.startswith("model name"))
    return {
        "cpu": cpu,
        "threads": torch.get_num_threads(),
        "load_average": list(os.getloadavg()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def pytest_summary(extra: list[str]) -> str:
    """The last line of a pytest run, e.g. '166 passed, 5 deselected in 38.8s'."""
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *extra],
        cwd=paths.REPO,
        capture_output=True,
        text=True,
    )
    return run.stdout.strip().splitlines()[-1].strip("= ")
