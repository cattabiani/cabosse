# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""What the report scripts (scripts/report_m*.py) share when they measure:
the commit the data comes from, and pytest's summary of a suite."""

import subprocess
import sys

import paths


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=paths.REPO, capture_output=True, text=True, check=True
    ).stdout


def commit_and_dirty() -> tuple[str, bool]:
    """The checked-out commit, and whether there are uncommitted changes."""
    commit, dirty = git("rev-parse", "HEAD").strip(), bool(git("status", "--porcelain").strip())
    if dirty:
        print("warning: uncommitted changes; the report will say so")
    return commit, dirty


def pytest_summary(extra: list[str]) -> str:
    """The last line of a pytest run, e.g. '166 passed, 5 deselected in 38.8s'."""
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *extra],
        cwd=paths.REPO,
        capture_output=True,
        text=True,
    )
    return run.stdout.strip().splitlines()[-1].strip("= ")
