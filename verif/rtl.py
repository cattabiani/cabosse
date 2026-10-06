# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Run the RTL checks from pytest: cocotb testbenches under Verilator, the
Verilator lint and a Yosys synthesis check.

The OSS CAD Suite in .tools/ (README, "Development setup") goes on PATH after
the venv's own bin, so the venv's cocotb comes first. Without Verilator or
Yosys the tests that need them skip.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner

REPO = Path(__file__).resolve().parent.parent
RTL = REPO / "rtl"
BUILD = REPO / "verif" / "sim_build"  # git-ignored
TOOLS_BIN = REPO / ".tools" / "oss-cad-suite" / "bin"
# The golden model and the testbench modules, for the Python inside the simulator.
PYTHONPATH = [REPO / "model", REPO / "verif", REPO / "verif" / "tests"]

if TOOLS_BIN.is_dir():
    os.environ["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), str(TOOLS_BIN), os.environ["PATH"]]
    )

needs_verilator = pytest.mark.skipif(shutil.which("verilator") is None, reason="no verilator")
needs_yosys = pytest.mark.skipif(shutil.which("yosys") is None, reason="no yosys")


def sources() -> list[Path]:
    return sorted(RTL.glob("*.sv"))


def simulate(top: str, test_module: str) -> None:
    """Build `top` (one module per file: rtl/<top>.sv) and run the cocotb tests
    in verif/tests/<test_module>.py against it; a failing test fails here."""
    runner = get_runner("verilator")
    build_dir = BUILD / top
    runner.build(sources=[RTL / f"{top}.sv"], hdl_toplevel=top, build_dir=build_dir, always=True)
    pythonpath = os.pathsep.join(str(p) for p in PYTHONPATH)
    runner.test(
        hdl_toplevel=top,
        test_module=test_module,
        build_dir=build_dir,
        test_dir=build_dir,
        extra_env={"PYTHONPATH": pythonpath},
    )


def lint(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["verilator", "--lint-only", "-Wall", str(path)], capture_output=True, text=True
    )


def synthesize(path: Path) -> subprocess.CompletedProcess:
    """Generic Yosys synthesis of one module; `check -assert` fails on
    problems such as latches or undriven signals."""
    script = f"read_slang {path}; synth -top {path.stem}; check -assert"
    return subprocess.run(
        ["yosys", "-q", "-m", "slang", "-p", script], capture_output=True, text=True
    )
