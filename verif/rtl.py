# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Run the RTL checks from pytest: cocotb testbenches under Verilator, the
Verilator lint and a Yosys synthesis check. Every tool gets all of rtl/ and
the name of the top module, so blocks can instantiate each other.

The OSS CAD Suite in .tools/ goes on PATH for this process (the cocotb runner
finds Verilator there). The simulator's Python imports what pytest's sys.path
has (pyproject.toml, `pythonpath`): the runner passes it on. Without the tools
the RTL tests skip, unless CABOSSE_REQUIRE_RTL=1 makes that a failure.
"""

import os
import shutil
import subprocess

import pytest
from cocotb_tools.runner import get_runner
from paths import REPO

RTL = REPO / "rtl"
BUILD = REPO / "verif" / "sim_build"  # git-ignored
TOOLS_BIN = REPO / ".tools" / "oss-cad-suite" / "bin"

if TOOLS_BIN.is_dir():
    os.environ["PATH"] = os.pathsep.join([str(TOOLS_BIN), os.environ["PATH"]])


def needs(tool: str) -> pytest.MarkDecorator:
    missing = shutil.which(tool) is None
    if missing and os.environ.get("CABOSSE_REQUIRE_RTL") == "1":
        raise RuntimeError(f"CABOSSE_REQUIRE_RTL=1 but {tool} is not on PATH")
    return pytest.mark.skipif(missing, reason=f"no {tool}")


needs_verilator, needs_yosys = needs("verilator"), needs("yosys")


def sources() -> list[str]:
    return [str(p) for p in sorted(RTL.glob("*.sv"))]


def simulate(top: str, test_module: str | None = None) -> None:
    """Build rtl/ with `top` as the top module and run the cocotb tests in
    verif/tests/<test_module>.py (default test_<top>) against it; a failing
    test fails here."""
    runner = get_runner("verilator")
    build_dir = BUILD / top
    runner.build(sources=sources(), hdl_toplevel=top, build_dir=build_dir, always=True)
    runner.test(hdl_toplevel=top, test_module=test_module or f"test_{top}", build_dir=build_dir)


def lint(top: str) -> subprocess.CompletedProcess:
    argv = ["verilator", "--lint-only", "-Wall", "--top-module", top, *sources()]
    return subprocess.run(argv, capture_output=True, text=True)


def synthesize(top: str) -> subprocess.CompletedProcess:
    """Generic Yosys synthesis with `top` as the top module; `check -assert`
    fails on problems such as latches or undriven signals."""
    script = f"read_slang {' '.join(sources())} --top {top}; synth -top {top}; check -assert"
    return subprocess.run(
        ["yosys", "-q", "-m", "slang", "-p", script], capture_output=True, text=True
    )
