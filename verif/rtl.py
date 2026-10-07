# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Run the RTL checks from pytest: cocotb testbenches under Verilator, the
Verilator lint and a Yosys synthesis check. Every tool reads all of rtl/ from
rtl/sources.f and is told the top module, so blocks can instantiate each
other.

The OSS CAD Suite in .tools/ goes on PATH for this process (the cocotb runner
finds Verilator there). The simulator's Python imports what pytest's sys.path
has (pyproject.toml, `pythonpath`): the runner passes it on. Without the tools
the RTL tests skip, unless CABOSSE_REQUIRE_RTL=1 makes that a failure.
"""

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner
from paths import REPO

RTL = REPO / "rtl"
SOURCES = RTL / "sources.f"  # every RTL file in compile order
VENDOR_LINT = RTL / "vendor" / "lint.vlt"  # vendored code is not held to -Wall
BUILD = REPO / "verif" / "sim_build"  # git-ignored
TOOLS_BIN = REPO / ".tools" / "oss-cad-suite" / "bin"
VERILATOR_FILES = [str(VENDOR_LINT), "-F", str(SOURCES)]

if TOOLS_BIN.is_dir():
    os.environ["PATH"] = os.pathsep.join([str(TOOLS_BIN), os.environ["PATH"]])


def needs(tool: str) -> pytest.MarkDecorator:
    missing = shutil.which(tool) is None
    if missing and os.environ.get("CABOSSE_REQUIRE_RTL") == "1":
        raise RuntimeError(f"CABOSSE_REQUIRE_RTL=1 but {tool} is not on PATH")
    return pytest.mark.skipif(missing, reason=f"no {tool}")


needs_verilator, needs_yosys = needs("verilator"), needs("yosys")


def modules() -> list[str]:
    """Our modules: one per file in rtl/ (not rtl/vendor/), named after it;
    packages (*_pkg.sv) are not modules."""
    return [p.stem for p in sorted(RTL.glob("*.sv")) if not p.stem.endswith("_pkg")]


def harnesses() -> list[Path]:
    """Test-only modules in platforms/ built on rtl/ (such as timing
    harnesses): one per file, named after it."""
    return sorted((REPO / "platforms").rglob("*.sv"))


def simulate(top: str, test_module: str | None = None) -> None:
    """Build rtl/ with `top` as the top module and run the cocotb tests in
    verif/tests/<test_module>.py (default test_<top>) against it; a failing
    test fails here."""
    runner = get_runner("verilator")
    build_dir = BUILD / top
    runner.build(build_args=VERILATOR_FILES, hdl_toplevel=top, build_dir=build_dir)
    runner.test(
        hdl_toplevel=top,
        hdl_toplevel_lang="verilog",  # the runner cannot infer it from a file list
        test_module=test_module or f"test_{top}",
        build_dir=build_dir,
    )


def lint(top: str, extra: Sequence[Path] = ()) -> subprocess.CompletedProcess:
    """Verilator -Wall lint of rtl/ plus `extra` files, with `top` as the top."""
    argv = ["verilator", "--lint-only", "-Wall", "--top-module", top, *VERILATOR_FILES]
    argv += [str(p) for p in extra]
    return subprocess.run(argv, capture_output=True, text=True)


def synthesize(top: str, extra: Sequence[Path] = ()) -> subprocess.CompletedProcess:
    """Generic Yosys synthesis of rtl/ plus `extra` files with `top` as the
    top module; `check -assert` fails on problems such as latches or undriven
    signals."""
    files = " ".join(str(p) for p in extra)
    script = f"read_slang -F {SOURCES} {files} --top {top}; synth -top {top}; check -assert"
    return subprocess.run(
        ["yosys", "-q", "-m", "slang", "-p", script], capture_output=True, text=True
    )
