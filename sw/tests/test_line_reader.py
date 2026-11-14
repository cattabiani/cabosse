# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The F2 host's PCIS read plans (platforms/f2/host/line_reader.h, M11 step
3) on plain memory: each load and thread count copies every byte of whole
lines and nothing past them, read after read; plan texts parse or are
refused. The rates are F2's (pcis_bench.cpp)."""

import platform
import subprocess
from pathlib import Path

import bulk
import pytest

import rtl

CHECK = Path(__file__).parent / "line_reader_check.cpp"
X86 = platform.machine() in ("x86_64", "AMD64")
LOADS = ["memcpy", "load32", "stream32"] if X86 else ["memcpy"]
# Sizes in lines: one, fewer than the threads, not a multiple of the
# unrolled 4 lines, and SmolLM2's logits (49,152 words).
SIZES = [64, 3 * 64, 7 * 64, 4096 + 64, 196_608]
# A C++23 compiler with <print>, as the RTL tests' (CI's rtl job sets CXX).
pytestmark = rtl.needs_verilator


@pytest.fixture(scope="module")
def check() -> Path:
    return bulk.build_program("line_reader_check", (CHECK,))


@pytest.mark.parametrize("threads", [1, 2, 5, 8])
@pytest.mark.parametrize("load", LOADS)
def test_copy(check: Path, load: str, threads: int) -> None:
    run = subprocess.run(
        [check, "copy", f"{load}:{threads}", *map(str, SIZES)], capture_output=True
    )
    assert run.returncode == 0, run.stderr.decode()


@pytest.mark.parametrize(
    "text, plan",
    [
        ("memcpy", "memcpy:1"),
        ("memcpy:16", "memcpy:16"),
        ("load32:4", "load32:4" if X86 else "none"),
        ("stream32:2", "stream32:2" if X86 else "none"),
        ("memcpy:0", "none"),
        ("memcpy:17", "none"),
        ("memcpy:", "none"),
        ("memcpy:x", "none"),
        ("load64:1", "none"),
    ],
)
def test_parse(check: Path, text: str, plan: str) -> None:
    run = subprocess.run([check, "parse", text], capture_output=True, text=True, check=True)
    assert run.stdout.strip() == plan
