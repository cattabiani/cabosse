# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Every RTL file passes the Verilator lint with no warnings and synthesizes
in Yosys (AGENTS.md, SystemVerilog conventions)."""

from pathlib import Path

import pytest

import rtl


def test_there_is_rtl() -> None:
    assert rtl.sources()


@rtl.needs_verilator
@pytest.mark.parametrize("path", rtl.sources(), ids=lambda p: p.name)
def test_lint_is_clean(path: Path) -> None:
    result = rtl.lint(path)
    assert result.returncode == 0 and not result.stderr.strip(), result.stderr


@rtl.needs_yosys
@pytest.mark.parametrize("path", rtl.sources(), ids=lambda p: p.name)
def test_synthesizes(path: Path) -> None:
    result = rtl.synthesize(path)
    assert result.returncode == 0, result.stdout + result.stderr
