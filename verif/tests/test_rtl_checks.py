# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Every RTL module passes the Verilator lint with no warnings and
synthesizes in Yosys (AGENTS.md, SystemVerilog conventions), as the top of
all of rtl/. One module per file, named after it."""

from pathlib import Path

import pytest

import rtl


def test_there_is_rtl() -> None:
    assert rtl.sources()


MODULES = [Path(p).stem for p in rtl.sources()]


@rtl.needs_verilator
@pytest.mark.parametrize("top", MODULES)
def test_lint_is_clean(top: str) -> None:
    result = rtl.lint(top)
    assert result.returncode == 0 and not result.stderr.strip(), result.stderr


@rtl.needs_yosys
@pytest.mark.parametrize("top", MODULES)
def test_synthesizes(top: str) -> None:
    result = rtl.synthesize(top)
    assert result.returncode == 0, result.stdout + result.stderr
