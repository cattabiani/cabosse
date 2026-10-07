# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Every RTL module passes the Verilator lint with no warnings and
synthesizes in Yosys (AGENTS.md, SystemVerilog conventions), as the top of
all of rtl/. One module per file, named after it."""

import pytest

import rtl

MODULES = rtl.modules()
# (top, extra files): our modules, then the harnesses in platforms/.
TOPS = [(m, ()) for m in MODULES] + [(p.stem, (p,)) for p in rtl.harnesses()]


def test_there_is_rtl() -> None:
    assert MODULES


def test_sources_lists_every_rtl_file() -> None:
    """rtl/sources.f names every .sv file under rtl/, and nothing else."""
    lines = [ln.strip() for ln in rtl.SOURCES.read_text().splitlines()]
    listed = {rtl.RTL / ln for ln in lines if ln.endswith(".sv")}
    assert listed == set(rtl.RTL.rglob("*.sv"))


@rtl.needs_verilator
@pytest.mark.parametrize(("top", "extra"), TOPS, ids=[t for t, _ in TOPS])
def test_lint_is_clean(top: str, extra: tuple) -> None:
    result = rtl.lint(top, extra)
    assert result.returncode == 0 and not result.stderr.strip(), result.stderr


@rtl.needs_yosys
@pytest.mark.parametrize(("top", "extra"), TOPS, ids=[t for t, _ in TOPS])
def test_synthesizes(top: str, extra: tuple) -> None:
    result = rtl.synthesize(top, extra)
    assert result.returncode == 0, result.stdout + result.stderr
