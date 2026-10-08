# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Report generation (model/reporting/): generated blocks, and the committed
reports agree with their data."""

import re

import paths
import pytest
from reporting import blocks, m1, m2, m3_subnormal

TEXT = """# Title

prose 1.5
<!-- begin: a -->
old
<!-- end: a -->
middle
<!-- begin: b -->
<!-- end: b -->
"""


def test_fill_replaces_only_the_blocks() -> None:
    out = blocks.fill(TEXT, {"a": "new a", "b": "| x |"})
    assert out == TEXT.replace("old\n", "new a\n").replace(
        "<!-- begin: b -->\n", "<!-- begin: b -->\n| x |\n"
    )
    assert blocks.fill(out, {"a": "new a", "b": "| x |"}) == out  # idempotent


@pytest.mark.parametrize("given", [{"a": "x"}, {"a": "x", "b": "y", "c": "z"}])
def test_fill_needs_exactly_the_blocks_in_the_text(given: dict[str, str]) -> None:
    with pytest.raises(AssertionError):
        blocks.fill(TEXT, given)


@pytest.mark.parametrize(
    ("summary", "passed"),
    [
        ("166 passed, 5 deselected in 38.82s", True),
        ("5 passed, 165 deselected in 727.44s (0:12:07)", True),
        ("1 failed, 165 passed in 40.00s", False),
        ("165 passed, 1 error in 40.00s", False),
        ("no tests ran in 0.01s", False),
    ],
)
def test_tests_passed(summary: str, passed: bool) -> None:
    assert m1.tests_passed(summary) == passed


def test_m1_report_is_up_to_date() -> None:
    """reports/M1.md shows what reports/data/M1.json says; after a new
    measurement, run: python scripts/report_m1.py render"""
    assert m1.render() == m1.PATH.read_text()


def test_m2_report_is_up_to_date() -> None:
    """reports/M2.md shows the current perf model, command list and
    comparison data; after a change, run: python scripts/report_m2.py"""
    assert m2.render() == m2.PATH.read_text()


def test_m3_subnormal_report_is_up_to_date() -> None:
    """reports/M3-interim-subnormal.md shows what its Yosys log says; after a
    new measurement, run: python scripts/report_m3_subnormal.py render"""
    assert m3_subnormal.render() == m3_subnormal.PATH.read_text()


def test_measure_deselects_a_test_that_exists() -> None:
    """scripts/report_m1.py measure leaves out the up-to-date test by its pytest
    id; pytest ignores an id that matches nothing, so check it here."""
    script = (paths.REPO / "scripts" / "report_m1.py").read_text()
    path, name = re.search(r'UP_TO_DATE = "(.+)::(\w+)"', script).groups()
    assert f"def {name}(" in (paths.REPO / path).read_text()
