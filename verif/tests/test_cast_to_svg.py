# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""scripts/cast_to_svg.py: runs on rows by CR, LF and wrapping, user lines
marked, and a well-formed SVG with each run shown at its time."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import cast_to_svg  # noqa: E402

EVENTS = [[0.0, "o", "> hi\r\n\r\n"], [0.5, "o", "abc"], [0.6, "o", "de\r\n"], [0.7, "i", "x"]]


def test_runs_fall_on_their_rows_and_wrap() -> None:
    runs = cast_to_svg.place(EVENTS, width=4)
    assert runs == [
        (0.0, 0, 0, "> hi", True),
        (0.5, 2, 0, "abc", False),
        (0.6, 2, 3, "d", False),
        (0.6, 3, 0, "e", False),
    ]


def test_the_svg_is_well_formed_and_timed() -> None:
    root = ET.fromstring(cast_to_svg.svg({"width": 4, "title": "t"}, EVENTS))
    texts = root.findall("{http://www.w3.org/2000/svg}text")
    assert [t.text for t in texts] == ["> hi", "abc", "d", "e"]
    shows = [t[1].get("begin") for t in texts]
    assert shows == ["clock.begin+0.000s", "clock.begin+0.500s"] + ["clock.begin+0.600s"] * 2
