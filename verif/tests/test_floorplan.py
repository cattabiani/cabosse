# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""scripts/f2/floorplan.py: the committed XDC is its output, every engine
port and lane has one SLR, and a group's lanes share their DMA's."""

import re

import floorplan

LANE_OR_DMA = re.compile(r"SLR(\d) \[get_cells \{.*/u_engine/gen_(lane|dma)\?(\d+)\?\.\*\}\]")


def test_committed_file_is_current() -> None:
    assert floorplan.OUT.read_text() == "\n".join(floorplan.lines()) + "\n"


def test_lanes_with_their_dma() -> None:
    slr: dict[tuple[str, int], int] = {}
    for line in floorplan.lines():
        m = LANE_OR_DMA.search(line)
        if m:
            key = (m[2], int(m[3]))
            assert key not in slr, f"{key} assigned twice"
            slr[key] = int(m[1])
    assert sorted(i for kind, i in slr if kind == "dma") == list(range(floorplan.PORTS))
    lanes = floorplan.PORTS * floorplan.LANES_PER_PORT
    assert sorted(i for kind, i in slr if kind == "lane") == list(range(lanes))
    for (kind, i), s in slr.items():
        if kind == "lane":
            assert s == slr[("dma", i // floorplan.LANES_PER_PORT)]
