# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""scripts/f2/floorplan.py: the committed XDC is its output, every engine
port and lane has one SLR, a group's lanes share their DMA's, and in SLR1
only the host's stages may use the corridor (column X5)."""

import re

import floorplan

LANE_OR_DMA = re.compile(
    r"pblock_slr(\d)(?:_core)? \[get_cells \{.*/u_engine/gen_(lane|dma)\?(\d+)\?\.\*\}\]"
)
IN_PBLOCK = re.compile(r"add_cells_to_pblock (\S+) \[get_cells .*\{(\S+)\}\]")


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


def test_slr1_corridor() -> None:
    """SLR1's child pblock is clock regions west of the corridor, and the
    only cells placed in SLR1's pblock outside it are the host's stages."""
    text = floorplan.pblocks()
    ranges = re.search(rf"resize_pblock {floorplan.SLR1_CORE} -add \{{(.*)\}}", text)[1].split()
    assert ranges, text
    for r in ranges:
        m = re.fullmatch(r"CLOCKREGION_X(\d+)Y(\d+):CLOCKREGION_X(\d+)Y(\d+)", r)
        assert m and int(m[3]) < floorplan.CORRIDOR_X and 4 <= int(m[2]) <= int(m[4]) <= 7, r
    outside = [
        m[2]
        for line in floorplan.lines()
        if (m := IN_PBLOCK.search(line)) and m[1] == "pblock_slr1"
    ]
    assert outside and all("/u_host/" in c for c in outside), outside
