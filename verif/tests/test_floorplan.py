# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""scripts/f2/floorplan.py: the committed XDC is its output, every engine
port and lane has one SLR, a group's lanes share their DMA's, and in SLR1
only the host's stages may use the corridor (column X5)."""

import re

import floorplan

F2_CORE = floorplan.OUT.parent / "f2_core.sv"

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
    """SLR1's child pblock is SLR1's CL clock regions west of the corridor;
    the only site ranges it leaves out are the X4Y5 strip's; the only cells
    in SLR1's pblock outside it are the host's stages."""
    ranges = floorplan.clip(1)
    kept = floorplan.west_of_corridor(ranges)
    rows = floorplan.ROWS_PER_SLR["CLOCKREGION"]
    assert kept
    for kind, x0, y0, x1, y1 in kept:
        assert kind == "CLOCKREGION" and x0 <= x1 < floorplan.CORRIDOR_X, kept
        assert rows <= y0 <= y1 < 2 * rows, kept
    sites = [r for r in ranges if r[0] != "CLOCKREGION"]
    assert [r for r in sites if r[0] == "SLICE"] == [floorplan.STRIP], sites
    outside = [
        m[2]
        for line in floorplan.lines()
        if (m := IN_PBLOCK.search(line)) and m[1] == floorplan.slr_pblock(1)
    ]
    assert outside and all("/u_host/" in c for c in outside), outside


def test_regions_match_f2_core() -> None:
    """REGION1 and REGION2 are f2_core.sv's defaults at PORTS."""
    src = F2_CORE.read_text()
    for name, value in (("Region1", floorplan.REGION1), ("Region2", floorplan.REGION2)):
        expr = re.search(rf"parameter\s+int unsigned {name}\s*=\s*([^,]+),", src)[1]
        assert re.fullmatch(r"[\sPorts0-9/+-]+", expr), expr
        assert eval(expr.replace("/", "//"), {"Ports": floorplan.PORTS}) == value, (name, expr)
