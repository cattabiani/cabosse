# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The tile buffer's bank mapping (docs/architecture.md, "Tile buffer
banks"), restated here: entry u (elements 4u to 4u+3) of position q sits in
bank (q + 4·alpha(u)) mod L, L = 4·ports. Every access pattern of
rtl/engine.sv touches each bank at most once per cycle:

- fill, from every port one beat (4 entries) of row r, head slot x, beat b:
  position c + r·ports from port c; or one entry per port when ports is not
  a multiple of 4 (the quarter-rate fill);
- q·K: one entry u of every position of the tile;
- p·V: 4 positions 4k ... 4k+3 × the pass's entries: those of up to
  `heads` heads (heads·d <= L), or L / 4 entries of one head when d > L.

Exhaustive over the ranges below, wider than the RTL's (2 heads, d <= 128)."""

import pytest

PORTS = range(1, 70)
HEAD_DIMS = (16, 32, 48, 64, 128, 256)
MAX_HEADS = 4


def alpha(u: int, ports: int) -> int:
    if ports % 4 == 0:
        return ((ports // 4) * (u % 4) + u // 4) % ports
    return u % ports


def bank(q: int, u: int, ports: int) -> int:
    return (q + 4 * alpha(u, ports)) % (4 * ports)


def distinct(banks: list[int]) -> bool:
    return len(set(banks)) == len(banks)


def pass_entries(d: int, heads: int, lanes: int) -> list[range]:
    """The entries of each p.V pass's lanes: every head's when they fit,
    else L / 4 entries of one head at a time."""
    entries = d // 4
    if d > lanes:
        return [range(c, min(c + lanes // 4, entries)) for c in range(0, entries, lanes // 4)]
    return [range(heads * entries)]


@pytest.mark.parametrize("ports", PORTS)
def test_no_bank_conflict(ports: int) -> None:
    lanes = 4 * ports
    full = ports % 4 == 0
    for d in HEAD_DIMS:
        entries = d // 4
        for heads in range(1, MAX_HEADS + 1):
            if d * heads > lanes and heads > 1:
                continue
            for x in range(heads):
                for r in range(4):
                    for b in range(d // 16):
                        for ws in [range(4)] if full else [[w] for w in range(4)]:
                            fill = [
                                bank(c + r * ports, x * entries + 4 * b + w, ports)
                                for c in range(ports)
                                for w in ws
                            ]
                            assert distinct(fill), ("fill", ports, d, heads, x, r, b)
            for u in range(heads * entries):
                assert distinct([bank(q, u, ports) for q in range(lanes)]), ("q·K", ports, d, u)
            for us in pass_entries(d, heads, lanes):
                for k in range(ports):
                    values = [bank(4 * k + e, u, ports) for e in range(4) for u in us]
                    assert distinct(values), ("p·V", ports, d, heads, k)
