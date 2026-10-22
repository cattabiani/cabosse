# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The tile buffer's bank mapping (docs/architecture.md, "Tile buffer
banks"), restated here: with ports a multiple of 4 and L = 4·ports, entry u
(elements 4u to 4u+3) of position q sits in bank (q + 4·alpha(u)) mod L,
alpha(u) = (ports/4)·(u mod 4) + u div 4 (mod ports). Every access pattern of
rtl/engine.sv touches each bank at most once per cycle:

- fill, from every port one beat (4 entries) of row r, head slot x, beat b:
  position c + r·ports from port c (any d for one head: q·K);
- q·K: one entry u of every position of the tile;
- p·V: 4 positions 4k ... 4k+3 × the entries of up to `heads` heads
  (heads·d <= L).

Exhaustive over the ranges below, wider than the RTL's (2 heads, d <= 128)."""

import pytest

PORTS = range(4, 70, 4)
HEAD_DIMS = (16, 32, 48, 64, 128, 256)
MAX_HEADS = 4


def alpha(u: int, ports: int) -> int:
    return ((ports // 4) * (u % 4) + u // 4) % ports


def bank(q: int, u: int, ports: int) -> int:
    return (q + 4 * alpha(u, ports)) % (4 * ports)


def distinct(banks: list[int]) -> bool:
    return len(set(banks)) == len(banks)


@pytest.mark.parametrize("ports", PORTS)
def test_no_bank_conflict(ports: int) -> None:
    lanes = 4 * ports
    for d in HEAD_DIMS:
        entries = d // 4
        for u in range(entries):
            assert distinct([bank(q, u, ports) for q in range(lanes)]), ("q·K", ports, d, u)
        for heads in range(1, MAX_HEADS + 1):
            if heads > 1 and d * heads > lanes:  # one head: q·K takes any d
                break
            for x in range(heads):
                for r in range(4):
                    for b in range(d // 16):
                        fill = [
                            bank(c + r * ports, x * entries + 4 * b + w, ports)
                            for c in range(ports)
                            for w in range(4)
                        ]
                        assert distinct(fill), ("fill", ports, d, heads, x, r, b)
            if d * heads > lanes:
                continue
            for k in range(ports):
                values = [
                    bank(4 * k + e, u, ports) for e in range(4) for u in range(heads * entries)
                ]
                assert distinct(values), ("p·V", ports, d, heads, k)
