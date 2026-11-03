# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""scripts/f2/bringup_report.py on made-up counters: a port's rate is its
beats over its cycles, the total is all beats over the slowest port's
cycles (not the sum of the ports' rates), the mean wait follows Little's
law, and a run's summary follows the run's order, reads and writes as
their mode says, error responses per direction, and a failed clock step
or a port that read nothing reported, not fatal."""

import bringup_report as report
import pytest
from bringup_plan import READ, WRITE

# The host tool's counters (platforms/f2/host/bringup.cpp, kCounters).
COUNTERS = (
    "rd_beats", "rd_cycles", "rd_first", "rd_stalls", "rd_maxgap", "rd_bad", "rd_occ_lo",
    "rd_occ_hi", "wr_beats", "wr_cycles", "wr_first", "wr_maxout", "wr_occ_lo", "wr_occ_hi",
    "flags",
)  # fmt: skip
PROVENANCE = {
    "commit": "abc",
    "agfi": "agfi-x",
    "afi": "afi-x",
    "instance": "i-x",
    "instance_type": "f2",
    "cpu": "AMD EPYC",
}


def port(
    rd_beats: int = 0, rd_cycles: int = 0, wr_beats: int = 0, wr_cycles: int = 0, **more: int
) -> dict:
    """A port's counters: these, the rest zero."""
    given = dict(rd_beats=rd_beats, rd_cycles=rd_cycles, wr_beats=wr_beats, wr_cycles=wr_cycles)
    return dict.fromkeys(COUNTERS, 0) | given | more


def row(step: str, params: dict, result: dict) -> dict:
    return {"step": step, "params": params, "result": result} | PROVENANCE


def test_rates_and_total():
    # Port 0 reads 100 beats in 100 cycles, port 1 in 400: at 250 MHz, 8 and 2 GB/s.
    result = {"ports": [port(100, 100), port(100, 400)]}
    assert report.rates(result, 250, "rd") == pytest.approx([8.0, 2.0])
    # Both ports' 200 beats over 400 cycles, not 8 + 2.
    assert report.total_gb_per_s(result, 250, "rd") == pytest.approx(4.0)


def test_mean_wait():
    # 10 one-beat writes, 50 burst-cycles unanswered in all: 5 cycles each.
    result = {"ports": [port(wr_beats=6, wr_occ_lo=30), port(wr_beats=4, wr_occ_lo=20)]}
    assert report.mean_wait(result, 1, "wr") == pytest.approx(5.0)
    assert report.mean_wait(result, 2, "wr") == pytest.approx(10.0)  # 5 bursts of 2
    high = {"ports": [port(wr_beats=1, wr_occ_hi=1)]}  # the occupancy's high word
    assert report.mean_wait(high, 1, "wr") == 1 << 32


def test_report_blocks():
    full = {"ports": [port(1000, 1000, 1000, 1000, wr_maxout=12)]}
    regs = {
        "rd_burst": 16,
        "wr_burst": 16,
        "rd_limit": 64,
        "wr_limit": 64,
        "port_en": 1,
        "base_lo": 0,
        "base_hi": 0,
    }
    clock = {"set_mhz": 250, "measured_mhz": 249.99}
    rows = [
        row("clock", {"mhz": 250}, clock),
        row(
            "pcis_write",
            {"bytes": 64, "offset": 0},
            {"error": "PCIS writes need AVX2 (32-byte stores)"},
        ),
        row(
            "pcis_write",
            {"bytes": 64, "offset": 64},
            {"gb_per_s": 2.5, "last_ok": 1},
        ),
        row("pcis_read", {"bytes": 64, "offset": 1 << 33}, {"gb_per_s": 0.005, "bad_words": 3}),
        row("clock", {"mhz": 250}, clock),
        row("fill", regs | {"mode": WRITE}, full),
        row("read_write_all", regs | {"mode": READ | WRITE}, full),
        row("write_burst", regs | {"mode": WRITE, "wr_burst": 4}, full),
        row("read_one", regs | {"mode": READ}, full),
        row("clock", {"mhz": 125}, clock | {"set_mhz": 125}),
        row("read_limit", regs | {"mode": READ, "rd_limit": 8}, full),
        row(
            "read_latency",
            regs | {"mode": READ, "port_en": 1 << 31},
            {"rd_first": [40] * 99 + [80]},
        ),
    ]
    text = "\n".join(report.report(rows))
    assert "pcis_write" in text and "error PCIS writes need AVX2" in text
    assert "CPU AMD EPYC" in text
    assert "pcis_write offset 0x40: 2500.0 MB/s, last ok 1" in text
    assert "pcis_read offset 0x200000000: 5.0 MB/s, bad words 3" in text
    assert "clock 250 MHz (measured 249.990" in text and "clock 125 MHz" in text
    assert "fill                   wr    8.0 GB/s total" in text
    assert "fill                   rd" not in text  # a write-only run
    assert "read_write_all         rd" in text and "read_write_all         wr" in text
    assert "write_burst 4          wr" in text and "max out 12" in text
    assert "read_one (each alone)  rd  8.00- 8.00 GB/s, 1 ports" in text
    assert "read_limit 8           rd    4.0 GB/s total" in text
    assert "read_latency port 31" in text and "p50 320 ns" in text


def test_report_faults():
    regs = {"rd_burst": 16, "wr_burst": 16, "port_en": 1, "base_lo": 0, "base_hi": 0}
    dead = {"ports": [port()]}  # read nothing
    read_err = {"ports": [port(10, 10, 10, 10, flags=0b0100, rd_bad=2)]}
    rows = [
        row("clock", {"mhz": 250}, {"set_mhz": 250, "measured_mhz": 250.0}),
        row("read_one", regs | {"mode": READ}, dead),
        row("read_one", regs | {"mode": READ}, {"ports": [port(10, 10)]}),
        row("read_write_all", regs | {"mode": READ | WRITE}, read_err),
        row("clock", {"mhz": 125}, {"error": "clock generator timeout"}),
        row("fill", regs | {"mode": WRITE}, {"ports": [port(wr_beats=10, wr_cycles=10)]}),
    ]
    text = "\n".join(report.report(rows))
    assert "read_one (each alone)  rd  0.00- 8.00 GB/s, 2 ports, error ports 0" in text
    assert "read_write_all         rd" in text and "error ports 1, bad 2" in text
    assert (
        "read_write_all         wr" in text
        and "wr    8.0 GB/s total,  1 ports  8.00- 8.00 GB/s, error ports 0" in text
    )
    assert "clock {" in text and "error clock generator timeout" in text
    assert "fill: not reported, the core clock is not known" in text
