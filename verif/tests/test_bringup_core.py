# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""platforms/f2/bringup/bringup_core.sv (PLAN.md, M8; platforms/f2/README.md)
on the memory model of verif/bulk/axi_mem.h, through the driver
verif/bulk/bringup_core.cpp: the registers, runs that write and read the
pattern on every port (with the memory's checks of the AXI rules), the
counters, the limits, partial writes and error responses, the host's
port to HBM port 0; and the CL top's elaboration against the shell."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import bulk
import numpy as np
import pytest

import rtl

PORTS = 4
PARAMS = (("Ports", PORTS),)
MEM_BYTES = 256 * 1024
STRIDE = 64 * 1024  # bytes between the ports' regions
ALL = (1 << PORTS) - 1
SEED = 0x5EED_0001

# Registers (README.md).
REGS = {
    "id": 0x000, "build": 0x004, "scratch": 0x008, "status": 0x00C, "cycles_lo": 0x010,
    "cycles_hi": 0x014, "ctrl": 0x020, "port_en": 0x024, "mode": 0x028, "base_lo": 0x02C,
    "base_hi": 0x030, "stride": 0x034, "beats": 0x038, "rd_burst": 0x03C, "wr_burst": 0x040,
    "rd_limit": 0x044, "wr_limit": 0x048, "strb": 0x04C, "seed": 0x050, "host_port": 0x054,
}  # fmt: skip
READ, WRITE, CHECK = 1, 2, 4  # MODE
COUNTERS = [
    "rd_beats", "rd_cycles", "rd_first", "rd_stalls", "rd_maxgap", "rd_bad", "rd_occ_lo",
    "rd_occ_hi", "wr_beats", "wr_cycles", "wr_first", "wr_maxout", "wr_occ_lo", "wr_occ_hi",
    "flags",
]  # fmt: skip
WRITE_OP, READ_OP, POLL_OP, WAIT_OP, HOST_WRITE_OP, HOST_READ_OP = range(6)  # the driver's
Op = tuple[int, int, int]


@dataclass
class Result:
    reads: list[int]
    mem: np.ndarray  # uint32 words


def run(mem: np.ndarray, ops: list[Op], **plusargs: int) -> Result:
    words = np.concatenate([[len(mem)], mem, [len(ops)], np.array(ops, dtype=np.uint32).ravel()])
    result = bulk.run_driver("bringup_core", words.astype(np.uint32), PARAMS, **plusargs)
    assert result.returncode == 0, f"{plusargs}: {result.stderr.decode()}"
    out = np.frombuffer(result.stdout, dtype="<u4")
    n_reads = len(out) - len(mem)
    return Result([int(v) for v in out[:n_reads]], out[n_reads:].copy())


def start(mode: int, poll: bool = True, **regs: int) -> list[Op]:
    """Writes of a run's registers (REGS names override the defaults), then
    start and, with poll, the wait for its end."""
    values = {"host_port": 0, "port_en": ALL, "mode": mode, "base_lo": 0, "base_hi": 0,
              "stride": STRIDE, "seed": SEED} | regs  # fmt: skip
    ops = [(WRITE_OP, REGS[k], v) for k, v in values.items()] + [(WRITE_OP, REGS["ctrl"], 1)]
    return ops + [(POLL_OP, 0, 0)] if poll else ops


def counters(n_ports: int = PORTS) -> list[Op]:
    """Reads of the first n_ports ports' counters."""
    return [
        (READ_OP, 0x100 + 0x40 * p + 4 * i, 0) for p in range(n_ports) for i in range(len(COUNTERS))
    ]


def runs(reads: list[int], n_ports: int = PORTS) -> list[list[dict[str, int]]]:
    """The counters() reads of each run, by port: when they are all the reads."""
    table = np.reshape(reads, (-1, n_ports, len(COUNTERS)))
    return [[dict(zip(COUNTERS, map(int, port), strict=True)) for port in run] for run in table]


def pattern(byte_addr: int, n_words: int, seed: int = SEED) -> np.ndarray:
    return (np.arange(n_words, dtype=np.uint64) + byte_addr // 4).astype(np.uint32) ^ np.uint32(
        seed
    )


def blank() -> np.ndarray:
    return np.random.default_rng(1).integers(0, 1 << 32, MEM_BYTES // 4, dtype=np.uint32)


@rtl.needs_verilator
def test_registers():
    r = REGS
    ops = [(READ_OP, r["id"], 0), (READ_OP, r["build"], 0), (WRITE_OP, r["scratch"], 0x1234_5678)]
    ops += [(READ_OP, r[k], 0) for k in ("scratch", "rd_burst", "wr_burst", "rd_limit", "wr_limit")]
    ops += [(READ_OP, r[k], 0) for k in ("strb", "host_port")]
    ops += [(WRITE_OP, r["host_port"], 0), (READ_OP, r["host_port"], 0), (READ_OP, r["status"], 0)]
    ops += [(READ_OP, r["cycles_lo"], 0), (READ_OP, r["cycles_hi"], 0), (WAIT_OP, 0, 100)]
    ops += [(READ_OP, r["cycles_lo"], 0)]
    # No register there: past the last port, past 32 ports, below the ports.
    ops += [(READ_OP, a, 0) for a in (0x100 + 0x40 * PORTS, 0x900, 0xFC0, 0x0FC)]
    got = run(blank(), ops).reads
    assert got[:3] == [0xCAB0_B001, 0, 0x1234_5678]
    assert got[3:9] == [16, 16, 8, 8, 0xFFFF_FFFF, 1]  # the reset values
    assert got[9:11] == [0, 0b10]  # HOST_PORT now 0; HBM ready, idle
    assert got[12] == 0 and got[13] - got[11] >= 100  # CYCLES_HI latched with LO
    assert got[14:] == [0, 0, 0, 0]


@rtl.needs_verilator
@pytest.mark.parametrize("burst", [1, 2, 4, 8, 16])
def test_write_then_read(burst: int):
    """Every port writes its region and reads it back, checked; the memory
    holds the pattern there and nothing else changed."""
    beats = 256  # 8 KB a port: two 4 KB pages
    mem = blank()
    ops = start(WRITE, beats=beats, wr_burst=burst) + counters()
    ops += start(READ | CHECK, beats=beats, rd_burst=burst) + counters()
    res = run(mem, ops, latency=20, outstanding=16, pause=20, pause_max=5, rate=900, wlatency=10)
    want = mem.copy()
    for p in range(PORTS):
        lo = p * STRIDE // 4
        want[lo : lo + beats * 8] = pattern(p * STRIDE, beats * 8)
    np.testing.assert_array_equal(res.mem, want)
    wrote, read = runs(res.reads)
    for c in wrote:
        assert c["wr_beats"] == beats and c["flags"] == 0, c
        assert 0 < c["wr_first"] <= c["wr_cycles"], c
        assert 1 <= c["wr_maxout"] <= 8, c  # the reset limit
    for c in read:
        assert c["rd_beats"] == beats and c["rd_bad"] == 0 and c["flags"] == 0, c
        assert 20 <= c["rd_first"] <= c["rd_cycles"], c
        assert c["rd_stalls"] >= c["rd_maxgap"] >= 1, c  # the memory pauses
        assert c["rd_occ_lo"] > 0, c


@rtl.needs_verilator
def test_check_counts_bad_beats():
    """Read with another seed: every beat is unlike the pattern."""
    beats = 64
    ops = start(WRITE, beats=beats) + start(READ | CHECK, beats=beats, seed=SEED ^ 1) + counters()
    for c in runs(run(blank(), ops).reads)[0]:
        assert c["rd_bad"] == beats and c["rd_beats"] == beats, c


@rtl.needs_verilator
def test_partial_write():
    """Strobes: only the enabled bytes change."""
    beats, strb = 32, 0x0000_00F0  # bytes 4 to 7: word 1 of each beat
    mem = blank()
    res = run(mem, start(WRITE, beats=beats, strb=strb, port_en=1))
    want = mem.copy()
    want[1 : beats * 8 : 8] = pattern(0, beats * 8)[1::8]
    np.testing.assert_array_equal(res.mem, want)


@rtl.needs_verilator
@pytest.mark.parametrize("limit", [1, 3, 64])
def test_limits(limit: int):
    """With a slow memory the bursts unanswered reach the limit and stay
    within it; reads see the memory's latency first."""
    beats = 512
    ops = start(WRITE, beats=beats, wr_burst=1, wr_limit=limit, port_en=1) + counters(1)
    ops += start(READ, beats=beats, rd_burst=1, rd_limit=limit, port_en=1) + counters(1)
    res = run(blank(), ops, latency=100, outstanding=128, wlatency=100, woutstanding=128)
    [w], [r] = runs(res.reads, 1)
    assert w["wr_maxout"] == limit, w
    assert 100 <= r["rd_first"] <= 110, r
    # Little's law: the bursts unanswered summed over cycles, over the
    # bursts, is the mean latency; the rate is the limit over it, at most
    # half a beat a cycle (1-beat commands, port_dma).
    latency = r["rd_occ_lo"] / beats
    assert 100 <= latency <= 110, r
    assert beats / r["rd_cycles"] == pytest.approx(min(0.5, limit / latency), rel=0.1), r


@rtl.needs_verilator
def test_error_responses():
    """An error response sets its side's flag, until the next start."""
    beats = 32
    ops = (
        start(WRITE, beats=beats, port_en=1)
        + counters(1)
        + start(READ, beats=beats, port_en=1)
        + counters(1)
    )
    [w], [r] = runs(run(blank(), ops, bad=5, wbad=1).reads, 1)
    assert w["flags"] == 0b1000 and r["flags"] == 0b0100, (w, r)


@rtl.needs_verilator
def test_read_and_write_at_once():
    beats = 256
    ops = start(READ | WRITE, beats=beats) + counters()
    for c in runs(run(blank(), ops, latency=30, outstanding=32).reads)[0]:
        assert c["rd_beats"] == beats and c["wr_beats"] == beats and c["flags"] == 0, c


@rtl.needs_verilator
def test_start_ignored_while_busy():
    """A second start during a run changes nothing; the run ends normally."""
    beats = 512
    ops = start(WRITE, beats=beats, poll=False, port_en=1) + [
        (WRITE_OP, REGS["ctrl"], 1),
        (POLL_OP, 0, 0),
    ]
    [[c]] = runs(run(blank(), ops + counters(1), wlatency=50).reads, 1)
    assert c["wr_beats"] == beats, c


@rtl.needs_verilator
@pytest.mark.parametrize(
    "bad",
    [{"rd_burst": 3}, {"wr_burst": 32}, {"rd_limit": 0}, {"wr_limit": 0}, {"beats": 100},
     {"beats": 1 << 24}, {"base_lo": 0x800}, {"stride": 0x1100}],
)  # fmt: skip
def test_bad_settings_refused(bad: dict[str, int]):
    """A start whose settings would hang a port does nothing and sets
    STATUS.refused; the next good start clears it."""
    good = {"beats": 64, "rd_burst": 16, "wr_burst": 16, "rd_limit": 8, "wr_limit": 8,
            "base_lo": 0, "stride": STRIDE}  # fmt: skip
    status = (READ_OP, REGS["status"], 0)
    ops = start(WRITE, **(good | bad)) + [status] + start(WRITE, **good) + [status] + counters(1)
    got = run(blank(), ops).reads
    assert got[0] == 0b110 and got[1] == 0b010, got[:2]
    [[c]] = runs(got[2:], 1)
    assert c["wr_beats"] == 64, c  # the good run, on port 0


@rtl.needs_verilator
def test_host_port_keeps_port_0():
    """With HOST_PORT set, port 0 stays idle on a start; HOST_PORT written
    during a run keeps its value."""
    host = (READ_OP, REGS["host_port"], 0)
    ops = start(WRITE, beats=512, poll=False, host_port=1) + [
        (WRITE_OP, REGS["host_port"], 0),
        host,
    ]
    ops += [(POLL_OP, 0, 0), host] + counters(2)
    got = run(blank(), ops, wlatency=50).reads
    assert got[:2] == [1, 1]
    [[port0, port1]] = runs(got[2:], 2)
    assert port0["wr_beats"] == 0 and port1["wr_beats"] == 512, (port0, port1)


@rtl.needs_verilator
def test_host_port():
    """The host reaches any address through port 0 while HOST_PORT is set
    (the reset value); with it clear port 0's engine runs, and the host then
    reads what the engine wrote."""
    mem = blank()
    addrs = [0, 0x20, STRIDE + 0x40, MEM_BYTES - 32]  # port 0's region and others'
    ops = [(HOST_WRITE_OP, a, 0x1000 * (k + 1)) for k, a in enumerate(addrs)]
    ops += [(HOST_READ_OP, a, 0) for a in addrs]
    ops += start(WRITE, beats=16, port_en=1) + [(WRITE_OP, REGS["host_port"], 1)]
    ops += [(HOST_READ_OP, 0x40, 0)]
    res = run(mem, ops, latency=20, wlatency=10)
    beats = np.reshape(res.reads, (-1, 8))
    for k, a in enumerate(addrs):
        want = 0x1000 * (k + 1) + np.arange(8)
        np.testing.assert_array_equal(beats[k], want)
        if a >= 16 * 32:  # outside what the engine writes after
            np.testing.assert_array_equal(res.mem[a // 4 : a // 4 + 8], want)
    np.testing.assert_array_equal(beats[-1], pattern(0x40, 8))


@rtl.needs("slang")
@pytest.mark.skipif(not os.environ.get("CABOSSE_AWS_FPGA"), reason="needs CABOSSE_AWS_FPGA")
def test_cl_top_elaborates() -> None:
    """The CL top (cabosse_bringup.sv) elaborates against the shell's ports
    (cl_ports.vh) and AWS's clock generator and DDR stub, from the aws-fpga
    checkout (branch f2) that CABOSSE_AWS_FPGA names; AMD's IP stays unknown
    (Vivado checks it)."""
    hdk = Path(os.environ["CABOSSE_AWS_FPGA"]) / "hdk" / "common"
    aws = [
        hdk / "lib" / f
        for f in ("interfaces.sv", "lib_pipe.sv", "aws_clk_regs.sv", "aws_clk_gen.sv")
    ]
    aws.append(hdk / "shell_stable" / "design" / "sh_ddr" / "sh_ddr.stub.sv")
    ours = [*rtl.file_list(rtl.SOURCES), *rtl.BRINGUP, rtl.BRINGUP_TOP]
    argv = ["slang", "--ignore-unknown-modules", "--top", rtl.BRINGUP_TOP.stem]
    argv += ["-I", str(hdk / "shell_stable" / "design" / "interfaces"), *map(str, aws + ours)]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0 and "0 errors, 0 warnings" in result.stdout, result.stdout
