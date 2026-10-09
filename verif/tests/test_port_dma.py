# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/port_dma.sv (PLAN.md, M5) against the memory model of
verif/bulk/axi_mem.h, through the driver verif/bulk/port_dma.cpp: every beat
and every command's last beat, the bursts it asks for (as long as AXI and the
4 KB rule allow), no wait on the data channel (credits), and full rate when
its FIFO covers the memory's latency. The memory's own checks (alignment,
4 KB, range, burst length, a request held until taken) end the run on a
broken rule; a test shows they fire."""

import subprocess
from dataclasses import dataclass

import bulk
import numpy as np
import pytest

import rtl

SEED = 20261010
BEAT_BYTES = 32
BEAT_WORDS = BEAT_BYTES // 4
PAGE_BYTES = 4096
MAX_BURST = 16
MEM_BYTES = 64 * 1024
FAST_SLACK_CYCLES = 8  # full rate: cycles at most beats + latency + this


@dataclass
class Result:
    beats: np.ndarray  # uint32 [n, 8]
    last: np.ndarray  # bool [n]
    cycles: int
    bursts: int
    max_in_flight: int
    r_stalls: int
    err: bool


def run(
    mem: np.ndarray,
    commands: list[tuple[int, int]],
    params: tuple[tuple[str, int], ...] = (),
    **plusargs: int,
) -> subprocess.CompletedProcess:
    """The driver on `mem` (uint32 words) and (address, beats) commands."""
    binary = bulk.build("port_dma", params)
    cmds = [w for addr, beats in commands for w in (addr & 0xFFFFFFFF, addr >> 32, beats)]
    words = np.concatenate([[len(mem)], mem, [len(commands)], cmds]).astype("<u4")
    argv = [str(binary), *(f"+{k}={v}" for k, v in plusargs.items())]
    return subprocess.run(argv, input=memoryview(words), capture_output=True)


def fetch(mem: np.ndarray, commands: list[tuple[int, int]], params=(), **plusargs: int) -> Result:
    result = run(mem, commands, params, **plusargs)
    assert result.returncode == 0, f"{plusargs}: {result.stderr.decode()}"
    words = np.frombuffer(result.stdout, dtype="<u4")
    beats, tail = words[:-5].reshape(-1, BEAT_WORDS + 1), words[-5:]
    return Result(
        beats[:, :BEAT_WORDS], beats[:, BEAT_WORDS] == 1, *map(int, tail[:4]), tail[4] == 1
    )


def expected(mem: np.ndarray, commands: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """The beats the commands name, in order, and which ends a command."""
    beats, last = [], []
    for addr, n in commands:
        first = addr // BEAT_BYTES
        beats += [mem[(first + k) * BEAT_WORDS : (first + k + 1) * BEAT_WORDS] for k in range(n)]
        last += [k == n - 1 for k in range(n)]
    return np.array(beats, dtype=np.uint32).reshape(-1, BEAT_WORDS), np.array(last, bool)


def n_bursts(commands: list[tuple[int, int]], max_burst: int = MAX_BURST) -> int:
    """Bursts as long as allowed: up to max_burst beats, ending at a 4 KB
    boundary at the latest."""
    count = 0
    for addr, n in commands:
        while n:
            to_page = (PAGE_BYTES - addr % PAGE_BYTES) // BEAT_BYTES
            k = min(n, max_burst, to_page)
            addr, n, count = addr + k * BEAT_BYTES, n - k, count + 1
    return count


def edge_commands(rng: np.random.Generator) -> list[tuple[int, int]]:
    """Lengths around a burst and a page, starts just before, at and after
    a 4 KB boundary, zero-beat commands, and random ones."""
    page = PAGE_BYTES
    fixed = [
        (0, 1), (0, 15), (0, 16), (0, 17), (0, 0), (BEAT_BYTES, 31), (0, 128), (0, 129),
        (page - BEAT_BYTES, 1), (page - BEAT_BYTES, 2), (page - 15 * BEAT_BYTES, 16),
        (page - 16 * BEAT_BYTES, 16), (page - 16 * BEAT_BYTES, 17), (page, 300), (0, 0),
        (3 * page + 7 * BEAT_BYTES, 260), (MEM_BYTES - BEAT_BYTES, 1), (0, MEM_BYTES // BEAT_BYTES),
    ]  # fmt: skip
    rand = []
    for _ in range(40):
        n = int(rng.integers(0, 70))
        start = int(rng.integers(0, MEM_BYTES // BEAT_BYTES - n + 1))
        rand.append((start * BEAT_BYTES, n))
    return fixed + rand


def memory(rng: np.random.Generator, n_bytes: int = MEM_BYTES) -> np.ndarray:
    return rng.integers(0, 2**32, n_bytes // 4, dtype=np.uint32)


# Memory behaviours: ideal, slow to answer, few bursts in flight, low
# bandwidth, pauses, a slow consumer, everything at once.
TIMINGS = {
    "ideal": dict(latency=1, outstanding=16),
    "latency": dict(latency=40, outstanding=8),
    "one-burst": dict(latency=5, outstanding=1),
    "bandwidth": dict(latency=3, rate=300),
    "pauses": dict(latency=10, pause=30, pause_max=60),
    "consumer": dict(latency=10, ready=200),
    "all": dict(latency=25, outstanding=3, rate=600, pause=10, pause_max=40, ready=500),
}
# DMA shapes: default; FIFO of one burst; one-beat bursts; one command at a time.
SHAPES = {
    "default": (),
    "fifo-16": (("Depth", 16),),
    "burst-1": (("MaxBurst", 1), ("Depth", 4)),
    "commands-1": (("Commands", 1),),
}


@rtl.needs_verilator
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("timing", TIMINGS)
def test_every_beat_in_order(timing: str, shape: str) -> None:
    seed = SEED + list(TIMINGS).index(timing) * 10 + list(SHAPES).index(shape)
    rng = np.random.default_rng(seed)
    mem, commands = memory(rng), edge_commands(rng)
    params = SHAPES[shape]
    max_burst = dict(params).get("MaxBurst", MAX_BURST)
    got = fetch(mem, commands, params, seed=seed, **TIMINGS[timing])
    beats, last = expected(mem, commands)
    assert got.beats.shape == beats.shape, f"seed {seed}: {len(got.beats)} beats, want {len(beats)}"
    bad = np.flatnonzero((got.beats != beats).any(axis=1))
    assert not len(bad), f"seed {seed}: {len(bad)} beats differ, first at {bad[0]}"
    assert np.array_equal(got.last, last), f"seed {seed}: last flags"
    assert got.bursts == n_bursts(commands, max_burst), f"seed {seed}"
    assert got.r_stalls == 0, f"seed {seed}: the memory waited on RREADY"
    assert not got.err


@rtl.needs_verilator
@pytest.mark.parametrize("latency", [1, 20, 40])
def test_full_rate(latency: int) -> None:
    """One long command with an ideal consumer: a beat every cycle once the
    first arrives, when the FIFO (64 beats) covers the latency plus a burst."""
    rng = np.random.default_rng(SEED)
    n = 2048
    got = fetch(memory(rng), [(0, n)], latency=latency, outstanding=8)
    assert got.cycles <= n + latency + FAST_SLACK_CYCLES, got.cycles


@rtl.needs_verilator
def test_a_small_fifo_limits_the_rate() -> None:
    """A FIFO of one burst cannot cover a 40-cycle latency: about one burst
    per round trip. Shows that full rate above comes from the credits, not
    from the test."""
    rng = np.random.default_rng(SEED)
    n = 1024
    got = fetch(memory(rng), [(0, n)], (("Depth", 16),), latency=40, outstanding=8)
    assert got.cycles > 2 * n, got.cycles


@rtl.needs_verilator
def test_an_error_response_is_kept() -> None:
    rng = np.random.default_rng(SEED)
    got = fetch(memory(rng), [(0, 40)], bad=17)
    assert got.err and len(got.beats) == 40


BROKEN = {
    "past the memory": [(MEM_BYTES - BEAT_BYTES, 2)],
    "not beat-aligned": [(BEAT_BYTES // 2, 1)],
}


@rtl.needs_verilator
@pytest.mark.parametrize("case", BROKEN)
def test_the_memory_rejects_a_broken_request(case: str) -> None:
    """The model ends the run on a request outside the AXI subset or the
    memory (exit 5), naming the rule."""
    rng = np.random.default_rng(SEED)
    result = run(memory(rng), BROKEN[case])
    assert result.returncode == 5, result.stderr.decode()
    assert case in result.stderr.decode()
