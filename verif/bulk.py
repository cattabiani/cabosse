# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""High-volume numerics tests (PLAN.md, M3): a block runs under Verilator's
own loop, so nothing returns to Python per input. Python makes the inputs and
the expected bits with the golden model and compares whole arrays.

Protocol: a driver (verif/bulk/<top>.cpp, built on verif/bulk/stream.h) reads
records of little-endian uint32 from stdin, streams one per clock cycle into
the block, and writes one uint32 result per record to stdout, in order,
collected by valid_o. A record's fields are the order the driver documents.

For blocks with clk_i, rst_ni, valid_i, valid_o, fields of at most 32 bits
and no backpressure. The lane (dot_lane.cpp) has a valid/ready handshake and
its own loop and protocol, documented there; build() serves it too. Other
handshakes and wide or AXI ports belong in cocotb.
"""

import functools
import os
import subprocess
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import rtl

DRIVERS = Path(__file__).parent / "bulk"
CHUNK = 1 << 21  # records per run of the driver: one run per test chunk
WORKERS = 8  # drivers run at once by in_parallel
_BUILD_LOCK = threading.Lock()  # tests may run chunks from several threads


def build(top: str, params: tuple[tuple[str, int], ...] = ()) -> Path:
    """Verilate `top` (with parameter overrides) and its driver, and compile
    it, once per test session (Verilator's make rebuilds only what changed).
    The drivers need C++23 with <print> (GCC 14 or newer); the standard CXX
    variable picks the compiler, which Verilator's makefile otherwise fixes.
    Thread-safe: the first caller builds, the others wait for it."""
    with _BUILD_LOCK:
        return _build(top, params)


@functools.cache
def _build(top: str, params: tuple[tuple[str, int], ...]) -> Path:
    build_dir = rtl.BUILD / ("_".join(["bulk", top, *(f"{k}{v}" for k, v in params)]))
    build_dir.mkdir(parents=True, exist_ok=True)  # Verilator makes only the last level
    cxx = os.environ.get("CXX")
    argv = [
        "verilator", "--cc", "--exe", "--build", "-j", "0", "-O3",
        "--x-assign", "fast", "--x-initial", "fast", "-CFLAGS", f"-std=c++23 -O3 -I{DRIVERS}",
        "--top-module", top, "-Mdir", str(build_dir), "-o", "bulk",
        *(f"-G{k}={v}" for k, v in params),
        *(["-MAKEFLAGS", f"CXX={cxx}", "-MAKEFLAGS", f"LINK={cxx}"] if cxx else []),
        *rtl.VERILATOR_FILES, str(DRIVERS / f"{top}.cpp"),
    ]  # fmt: skip
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    return build_dir / "bulk"


def run(top: str, records: np.ndarray, params: tuple[tuple[str, int], ...] = ()) -> np.ndarray:
    """Stream `records` (uint32, one row per input, in the driver's field
    order) through `top`; one uint32 result per row, in order."""
    binary = build(top, params)
    records = np.ascontiguousarray(records, dtype="<u4")
    out = []
    for start in range(0, len(records), CHUNK):
        chunk = records[start : start + CHUNK]
        result = subprocess.run([str(binary)], input=memoryview(chunk), capture_output=True)
        assert result.returncode == 0, result.stderr.decode()
        out.append(np.frombuffer(result.stdout, dtype="<u4"))
    return np.concatenate(out) if out else np.empty(0, dtype="<u4")


def check(top: str, records: np.ndarray, want: np.ndarray, label: str, params: tuple = ()) -> None:
    """Assert that `records` through `top` give `want` (uint32), bit for bit;
    the failure names `label` (say, the seed) and the first rows that differ."""
    got = run(top, records, params)
    report = mismatches(got, want, records)
    assert not report, f"{label}: {report}"


def mismatches(got: np.ndarray, want: np.ndarray, inputs: np.ndarray, limit: int = 10) -> str:
    """The first `limit` rows where got and want differ, in hex, or ''."""
    bad = np.flatnonzero(got != want)
    rows = [
        " ".join(f"0x{v:08X}" for v in inputs[i]) + f" -> got 0x{got[i]:08X}, want 0x{want[i]:08X}"
        for i in bad[:limit]
    ]
    return f"{len(bad)} of {len(got)} differ:\n" + "\n".join(rows) if len(bad) else ""


def in_parallel[T](fn: Callable[[T], None], items: Iterable[T]) -> None:
    """fn(item) for every item, WORKERS at a time (each fn runs drivers);
    re-raises the first failure."""
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(fn, items))


def chunk_seeds(seed: int, n_total: int, per_chunk: int) -> list[int]:
    """One seed per chunk of `per_chunk` records, enough chunks for `n_total`:
    seed + 1, seed + 2, ..."""
    return [seed + 1 + k for k in range(-(-n_total // per_chunk))]
