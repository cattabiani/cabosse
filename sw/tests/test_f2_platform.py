# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The F2 platform's driver (platforms/f2/host/core_driver.cpp) on a fake
card (card_fake.cpp): the protocol, memory at any alignment, the logits
read where LOGITS_HI:LOGITS_LO points, the handovers of HBM port 0 (the
fake fails a start while the host holds it, memory while the core does),
and bad requests. The real card is M9's F2 run."""

import os
import subprocess
from pathlib import Path

import pytest
import torch
from commands import REG
from driver_process import DriverError
from f2_platform import F2Platform

import rtl

HOST = rtl.REPO / "platforms" / "f2" / "host"
FAKE = {"FAKE_PORTS": "4", "FAKE_CHAN_W": "12", "FAKE_LEN": "1024"}
SEED = 20261104


@pytest.fixture(scope="module")
def driver() -> Path:
    """core_driver built against the fake card, with CXX (else g++)."""
    binary = rtl.BUILD / "core_driver_fake"
    binary.parent.mkdir(parents=True, exist_ok=True)
    cxx = os.environ.get("CXX", "g++")
    argv = [cxx, "-std=c++23", "-O1", "-Wall", "-Wextra", "-Werror", "-o", str(binary)]
    argv += [str(HOST / "core_driver.cpp"), str(Path(__file__).parent / "card_fake.cpp")]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-4000:]
    return binary


@pytest.fixture
def platform(driver: Path, monkeypatch: pytest.MonkeyPatch):
    for k, v in FAKE.items():
        monkeypatch.setenv(k, v)
    with F2Platform(driver) as p:
        yield p


def test_hello(platform: F2Platform) -> None:
    assert (platform.ports, platform.channel_bytes, platform.buffer_elements) == (4, 4096, 1024)


@pytest.mark.parametrize(("addr", "n"), [(0, 64), (3, 1), (61, 7), (100, 333), (16383 - 5, 5)])
def test_memory_round_trip(platform: F2Platform, addr: int, n: int) -> None:
    g = torch.Generator().manual_seed(SEED + addr)
    data = torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g)
    platform.mem_write(addr, data)
    assert torch.equal(platform.mem_read(addr, n), data), f"seed {SEED + addr}"


def test_token_and_logits(platform: F2Platform) -> None:
    """Memory, a token, memory again: each needs its handover."""
    logits = torch.arange(10, dtype=torch.float32) - 3.5
    platform.mem_write(0x2000, logits.view(torch.uint8))
    platform.reg_write(REG["LOGITS_LO"], 0x2000)
    platform.reg_write(REG["LOGITS_HI"], 0)
    platform.reg_write(REG["CONTROL"], 1)
    platform.run_until_done()
    assert platform.reg_read(REG["STATUS"]) == 2
    assert torch.equal(platform.read_logits(10), logits)


@pytest.mark.parametrize(
    "words",
    [(3, 0x4000 - 4, 0, 8), (5, 15, 0, 16), (7,), (99,)],
    ids=["past the memory", "buffer read", "trace", "unknown"],
)
def test_bad_request(platform: F2Platform, words: tuple[int, ...]) -> None:
    platform.send(*words)
    with pytest.raises(DriverError, match=r"exited \(2\)"):
        platform.words(1)
