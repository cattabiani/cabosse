# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The F2 platform's driver (platforms/f2/host/core_driver.cpp) on a fake
card (card_fake.cpp): the protocol, memory at any alignment, the
handovers of HBM port 0 (the fake fails a start while the host holds it,
memory while the core does), a token's done only once it ran, and bad
requests. The real card is M9's F2 run."""

import json
import subprocess
import time
from pathlib import Path

import bulk
import pytest
import torch
from commands import REG
from driver_process import DriverError
from f2_platform import F2Platform

import rtl

HOST = rtl.REPO / "platforms" / "f2" / "host"
FAKE_CARD = Path(__file__).parent / "card_fake.cpp"
FAKE = {"FAKE_PORTS": "4", "FAKE_CHAN_W": "12", "FAKE_LEN": "1024"}
SEED = 20261104
# A C++23 compiler with <print>, as the RTL tests' (CI's rtl job sets CXX).
pytestmark = rtl.needs_verilator


@pytest.fixture(scope="module")
def driver() -> Path:
    """core_driver built against the fake card."""
    return bulk.build_program("core_driver_fake", (HOST / "core_driver.cpp", FAKE_CARD))


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


def test_token_between_memory(platform: F2Platform) -> None:
    """Memory, two tokens, memory again: each needs its handover, and a
    token is done only once STATUS showed it running (the fake's first read
    after a start is the last token's STATUS)."""
    logits = torch.arange(16, dtype=torch.float32) - 3.5
    platform.mem_write(0x2000, logits.view(torch.uint8))
    platform.reg_write(REG["CONTROL"], 1)
    platform.run_until_done()
    platform.reg_write(REG["CONTROL"], 1)
    platform.run_until_done()
    assert platform.reg_read(REG["STATUS"]) == 2
    assert torch.equal(platform.mem_read(0x2000, 64).view(torch.float32), logits)


def test_output_through_pcim(platform: F2Platform) -> None:
    """The first output read arms the PCIM window and reads card memory; after
    a token, the next comes from the host page (the fake's copy at the
    token's end): card memory changed after the token does not show."""
    addr, n = 0x2000, 256
    first = torch.arange(n // 4, dtype=torch.float32)
    platform.mem_write(addr, first.view(torch.uint8))
    for _ in range(2):  # the second from the page, as the first left it
        assert torch.equal(platform.read_output(addr, n).view(torch.float32), first)
    for k in range(2):
        token = first + 100 * (k + 1)
        platform.mem_write(addr, token.view(torch.uint8))
        platform.reg_write(REG["CONTROL"], 1)
        platform.run_until_done()
        platform.mem_write(addr, torch.zeros(n, dtype=torch.uint8))  # after the copy
        assert torch.equal(platform.read_output(addr, n).view(torch.float32), token), k


def test_output_after_a_pcim_error_reads_card_memory(
    driver: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """An error response turns PCIM off for good: that read and the later
    ones go through PCIS, and the driver says so on stderr."""
    for k, v in {**FAKE, "FAKE_PCIM_ERR": "1"}.items():
        monkeypatch.setenv(k, v)
    addr, n = 0x2000, 64
    with F2Platform(driver) as platform:
        platform.read_output(addr, n)  # arms
        for k in range(2):
            token = torch.full((n,), k + 1, dtype=torch.uint8)
            platform.mem_write(addr, token)
            platform.reg_write(REG["CONTROL"], 1)
            platform.run_until_done()
            assert torch.equal(platform.read_output(addr, n), token), k
    assert "PCIM off for good" in capfd.readouterr().err


def test_output_without_pcim_reads_card_memory_at_once(
    driver: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """A wrapper built before PCIM (build 9's AFI): arming finds no PCIM and
    turns it off for good, with no wait for copies that never come."""
    for k, v in {**FAKE, "FAKE_NO_PCIM": "1"}.items():
        monkeypatch.setenv(k, v)
    addr, n = 0x2000, 64
    with F2Platform(driver) as platform:
        platform.read_output(addr, n)  # arms, finds none
        for k in range(2):
            token = torch.full((n,), k + 1, dtype=torch.uint8)
            platform.mem_write(addr, token)
            platform.reg_write(REG["CONTROL"], 1)
            platform.run_until_done()
            start = time.monotonic()
            assert torch.equal(platform.read_output(addr, n), token), k
            assert time.monotonic() - start < 1, "waited for copies"
    assert "has no PCIM" in capfd.readouterr().err


def test_pcis_bench(monkeypatch: pytest.MonkeyPatch) -> None:
    """pcis_bench on the fake card: a JSON line a plan, all reads matching;
    a plan this machine lacks is skipped and fails the run."""
    bench = bulk.build_program("pcis_bench_fake", (HOST / "pcis_bench.cpp", FAKE_CARD))
    for k, v in FAKE.items():
        monkeypatch.setenv(k, v)
    run = subprocess.run(
        [bench, "4096", "3", "memcpy:1", "memcpy:2"], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    lines = [json.loads(line) for line in run.stdout.splitlines()]
    assert [x["plan"] for x in lines] == ["memcpy:1", "memcpy:2"]
    assert all(x["ok"] and x["bytes"] == 4096 and len(x["us"]) == 3 for x in lines)
    run = subprocess.run([bench, "64", "1", "load64:1"], capture_output=True, text=True)
    assert run.returncode == 1 and "skipped" in run.stderr


def test_output_off_the_beats_reads_card_memory(platform: F2Platform) -> None:
    """A window not in whole 32-byte beats is never armed: PCIS each time."""
    data = torch.arange(7, dtype=torch.uint8)
    platform.mem_write(3, data)
    platform.reg_write(REG["CONTROL"], 1)
    platform.run_until_done()
    for _ in range(2):
        assert torch.equal(platform.read_output(3, 7), data)


@pytest.mark.parametrize(
    "words",
    [(3, 0x4000 - 4, 0, 8), (5, 15, 0, 16), (7,), (99,)],
    ids=["past the memory", "buffer read", "trace", "unknown"],
)
def test_bad_request(platform: F2Platform, words: tuple[int, ...]) -> None:
    platform.send(*words)
    with pytest.raises(DriverError, match=r"exited \(2\)"):
        platform.words(1)
