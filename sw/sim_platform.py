# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The simulation platform (PLAN.md, M7's answers): rtl/core.sv under
Verilator, its driver (verif/bulk/core.cpp) a child process serving the
six calls on stdin and stdout, so a crash in the simulation does not take
Python down. The build's parameters fix the ports, the channels' size and
the buffers' length (one Len for every buffer)."""

import subprocess
from pathlib import Path

import bulk
import numpy as np
import torch
from commands import Buf

HELLO = 0x43425349  # 'CBSI'
REG_WRITE, REG_READ, MEM_WRITE, MEM_READ, RUN, BUF_READ, QUIT = range(7)


class SimError(RuntimeError):
    """The simulation stopped: its exit code and message."""


class SimPlatform:
    """A Platform on rtl/core.sv: `ports` HBM ports (one channel each, of
    2^chan_w bytes) and buffers of `length` elements."""

    def __init__(self, ports: int, chan_w: int, length: int, **plusargs: int) -> None:
        params = (("Ports", ports), ("ChanW", chan_w), ("Len", length))
        binary: Path = bulk.build("core", params)
        argv = [str(binary), *(f"+{k}={v}" for k, v in plusargs.items())]
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        hello = self.words(4)
        assert hello[0] == HELLO, hex(hello[0])
        self.ports, chan_w_got, self.buffer_elements = map(int, hello[1:])
        assert (self.ports, chan_w_got, self.buffer_elements) == (ports, chan_w, length), hello
        self.channel_bytes = 1 << chan_w

    def send(self, *words: int, data: bytes = b"") -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.array(words, dtype="<u4").tobytes() + data)
        self.proc.stdin.flush()

    def words(self, n: int) -> np.ndarray:
        assert self.proc.stdout is not None
        raw = self.proc.stdout.read(4 * n)
        if len(raw) != 4 * n:
            code = self.proc.wait()
            raise SimError(f"the simulation exited ({code}): see its stderr above")
        return np.frombuffer(raw, dtype="<u4")

    def reg_write(self, offset: int, value: int) -> None:
        self.send(REG_WRITE, offset, value)

    def reg_read(self, offset: int) -> int:
        self.send(REG_READ, offset)
        return int(self.words(1)[0])

    def mem_write(self, addr: int, data: torch.Tensor) -> None:
        raw = data.numpy().tobytes()
        self.send(
            MEM_WRITE, addr & 0xFFFFFFFF, addr >> 32, len(raw), data=raw + bytes(-len(raw) % 4)
        )

    def mem_read(self, addr: int, n_bytes: int) -> torch.Tensor:
        self.send(MEM_READ, addr & 0xFFFFFFFF, addr >> 32, n_bytes)
        raw = self.words(-(-n_bytes // 4)).view(np.uint8)[:n_bytes]
        return torch.from_numpy(raw.copy())

    def run_until_done(self) -> None:
        self.send(RUN)
        self.words(1)  # STATUS: the runtime reads it

    def read_buffer(self, buf: Buf, first: int, n: int) -> np.ndarray:
        """A buffer's raw bits (BF16 in the low half), a uint32 each."""
        self.send(BUF_READ, buf, first, n)
        return self.words(n).copy()

    def read_logits(self, n: int) -> torch.Tensor:
        return torch.from_numpy(self.read_buffer(Buf.LOGITS, 0, n).view(np.float32).copy())

    def close(self) -> None:
        if self.proc.poll() is None:
            self.send(QUIT)
            self.proc.wait()

    def __enter__(self) -> SimPlatform:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
