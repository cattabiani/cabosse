# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""A platform's driver as a child process, in the protocol of
sw/driver_protocol.h: the simulation's (sw/sim_platform.py) and F2's
(sw/f2_platform.py). The calls both serve live here."""

from __future__ import annotations

import subprocess

import numpy as np
import torch

HELLO = 0x43425349  # 'CBSI'
REG_WRITE, REG_READ, MEM_WRITE, MEM_READ, RUN, BUF_READ, QUIT, TRACE = range(8)


class DriverError(RuntimeError):
    """The driver stopped: its exit code and message."""


class DriverProcess:
    """`argv` started as the driver; after its hello (HELLO, Ports, ChanW,
    Len), `ports`, `channel_bytes` and `buffer_elements` hold the rest."""

    def __init__(self, argv: list[str]) -> None:
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            hello, ports, chan_w, length = map(int, self.words(4))
            if hello != HELLO:
                raise DriverError(f"hello 0x{hello:08X}, not 0x{HELLO:08X}")
        except BaseException:
            self.proc.kill()
            self.proc.wait()
            raise
        self.ports, self.buffer_elements = ports, length
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
            raise DriverError(f"the driver exited ({code}): see its stderr above")
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

    def close(self) -> None:
        if self.proc.poll() is None:
            self.send(QUIT)
            self.proc.wait()

    def __enter__(self) -> DriverProcess:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
