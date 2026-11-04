# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The simulation platform (PLAN.md, M7's answers): rtl/core.sv under
Verilator, its driver (verif/bulk/core.cpp) a child process serving the
five calls on stdin and stdout, so a crash in the simulation does not take
Python down. The build's parameters fix the ports, the channels' size and
the buffers' length (one Len for every buffer)."""

from pathlib import Path

import bulk
import numpy as np
from commands import REG, Buf
from driver_process import BUF_READ, RUN, TRACE, DriverError, DriverProcess

# The controller's handshakes with the units, in the driver's order.
EVENTS = ("eng_take", "eng_go", "eng_done", "vec_take", "vec_done")


class SimPlatform(DriverProcess):
    """A Platform on rtl/core.sv, or on `top` f2_core (the core in its F2
    wrapper, platforms/f2/core/: one memory at global addresses, HBM port 0
    handed over as on F2; verif/bulk/f2_core.cpp): `ports` HBM ports (one
    channel each, of 2^chan_w bytes) and buffers of `length` elements.
    `runs` holds each run's POSITION, trace and CYCLES (verif/breakdown.py)."""

    def __init__(
        self, ports: int, chan_w: int, length: int, top: str = "core", **plusargs: int
    ) -> None:
        params = (("Ports", ports), ("ChanW", chan_w), ("Len", length))
        binary: Path = bulk.build(top, params)
        super().__init__([str(binary), *(f"+{k}={v}" for k, v in plusargs.items())])
        found = (self.ports, self.channel_bytes, self.buffer_elements)
        if found != (ports, 1 << chan_w, length):
            self.close()
            raise DriverError(f"the driver's build is {found}, not {(ports, 1 << chan_w, length)}")
        self.runs: list[tuple[int, dict[str, np.ndarray], int]] = []

    def run_until_done(self) -> None:
        self.send(RUN)
        self.words(1)  # STATUS: the runtime reads it
        position = self.reg_read(REG["POSITION"])
        cycles = self.reg_read(REG["CYCLES_LO"]) | self.reg_read(REG["CYCLES_HI"]) << 32
        self.runs.append((position, self.trace(), cycles))

    def read_buffer(self, buf: Buf, first: int, n: int) -> np.ndarray:
        """A buffer's raw bits (BF16 in the low half), a uint32 each."""
        self.send(BUF_READ, buf, first, n)
        return self.words(n).copy()

    def trace(self) -> dict[str, np.ndarray]:
        """The cycles (the simulation's count) of each handshake in EVENTS
        since the last trace (the last run), as uint64."""
        self.send(TRACE)
        out = {}
        for name in EVENTS:
            n = int(self.words(1)[0])
            out[name] = self.words(2 * n).copy().view("<u8")
        return out
