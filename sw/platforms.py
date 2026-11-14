# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""What the runtime needs from a platform (PLAN.md, M7's answers): five
calls, the same for the simulation (a Verilator driver in a child process)
and for F2 (M9); the logits are read where OUTPUT writes them: card memory
(D-071), or on F2 host memory, where the wrapper copies them through PCIM
(M11, Q-34). Registers are 32-bit, at the offsets of
commands.REGISTERS; memory addresses are HBM byte addresses, channel c at
c · channel_bytes (commands.Layout)."""

from typing import Protocol

import torch


class Platform(Protocol):
    ports: int  # the engine's HBM ports, one channel each
    channel_bytes: int
    buffer_elements: int | None  # each on-chip buffer's length; None: no limit

    def reg_write(self, offset: int, value: int) -> None: ...

    def reg_read(self, offset: int) -> int: ...

    def mem_write(self, addr: int, data: torch.Tensor) -> None:
        """data: bytes as uint8."""
        ...

    def mem_read(self, addr: int, n_bytes: int) -> torch.Tensor:
        """Bytes as uint8."""
        ...

    def read_output(self, addr: int, n_bytes: int) -> torch.Tensor:
        """What OUTPUT wrote at addr, after a run: bytes as uint8. A memory
        read, but on F2, from the second call for a window on: from host
        memory (sw/driver_protocol.h, kOutputRead)."""
        ...

    def run_until_done(self) -> None:
        """Run until STATUS is done or error, after CONTROL's start."""
        ...
