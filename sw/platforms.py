# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""What the runtime needs from a platform (PLAN.md, M7's answers): six
calls, the same for the simulation (a Verilator driver in a child process)
and for F2 (M9). Registers are 32-bit, at the offsets of
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

    def run_until_done(self) -> None:
        """Run until STATUS is done or error, after CONTROL's start."""
        ...

    def read_logits(self, n: int) -> torch.Tensor:
        """The LOGITS buffer's first n elements (FP32), after END. In M7
        the simulation reads them; how F2 gets them is Q-33 (M9)."""
        ...
