# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The F2 platform (PLAN.md, M9's answers): the card in slot 0 through
platforms/f2/host/core_driver.cpp, a child process on the F2 host serving
the five calls in the simulation's protocol (sw/driver_protocol.h), so a
card error does not take Python down. The driver reads the build's Ports,
ChanW and Len from the wrapper; the runtime reads the logits from card
memory through PCIS, where OUTPUT writes them (D-071)."""

from pathlib import Path

from driver_process import DriverProcess

DRIVER = Path(__file__).resolve().parent.parent / "platforms" / "f2" / "host" / "core_driver"


class F2Platform(DriverProcess):
    """A Platform on the F2 card, through `driver` (the binary)."""

    def __init__(self, driver: Path = DRIVER) -> None:
        super().__init__([str(driver)])
