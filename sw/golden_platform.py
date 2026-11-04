# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""A Platform on the golden model: registers and memory in Python, a token
run by commands.run(), the reference controller. It tests the runtime in
seconds, with no RTL: the commands it runs are the ones the runtime wrote
into the command buffer, the weights the bytes it wrote into memory
(commands.channel_reads). It knows the Layout only to read those bytes back
as tensors; the hardware needs no such thing, its commands carry the
addresses."""

import dataclasses

import commands
import numpy as np
import torch
from commands import REG, Command, Layout

COMMAND_BUFFER = REG["COMMAND_BUFFER"]


class Memory:
    """Sparse HBM: each channel's bytes up to the highest one written (zeros
    elsewhere)."""

    def __init__(self, channel_bytes: int) -> None:
        self.channel_bytes = channel_bytes
        self.channels: dict[int, bytearray] = {}

    def write(self, addr: int, data: torch.Tensor) -> None:
        c, offset = divmod(addr, self.channel_bytes)
        assert offset + len(data) <= self.channel_bytes, (addr, len(data))
        mem = self.channels.setdefault(c, bytearray())
        if len(mem) < offset + len(data):
            mem.extend(bytes(offset + len(data) - len(mem)))
        mem[offset : offset + len(data)] = memoryview(data.numpy())

    def read(self, addr: int, n_bytes: int) -> torch.Tensor:
        c, offset = divmod(addr, self.channel_bytes)
        raw = bytes(self.channels.get(c, b"")[offset : offset + n_bytes]).ljust(n_bytes, b"\0")
        return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())


class GoldenPlatform:
    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.ports, self.channel_bytes = layout.ports, layout.channel_bytes
        self.buffer_elements = None
        self.mem = Memory(layout.channel_bytes)
        self.regs = {REG["ID"]: commands.ID, REG["VERSION"]: commands.VERSION}
        self.command_buffer = bytearray(commands.COMMAND_BUFFER_BYTES)
        self.hbm: dict[int, torch.Tensor] | None = None  # memory as tensors, once read
        self.stale = False  # the stores and OUTPUT wrote self.hbm, not yet self.mem
        # Rows no store has written hold NaN, as in commands.load(), so a
        # read of one shows (the hardware's are whatever was there).
        stale = {
            addr: torch.full(shape, torch.nan, dtype=dtype)
            for name, (addr, shape, dtype) in layout.tensors.items()
            if layout.kinds[name] in commands.CACHES
        }
        for addr, data in commands.channel_writes(layout, stale, commands.CACHES):
            self.mem.write(addr, data)

    def reg_write(self, offset: int, value: int) -> None:
        assert 0 <= value < 2**32, value
        if offset >= COMMAND_BUFFER:
            at = offset - COMMAND_BUFFER
            assert at % 4 == 0 and at < len(self.command_buffer), hex(offset)
            self.command_buffer[at : at + 4] = value.to_bytes(4, "little")
        elif offset == REG["CONTROL"]:
            if value & commands.CONTROL_RESET:
                self.regs[REG["STATUS"]] = 0
            if value & commands.CONTROL_START:
                self.regs[REG["STATUS"]] = commands.STATUS_BUSY
        else:
            self.regs[offset] = value

    def reg_read(self, offset: int) -> int:
        return self.regs.get(offset, 0)

    def mem_write(self, addr: int, data: torch.Tensor) -> None:
        self.sync()
        self.mem.write(addr, data)
        self.hbm = None

    def mem_read(self, addr: int, n_bytes: int) -> torch.Tensor:
        logits, (_, shape, dtype) = self.layout.addr("logits"), self.layout.tensors["logits"]
        if logits <= addr and addr + n_bytes <= logits + commands.n_bytes(shape, dtype):
            self.sync(frozenset({commands.Kind.OUTPUT}))  # each token's: not the caches
        else:
            self.sync()
        return self.mem.read(addr, n_bytes)

    def sync(self, kinds: frozenset[commands.Kind] = commands.WRITTEN) -> None:
        """Write what the stores and OUTPUT wrote (of `kinds`) into the bytes
        when they are next used (whole caches: slow at long caps, fine for
        tests)."""
        if self.stale and self.hbm is not None:
            for addr, data in commands.channel_writes(self.layout, self.hbm, kinds):
                self.mem.write(addr, data)
        self.stale = self.stale and kinds != commands.WRITTEN

    def run_until_done(self) -> None:
        assert self.regs.get(REG["STATUS"]) == commands.STATUS_BUSY, "no CONTROL start"
        cmds, size = [], commands.COMMAND_BYTES
        for i in range(self.regs.get(REG["COMMANDS"], 0)):
            raw = bytes(self.command_buffer[i * size : (i + 1) * size])
            if raw[0] not in commands.Op:  # the controller's one check
                self.regs[REG["STATUS"]] = commands.STATUS_ERROR
                self.regs[REG["ERROR"]] = commands.error_value(commands.ERROR_OPCODE, i)
                return
            cmds.append(Command.decode(raw))
        if self.hbm is None:
            self.hbm = commands.channel_reads(self.layout, self.mem.read)
        token, position = self.regs.get(REG["TOKEN"], 0), self.regs.get(REG["POSITION"], 0)
        layout = dataclasses.replace(self.layout, stride=self.regs[REG["STRIDE"]])
        commands.run(cmds, layout, self.hbm, token, position)
        self.stale = True
        self.regs[REG["STATUS"]] = commands.STATUS_DONE
