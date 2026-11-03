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
        mem[offset : offset + len(data)] = data.numpy().tobytes()

    def read(self, addr: int, n_bytes: int) -> torch.Tensor:
        c, offset = divmod(addr, self.channel_bytes)
        mem = self.channels.get(c, bytearray())
        out = torch.zeros(n_bytes, dtype=torch.uint8)
        raw = mem[offset : offset + n_bytes]
        out[: len(raw)] = torch.frombuffer(raw, dtype=torch.uint8) if raw else out[:0]
        return out


class GoldenPlatform:
    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.ports, self.channel_bytes = layout.ports, layout.channel_bytes
        self.mem = Memory(layout.channel_bytes)
        self.regs = {REG["ID"]: commands.ID, REG["VERSION"]: commands.VERSION}
        self.command_buffer = bytearray(commands.COMMAND_BUFFER_BYTES)
        self.hbm: dict[int, torch.Tensor] | None = None  # memory as tensors, once read
        self.started = False
        self.logits: torch.Tensor | None = None

    def reg_write(self, offset: int, value: int) -> None:
        assert 0 <= value < 2**32, value
        if offset >= COMMAND_BUFFER:
            at = offset - COMMAND_BUFFER
            self.command_buffer[at : at + 4] = value.to_bytes(4, "little")
        elif offset == REG["CONTROL"]:
            if value & commands.CONTROL_RESET:
                self.regs[REG["STATUS"]], self.started = 0, False
            if value & commands.CONTROL_START:
                self.regs[REG["STATUS"]], self.started = commands.STATUS_BUSY, True
        else:
            self.regs[offset] = value

    def reg_read(self, offset: int) -> int:
        return self.regs.get(offset, 0)

    def mem_write(self, addr: int, data: torch.Tensor) -> None:
        self.mem.write(addr, data)
        self.hbm = None

    def mem_read(self, addr: int, n_bytes: int) -> torch.Tensor:
        return self.mem.read(addr, n_bytes)

    def run_until_done(self) -> None:
        assert self.started, "run_until_done without CONTROL start"
        self.started = False
        cmds = []
        for i in range(self.regs.get(REG["COMMANDS"], 0)):
            raw = self.command_buffer[i * commands.COMMAND_BYTES :][: commands.COMMAND_BYTES]
            if raw[0] not in commands.Op:  # the controller's one check
                self.regs[REG["STATUS"]], self.regs[REG["ERROR"]] = commands.STATUS_ERROR, i
                return
            cmds.append(Command.decode(bytes(raw)))
        if self.hbm is None:
            self.hbm = commands.channel_reads(self.layout, self.mem.read)
        token, position = self.regs.get(REG["TOKEN"], 0), self.regs.get(REG["POSITION"], 0)
        layout = dataclasses.replace(self.layout, stride=self.regs[REG["STRIDE"]])
        self.logits = commands.run(cmds, layout, self.hbm, token, position)
        caches = frozenset({commands.Kind.K_CACHE, commands.Kind.V_CACHE})
        for addr, data in commands.channel_writes(self.layout, self.hbm, caches):
            self.mem.write(addr, data)  # what the stores wrote, for mem_read
        self.regs[REG["STATUS"]] = commands.STATUS_DONE

    def read_logits(self, n: int) -> torch.Tensor:
        assert self.logits is not None, "no token run"
        return self.logits[:n].clone()
