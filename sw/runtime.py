# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The host runtime (PLAN.md, M7): lays a model out in the accelerator's
memory, writes its command list once, then runs one decode step per token
with the token id and the position in registers (D-003). The same code
drives every Platform (platforms.py).

It validates the command list and the layout before writing them (M7's
answers): the hardware checks only for unknown opcodes, so whatever would
hang it or silently give wrong bits is refused here."""

import struct
from dataclasses import dataclass

import commands
import torch
from commands import REG, Layout
from golden import decoder

from platforms import Platform


class ContextFull(Exception):
    """The conversation reached the end of the cache: contexts beyond it
    are not supported yet (PLAN.md, Non-goals)."""


class DeviceError(Exception):
    """The controller stopped with STATUS error."""


@dataclass
class Runtime:
    """A model loaded on a platform, and the position of its conversation."""

    platform: Platform
    layout: Layout
    position: int = 0
    cycles: int = 0  # the last token's, from CYCLES

    @staticmethod
    def open(platform: Platform, model: decoder.Model, cap: int | None = None) -> Runtime:
        """Lay `model` out for `cap` positions (by default its whole window),
        validate, and write the weights and the command list. The KV cache is
        not written: a token reads only the positions it and earlier ones
        stored."""
        c = model.config
        cap = c.max_position_embeddings if cap is None else cap
        layout = Layout.of(c, cap, ports=platform.ports, channel_bytes=platform.channel_bytes)
        cmds = commands.build(layout)
        commands.validate(layout, cmds, platform.buffer_elements)
        p = platform
        found = p.reg_read(REG["ID"]), p.reg_read(REG["VERSION"])
        if found != (commands.ID, commands.VERSION):
            raise DeviceError(
                f"ID/VERSION 0x{found[0]:08X}/{found[1]}, not a Cabosse of this version"
            )
        p.reg_write(REG["CONTROL"], commands.CONTROL_RESET)
        hbm = commands.load(model, layout)
        for addr, data in commands.channel_writes(layout, hbm, commands.WEIGHTS):
            p.mem_write(addr, data)
        raw = b"".join(cmd.encode() for cmd in cmds)
        for k, (word,) in enumerate(struct.iter_unpack("<I", raw)):
            p.reg_write(REG["COMMAND_BUFFER"] + 4 * k, word)
        p.reg_write(REG["COMMANDS"], len(cmds))
        p.reg_write(REG["STRIDE"], layout.stride)
        return Runtime(platform, layout)

    @property
    def context(self) -> int:
        """Positions a conversation can hold."""
        return min(self.layout.cap, self.layout.stride)

    def reset(self) -> None:
        """A new conversation (stale cache rows need no clearing)."""
        self.position = 0

    def step(self, token: int) -> torch.Tensor:
        """One decode step of `token` at the next position: its FP32 logits."""
        if self.position >= self.context:
            raise ContextFull(
                f"the conversation reached {self.context} tokens, the end of the context;"
                " longer contexts are not supported yet. Start a new conversation."
            )
        vocab = self.layout.config.vocab_size
        if not 0 <= token < vocab:
            raise ValueError(f"token {token} outside the vocabulary of {vocab}")
        p = self.platform
        p.reg_write(REG["TOKEN"], token)
        p.reg_write(REG["POSITION"], self.position)
        p.reg_write(REG["CONTROL"], commands.CONTROL_START)
        p.run_until_done()
        status = p.reg_read(REG["STATUS"])
        if status & commands.STATUS_ERROR or not status & commands.STATUS_DONE:
            raise DeviceError(f"STATUS 0x{status:X}, ERROR 0x{p.reg_read(REG['ERROR']):X}")
        self.cycles = p.reg_read(REG["CYCLES_LO"]) | p.reg_read(REG["CYCLES_HI"]) << 32
        self.position += 1
        return p.read_logits(vocab)

    def generate(self, prompt: list[int], n_new: int) -> tuple[list[int], torch.Tensor]:
        """Greedy decoding, as decoder.generate (the lowest index on a tie):
        the prompt one token at a time through the decode step (PLAN.md,
        Non-goals), then n_new tokens. Returns them and the logits of every
        position run but the last, [len(prompt) + n_new - 1, vocab]. Raises
        ContextFull at the end of the context."""
        assert prompt and n_new > 0, (len(prompt), n_new)
        logits = [self.step(t) for t in prompt]
        tokens = [int(logits[-1].argmax())]
        while len(tokens) < n_new:
            logits.append(self.step(tokens[-1]))
            tokens.append(int(logits[-1].argmax()))
        return tokens, torch.stack(logits)

    def kv_cache(self) -> dict[int, torch.Tensor]:
        """The KV caches as the device holds them (address -> [kv_heads, cap,
        d], BF16), read back for checks."""
        return commands.channel_reads(self.layout, self.platform.mem_read, commands.CACHES)
