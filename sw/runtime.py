# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The host runtime (PLAN.md, M7): lays a model out in the accelerator's
memory, writes its command list once, then runs one decode step per token
with the token id and the position in registers (D-003). The same code
drives every Platform (platforms.py).

It validates the command list and the layout before writing them (M7's
answers): the hardware checks only for unknown opcodes, so whatever would
hang it or silently give wrong bits is refused here."""

import dataclasses
from dataclasses import dataclass

import commands
import torch
from commands import REG, Command, Flag, Kind, Layout, Op
from golden import decoder

from platforms import Platform

# Field widths of the RTL's defaults (rtl/engine.sv, rtl/vec_unit.sv).
ROW_BITS = 20  # rows, n, m, cap, stride, token, positions
COL_BITS = 16  # the engine's columns
HEAD_BITS = 8  # attention's heads and KV heads
HEAD_DIMS = (16, 32, 64, 128)  # the engine's: a power of two, 16 to 128
ENGINE_ADDR_ALIGN = commands.BEAT_BYTES


class ContextFull(Exception):
    """The conversation reached the end of the cache: contexts beyond it
    are not supported yet (PLAN.md, Non-goals)."""


class DeviceError(Exception):
    """The controller stopped with STATUS error."""


def validate(layout: Layout, cmds: list[Command]) -> None:
    """Raise ValueError on the first command the hardware cannot run
        correctly, or on a layout it cannot hold. Ordering needs no check: the
    controller derives its waits and stops from the list itself (D-068), so
    commands.hazards() finds none in any list."""
    lanes = layout.lanes

    def need(ok: bool, i: int, why: str) -> None:
        if not ok:
            raise ValueError(f"command {i} ({cmds[i].op!r}): {why}")

    most = commands.COMMAND_BUFFER_BYTES // commands.COMMAND_BYTES
    if not 0 < len(cmds) <= most:
        raise ValueError(f"{len(cmds)} commands: the buffer holds 1 to {most}")
    if layout.stride % commands.STRIDE_ALIGN or not 0 < layout.stride < 2**ROW_BITS:
        raise ValueError(
            f"stride {layout.stride}: a multiple of {commands.STRIDE_ALIGN} below 2^{ROW_BITS}"
        )
    if layout.end_bytes > layout.channel_bytes:
        raise ValueError(
            f"layout: {layout.end_bytes:,} bytes, channel 0 holds {layout.channel_bytes:,}"
        )
    for i, cmd in enumerate(cmds):
        need(cmd.op in commands.OPS, i, "unknown opcode")
        need((cmd.op == Op.END) == (i == len(cmds) - 1), i, "END must be the last command, once")
        need(commands.flags_ok(cmd), i, f"flags {cmd.flags!r} not accepted")
        need(commands.overlap_ok(cmd), i, "dst overlaps a source it reads again")
        need(max(cmd.n, cmd.m, cmd.cap) < 2**ROW_BITS, i, f"n, m or cap of {ROW_BITS} bits or more")
        if cmd.op in (Op.END, Op.OUTPUT):
            continue
        need(cmd.m >= 1 or bool(cmd.flags & Flag.LEN_T), i, "no columns (m = 0)")  # LEN_T: t
        if cmd.op != Op.LOAD:
            need(cmd.n >= 1, i, "no rows (n = 0)")
        if cmd.op == Op.ROTATE_HALF:
            need(cmd.m % 2 == 0, i, "odd row length")
        if cmd.op in commands.ENGINE_OPS:
            need(cmd.m < 2**COL_BITS, i, f"m of {COL_BITS} bits or more")
            need(cmd.addr % ENGINE_ADDR_ALIGN == 0, i, "address not beat-aligned")
        if cmd.op in commands.ENGINE_OPS - {Op.MATVEC} | commands.STORE_OPS:
            need(cmd.cap % lanes == 0, i, f"cap {cmd.cap} not a multiple of the {lanes} lanes")
            need(cmd.m in HEAD_DIMS, i, f"head dimension {cmd.m} not one of {HEAD_DIMS}")
        if cmd.op in (Op.VALUES, Op.V_STORE):
            need(cmd.m <= lanes, i, f"head dimension {cmd.m} above the {lanes} lanes")
        if cmd.op in (Op.SCORES, Op.VALUES):
            need(0 < cmd.kv_heads and cmd.n % cmd.kv_heads == 0, i, "KV heads not dividing heads")
            need(cmd.n < 2**HEAD_BITS, i, f"heads of {HEAD_BITS} bits or more")


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
        validate(layout, cmds)
        p = platform
        if (p.reg_read(REG["ID"]), p.reg_read(REG["VERSION"])) != (commands.ID, commands.VERSION):
            raise DeviceError(
                f"ID/VERSION 0x{p.reg_read(REG['ID']):08X}/{p.reg_read(REG['VERSION'])}"
            )
        p.reg_write(REG["CONTROL"], commands.CONTROL_RESET)
        hbm = commands.load(model, layout)
        weights = frozenset({Kind.LOADED, Kind.MATRIX})
        for addr, data in commands.channel_writes(layout, hbm, weights):
            p.mem_write(addr, data)
        raw = b"".join(cmd.encode() for cmd in cmds)
        words = torch.frombuffer(bytearray(raw), dtype=torch.int32).tolist()
        for k, word in enumerate(words):
            p.reg_write(REG["COMMAND_BUFFER"] + 4 * k, word & 0xFFFFFFFF)
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
        assert 0 <= token < vocab, (token, vocab)
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
        names = [n for n, k in self.layout.kinds.items() if k in (Kind.K_CACHE, Kind.V_CACHE)]
        sub = dataclasses.replace(self.layout, tensors={n: self.layout.tensors[n] for n in names})
        return commands.channel_reads(sub, self.platform.mem_read)
