# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""What the vector unit's tests share: its buffers as the golden model holds
them, the memory behind its AXI port as words, the driver
verif/bulk/vec_unit.cpp run on commands, and the bit-exact comparison of
every buffer and of the memory afterwards (rtl/vec_unit.sv)."""

from dataclasses import dataclass

import bulk
import commands
import numpy as np
import torch
from commands import Buf, Command, Flag, Op
from engine import port_bits as bits  # a buffer's raw bits: the engine's write port's
from fp_inputs import random_bits, with_exponents
from golden import arith, tiny

LANES = 16  # rtl/vec_unit.sv's default
# The memory's channels in the tests: KV_PORTS of 2^CHAN_W bytes (F2: 32 of
# 512 MiB), one byte array from address 0.
KV_PORTS, CHAN_W = 4, 14
MEM_WORDS = (KV_PORTS << CHAN_W) // 4
# Each buffer's format, from the command list's (any model: formats do not
# depend on it); the RTL takes them as Bf16Mask.
BF16_BUFS = frozenset(
    b for b, (dt, _) in commands.buffers(tiny.tiny_config(), 16).items() if dt == commands.BF16
)
FP32_BUFS = tuple(b for b in Buf if b != Buf.NONE and b not in BF16_BUFS)
BF16_MASK = sum(1 << b for b in BF16_BUFS)


def dtype(b: Buf) -> torch.dtype:
    return torch.bfloat16 if b in BF16_BUFS else torch.float32


def from_bits(raw: np.ndarray, b: Buf) -> torch.Tensor:
    t = torch.from_numpy(raw.astype(np.int64))
    return arith.bf16_from_bits(t) if dtype(b) == torch.bfloat16 else arith.f32_from_bits(t)


def random_buffers(rng: np.random.Generator, length: int) -> dict[Buf, torch.Tensor]:
    """Every buffer of `length` elements, random bit patterns: the sources'
    values and the stale data a command must leave alone. NaN and infinite
    patterns are kept to a few percent, so most results are numbers."""
    out = {}
    for b in Buf:
        if b == Buf.NONE:
            continue
        width = 16 if dtype(b) == torch.bfloat16 else 32
        raw = random_bits(rng, length) >> (32 - width)
        # Most values moderate: an exponent near the middle of the range.
        moderate = rng.random(length) < 0.95
        raw = np.where(moderate, with_exponents(rng, length, 127 - 20, 126 + 20, width), raw)
        out[b] = from_bits(raw, b)
    return out


@dataclass
class Step:
    """One command as the driver takes it: the command, the registers it
    runs with, and whether it waits for every earlier command to be done."""

    cmd: Command
    t: int = 1
    stride: int = 16
    wait: bool = False
    token: int = 0


@dataclass
class Result:
    buffers: dict[Buf, np.ndarray]  # raw bits, uint32
    mem: np.ndarray  # the memory's words
    taken: np.ndarray  # per command, the cycle it was taken
    done: np.ndarray  # per command, the cycle of its done_o
    answers: np.ndarray  # the cycle of each write response
    err: bool
    cycles: int


def tensor_at(
    mem: np.ndarray, addr: int, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """The memory's bytes at addr as a tensor (a copy)."""
    n_bytes = commands.n_bytes(shape, dtype)
    raw = mem.view(np.uint8)[addr : addr + n_bytes].copy()
    return torch.from_numpy(raw).view(dtype).reshape(shape)


def cache_rows(cmd: Command) -> list[tuple[int, int, int]]:
    """Where KV_STORE's cache sits in the memory (D-043): per head h and
    position p, (h, p, byte address); position p in channel p mod KV_PORTS,
    row p div KV_PORTS of h's region of cap / KV_PORTS rows."""
    row_bytes = cmd.m * 2
    region = cmd.cap // KV_PORTS * row_bytes
    return [
        (h, p, ((p % KV_PORTS) << CHAN_W) + cmd.addr + h * region + p // KV_PORTS * row_bytes)
        for h in range(cmd.n)
        for p in range(cmd.cap)
    ]


def memory_command(s: Step, buf: dict[Buf, torch.Tensor], mem: np.ndarray) -> None:
    """LOAD or KV_STORE on the buffers and the memory's words, in place, by
    commands.load_command and kv_store_command."""
    c, position = s.cmd, s.t - 1
    if c.op == Op.LOAD:
        dtype = commands.BF16 if c.flags & Flag.SRC_BF16 else commands.F32
        row = s.token if c.flags & Flag.BY_TOKEN else position if c.flags & Flag.BY_POSITION else 0
        by_row = c.flags & (Flag.BY_TOKEN | Flag.BY_POSITION)
        shape = (row + 1, c.m) if by_row else (c.m,)
        commands.load_command(c, buf, tensor_at(mem, c.addr, shape, dtype), s.token, position)
        return
    rows = cache_rows(c)
    cache = torch.empty(c.n, c.cap, c.m, dtype=commands.BF16)
    for h, p, addr in rows:
        cache[h, p] = tensor_at(mem, addr, (c.m,), commands.BF16)
    commands.kv_store_command(c, buf, cache, position)
    raw = mem.view(np.uint8)
    for h, p, addr in rows:
        raw[addr : addr + 2 * c.m] = cache[h, p].view(torch.uint8).numpy()


def golden(
    steps: list[Step], start: dict[Buf, torch.Tensor], mem: np.ndarray
) -> tuple[dict[Buf, np.ndarray], np.ndarray]:
    """The buffers (raw bits) and the memory's words after the commands."""
    buf = {b: x.clone() for b, x in start.items()}
    mem = mem.copy()
    for s in steps:
        if s.cmd.op in (Op.LOAD, Op.KV_STORE):
            memory_command(s, buf, mem)
        else:
            commands.vector_command(s.cmd, buf, s.t, s.stride)
    return {b: bits(x) for b, x in buf.items()}, mem


def words(
    steps: list[Step], start: dict[Buf, torch.Tensor], length: int, mem: np.ndarray
) -> np.ndarray:
    """The driver's input: the memory, every buffer written, the commands,
    every buffer read."""
    groups = range(0, length, LANES)
    out = [len(mem), *mem.tolist(), len(start) * len(groups)]
    for b, x in start.items():
        raw = np.zeros(-(-length // LANES) * LANES, dtype=np.uint32)
        raw[:length] = bits(x)
        for g in groups:
            mask = (1 << min(LANES, length - g)) - 1
            out += [int(b), g, mask, *raw[g : g + LANES].tolist()]
    out.append(len(steps))
    for s in steps:
        c = s.cmd
        addr = [c.addr & 0xFFFFFFFF, c.addr >> 32]
        out += [c.op, c.dst, c.a, c.b, c.c, c.flags, c.n, c.m, c.scalar, c.cap, *addr, s.token]
        out += [s.t, s.stride, int(s.wait)]
    out.append(len(start) * len(groups))
    for b in start:
        out += [x for g in groups for x in (int(b), g)]
    return np.array(out, dtype=np.uint32)


def params(length: int) -> tuple[tuple[str, int], ...]:
    return (("Len", length), ("Bf16Mask", BF16_MASK), ("KvPorts", KV_PORTS), ("ChanW", CHAN_W))


def run(
    steps: list[Step],
    start: dict[Buf, torch.Tensor],
    length: int,
    mem: np.ndarray | None = None,
    **plusargs: int,
) -> Result:
    """The commands on the unit, every buffer filled with `start` first and
    the memory with `mem` (words; by default MEM_WORDS of zeros)."""
    mem = np.zeros(MEM_WORDS, dtype=np.uint32) if mem is None else mem
    argv = words(steps, start, length, mem)
    result = bulk.run_driver("vec_unit", argv, params(length), **plusargs)
    assert result.returncode == 0, result.stderr.decode()
    out = np.frombuffer(result.stdout, dtype="<u4")
    at = 2 * len(steps)
    times = out[:at].reshape(-1, 2)
    n_reads = len(start) * -(-length // LANES) * LANES
    reads = out[at : at + n_reads].reshape(len(start), -1)[:, :length]
    at += n_reads
    got_mem = out[at : at + len(mem)].copy()
    at += len(mem)
    n_answers = int(out[at])
    answers = out[at + 1 : at + 1 + n_answers].copy()
    buffers = {b: r.copy() for b, r in zip(start, reads, strict=True)}
    return Result(buffers, got_mem, times[:, 0], times[:, 1], answers, bool(out[-2]), int(out[-1]))


def assert_same(got: dict[Buf, np.ndarray], want: dict[Buf, np.ndarray], label: str) -> None:
    """Every buffer bit for bit; else the first buffer and element that differ."""
    for b, w in want.items():
        bad = np.flatnonzero(got[b] != w)
        assert not len(bad), (
            f"{label}: {b.name}: {len(bad)} elements differ, first {bad[0]}: "
            f"got 0x{got[b][bad[0]]:08X}, want 0x{w[bad[0]]:08X}"
        )


def check(
    steps: list[Step],
    start: dict[Buf, torch.Tensor],
    length: int,
    label: str,
    mem: np.ndarray | None = None,
    **plusargs: int,
) -> Result:
    """Run, then compare every buffer and the memory with golden()."""
    mem = np.zeros(MEM_WORDS, dtype=np.uint32) if mem is None else mem
    result = run(steps, start, length, mem, **plusargs)
    want_buf, want_mem = golden(steps, start, mem)
    assert_same(result.buffers, want_buf, label)
    bad = np.flatnonzero(result.mem != want_mem)
    assert not len(bad), (
        f"{label}: memory: {len(bad)} words differ, first at byte 0x{4 * bad[0]:X}: "
        f"got 0x{result.mem[bad[0]]:08X}, want 0x{want_mem[bad[0]]:08X}"
    )
    assert not result.err, f"{label}: err_o"
    return result
