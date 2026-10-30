# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""What the vector unit's tests share: its buffers as the golden model holds
them, the memory behind its AXI port as words, the driver
verif/bulk/vec_unit.cpp run on commands, and the bit-exact comparison of
every buffer and of the memory afterwards (rtl/vec_unit.sv). An EngineStep
stands in for an engine command among the commands."""

from dataclasses import dataclass

import bulk
import commands
import numpy as np
import torch
from commands import Buf, Command, Flag, Op
from engine import cache_by_channel
from engine import port_bits as bits  # a buffer's raw bits: the engine's write port's
from fp_inputs import random_bits, with_exponents
from golden import arith, tiny

LANES = 16  # rtl/vec_unit.sv's default
# The memory's channels in the tests: KV_PORTS of 2^CHAN_W bytes (F2: 32 of
# 512 MiB), one byte array from address 0.
KV_PORTS, CHAN_W = 4, 14
MEM_WORDS = (KV_PORTS << CHAN_W) // 4
CMD_WORDS = 16  # a command's words in the driver's input
WRITE_WORDS = 3 + LANES  # an ext write's: buffer, first element, mask, data
# Each buffer's format, from the command list's (any model: formats do not
# depend on it); the RTL takes them as Bf16Mask.
BF16_BUFS = frozenset(
    b for b, (dt, _) in commands.buffers(tiny.tiny_config(), 16).items() if dt == commands.BF16
)
BUFS = tuple(b for b in Buf if b != Buf.NONE)
FP32_BUFS = tuple(b for b in BUFS if b not in BF16_BUFS)
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
    for b in BUFS:
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
class EngineStep:
    """An engine command's stand-in, the driver's engine step: once every
    earlier command is done, read the spans `reads` (buffer, first element,
    length) through the ext port, then write `writes` (buffer, first element,
    the elements)."""

    reads: list[tuple[Buf, int, int]]
    writes: list[tuple[Buf, int, torch.Tensor]]


@dataclass
class Result:
    buffers: dict[Buf, np.ndarray]  # raw bits, uint32
    engine_reads: list[list[np.ndarray]]  # per EngineStep, per read span: raw bits
    mem: np.ndarray  # the memory's words
    taken: np.ndarray  # per command, the cycle it was taken
    done: np.ndarray  # per command, the cycle of its done_o
    answers: np.ndarray  # the cycle of each write response
    err: bool
    cycles: int


def tensor_at(mem: np.ndarray, at: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """The memory's bytes from byte `at` of its image as a tensor (a copy)."""
    raw = mem.view(np.uint8)[at : at + commands.n_bytes(shape, dtype)].copy()
    return torch.from_numpy(raw).view(dtype).reshape(shape)


def cache_view(mem: np.ndarray, cmd: Command) -> np.ndarray:
    """KV_STORE's cache in the memory's image (from address 0), as BF16 bits
    [KV_PORTS, n, cap / KV_PORTS, m] (engine.cache_by_channel's order): the
    same offset in every channel (D-043). A view: writes go to the memory."""
    assert cmd.cap % KV_PORTS == 0, cmd  # whole rows in every channel, as the unit needs
    channels = mem.view(np.uint16).reshape(KV_PORTS, -1)
    rows = cmd.n * cmd.cap // KV_PORTS * cmd.m
    region = channels[:, cmd.addr // 2 : cmd.addr // 2 + rows]
    return region.reshape(KV_PORTS, cmd.n, cmd.cap // KV_PORTS, cmd.m)


def memory_command(s: Step, buf: dict[Buf, torch.Tensor], mem: np.ndarray, base: int) -> None:
    """LOAD or KV_STORE on the buffers and the memory's words (an image of
    the bytes from `base`), in place, by commands.load_command and
    kv_store_command."""
    c, position = s.cmd, s.t - 1
    if c.op == Op.LOAD:
        # The row LOAD reads, as every row of a tensor (stride 0): the image
        # need not hold the rest.
        dtype = commands.BF16 if c.flags & Flag.SRC_BF16 else commands.F32
        row = s.token if c.flags & Flag.BY_TOKEN else position if c.flags & Flag.BY_POSITION else 0
        at = c.addr + row * commands.n_bytes(c.m, dtype) - base
        tensor = tensor_at(mem, at, (c.m,), dtype)
        if c.flags & (Flag.BY_TOKEN | Flag.BY_POSITION):
            tensor = tensor.expand(row + 1, c.m)
        commands.load_command(c, buf, tensor, s.token, position)
        return
    assert base == 0, base  # the cache's channels start at address 0
    view = cache_view(mem, c)
    bits = view.transpose(1, 2, 0, 3).reshape(c.n, c.cap, c.m)  # back to [n, cap, m]
    cache = torch.from_numpy(bits.astype(np.int16)).view(commands.BF16)
    commands.kv_store_command(c, buf, cache, position)
    view[...] = cache_by_channel(cache.view(torch.int16).numpy().view(np.uint16), KV_PORTS)


def golden(
    steps: list[Step], start: dict[Buf, torch.Tensor], mem: np.ndarray, base: int = 0
) -> tuple[dict[Buf, np.ndarray], np.ndarray]:
    """The buffers (raw bits) and the memory's words after the commands."""
    buf = {b: x.clone() for b, x in start.items()}
    mem = mem.copy()
    for s in steps:
        if s.cmd.op in (Op.LOAD, Op.KV_STORE):
            memory_command(s, buf, mem, base)
        else:
            commands.vector_command(s.cmd, buf, s.t, s.stride)
    return {b: bits(x) for b, x in buf.items()}, mem


def span_groups(first: int, length: int) -> range:
    """The groups (their first elements) that hold elements first ... first
    + length - 1."""
    return range(first // LANES * LANES, first + length, LANES)


def group_writes(b: Buf, first: int, x: torch.Tensor) -> np.ndarray:
    """The driver's ext writes of x into b from element `first`, a row per
    group: buffer, first element, lane mask and LANES words."""
    raw = bits(x)
    groups = span_groups(first, len(raw))
    at = first - groups[0]
    lanes = np.zeros(LANES * len(groups), dtype=np.uint32)
    lanes[at : at + len(raw)] = raw
    valid = np.zeros(len(lanes), dtype=np.uint32)
    valid[at : at + len(raw)] = 1
    masks = valid.reshape(-1, LANES) @ (np.uint32(1) << np.arange(LANES, dtype=np.uint32))
    head = np.column_stack([np.full(len(groups), int(b)), np.array(groups), masks])
    return np.hstack([head.astype(np.uint32), lanes.reshape(-1, LANES)])


def group_reads(spans: list[tuple[Buf, int, int]]) -> np.ndarray:
    """The driver's ext reads of the spans (buffer, first, length), a row per
    group: buffer and first element."""
    rows = [(int(b), g) for b, first, n in spans for g in span_groups(first, n)]
    return np.array(rows, dtype=np.uint32).reshape(-1, 2)


def counted(rows: np.ndarray) -> np.ndarray:
    """The driver's list of records: their number, then each."""
    return np.concatenate([np.array([len(rows)], dtype=np.uint32), rows.ravel()])


def words(
    steps: list[Step | EngineStep], start: dict[Buf, torch.Tensor], length: int, mem: np.ndarray
) -> np.ndarray:
    """The driver's input: the memory, every buffer written, the commands,
    the engine steps, every buffer read."""
    no_writes = np.empty((0, WRITE_WORDS), dtype=np.uint32)
    writes = [group_writes(b, 0, v[:length]) for b, v in start.items()]
    cmd_rows, engine = [], []
    for s in steps:
        if isinstance(s, EngineStep):
            writes_s = [group_writes(b, first, v) for b, first, v in s.writes]
            before = np.array([len(cmd_rows)], dtype=np.uint32)
            engine += [
                before,
                counted(group_reads(s.reads)),
                counted(np.vstack([no_writes, *writes_s])),
            ]
            continue
        c = s.cmd
        addr = [c.addr & 0xFFFFFFFF, c.addr >> 32]
        fields = [c.op, c.dst, c.a, c.b, c.c, c.flags, c.n, c.m, c.scalar, c.cap, *addr, s.token]
        cmd_rows.append(fields + [s.t, s.stride, int(s.wait)])
    return np.concatenate(
        [
            counted(mem),
            counted(np.vstack([no_writes, *writes])),
            counted(np.array(cmd_rows, dtype=np.uint32).reshape(-1, CMD_WORDS)),
            np.array([len(engine) // 3], dtype=np.uint32),  # 3 arrays a step
            *engine,
            counted(group_reads([(b, 0, length) for b in start])),
        ]
    )


def read_spans(
    out: np.ndarray, at: int, spans: list[tuple[Buf, int, int]]
) -> tuple[list[np.ndarray], int]:
    """The spans' elements from the driver's reads (group_reads' order) from
    word `at` of its output, and the word after them."""
    got = []
    for _, first, n in spans:
        groups = span_groups(first, n)
        start = at + first - groups[0]
        got.append(out[start : start + n].copy())
        at += LANES * len(groups)
    return got, at


def with_waits(steps: list[Step]) -> list[Step]:
    """Each command waits for every earlier one to be done where the
    controller stops (commands.stops, D-045)."""
    for s, stop in zip(steps, commands.stops([s.cmd for s in steps]), strict=True):
        s.wait = stop
    return steps


def params(length: int, chan_w: int = CHAN_W) -> tuple[tuple[str, int], ...]:
    return (("Len", length), ("Bf16Mask", BF16_MASK), ("KvPorts", KV_PORTS), ("ChanW", chan_w))


def zero_memory() -> np.ndarray:
    return np.zeros(MEM_WORDS, dtype=np.uint32)


def run(
    steps: list[Step | EngineStep],
    start: dict[Buf, torch.Tensor],
    length: int,
    mem: np.ndarray,
    base: int = 0,
    chan_w: int = CHAN_W,
    **plusargs: int,
) -> Result:
    """The commands and engine steps on the unit, every buffer filled with
    `start` first and the memory (an image from byte `base`, channels of
    2^chan_w bytes) with `mem`'s words."""
    argv = words(steps, start, length, mem)
    plusargs = {"base": base, **plusargs} if base else plusargs
    result = bulk.run_driver("vec_unit", argv, params(length, chan_w), **plusargs)
    assert result.returncode == 0, result.stderr.decode()
    out = np.frombuffer(result.stdout, dtype="<u4")
    at = 2 * sum(isinstance(s, Step) for s in steps)
    times = out[:at].reshape(-1, 2)
    engine_reads = []
    for s in steps:
        if isinstance(s, EngineStep):
            got, at = read_spans(out, at, s.reads)
            engine_reads.append(got)
    reads, at = read_spans(out, at, [(b, 0, length) for b in start])
    got_mem = out[at : at + len(mem)]  # a view of the driver's output, no copy
    at += len(mem)
    n_answers = int(out[at])
    answers = out[at + 1 : at + 1 + n_answers].copy()
    buffers = dict(zip(start, reads, strict=True))
    return Result(
        buffers,
        engine_reads,
        got_mem,
        times[:, 0],
        times[:, 1],
        answers,
        bool(out[-2]),
        int(out[-1]),
    )


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
    base: int = 0,
    **plusargs: int,
) -> Result:
    """Run, then compare every buffer and the memory (zeros by default) with
    golden()."""
    mem = zero_memory() if mem is None else mem
    result = run(steps, start, length, mem, base, **plusargs)
    want_buf, want_mem = golden(steps, start, mem, base)
    assert_same(result.buffers, want_buf, label)
    bad = np.flatnonzero(result.mem != want_mem)
    assert not len(bad), (
        f"{label}: memory: {len(bad)} words differ, first at byte 0x{4 * bad[0]:X}: "
        f"got 0x{result.mem[bad[0]]:08X}, want 0x{want_mem[bad[0]]:08X}"
    )
    assert not result.err, f"{label}: err_o"
    return result
