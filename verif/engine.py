# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""What the engine's tests share: the matrices placed in every port's
channel (the same offset in each, docs/architecture.md, "Weight layout"),
the driver verif/bulk/engine.cpp run on them, and the golden results."""

from collections.abc import Sequence
from dataclasses import dataclass

import bulk
import commands
import numpy as np
import torch
from golden import arith, dot

BEAT_BYTES = commands.BEAT_BYTES
BEAT_WORDS = BEAT_BYTES // 4
E = commands.E
LANES_PER_PORT = commands.LANES_PER_PORT
STATS = 3  # words after the commands: cycles, stalls, err


@dataclass
class Matvec:
    """One MATVEC: dst = w · x[a], rounded to BF16 if bf16."""

    w: torch.Tensor  # BF16 [rows, cols]
    a: int = 0
    dst: int = 1
    bf16: bool = False


@dataclass
class Result:
    out: list[np.ndarray]  # per command, uint32 [rows]: FP32, or BF16 in the low half
    taken: np.ndarray  # per command, the cycle it was taken
    done: np.ndarray  # per command, the cycle of its done_o
    cycles: int
    r_stalls: int  # cycles any memory waited on RREADY
    err: int  # ports with err_o, a bit each


def n_beats(m: Matvec, ports: int) -> int:
    """Beats each port reserves for the matrix: every pass's, as if full."""
    rows, cols = m.w.shape
    return -(-rows // (ports * LANES_PER_PORT)) * -(-cols // E)


def place(mats: Sequence[Matvec], ports: int, gaps: Sequence[int] | None = None) -> list[int]:
    """Each matrix's offset in every channel, one after another, with
    gaps[i] free beats before matrix i (to move it across 4 KB boundaries)."""
    addrs, beat = [], 0
    for i, m in enumerate(mats):
        beat += gaps[i] if gaps else 0
        addrs.append(beat * BEAT_BYTES)
        beat += n_beats(m, ports)
    return addrs


def channel_images(mats: Sequence[Matvec], addrs: Sequence[int], ports: int) -> np.ndarray:
    """Every port's channel, uint32 [ports, words]: each matrix in the lanes'
    order (commands.lane_order) at its offset."""
    end = max(a // BEAT_BYTES + n_beats(m, ports) for m, a in zip(mats, addrs, strict=True))
    images = np.zeros((ports, end * BEAT_WORDS), dtype=np.uint32)
    for m, addr in zip(mats, addrs, strict=True):
        for p, beats in enumerate(commands.lane_order(m.w, ports * LANES_PER_PORT)):
            words = arith.bits_bf16(beats).numpy().astype(np.uint16).view(np.uint32).ravel()
            start = addr // 4
            images[p, start : start + len(words)] = words
    return images


def fetch(
    mats: Sequence[Matvec],
    xs: dict[int, torch.Tensor],
    ports: int,
    addrs: Sequence[int] | None = None,
    params: tuple[tuple[str, int], ...] = (),
    **plusargs: int,
) -> Result:
    """The driver on these commands, with the x buffers xs (id -> BF16) and
    the matrices at addrs (default: place()); checked to finish, its output
    split per command."""
    addrs = place(mats, ports) if addrs is None else addrs
    images = channel_images(mats, addrs, ports)
    words: list[np.ndarray] = [np.array([images.shape[1]]), images.ravel(), np.array([len(xs)])]
    for buf, x in xs.items():
        bits = arith.bits_bf16(x).numpy().astype(np.uint16)
        packed = np.concatenate([bits, np.zeros(len(bits) % 2, np.uint16)]).view(np.uint32)
        words += [np.array([buf, len(bits)]), packed]
    fields = [
        (m.w.shape[0], m.w.shape[1], a & 0xFFFFFFFF, a >> 32, m.a, m.dst, int(m.bf16))
        for m, a in zip(mats, addrs, strict=True)
    ]
    words += [np.array([len(mats)]), np.array(fields, dtype=np.int64).ravel()]
    stdin = np.concatenate(words).astype(np.uint32)
    result = bulk.run_driver("engine", stdin, (("Ports", ports), *params), **plusargs)
    assert result.returncode == 0, f"{plusargs}: {result.stderr.decode()}"
    words = np.frombuffer(result.stdout, dtype="<u4")
    out, taken, done, at = [], [], [], 0
    for m in mats:
        rows = m.w.shape[0]
        out.append(words[at : at + rows])
        taken.append(words[at + rows])
        done.append(words[at + rows + 1])
        at += rows + 2
    assert len(words) == at + STATS, (len(words), at)
    cycles, stalls, err = map(int, words[at:])
    return Result(out, np.array(taken), np.array(done), cycles, stalls, err)


def expected(m: Matvec, x: torch.Tensor) -> np.ndarray:
    """The golden result as the write port gives it: golden.dot.dot of each
    row with x, FP32 bits, or BF16 bits in the low half."""
    y = dot.dot(m.w, x[: m.w.shape[1]])
    bits = arith.bits_bf16(arith.bf16(y)) if m.bf16 else arith.bits_f32(y)
    return bits.numpy().astype(np.uint32) & (0xFFFF if m.bf16 else 0xFFFFFFFF)
