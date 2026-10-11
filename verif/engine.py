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
import perf
import torch
from golden import arith, decoder, dot

BEAT_BYTES = commands.BEAT_BYTES
BEAT_WORDS = BEAT_BYTES // 4
E = commands.E
LANES_PER_PORT = commands.LANES_PER_PORT
STATS = 4  # words after the commands: cycles, stalls, err, wall time
DESIGN = perf.load_design("v0").values()  # its lanes are set per engine (design())
# F2's HBM read latency, 280 ns (measured by AWS, docs/f2.md), in core
# cycles; our path's latency (switch, clock crossing) waits for M8.
F2_LATENCY_CYCLES = round(280e-9 * DESIGN["clock_hz"])
F2_MEMORY = dict(latency=F2_LATENCY_CYCLES, outstanding=8)  # the real-data and speed runs
SMOLLM2_PROMPT = list(range(1000, 1063))  # the step runs at position 63 (also M4's lane test)
# Large matrices for the bandwidth criterion: SmolLM2's gate and up and a
# slice of its classifier fill whole passes; down (576 rows, 4.5 passes)
# leaves half the ports idle in its last pass, so the perf model's engine
# use (90%) bounds it, not the memory.
LARGE = ((1536, 576), (8192, 576), (576, 1536))
RATES = (1000, 500)  # permille of cycles each port may send a beat


@dataclass
class Matvec:
    """One MATVEC: dst = w · x[a], rounded to BF16 if bf16; with wait, its
    weights are read ahead and it steps only after its go, which the driver
    gives once every earlier command is done, or at once (+go_early); x[a]
    reads as NaN until then, and only later commands may share x[a]."""

    w: torch.Tensor  # BF16 [rows, cols]
    a: int = 0
    dst: int = 1
    bf16: bool = False
    wait: bool = False


@dataclass
class Result:
    out: list[np.ndarray]  # per command, uint32 [rows]: FP32, or BF16 in the low half
    taken: np.ndarray  # per command, the cycle it was taken
    done: np.ndarray  # per command, the cycle of its done_o
    cycles: int
    r_stalls: int  # cycles any memory waited on RREADY
    err: int  # ports with err_o, a bit each
    wall_us: int  # the driver's simulation loop, wall-clock microseconds


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
        (m.w.shape[0], m.w.shape[1], a & 0xFFFFFFFF, a >> 32, m.a, m.dst, int(m.bf16), int(m.wait))
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
    cycles, stalls, err, wall_us = map(int, words[at:])
    return Result(out, np.array(taken), np.array(done), cycles, stalls, err, wall_us)


def expected(m: Matvec, x: torch.Tensor) -> np.ndarray:
    """The golden result as the write port gives it: golden.dot.dot of each
    row with x, FP32 bits, or BF16 bits in the low half."""
    y = dot.dot(m.w, x[: m.w.shape[1]])
    bits = arith.bits_bf16(arith.bf16(y)) if m.bf16 else arith.bits_f32(y)
    return bits.numpy().astype(np.uint32) & (0xFFFF if m.bf16 else 0xFFFFFFFF)


@dataclass
class RealMatvec:
    """A MATVEC recorded from the golden model's command list."""

    name: str  # the Layout name of its matrix
    matvec: Matvec
    want: np.ndarray  # its output as run() wrote it, uint32 FP32 bits


def smollm2_step(model: decoder.Model) -> list[commands.EngineCall]:
    """The engine commands of one SmolLM2 decode step at position 63
    (commands.record_step after SMOLLM2_PROMPT)."""
    return commands.record_step(model, SMOLLM2_PROMPT, SMOLLM2_PROMPT[-1] + 1)[1]


def smollm2_matvecs(model: decoder.Model) -> tuple[list[RealMatvec], dict[int, torch.Tensor]]:
    """The MATVECs of one decode step (commands.record_step at position 63):
    the first and last layers' and the classifier, in command-list order,
    waiting where the controller would (commands.engine_waits), and their x
    buffers: a waiting command opens one, the commands after it that read
    the same buffer share it (k and v with q, up with gate)."""
    last = f"layers.{len(model.layers) - 1}."
    found, xs, x_of = [], {}, {}  # x_of: a command buffer -> its current x buffer
    for c in smollm2_step(model):
        if c.cmd.op != commands.Op.MATVEC or not c.name.startswith(("layers.0.", last, "lm_head")):
            continue
        assert c.out.dtype == torch.float32, (c.name, c.out.dtype)
        if c.wait:
            x_of[c.cmd.a] = len(xs)
            xs[len(xs)] = c.a
        assert c.cmd.a in x_of, f"{c.name}: its buffer's first reader kept here does not wait"
        a = x_of[c.cmd.a]
        assert torch.equal(xs[a], c.a), c.name  # shared, so the same bits
        m = Matvec(c.mem, a=a, dst=len(found), wait=c.wait)
        found.append(RealMatvec(c.name, m, arith.bits_f32(c.out).numpy().astype(np.uint32)))
    return found, xs


def run_smollm2(
    model: decoder.Model, ports: int, **plusargs: int
) -> tuple[list[RealMatvec], Result]:
    """smollm2_matvecs on the engine, checked bit for bit."""
    found, xs = smollm2_matvecs(model)
    result = fetch([f.matvec for f in found], xs, ports, **plusargs)
    for f, got in zip(found, result.out, strict=True):
        assert_bits(got, f.want, f.name)
    assert result.err == 0 and result.r_stalls == 0, (result.err, result.r_stalls)
    return found, result


def assert_bits(got: np.ndarray, want: np.ndarray, label: str) -> None:
    """Results equal, bit for bit; else the count and the first row that differs."""
    bad = np.flatnonzero(got != want)
    assert not len(bad), (
        f"{label}: {len(bad)} rows differ, first row {bad[0]}: "
        f"got 0x{got[bad[0]]:08X}, want 0x{want[bad[0]]:08X}"
    )


def command_cycles(result: Result) -> list[int]:
    """Each command's cycles, from the previous command's done_o (the first
    command's from its start) to its own: for a waiting command, from the
    cycle before its go, so its start-up counts."""
    return np.diff(np.r_[result.taken[0], result.done]).tolist()


def design(ports: int) -> dict[str, float]:
    """Design v0 with this engine's lanes."""
    return DESIGN | {"lanes": ports * LANES_PER_PORT}


def model_cycles(m: Matvec, ports: int, startup: bool = True) -> float:
    """The perf model's cycles for the command, with the start-up of a
    waiting one (engine_startup_cycles, measured at 128 lanes) if startup."""
    return perf.engine_cycles((tuple(m.w.shape),), design(ports), int(m.wait and startup))


def read_share(m: Matvec, result: Result, ports: int, rate: int) -> float:
    """Bytes per cycle of a command run alone, from its start to its done,
    as a share of min(the memory's bytes per cycle at `rate` permille,
    2·E·L) (PLAN.md, M5)."""
    peak = min(ports * BEAT_BYTES * rate / 1000, 2 * E * ports * LANES_PER_PORT)
    rows, cols = m.w.shape
    return 2 * rows * cols / int(result.done[0] - result.taken[0]) / peak


def bandwidth(rows: int, cols: int, ports: int, rate: int, seed: int) -> dict[str, float]:
    """A random rows × cols matrix alone (F2_MEMORY), each port sending a
    beat on `rate` permille of cycles: its read_share, checked bit for bit,
    and the perf model's engine use (the most a partial last pass allows)."""
    gen = torch.Generator().manual_seed(seed)
    m = Matvec(torch.randn(rows, cols, generator=gen).to(torch.bfloat16))
    x = torch.randn(cols, generator=gen).to(torch.bfloat16)
    result = fetch([m], {0: x}, ports, rate=rate, seed=seed, **F2_MEMORY)
    assert_bits(result.out[0], expected(m, x), f"{rows} x {cols}, rate {rate}, seed {seed}")
    assert result.err == 0 and result.r_stalls == 0, (result.err, result.r_stalls)
    use = perf.engine_use(((rows, cols),), design(ports))
    return {"share": read_share(m, result, ports, rate), "engine_use": use}
