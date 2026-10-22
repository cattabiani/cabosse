# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""What the engine's tests share: the matrices and KV caches placed in every
port's channel (the same offset in each, docs/architecture.md, "Weight
layout" and "KV cache layout"), the driver verif/bulk/engine.cpp run on
them, and the golden results."""

from collections.abc import Sequence
from dataclasses import dataclass

import bulk
import commands
import long_cache
import numpy as np
import paths
import perf
import torch
from golden import arith, decoder, dot

BEAT_BYTES = commands.BEAT_BYTES
BEAT_WORDS = BEAT_BYTES // 4
E = commands.E
LANES_PER_PORT = commands.LANES_PER_PORT
STATS = 4  # words after the commands: cycles, stalls, err, wall time
MAX_D = 128  # rtl/engine.sv's MaxD
DESIGN = perf.load_design("v0").values()  # its lanes are set per engine (design())
# F2's HBM read latency, 280 ns (measured by AWS, docs/f2.md), in core
# cycles; our path's latency (switch, clock crossing) waits for M8.
F2_LATENCY_CYCLES = round(280e-9 * DESIGN["clock_hz"])
F2_MEMORY = dict(latency=F2_LATENCY_CYCLES, outstanding=8)  # the real-data and speed runs
SMOLLM2_PROMPT = list(range(1000, 1063))  # the step runs at position 63 (also M4's lane test)
SMOLLM2_POSITIONS = (63, 1023, 8191)  # PLAN.md, M5: 1023 and 8191 read the long cache
# Large matrices for the bandwidth criterion: SmolLM2's gate and up and a
# slice of its classifier fill whole passes; down (576 rows, 4.5 passes)
# leaves half the ports idle in its last pass, so the perf model's engine
# use (90%) bounds it, not the memory.
LARGE = ((1536, 576), (8192, 576), (576, 1536))
RATES = (1000, 500)  # permille of cycles each port may send a beat
OP_MATVEC, OP_SCORES, OP_VALUES = 0, 1, 2  # rtl/engine.sv's cmd_op_i


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
class Attention:
    """SCORES (values=False) or VALUES on a KV cache of kv_heads × cap
    positions × d (positions past t may hold anything): x[a] is q (heads × d)
    or p (heads rows of cap, past t anything). SCORES writes dst[h * cap +
    i] for i < t, VALUES dst[h * d + c]. wait as for Matvec."""

    values: bool
    cache: torch.Tensor  # BF16 [kv_heads, cap, d]
    heads: int
    t: int
    a: int = 0
    dst: int = 1
    scale: float = 0.125  # SCORES; rounded to FP32
    bf16: bool = False
    wait: bool = False

    @property
    def group(self) -> int:
        return self.heads // self.cache.shape[0]


Command = Matvec | Attention


def n_rows(c: Command) -> int:
    """The rows of dst the command's output spans (the driver's output)."""
    if isinstance(c, Matvec):
        return c.w.shape[0]
    _, cap, d = c.cache.shape
    return c.heads * (d if c.values else cap)


def fields(c: Command, addr: int) -> tuple[int, ...]:
    """The driver's command words: op, rows, cols, address (low, high), a,
    dst, bf16, wait, heads, group, t, cap, scale."""
    common = (addr & 0xFFFFFFFF, addr >> 32, c.a, c.dst, int(c.bf16), int(c.wait))
    if isinstance(c, Matvec):
        return (OP_MATVEC, *c.w.shape, *common, 0, 0, 0, 0, 0)
    _, cap, d = c.cache.shape
    scale = commands.f32_bits(c.scale)
    op = OP_VALUES if c.values else OP_SCORES
    return (op, 0, d, *common, c.heads, c.group, c.t, cap, scale)


@dataclass
class Result:
    out: list[np.ndarray]  # per command, uint32 [rows]: FP32, or BF16 in the low half
    taken: np.ndarray  # per command, the cycle it was taken
    done: np.ndarray  # per command, the cycle of its done_o
    cycles: int
    r_stalls: int  # cycles any memory waited on RREADY
    err: int  # ports with err_o, a bit each
    wall_us: int  # the driver's simulation loop, wall-clock microseconds


def n_beats(m: Command, ports: int) -> int:
    """Beats each port reserves: a matrix's passes, as if full; a cache's
    regions of cap / ports rows per KV head (D-043)."""
    if isinstance(m, Attention):
        kv_heads, cap, d = m.cache.shape
        return commands.n_bytes((kv_heads, cap // ports, d), torch.bfloat16) // BEAT_BYTES
    rows, cols = m.w.shape
    return -(-rows // (ports * LANES_PER_PORT)) * -(-cols // E)


def bf16_bits(t: torch.Tensor) -> np.ndarray:
    """t's BF16 bit patterns, uint16, in t's shape."""
    return arith.bits_bf16(t.contiguous()).numpy().astype(np.uint16)


def check_attention(c: Attention, lanes: int) -> None:
    """The attention requirements the engine does not check (rtl/engine.sv's
    header): a command past them fails here, not as wrong bits."""
    kv_heads, cap, d = c.cache.shape
    assert 16 <= d <= MAX_D and d & (d - 1) == 0, f"d a power of two from 16 to {MAX_D}: d = {d}"
    assert not c.values or d <= lanes, f"VALUES needs d <= L: d = {d}, L = {lanes}"
    assert cap % lanes == 0, f"cap a multiple of L: cap = {cap}, L = {lanes}"
    assert 1 <= c.t <= cap, f"1 <= t <= cap: t = {c.t}, cap = {cap}"
    assert c.heads % kv_heads == 0, f"group dividing heads: {c.heads} heads, {kv_heads} KV"


def place(mats: Sequence[Command], ports: int, gaps: Sequence[int] | None = None) -> list[int]:
    """Each matrix's offset in every channel, one after another, with
    gaps[i] free beats before matrix i (to move it across 4 KB boundaries)."""
    addrs, beat = [], 0
    for i, m in enumerate(mats):
        beat += gaps[i] if gaps else 0
        addrs.append(beat * BEAT_BYTES)
        beat += n_beats(m, ports)
    return addrs


def channel_images(mats: Sequence[Command], addrs: Sequence[int], ports: int) -> np.ndarray:
    """Every port's channel, uint32 [ports, words]: each matrix in the lanes'
    order (commands.lane_order), each cache by D-043 (position i of a KV
    head on channel i mod ports, row i div ports of its region), at its
    offset."""
    end = max(a // BEAT_BYTES + n_beats(m, ports) for m, a in zip(mats, addrs, strict=True))
    images = np.zeros((ports, end * BEAT_WORDS), dtype=np.uint32)
    for m, addr in zip(mats, addrs, strict=True):
        if isinstance(m, Attention):
            kv_heads, cap, d = m.cache.shape
            bits = bf16_bits(m.cache)
            # [kv, row, channel, d] -> channel-major: each channel's rows in order
            rows = bits.reshape(kv_heads, cap // ports, ports, d).transpose(2, 0, 1, 3)
            words = np.ascontiguousarray(rows).view(np.uint32).reshape(ports, -1)
            images[:, addr // 4 : addr // 4 + words.shape[1]] = words
            continue
        for p, beats in enumerate(commands.lane_order(m.w, ports * LANES_PER_PORT)):
            words = bf16_bits(beats).view(np.uint32).ravel()
            start = addr // 4
            images[p, start : start + len(words)] = words
    return images


def fetch(
    mats: Sequence[Command],
    xs: dict[int, torch.Tensor],
    ports: int,
    addrs: Sequence[int] | None = None,
    params: tuple[tuple[str, int], ...] = (),
    **plusargs: int,
) -> Result:
    """The driver on these commands, with the x buffers xs (id -> BF16) and
    the matrices at addrs (default: place()); checked to finish, its output
    split per command."""
    for m in mats:
        if isinstance(m, Attention):
            check_attention(m, ports * LANES_PER_PORT)
    addrs = place(mats, ports) if addrs is None else addrs
    images = channel_images(mats, addrs, ports)
    words: list[np.ndarray] = [np.array([images.shape[1]]), images.ravel(), np.array([len(xs)])]
    for buf, x in xs.items():
        bits = bf16_bits(x)
        packed = np.concatenate([bits, np.zeros(len(bits) % 2, np.uint16)]).view(np.uint32)
        words += [np.array([buf, len(bits)]), packed]
    cmd_words = [fields(m, a) for m, a in zip(mats, addrs, strict=True)]
    words += [np.array([len(mats)]), np.array(cmd_words, dtype=np.int64).ravel()]
    stdin = np.concatenate(words).astype(np.uint32)
    result = bulk.run_driver("engine", stdin, (("Ports", ports), *params), **plusargs)
    assert result.returncode == 0, f"{plusargs}: {result.stderr.decode()}"
    words = np.frombuffer(result.stdout, dtype="<u4")
    out, taken, done, at = [], [], [], 0
    for m in mats:
        rows = n_rows(m)
        out.append(words[at : at + rows])
        taken.append(words[at + rows])
        done.append(words[at + rows + 1])
        at += rows + 2
    assert len(words) == at + STATS, (len(words), at)
    cycles, stalls, err, wall_us = map(int, words[at:])
    return Result(out, np.array(taken), np.array(done), cycles, stalls, err, wall_us)


def golden_attention(c: Attention, x: torch.Tensor) -> torch.Tensor:
    """FP32 [n_rows(c)]: the golden SCORES (0 at positions past t) or VALUES
    (docs/numerics.md, attention), head h on KV head h // group."""
    kv_heads, cap, d = c.cache.shape
    kv = c.cache[torch.arange(c.heads) // c.group, : c.t]  # [heads, t, d]
    if c.values:
        p = x[: c.heads * cap].reshape(c.heads, cap)[:, : c.t]
        return decoder.attention_values(p, kv).reshape(-1)
    q = x[: c.heads * d].reshape(c.heads, d)
    scores = decoder.attention_scores(q, kv, torch.tensor(c.scale, dtype=torch.float32))
    out = torch.zeros(c.heads, cap, dtype=torch.float32)
    out[:, : c.t] = scores
    return out.reshape(-1)


def expected(m: Command, x: torch.Tensor) -> np.ndarray:
    """The golden result as the write port gives it: golden.dot.dot of each
    row with x (or golden_attention), FP32 bits, or BF16 bits in the low
    half."""
    y = golden_attention(m, x) if isinstance(m, Attention) else dot.dot(m.w, x[: m.w.shape[1]])
    bits = arith.bits_bf16(arith.bf16(y)) if m.bf16 else arith.bits_f32(y)
    return bits.numpy().astype(np.uint32) & (0xFFFF if m.bf16 else 0xFFFFFFFF)


@dataclass
class RealCommand:
    """An engine command recorded from the golden model's command list."""

    name: str  # the Layout name of what it reads in HBM
    command: Command
    want: np.ndarray  # its output as the write port gives it (expected()'s format)


def smollm2_step(
    model: decoder.Model, position: int = len(SMOLLM2_PROMPT)
) -> list[commands.EngineCall] | None:
    """The engine commands of one SmolLM2 decode step at `position`
    (commands.record_step): after SMOLLM2_PROMPT at 63, else after the long
    cache's first `position` tokens (model/long_cache.py; None if it was not
    computed). The step's token is the prompt's last plus one."""
    if position == len(SMOLLM2_PROMPT):
        prompt, cache = SMOLLM2_PROMPT, None
    else:
        saved = long_cache.load(model, paths.SMOLLM2)
        if saved is None:
            return None
        tokens, cache = saved
        prompt, cache.length = tokens[:position], position
    return commands.record_step(model, prompt, prompt[-1] + 1, cache)[1]


def real_attention(
    c: commands.EngineCall, a: int, dst: int, lanes: int
) -> tuple[RealCommand, torch.Tensor]:
    """A recorded SCORES or VALUES as the engine takes it: the cache padded
    to whole tiles of `lanes` positions, p from rows of t to rows of that
    cap (both NaN past t, which the engine leaves out), and the output as
    the write port gives it (scores at h·cap, 0 past t; p.V rounded to BF16
    where the buffer is BF16); and its x buffer."""
    kv_heads, t, d = c.mem.shape
    heads, values = c.cmd.n, c.cmd.op == commands.Op.VALUES
    cap = -(-t // lanes) * lanes
    cache = torch.full((kv_heads, cap, d), float("nan"), dtype=torch.bfloat16)
    cache[:, :t] = c.mem
    if values:
        x = torch.full((heads, cap), float("nan"), dtype=torch.bfloat16)
        x[:, :t] = c.a[: heads * t].reshape(heads, t)
        x = x.reshape(-1)
        out = c.out[: heads * d]
    else:
        x = c.a[: heads * d]
        out = torch.zeros(heads, cap, dtype=torch.float32)
        out[:, :t] = c.out[: heads * t].reshape(heads, t)
        out = out.reshape(-1)
    bf16 = out.dtype == torch.bfloat16
    scale = 1.0 if values else float(commands.f32_value(c.cmd.scalar))
    m = Attention(values, cache, heads, t, a=a, dst=dst, scale=scale, bf16=bf16, wait=c.wait)
    bits = arith.bits_bf16(out) if bf16 else arith.bits_f32(out)
    want = bits.numpy().astype(np.uint32) & (0xFFFF if bf16 else 0xFFFFFFFF)
    return RealCommand(c.name, m, want), x


def smollm2_commands(
    model: decoder.Model, ports: int, position: int = len(SMOLLM2_PROMPT)
) -> tuple[list[RealCommand], dict[int, torch.Tensor]] | None:
    """The engine commands of one decode step at `position` (smollm2_step;
    None without the long cache): the first and last layers' MATVECs,
    SCORES and VALUES and the classifier, in command-list order, waiting
    where the controller would (commands.engine_waits), and their x buffers:
    a waiting command opens one, the commands after it that read the same
    buffer share it (k and v with q, up with gate)."""
    step = smollm2_step(model, position)
    if step is None:
        return None
    last = f"layers.{len(model.layers) - 1}."
    found, xs, x_of = [], {}, {}  # x_of: a command buffer -> its current x buffer
    for c in step:
        if not c.name.startswith(("layers.0.", last, "lm_head")):
            continue
        if c.cmd.op != commands.Op.MATVEC:
            assert c.wait, f"{c.name}: attention waits for its KV_STORE (D-044)"
            f, xs[len(xs)] = real_attention(c, len(xs), len(found), ports * LANES_PER_PORT)
            found.append(f)
            continue
        assert c.out.dtype == torch.float32, (c.name, c.out.dtype)
        if c.wait:
            x_of[c.cmd.a] = len(xs)
            xs[len(xs)] = c.a
        assert c.cmd.a in x_of, f"{c.name}: its buffer's first reader kept here does not wait"
        a = x_of[c.cmd.a]
        assert torch.equal(xs[a], c.a), c.name  # shared, so the same bits
        m = Matvec(c.mem, a=a, dst=len(found), wait=c.wait)
        found.append(RealCommand(c.name, m, arith.bits_f32(c.out).numpy().astype(np.uint32)))
    return found, xs


def run_smollm2(
    model: decoder.Model, ports: int, position: int = len(SMOLLM2_PROMPT), **plusargs: int
) -> tuple[list[RealCommand], Result] | None:
    """smollm2_commands on the engine, checked bit for bit (None without
    the long cache)."""
    got = smollm2_commands(model, ports, position)
    if got is None:
        return None
    found, xs = got
    result = fetch([f.command for f in found], xs, ports, **plusargs)
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


def model_cycles(m: Command, ports: int, startup: bool = True) -> float:
    """The perf model's cycles for the command (perf.attention's shapes and
    tile fills for SCORES and VALUES), with the start-up of a waiting one
    (engine_startup_cycles, measured at 128 lanes) if startup."""
    fill_cols = 0
    if isinstance(m, Attention):
        kv_heads, _, d = m.cache.shape
        op = commands.Op.VALUES if m.values else commands.Op.SCORES
        cmd = commands.Command(op, n=m.heads, m=d, kv_heads=kv_heads)
        step = perf.attention(cmd, m.t, ports * LANES_PER_PORT)
        shapes, fill_cols = step.engine_shapes, step.fill_cols
    else:
        shapes = (tuple(m.w.shape),)
    return perf.engine_cycles(shapes, design(ports), int(m.wait and startup), fill_cols)


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
