# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/vec_unit.sv's LOAD and KV_STORE through its AXI port, against
commands.load_command and kv_store_command: every buffer and the whole memory
bit for bit afterwards (so a write past a row, or into the wrong channel,
shows), with the memory model's own checks (verif/bulk/axi_mem.h).

LOAD in every row mode (the tensor, its row token, its row position), from
BF16 and FP32, into FP32 and BF16 buffers, rows of one beat and many;
KV_STORE for several heads and positions in every channel, its done after
its last write response while later commands run; random sequences of both
mixed with the other commands, with slow and pausing memories; error
responses."""

import dataclasses

import numpy as np
import pytest
import vector
from commands import Buf, Command, Flag, Op
from fp_inputs import random_bits, with_exponents
from test_vec_unit import LENGTH, SEED, random_command, with_waits
from vector import Step

import rtl

LOAD_ADDR = 0x400  # channel 0, below the cache's offset
CACHE_ADDR = 0x2000  # the KV cache's offset in every channel
CAP = 16  # cache rows per head: 4 a channel
SLOW = {"latency": 30, "rate": 600, "pause": 20, "pause_max": 6}
SLOW_WRITES = {"wlatency": 40, "wrate": 600, "wpause": 20, "wpause_max": 6, "waw_after_w": 1}


def random_memory(rng: np.random.Generator) -> np.ndarray:
    """Random words, mostly moderate values whatever the format they are read
    in (as vector.random_buffers): an FP32 word's high half is a moderate
    BF16 value, and so is its low half."""
    n = vector.MEM_WORDS
    low = with_exponents(rng, n, 127 - 20, 126 + 20, 16).astype(np.uint32)
    moderate = with_exponents(rng, n, 127 - 20, 126 + 20, 32) & np.uint32(0xFFFF0000) | low
    return np.where(rng.random(n) < 0.95, moderate, random_bits(rng, n)).astype(np.uint32)


def load(dst: Buf, m: int, flags: Flag) -> Command:
    return Command(Op.LOAD, dst, m=m, flags=flags, addr=LOAD_ADDR)


def kv_store(a: Buf, n: int, m: int) -> Command:
    return Command(Op.KV_STORE, a=a, n=n, m=m, cap=CAP, addr=CACHE_ADDR)


# m: a beat (BF16: 16, FP32: 8), whole groups, a partial FP32 group, a long row.
LOAD_CASES = [
    (src, row, m)
    for src in (Flag.SRC_BF16, Flag.NONE)
    for row in (Flag.NONE, Flag.BY_TOKEN, Flag.BY_POSITION)
    for m in ((16, 48, 256) if src else (8, 24, 32, 256))
]


@rtl.needs_verilator
@pytest.mark.parametrize(("src", "row", "m"), LOAD_CASES)
@pytest.mark.parametrize("dst", [Buf.H, Buf.X])
def test_load(src: Flag, row: Flag, m: int, dst: Buf) -> None:
    """A LOAD alone, into an FP32 buffer (H) and a BF16 one (X, rounded)."""
    rng = np.random.default_rng(SEED + m)
    step = Step(load(dst, m, src | row), t=6, token=3)
    vector.check(
        [step], vector.random_buffers(rng, LENGTH), LENGTH, f"{dst.name} {m}", random_memory(rng)
    )


@rtl.needs_verilator
@pytest.mark.parametrize("slow", [False, True])
def test_loads_back_to_back(slow: bool) -> None:
    """LOADs into different buffers, back to back, from a fast or a slow and
    pausing memory."""
    rng = np.random.default_rng(SEED)
    steps = [
        Step(load(Buf.H, 64, Flag.SRC_BF16 | Flag.BY_TOKEN), token=5),
        Step(load(Buf.COS, 16, Flag.BY_POSITION), t=9),
        Step(load(Buf.SIN, 16, Flag.BY_POSITION), t=9),
        Step(load(Buf.W, 64, Flag.SRC_BF16)),
    ]
    vector.check(
        steps,
        vector.random_buffers(rng, LENGTH),
        LENGTH,
        f"slow {slow}",
        random_memory(rng),
        **(SLOW if slow else {}),
    )


@rtl.needs_verilator
def test_load_far_row() -> None:
    """A row at a token whose offset needs more than the 20 bits of a row
    index (Qwen2.5's last token is 151,935 rows of 56 beats): the memory's
    image starts at the row's 4 KB page, about 260 MiB in."""
    token, m = 151_935, 896
    row_addr = token * m * 2
    base = row_addr - row_addr % 4096
    cmd = Command(Op.LOAD, Buf.H, m=m, flags=Flag.SRC_BF16 | Flag.BY_TOKEN)
    rng = np.random.default_rng(SEED)
    length = 1024
    vector.check(
        [Step(cmd, token=token)],
        vector.random_buffers(rng, length),
        length,
        "far row",
        random_memory(rng),
        base=base,
    )


@rtl.needs_verilator
def test_load_speed() -> None:
    """A long LOAD of BF16 takes a group a cycle once its first beat is in."""
    length = 4096
    rng = np.random.default_rng(SEED)
    cmd = dataclasses.replace(load(Buf.H, length, Flag.SRC_BF16), addr=0)
    result = vector.check(
        [Step(cmd)], vector.random_buffers(rng, length), length, "speed", random_memory(rng)
    )
    took = int(result.done[0] - result.taken[0])
    groups = length // vector.LANES
    print(f"LOAD of {length} BF16: {took} cycles for {groups} groups")
    assert took <= groups + 70, took  # the reader's start, the chain's 50 cycles, the done queue


@rtl.needs_verilator
@pytest.mark.parametrize("m", [16, 64, 128])
@pytest.mark.parametrize("position", [0, 5, 6, 7, CAP - 1])
def test_kv_store(m: int, position: int) -> None:
    """Up to 3 heads at a position, from a slow, pausing memory: the right
    rows of the right channel (every other byte unchanged), done after the
    last write response."""
    rng = np.random.default_rng(SEED + m + position)
    heads = min(3, LENGTH // m)
    step = Step(kv_store(Buf.KR, heads, m), t=position + 1)
    result = vector.check(
        [step],
        vector.random_buffers(rng, LENGTH),
        LENGTH,
        f"m {m}, position {position}",
        random_memory(rng),
        **SLOW_WRITES,
    )
    assert len(result.answers) == heads, result.answers
    assert result.done[0] > result.answers.max(), (result.done, result.answers)


@rtl.needs_verilator
def test_kv_store_into_a_slave_that_waits_for_data() -> None:
    """A write port that takes an address only once its burst's data is in
    (AXI allows it): the unit's data must not wait for the address."""
    rng = np.random.default_rng(SEED)
    steps = [Step(kv_store(Buf.KR, 3, 64), t=7), Step(kv_store(Buf.V, 3, 64), t=7)]
    vector.check(
        steps,
        vector.random_buffers(rng, LENGTH),
        LENGTH,
        "aw after w",
        random_memory(rng),
        waw_after_w=1,
    )


@rtl.needs_verilator
def test_kv_store_lets_later_commands_run() -> None:
    """With slow write responses, the commands after a KV_STORE run while it
    waits (taken before its done), and their dones come after its, in order."""
    rng = np.random.default_rng(SEED)
    steps = [
        Step(kv_store(Buf.KR, 3, 64), t=4),
        Step(kv_store(Buf.V, 3, 64), t=4),
        Step(Command(Op.ROTATE_HALF, Buf.QT, Buf.Q, n=4, m=16)),
        Step(Command(Op.MUL, Buf.T, Buf.G, Buf.U, n=1, m=64)),
    ]
    result = vector.check(
        steps,
        vector.random_buffers(rng, LENGTH),
        LENGTH,
        "kv",
        random_memory(rng),
        wlatency=200,
    )
    assert (np.diff(result.done) > 0).all(), result.done
    assert result.taken[3] < result.done[1], (result.taken, result.done)
    assert result.done[1] > result.answers.max(), (result.done, result.answers)


def random_memory_command(rng: np.random.Generator) -> Step:
    """A random LOAD (into an FP32 buffer, so later commands may read it) or
    KV_STORE."""
    if rng.random() < 0.5:
        src = Flag.SRC_BF16 if rng.random() < 0.5 else Flag.NONE
        row = Flag(int(rng.choice([Flag.NONE, Flag.BY_TOKEN, Flag.BY_POSITION])))
        m = int(rng.choice([16, 32, 48, 64])) if src else int(rng.choice([8, 16, 24, 64]))
        dst = Buf(rng.choice(vector.FP32_BUFS))
        return Step(load(dst, m, src | row), t=int(rng.integers(1, 6)), token=int(rng.integers(6)))
    a = Buf(rng.choice(vector.FP32_BUFS))
    n, m = int(rng.integers(1, 4)), int(rng.choice([16, 32, 64]))
    return Step(kv_store(a, n, m), t=int(rng.integers(1, CAP + 1)))


@rtl.needs_verilator
@pytest.mark.parametrize("seed", range(4))
def test_random_sequences(seed: int) -> None:
    """60 commands, a third of them LOAD and KV_STORE: dependent ones wait,
    the rest run back to back, with slow memories in half of the runs and
    commands offered at random cycles in the other half."""
    rng = np.random.default_rng(SEED + 100 + seed)
    steps = [
        random_memory_command(rng) if rng.random() < 0.33 else random_command(rng, LENGTH)
        for _ in range(60)
    ]
    plusargs = {**SLOW, **SLOW_WRITES} if seed % 2 else {"offer": 300}
    vector.check(
        with_waits(steps),
        vector.random_buffers(rng, LENGTH),
        LENGTH,
        f"seed {SEED + 100 + seed}",
        random_memory(rng),
        seed=seed,
        **plusargs,
    )


@rtl.needs_verilator
@pytest.mark.parametrize("which", ["read", "write"])
def test_error_responses(which: str) -> None:
    """A read beat or a write burst answered with SLVERR raises err_o; the
    commands still finish."""
    rng = np.random.default_rng(SEED)
    steps = [Step(load(Buf.H, 64, Flag.SRC_BF16)), Step(kv_store(Buf.KR, 2, 16), t=3)]
    bad = {"bad": 2} if which == "read" else {"wbad": 1}
    result = vector.run(
        steps, vector.random_buffers(rng, LENGTH), LENGTH, random_memory(rng), **bad
    )
    assert result.err
    assert (result.done > 0).all()


@rtl.needs_verilator
def test_kv_queue_fills() -> None:
    """KV_STOREs of 4 heads of 256 back to back into a write port that takes
    a beat in 20 cycles: the writer holds 9 bursts of 16 beats, more than the
    queue's 64, so groups wait for a place."""
    length = 1024
    rng = np.random.default_rng(SEED)
    steps = [Step(kv_store(Buf.KR if i % 2 else Buf.V, 4, 256), t=i + 1) for i in range(3)]
    result = vector.check(
        steps, vector.random_buffers(rng, length), length, "queue", random_memory(rng), wrate=50
    )
    assert len(result.answers) == 12, result.answers
