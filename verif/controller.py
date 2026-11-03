# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/controller.sv on its driver (verif/bulk/controller.cpp), with stub
units: what it handed over when, and its checks against the reference
rules (commands.stops, engine_waits, kv_waits; D-068)."""

import struct
from dataclasses import dataclass

import bulk
import commands
import numpy as np
from commands import REG, Command, Op


@dataclass
class Token:
    """One token's events, cycles from its CONTROL write."""

    handovers: list[tuple[int, int, int]]  # (cycle, unit: 0 engine / 1 vector, wait)
    gos: list[int]
    eng_dones: list[int]
    vec_dones: list[int]
    seen: int  # STATUS first read done or error
    status: int
    error: int
    cycles: int


def register_writes(cmds: list[Command], stride: int) -> list[tuple[int, int]]:
    """What the runtime writes once: the command buffer, COMMANDS, STRIDE."""
    raw = b"".join(c.encode() if isinstance(c.op, Op) else raw_command(c) for c in cmds)
    words = [
        (REG["COMMAND_BUFFER"] + 4 * k, w) for k, (w,) in enumerate(struct.iter_unpack("<I", raw))
    ]
    return [*words, (REG["COMMANDS"], len(cmds)), (REG["STRIDE"], stride)]


def raw_command(c: Command) -> bytes:
    """A command with an opcode the set does not have (its op an int)."""
    return struct.pack(commands.FORMAT, *[int(getattr(c, f)) for f in commands.FIELD_NAMES])


def run(
    writes: list[tuple[int, int]], tokens: list[tuple[int, int]], **plusargs: int
) -> tuple[list[Token], int, int]:
    """The driver: register writes, then each (token, position). Returns
    each token's events, ID and VERSION."""
    words = [
        len(writes),
        *[x for w in writes for x in w],
        len(tokens),
        *[x for t in tokens for x in t],
    ]
    result = bulk.run_driver("controller", np.array(words, dtype=np.uint64), **plusargs)
    assert result.returncode == 0, f"{plusargs}: exit {result.returncode}: {result.stderr.decode()}"
    out = np.frombuffer(result.stdout, dtype="<u4").astype(np.int64).tolist()
    at = 0

    def take(n: int) -> list[int]:
        nonlocal at
        at += n
        return out[at - n : at]

    found = []
    for _ in tokens:
        (n,) = take(1)
        flat = take(3 * n)
        handovers = [tuple(flat[i : i + 3]) for i in range(0, 3 * n, 3)]
        gos, eng, vec = (take(take(1)[0]) for _ in range(3))
        seen, status, error, lo, hi = take(5)
        found.append(Token(handovers, gos, eng, vec, seen, status, error, lo | hi << 32))
    ident, version = take(2)
    assert at == len(out), (at, len(out))
    return found, ident, version


def check(cmds: list[Command], tok: Token) -> None:
    """Assert that the token ran `cmds` as D-068 says: every command but
    OUTPUT and END handed over once, in list order, to its unit; the
    engine's wait bits engine_waits'; each stop after every earlier done;
    each go after its writer's done and its command's handover, in order;
    SCORES and VALUES after their store's done; STATUS done after every
    done, with CYCLES."""
    units = [c.op not in commands.ENGINE_OPS for c in cmds]
    issued = [i for i, c in enumerate(cmds) if c.op not in (Op.OUTPUT, Op.END)]
    assert [u for _, u, _ in tok.handovers] == [int(units[i]) for i in issued], "units or order"
    at = {i: h[0] for i, h in zip(issued, tok.handovers, strict=True)}  # handover cycle
    assert all(at[a] < at[b] for a, b in zip(issued, issued[1:], strict=False)), "list order"
    done, k = {}, {0: 0, 1: 0}
    for i in issued:
        u = int(units[i])
        done[i] = (tok.vec_dones if u else tok.eng_dones)[k[u]]
        k[u] += 1
    assert k[0] == len(tok.eng_dones) and k[1] == len(tok.vec_dones), "dones"
    waits = commands.engine_waits(cmds)
    for i, (_, u, w) in zip(issued, tok.handovers, strict=True):
        assert u == 1 or bool(w) == waits[i], f"command {i}: wait {w}, engine_waits {waits[i]}"
    for i, stop in enumerate(commands.stops(cmds)):
        if stop and i in at:
            assert all(done[j] < at[i] for j in issued if j < i), f"command {i}: stop before a done"
    writer = last_writers(cmds)
    waiting = [i for i in issued if units[i] == 0 and waits[i]]
    assert len(tok.gos) == len(waiting), (len(tok.gos), len(waiting))
    for go, i in zip(tok.gos, waiting, strict=True):
        assert go > at[i], f"command {i}: go before its handover"
        assert go > done[writer[i]], f"command {i}: go before its writer {writer[i]}'s done"
    for i, store in enumerate(commands.kv_waits(cmds)):
        if store is not None:
            assert at[i] > done[store], f"command {i}: handed over before store {store}'s done"
    assert tok.status == commands.STATUS_DONE, f"STATUS {tok.status:#x}, ERROR {tok.error:#x}"
    assert tok.seen > max(done.values(), default=0)
    assert 0 < tok.cycles <= tok.seen, (tok.cycles, tok.seen)


def last_writers(cmds: list[Command]) -> dict[int, int]:
    """Per engine command, the last command before it that wrote its input."""
    last, out = {}, {}
    for i, c in enumerate(cmds):
        if c.op in commands.ENGINE_OPS and c.a in last:
            out[i] = last[c.a]
        if c.dst != commands.Buf.NONE:
            last[c.dst] = i
    return out
