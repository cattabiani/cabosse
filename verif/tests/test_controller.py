# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""rtl/controller.sv with stub units (verif/controller.py): the registers,
the ladder's and the tiny model's lists run as D-068 says under random
readiness and latencies, and its errors."""

import dataclasses

import commands
import controller
import perf
import pytest
from commands import Command, Layout, Op
from golden import tiny

import rtl

SEEDS = (1, 2, 3)
STUBS = [  # plusargs: always ready and fast, random and slow
    {"ready": 1000, "latency": 1},
    {"ready": 300, "latency": 1, "latency_max": 60},
    {"ready": 700, "latency": 20, "latency_max": 200, "poll": 5},
]


def tiny_list(ports: int) -> tuple[list[Command], Layout]:
    layout = Layout.of(tiny.tiny_config(), 4 * ports, ports=ports)
    return commands.build(layout), layout


@rtl.needs_verilator
@pytest.mark.parametrize("stubs", STUBS, ids=["fast", "random", "slow"])
@pytest.mark.parametrize("seed", SEEDS)
def test_tiny_tokens_follow_the_rules(seed: int, stubs: dict) -> None:
    cmds, layout = tiny_list(4)
    tokens = [(5, 0), (6, 1), (7, 2)]
    found, _, _ = controller.run(
        controller.register_writes(cmds, layout.stride), tokens, seed=seed, **stubs
    )
    for tok in found:
        controller.check(cmds, tok)


@rtl.needs_verilator
def test_smollm2s_list_follows_the_rules() -> None:
    c = perf.load_config("SmolLM2-135M-Instruct")
    layout = Layout.of(c, 1024)
    cmds = commands.build(layout)
    stubs = STUBS[1]
    found, _, _ = controller.run(
        controller.register_writes(cmds, layout.stride), [(1, 10)], seed=4, **stubs
    )
    controller.check(cmds, found[0])


@rtl.needs_verilator
def test_attention_waits_for_its_store() -> None:
    """SCORES and VALUES right after their stores, with no stop between (in
    build()'s lists RoPE's stop comes first): handed over after the stores'
    dones (kv_waits)."""
    B = commands.Buf
    cmds = [
        Command(Op.MUL, B.QR, B.T, B.W, n=1, m=64),
        Command(Op.KV_STORE, a=B.KR, n=1, m=16, cap=16, addr=0x1000),
        Command(Op.V_STORE, a=B.V, n=1, m=16, cap=16, addr=0x2000),
        Command(Op.SCORES, B.S, B.QR, n=4, m=16, cap=16, kv_heads=1, addr=0x1000),
        Command(Op.VALUES, B.ATT, B.P, n=4, m=16, cap=16, kv_heads=1, addr=0x2000),
        Command(Op.END),
    ]
    found, _, _ = controller.run(
        controller.register_writes(cmds, 16), [(0, 3)], latency=20, latency_max=40
    )
    controller.check(cmds, found[0])


@rtl.needs_verilator
def test_registers_read_back() -> None:
    cmds, layout = tiny_list(4)
    writes = controller.register_writes(cmds, layout.stride)
    found, ident, version = controller.run(writes, [(11, 3)])
    assert (ident, version) == (commands.ID, commands.VERSION)
    assert found[0].cycles > len(cmds)  # at least a cycle a command


@rtl.needs_verilator
def test_an_unknown_opcode_stops_with_its_index() -> None:
    cmds, layout = tiny_list(4)
    cmds[7] = dataclasses.replace(cmds[7], op=99)
    found, _, _ = controller.run(controller.register_writes(cmds, layout.stride), [(1, 0), (2, 0)])
    for tok in found:  # and the next token starts afresh
        assert tok.status == commands.STATUS_ERROR
        assert tok.error == commands.error_value(commands.ERROR_OPCODE, 7), hex(tok.error)
        assert len(tok.handovers) == 7


@rtl.needs_verilator
def test_a_list_without_end_stops() -> None:
    cmds, layout = tiny_list(4)
    writes = controller.register_writes(cmds[:-1], layout.stride)
    found, _, _ = controller.run(writes, [(1, 0)])
    assert found[0].status == commands.STATUS_ERROR
    assert found[0].error == commands.error_value(commands.ERROR_NO_END, len(cmds) - 1), hex(
        found[0].error
    )


def test_check_catches_a_missing_wait() -> None:
    """The check is not vacuous: a token whose waits differ fails it."""
    cmds, _ = tiny_list(4)
    issued = [c for c in cmds if c.op not in (Op.OUTPUT, Op.END)]
    unit = [int(c.op not in commands.ENGINE_OPS) for c in issued]
    tok = controller.Token(
        handovers=[(10 * i, u, 0) for i, u in enumerate(unit)],
        gos=[],
        eng_dones=[10 * i + 5 for i, u in enumerate(unit) if u == 0],
        vec_dones=[10 * i + 5 for i, u in enumerate(unit) if u == 1],
        seen=10 * len(issued) + 10,
        status=commands.STATUS_DONE,
        error=0,
        cycles=10 * len(issued),
    )
    with pytest.raises(AssertionError, match="wait 0, engine_waits True"):
        controller.check(cmds, tok)
