# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The whole core (rtl/core.sv) driven by the runtime through the
simulation platform (PLAN.md, M7), and the core in its F2 wrapper (M9):
greedy decoding on the tiny model gives the golden model's tokens, logits
and KV cache bit for bit."""

import re
import time

import breakdown
import commands
import paths
import pytest
import torch
import vector
from commands import Buf
from generation import SEED, SMOLLM2_PORTS, check_generation, smollm2_on_sim, tiny_model
from runtime import Runtime
from sim_platform import SimPlatform

import rtl

CAP = 128  # the tiny model's whole window
CHAN_W = 20  # 1 MiB channels: the tiny model's layout fits
LENGTH = 1024  # buffer elements: the tiny model's largest is S, 4 heads x 128
N_NEW = 64
PROMPTS = 3


def prompts(n: int, vocab: int) -> list[list[int]]:
    gen = torch.Generator().manual_seed(SEED)
    return [torch.randint(0, vocab, (4 + 6 * k,), generator=gen).tolist() for k in range(n)]


def check_breakdown(cmds: list[commands.Command], runs: list, ports: int, label: str) -> None:
    """Each run's trace has every unit command's handover, go and done
    (breakdown.token asserts the counts), and the units' busy cycles fit in
    CYCLES. `runs` is a conversation's from its start."""
    assert [r[0] for r in runs] == list(range(len(runs))), (label, "positions")
    for position, trace, cycles in runs:
        b = breakdown.token(cmds, trace, cycles, position, ports)
        assert b["units"]["neither"] >= 0, (label, position, b["units"], cycles)
        assert all(r["cycles"] > 0 for r in b["commands"]), (label, position)


def generate_on_sim(ports: int, n_prompts: int, top: str = "core") -> None:
    """n_prompts conversations of N_NEW tokens on one simulated core (or
    `top`, the core in its F2 wrapper), each token's trace checked
    (check_breakdown)."""
    model = tiny_model()
    with SimPlatform(ports, CHAN_W, LENGTH, top) as p:
        rt = Runtime.open(p, model, CAP)
        cmds = commands.build(rt.layout)
        for k, prompt in enumerate(prompts(n_prompts, model.config.vocab_size)):
            label = f"seed {SEED}, {ports} ports, prompt {k}"
            rt.reset()
            p.runs.clear()
            start = time.monotonic()
            check_generation(rt, model, prompt, N_NEW, label)
            seconds = time.monotonic() - start
            check_breakdown(cmds, p.runs, ports, label)
            print(f"{label}: {rt.position + 1} tokens in {seconds:.1f} s, last {rt.cycles} cycles")


def test_the_cores_buffer_ids_are_the_command_lists() -> None:
    """rtl/core.sv's defaults (the F2 build's too): BF16 buffers and P's id."""
    src = (rtl.RTL / "core.sv").read_text()
    assert re.search(rf"Bf16Mask\s*=\s*32'h{vector.BF16_MASK:x}\b", src, re.IGNORECASE)
    assert re.search(rf"BufP\s*=\s*{int(Buf.P)}\b", src)
    assert re.search(rf"Bufs\s*=\s*{len(Buf)}\b", src)


@rtl.needs_verilator
def test_tiny_decodes_bit_exact_at_4_ports() -> None:
    """3 prompts x 64 tokens (M7's criterion), one head a VALUES pass."""
    generate_on_sim(4, PROMPTS)


@rtl.needs_verilator
def test_tiny_decodes_bit_exact_in_the_f2_wrapper() -> None:
    """The same in platforms/f2/core/f2_core.sv (M9): each engine port's
    channel in its addresses, engine port 0 and the vector unit sharing
    HBM port 0, the host's handovers of it, all on one memory."""
    generate_on_sim(4, 1, "f2_core")


@pytest.mark.wide
@rtl.needs_verilator
def test_tiny_decodes_bit_exact_at_32_ports() -> None:
    """One prompt on F2's 32 ports: two heads a VALUES pass (P's second read)."""
    generate_on_sim(32, 1)


@pytest.mark.slow
@pytest.mark.skipif(not paths.SMOLLM2.exists(), reason=f"needs the checkpoint in {paths.SMOLLM2}")
@rtl.needs_verilator
def test_smollm2_decodes_bit_exact_at_32_ports() -> None:
    """M7's criterion: SmolLM2 bit-exact for 8 tokens on F2's 32 ports (a
    build of about 6.5 minutes and 10 GB, then about 100 s a token: run it
    under MemoryMax=12G), each token's trace checked."""
    from golden import decoder

    rt, runs = smollm2_on_sim(decoder.load(paths.SMOLLM2))
    check_breakdown(commands.build(rt.layout), runs, SMOLLM2_PORTS, "SmolLM2")
