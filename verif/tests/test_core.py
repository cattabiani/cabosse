# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The whole core (rtl/core.sv) driven by the runtime through the
simulation platform (PLAN.md, M7): greedy decoding on the tiny model gives
the golden model's tokens, logits and KV cache bit for bit."""

import time

import pytest
import torch
from generation import SEED, check_generation, tiny_model
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


def generate_on_sim(ports: int, n_prompts: int) -> None:
    """n_prompts conversations of N_NEW tokens on one simulated core."""
    model = tiny_model()
    with SimPlatform(ports, CHAN_W, LENGTH) as p:
        rt = Runtime.open(p, model, CAP)
        for k, prompt in enumerate(prompts(n_prompts, model.config.vocab_size)):
            rt.reset()
            start = time.monotonic()
            try:
                check_generation(rt, model, prompt, N_NEW)
            except AssertionError as e:
                raise AssertionError(f"seed {SEED}, {ports} ports, prompt {k}: {e}") from e
            seconds = time.monotonic() - start
            print(f"{ports} ports, prompt {k}: {rt.position + 1} tokens in {seconds:.1f} s")


@rtl.needs_verilator
def test_tiny_decodes_bit_exact_at_4_ports() -> None:
    """3 prompts x 64 tokens (M7's criterion), one head a VALUES pass."""
    generate_on_sim(4, PROMPTS)


@pytest.mark.wide
@rtl.needs_verilator
def test_tiny_decodes_bit_exact_at_32_ports() -> None:
    """One prompt on F2's 32 ports: two heads a VALUES pass (P's second read)."""
    generate_on_sim(32, 1)
