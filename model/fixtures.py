# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Greedy-decode regression fixtures (M1): the golden model's tokens and a
hash of the bits of every logit, for a tiny random-weight config (checked in
CI, and later against the RTL) and for SmolLM2 (checked locally: the weights
are not downloaded in CI).

Pinned values come from the reference platform, x86-64 Linux (D-023).
Regenerate with scripts/make_fixtures.py after a deliberate numerics change,
and review the diff.
"""

from pathlib import Path

import compare
import torch
from golden import arith, decoder, tiny
from transformers import AutoTokenizer, LlamaForCausalLM

N_NEW = 64  # generated tokens per prompt
TINY_CONFIG = {"n_kv_heads": 2, "head_dim": 16, "tied": False}
TINY_SEED = 20261007
TINY_PROMPT_LENGTHS = (5, 9, 16)
# Untied output weights x4 (exact, a power of two): with random tied weights a
# token mostly predicts itself and greedy decoding repeats it; larger logits
# spread the choices.
TINY_LM_HEAD_SCALE = 4
SMOLLM2_PROMPTS = compare.PROMPTS[:3]


def greedy_run(model: decoder.Model, prompt: list[int]) -> dict:
    tokens, logits = decoder.generate(model, prompt, N_NEW)
    return {"prompt": prompt, "tokens": tokens, "logits_sha256": arith.bits_sha256(logits)}


def tiny_state() -> dict[str, torch.Tensor]:
    state = tiny.random_weights(tiny.tiny_config(**TINY_CONFIG), TINY_SEED)
    state["lm_head.weight"] = state["lm_head.weight"] * TINY_LM_HEAD_SCALE
    return state


def tiny_fixture() -> dict:
    """The tiny config with weights from TINY_SEED, three random-token prompts."""
    config, state = tiny.tiny_config(**TINY_CONFIG), tiny_state()
    gen = torch.Generator().manual_seed(TINY_SEED)
    vocab = config.vocab_size
    prompts = [torch.randint(0, vocab, (n,), generator=gen).tolist() for n in TINY_PROMPT_LENGTHS]
    model = decoder.from_state_dict(config, state)
    return {
        "config": TINY_CONFIG,
        "seed": TINY_SEED,
        "weights_sha256": arith.bits_sha256(*(state[k] for k in sorted(state))),
        "runs": [greedy_run(model, p) for p in prompts],
    }


def smollm2_fixture(path: Path) -> dict:
    """SmolLM2-135M-Instruct, three chat prompts. For information, each run
    also records FP32 `transformers`' greedy tokens and the first position
    where they differ from the golden model's (None if they never do)."""
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = decoder.load(path)
    reference = LlamaForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    runs = []
    for text in SMOLLM2_PROMPTS:
        run = greedy_run(model, compare.chat_prompt(tokenizer, text))
        theirs = compare.transformers_greedy(reference, run["prompt"], N_NEW)
        pairs = zip(run["tokens"], theirs, strict=True)
        diverge = next((i for i, (a, b) in enumerate(pairs) if a != b), None)
        run |= {
            "text": tokenizer.decode(run["tokens"]),
            "transformers_fp32_tokens": theirs,
            "first_difference": diverge,
        }
        runs.append(run)
        print(f"{text!r}: first difference from transformers FP32 at {diverge}")
    return {"weights": path.name, "runs": runs}
