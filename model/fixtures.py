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

import hashlib
import platform
import sys

import compare
import torch
from golden import arith, decoder, tiny
from transformers import LlamaForCausalLM

REFERENCE_PLATFORM = sys.platform == "linux" and platform.machine() == "x86_64"
N_NEW = 64  # generated tokens per prompt
TINY_SEED = 20261007
TINY_PROMPT_LENGTHS = (5, 9, 16)
SMOLLM2_PROMPTS = compare.PROMPTS[:3]


def bits_sha256(*tensors: torch.Tensor) -> str:
    """SHA-256 of the bit patterns of BF16 or FP32 tensors (little-endian)."""
    h = hashlib.sha256()
    for t in tensors:
        bits = arith.bits_bf16(t) if t.dtype == torch.bfloat16 else arith.bits_f32(t)
        width = "<u2" if t.dtype == torch.bfloat16 else "<u4"
        h.update(bits.numpy().astype(width).tobytes())
    return h.hexdigest()


def greedy_runs(model: decoder.Model, prompts: list[list[int]]) -> list[dict]:
    runs = []
    for prompt in prompts:
        tokens, logits = decoder.generate(model, prompt, N_NEW)
        runs.append({"prompt": prompt, "tokens": tokens, "logits_sha256": bits_sha256(logits)})
    return runs


def tiny_fixture() -> dict:
    """Tiny config (2 KV heads, head size 16), weights from TINY_SEED, three
    random-token prompts. The output projection is untied: with random tied
    weights a token mostly predicts itself, and greedy decoding repeats it."""
    config = tiny.tiny_config(tied=False)
    state = tiny.random_weights(config, TINY_SEED)
    gen = torch.Generator().manual_seed(TINY_SEED)
    prompts = [
        torch.randint(0, config.vocab_size, (n,), generator=gen).tolist()
        for n in TINY_PROMPT_LENGTHS
    ]
    model = decoder.from_state_dict(config, state)
    return {
        "config": config.to_diff_dict(),
        "seed": TINY_SEED,
        "weights_sha256": bits_sha256(*(state[k] for k in sorted(state))),
        "runs": greedy_runs(model, prompts),
    }


def transformers_greedy(model: LlamaForCausalLM, prompt: list[int]) -> list[int]:
    """N_NEW tokens by the golden model's rule: argmax (lowest index on a
    tie), no stopping at end of text, no token forbidden. transformers'
    generate() would stop at, or with min_new_tokens forbid, end of text."""
    seq = list(prompt)
    with torch.no_grad():
        for _ in range(N_NEW):
            seq.append(int(model(torch.tensor([seq])).logits[0, -1].argmax()))
    return seq[len(prompt) :]


def smollm2_fixture(path, log=print) -> dict:
    """SmolLM2-135M-Instruct, three chat prompts. For information, each run
    also records FP32 `transformers`' greedy tokens and the first position
    where they differ from the golden model's (None if they never do)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = decoder.load(path)
    reference = LlamaForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    runs = []
    for text in SMOLLM2_PROMPTS:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, return_dict=True
        )["input_ids"]
        (run,) = greedy_runs(model, [prompt])
        theirs = transformers_greedy(reference, prompt)
        pairs = zip(run["tokens"], theirs, strict=True)
        diverge = next((i for i, (a, b) in enumerate(pairs) if a != b), None)
        run |= {
            "text": tokenizer.decode(run["tokens"]),
            "transformers_fp32_tokens": theirs,
            "first_difference": diverge,
        }
        runs.append(run)
        log(f"{text!r}: first difference from transformers FP32 at {diverge}")
    return {"weights": path.name, "runs": runs}
