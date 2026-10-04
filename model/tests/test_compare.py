# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Tests for the comparison harness (model/compare.py): the metrics on
known inputs, and an end-to-end run on a tiny config."""

import math

import compare
import paths
import pytest
import torch
from golden import decoder, tiny
from transformers import AutoTokenizer, LlamaForCausalLM

SEED = 20261006
WEIGHTS = paths.SMOLLM2


def test_identical_logits_have_no_error() -> None:
    ref = torch.randn(5, 11, generator=torch.Generator().manual_seed(SEED))
    m = compare.summarize([compare.position_errors(ref, ref.clone())])
    assert (m.positions, m.top1, m.logit_err_max, m.kl_max) == (5, 1.0, 0.0, 0.0)


def test_metrics_on_a_known_case() -> None:
    """ref = [2, 1, 0], test = [0.5, 2, 0]: the argmax differs, the largest
    logit difference is 1.5 against a largest reference logit of 2, and
    KL(ref || test) is computed by hand (not symmetric here, so the direction
    is checked too)."""
    ref, test = [2.0, 1.0, 0.0], [0.5, 2.0, 0.0]
    e = compare.position_errors(torch.tensor([ref]), torch.tensor([test]))

    def softmax(x: list[float]) -> list[float]:
        z = sum(math.exp(v) for v in x)
        return [math.exp(v) / z for v in x]

    p, q = softmax(ref), softmax(test)
    kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True))
    reverse = sum(qi * math.log(qi / pi) for pi, qi in zip(p, q, strict=True))
    assert abs(kl - reverse) > 0.01
    assert e["top1"].item() == 0.0 and e["logit_err"].item() == 0.75
    assert e["kl"].item() == pytest.approx(kl, rel=1e-12)


def test_summarize_pools_positions() -> None:
    """Means are over all positions, not over sequences."""
    a = {
        "top1": torch.tensor([1.0, 0.0]),
        "logit_err": torch.tensor([0.1, 0.3]),
        "kl": torch.zeros(2),
    }
    b = {"top1": torch.tensor([1.0]), "logit_err": torch.tensor([0.2]), "kl": torch.tensor([0.5])}
    m = compare.summarize([a, b])
    assert m.positions == 3 and m.top1 == pytest.approx(2 / 3)
    assert m.logit_err_mean == pytest.approx(0.2) and m.logit_err_max == pytest.approx(0.3)
    assert m.kl_mean == pytest.approx(0.5 / 3) and m.kl_max == 0.5


def test_compare_end_to_end_on_tiny_config() -> None:
    config = tiny.tiny_config()
    state = tiny.random_weights(config, SEED)
    golden = decoder.from_state_dict(config, state)
    reference = tiny.transformers_model(config, state, torch.float32)
    bf16 = tiny.transformers_model(config, state, torch.bfloat16)
    gen = torch.Generator().manual_seed(SEED)
    sequences = [
        torch.randint(0, config.vocab_size, (12,), generator=gen).tolist() for _ in range(2)
    ]

    result = compare.compare(golden, reference, bf16, sequences, log=lambda _: None)
    assert result["golden"].positions == 24 and len(result["per_sequence"]) == 2
    assert result["golden"].logit_err_max <= 0.01  # the decoder's wiring bound
    assert "| top-1 agreement |" in compare.report(result)
    assert compare.as_json(result)["golden"]["positions"] == 24


def test_prompts_are_distinct() -> None:
    assert len(compare.PROMPTS) == 10 and len(set(compare.PROMPTS)) == 10


@pytest.mark.slow
@pytest.mark.skipif(not WEIGHTS.exists(), reason=f"needs the SmolLM2 checkpoint in {WEIGHTS}")
def test_chat_sequences_have_the_requested_length() -> None:
    """The prompt in the chat template, continued to exactly n positions."""
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS)
    reference = LlamaForCausalLM.from_pretrained(WEIGHTS, dtype=torch.float32).eval()
    (seq,) = compare.chat_sequences(tokenizer, reference, compare.PROMPTS[:1], 60)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": compare.PROMPTS[0]}],
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    assert len(seq) == 60 and seq[: len(prompt)] == prompt
