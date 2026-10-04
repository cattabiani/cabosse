# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Greedy-decode regression fixtures (model/fixtures.py): the golden model
must reproduce the stored tokens and logit bits. Pinned values are checked on
the reference platform only (D-023); regenerate them with
scripts/make_fixtures.py after a deliberate numerics change."""

import json

import compare
import fixtures
import paths
import pytest
import torch
from golden import arith, decoder, tiny

reference_only = pytest.mark.skipif(
    not paths.REFERENCE_PLATFORM, reason=paths.REFERENCE_PLATFORM_NOTE
)


def stored(name: str) -> dict:
    return json.loads((paths.FIXTURES / name).read_text())


def test_generate_is_deterministic_and_matches_forward() -> None:
    """Two runs give the same bits, and the logits are those of teacher
    forcing on the prompt plus the generated tokens (but the last)."""
    config = tiny.tiny_config(tied=False)
    model = decoder.from_state_dict(config, tiny.random_weights(config, 1))
    prompt = [3, 141, 59, 26]
    tokens, logits = decoder.generate(model, prompt, 12)
    again, logits_again = decoder.generate(model, prompt, 12)
    assert tokens == again and torch.equal(arith.bits_f32(logits), arith.bits_f32(logits_again))
    assert len(tokens) == 12 and logits.shape == (len(prompt) + 11, config.vocab_size)
    teacher = decoder.forward(model, prompt + tokens[:-1])
    assert torch.equal(arith.bits_f32(logits), arith.bits_f32(teacher))
    assert tokens == logits[len(prompt) - 1 :].argmax(-1).tolist()


@reference_only
def test_tiny_greedy_matches_fixture() -> None:
    """Tokens and logit bits of the tiny config, three prompts x 64 tokens
    (the M7 RTL target). The weights are regenerated from a seed and checked
    first, so a change in torch's random numbers is reported as such."""
    want = stored("tiny_greedy.json")
    got = fixtures.tiny_fixture()
    assert got["weights_sha256"] == want["weights_sha256"], (
        "the tiny random weights changed (torch's random number generator?): "
        "regenerate the fixture with scripts/make_fixtures.py --tiny and review"
    )
    assert got["runs"] == want["runs"]


@pytest.mark.slow
@reference_only
@pytest.mark.skipif(not paths.SMOLLM2.exists(), reason=f"needs the checkpoint in {paths.SMOLLM2}")
def test_smollm2_greedy_matches_fixture() -> None:
    """SmolLM2, three chat prompts x 64 tokens: tokens and logit bits.

    The stored tokens are known, so one teacher-forced pass gives the same
    logits as generating them one by one (test above), in about a fifth of
    the time: about 70 s. A wrong token changes the logits after it, so the
    hash still catches it. Runs locally, not in CI."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(paths.SMOLLM2)
    model = decoder.load(paths.SMOLLM2)
    for text, run in zip(
        fixtures.SMOLLM2_PROMPTS, stored("smollm2_greedy.json")["runs"], strict=True
    ):
        prompt = run["prompt"]
        assert compare.chat_prompt(tokenizer, text) == prompt
        logits = decoder.forward(model, prompt + run["tokens"][:-1])
        assert logits[len(prompt) - 1 :].argmax(-1).tolist() == run["tokens"], run["text"]
        assert arith.bits_sha256(logits) == run["logits_sha256"]
