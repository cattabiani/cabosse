# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Compare the golden model with `transformers` (M1 exit criterion, PLAN.md).

Every model is teacher-forced on the same token sequences: short chat prompts
continued greedily by FP32 `transformers` up to a fixed length. FP32
`transformers` is the reference; `transformers` in BF16 is the bar the golden
model must meet. Per position we measure top-1 agreement, the logit error
relative to the largest reference logit, and the KL divergence of the
predicted distributions.
"""

from dataclasses import asdict, dataclass

import torch
from golden import decoder

# Written for this project (no third-party text). Varied on purpose: facts,
# explanation, story, code, lists, a letter.
PROMPTS = [
    "What is the capital of France? Answer in one sentence, then explain why.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a refrigerator works to a ten-year-old.",
    "Write a Python function that checks whether a string is a palindrome, and explain it briefly.",
    "List five tips for learning a new language, with one sentence for each.",
    "What are the main differences between a virus and a bacterium?",
    "Describe the water cycle in a few short paragraphs.",
    "Give me a recipe for a simple vegetable soup.",
    "Why is the sky blue? Explain the physics briefly.",
    "Write a polite email asking a colleague to review a document by Friday.",
]


@dataclass(frozen=True)
class Metrics:
    """Against the reference, over a set of positions."""

    positions: int
    top1: float  # fraction of positions with the same argmax
    logit_err_mean: float  # max |logit - ref| / max |ref| per position, mean
    logit_err_max: float  # ... and max over positions
    kl_mean: float  # KL(ref || test) of the softmax distributions, nats
    kl_max: float


def position_errors(ref: torch.Tensor, test: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-position top-1 agreement, relative logit error and KL(ref || test),
    in float64. ref and test: logits [T, vocab]."""
    ref, test = ref.double(), test.double()
    log_p, log_q = ref.log_softmax(-1), test.log_softmax(-1)
    return {
        "top1": (ref.argmax(-1) == test.argmax(-1)).double(),
        "logit_err": (test - ref).abs().amax(-1) / ref.abs().amax(-1),
        "kl": (log_p.exp() * (log_p - log_q)).sum(-1),
    }


def summarize(errors: list[dict[str, torch.Tensor]]) -> Metrics:
    """Pool the per-position errors of several sequences."""
    pooled = {k: torch.cat([e[k] for e in errors]) for k in errors[0]}
    return Metrics(
        positions=len(pooled["top1"]),
        top1=pooled["top1"].mean().item(),
        logit_err_mean=pooled["logit_err"].mean().item(),
        logit_err_max=pooled["logit_err"].max().item(),
        kl_mean=pooled["kl"].mean().item(),
        kl_max=pooled["kl"].max().item(),
    )


def chat_sequences(tokenizer, reference, prompts: list[str], n_positions: int) -> list[list[int]]:
    """Each prompt in the chat template, continued greedily by the reference
    to exactly n_positions tokens (end of text is suppressed until then)."""
    sequences = []
    for prompt in prompts:
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        assert enc["input_ids"].shape[1] < n_positions, (prompt, enc["input_ids"].shape)
        with torch.no_grad():
            out = reference.generate(
                **enc,
                max_length=n_positions,
                min_length=n_positions,
                do_sample=False,
            )
        sequences.append(out[0].tolist())
    return sequences


def transformers_logits(model, tokens: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return model(torch.tensor([tokens])).logits[0].float()


def compare(golden: decoder.Model, reference, bf16, sequences: list[list[int]], log=print) -> dict:
    """Golden and BF16 `transformers` against the FP32 reference.

    Returns {"golden": Metrics, "transformers_bf16": Metrics, "per_sequence":
    [{"golden": Metrics, "transformers_bf16": Metrics}, ...]}.
    """
    models = {
        "golden": lambda tokens: decoder.forward(golden, tokens),
        "transformers_bf16": lambda tokens: transformers_logits(bf16, tokens),
    }
    errors = {name: [] for name in models}
    per_sequence = []
    for n, tokens in enumerate(sequences):
        ref = transformers_logits(reference, tokens)
        seq = {}
        for name, run in models.items():
            errors[name].append(position_errors(ref, run(tokens)))
            seq[name] = summarize(errors[name][-1:])
        per_sequence.append(seq)
        top1 = ", ".join(f"{name} {m.top1:.3f}" for name, m in seq.items())
        log(f"sequence {n + 1}/{len(sequences)}: top-1 {top1}")
    return {name: summarize(errs) for name, errs in errors.items()} | {"per_sequence": per_sequence}


def as_json(result: dict) -> dict:
    return {
        "golden": asdict(result["golden"]),
        "transformers_bf16": asdict(result["transformers_bf16"]),
        "per_sequence": [{k: asdict(v) for k, v in s.items()} for s in result["per_sequence"]],
    }


def report(result: dict) -> str:
    """Markdown table: golden next to BF16 `transformers`, both against FP32."""
    rows = [
        ("top-1 agreement", "top1", "{:.4f}"),
        ("logit error, mean", "logit_err_mean", "{:.2e}"),
        ("logit error, max", "logit_err_max", "{:.2e}"),
        ("KL, mean (nats)", "kl_mean", "{:.2e}"),
        ("KL, max (nats)", "kl_max", "{:.2e}"),
    ]
    g, b = result["golden"], result["transformers_bf16"]
    lines = [
        f"Against FP32 `transformers`, {g.positions} positions "
        f"({len(result['per_sequence'])} sequences).",
        "",
        "| | golden | transformers BF16 |",
        "|---|---|---|",
    ]
    lines += [
        f"| {name} | {fmt.format(getattr(g, k))} | {fmt.format(getattr(b, k))} |"
        for name, k, fmt in rows
    ]
    return "\n".join(lines)
