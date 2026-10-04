# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Compare the golden model with `transformers` on SmolLM2 (M1 exit criterion).

Teacher-forces the golden model, FP32 `transformers` (the reference) and BF16
`transformers` on the same sequences, prints a table and writes a Markdown
report and the raw numbers to runs/. The full run (10 prompts x 256
positions) takes about 22 minutes on the dev machine. Examples:
    python scripts/compare_transformers.py
    python scripts/compare_transformers.py --prompts 2 --positions 64   # quick
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "model"))

import compare  # noqa: E402
from golden import decoder  # noqa: E402

WEIGHTS = Path(os.environ.get("CABOSSE_WEIGHTS", REPO / "weights")) / "SmolLM2-135M-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument(
        "--prompts", type=int, default=len(compare.PROMPTS), help="how many prompts"
    )
    parser.add_argument("--positions", type=int, default=256, help="tokens per sequence")
    parser.add_argument("--out", type=Path, default=REPO / "runs")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.weights)
    reference = LlamaForCausalLM.from_pretrained(args.weights, dtype=torch.float32).eval()
    bf16 = LlamaForCausalLM.from_pretrained(args.weights, dtype=torch.bfloat16).eval()
    golden = decoder.load(args.weights)

    start = time.perf_counter()
    sequences = compare.chat_sequences(
        tokenizer, reference, compare.PROMPTS[: args.prompts], args.positions
    )
    result = compare.compare(golden, reference, bf16, sequences)
    minutes = (time.perf_counter() - start) / 60

    text = compare.report(result)
    print(f"\n{text}\n\n({minutes:.1f} min)")
    args.out.mkdir(exist_ok=True)
    stem = args.out / time.strftime("compare-%Y%m%d-%H%M%S")
    stem.with_suffix(".md").write_text(text + "\n")
    raw = compare.as_json(result) | {
        "prompts": compare.PROMPTS[: args.prompts],
        "sequences": sequences,
        "weights": str(args.weights),
        "minutes": minutes,
    }
    stem.with_suffix(".json").write_text(json.dumps(raw, indent=1))
    print(f"written: {stem}.md, {stem}.json")


if __name__ == "__main__":
    main()
