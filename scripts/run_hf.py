# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Run a Hugging Face checkpoint as is, with `transformers` on the CPU.

This is the unmodified reference behaviour that Cabosse is compared against.
Example:
    python scripts/run_hf.py --prompt "The capital of France is"
"""

import argparse
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path(os.environ.get("CABOSSE_WEIGHTS", REPO / "weights")) / "SmolLM2-135M"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="checkpoint directory")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--sample", action="store_true", help="sample instead of greedy decoding")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.eval()

    inputs = tok(args.prompt, return_tensors="pt")
    streamer = TextStreamer(tok, skip_special_tokens=True)  # prints tokens as they come
    n_prompt = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.sample,
            pad_token_id=tok.eos_token_id,
            streamer=streamer,
        )
    dt = time.perf_counter() - t0
    n_new = out.shape[1] - n_prompt

    print(f"\n[{args.model.name}, {args.dtype}, {'sampled' if args.sample else 'greedy'}] "
          f"{n_prompt} prompt + {n_new} new tokens in {dt:.2f} s "
          f"({n_new / dt:.1f} tokens/s)")


if __name__ == "__main__":
    main()
