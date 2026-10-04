# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Chat with SmolLM2-135M-Instruct as is, with `transformers` on the CPU.

This is the unmodified reference behaviour that Cabosse is compared against.
Output is streamed token by token. The Instruct variant runs as a chat that
remembers the conversation; the base variant continues the text. Examples:
    python scripts/run_smollm.py                     # interactive chat
    python scripts/run_smollm.py --prompt "Hi, who are you?"
    python scripts/run_smollm.py --greedy            # deterministic
    python scripts/run_smollm.py --variant base      # base model, text continuation
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

import paths  # noqa: E402

VARIANTS = {"instruct": paths.SMOLLM2, "base": paths.SMOLLM2_BASE}


def generate(model, tok, input_ids: torch.Tensor, args) -> torch.Tensor:
    """Stream a completion of `input_ids` and return only the new token ids."""
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    sampling = {} if args.greedy else {"temperature": args.temperature, "top_p": args.top_p}
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            streamer=streamer,
            **sampling,
        )
    dt = time.perf_counter() - t0
    new = out[0, input_ids.shape[1] :]
    mode = "greedy" if args.greedy else f"T={args.temperature}, top-p={args.top_p}"
    print(
        f"[{args.model.name}, {args.dtype}, {mode}] {input_ids.shape[1]} context + "
        f"{len(new)} new tokens in {dt:.2f} s ({len(new) / dt:.1f} tokens/s)"
    )
    return new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--variant",
        choices=VARIANTS,
        default="instruct",
        help="instruct = chat model (default), base = plain text continuation",
    )
    ap.add_argument("--prompt", help="run once with this prompt (default: interactive)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="sampling temperature (default 0.2, from the SmolLM2 model card)",
    )
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument(
        "--greedy", action="store_true", help="always pick the most likely token (deterministic)"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.model = VARIANTS[args.variant]

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    model.eval()
    chat = tok.chat_template is not None
    history: list[dict] = []

    def turn(text: str) -> None:
        if chat:
            history.append({"role": "user", "content": text})
            ids = tok.apply_chat_template(
                history, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )["input_ids"]
            reply = generate(model, tok, ids, args)
            history.append(
                {
                    "role": "assistant",
                    "content": tok.decode(reply, skip_special_tokens=True).strip(),
                }
            )
        else:
            print(text, end="", flush=True)
            generate(model, tok, tok(text, return_tensors="pt")["input_ids"], args)

    if args.prompt is not None:
        turn(args.prompt)
        return

    try:
        import readline  # noqa: F401  (line editing and history for input())
    except ImportError:
        pass
    if chat:
        print(
            f"{args.model.name}: chat mode. '/reset' clears the conversation, "
            f"Ctrl-D or 'exit' quits."
        )
    else:
        print(
            f"{args.model.name}: base model, it continues your text (no memory between "
            f"prompts). Ctrl-D or 'exit' quits."
        )
    while True:
        try:
            text = input("\n> ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            break
        if text in ("exit", "quit"):
            break
        if text == "/reset":
            history.clear()
            print("(conversation cleared)")
        elif text:
            turn(text)


if __name__ == "__main__":
    main()
