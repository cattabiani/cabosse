# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""M9's chat recording (PLAN.md): scripted user turns to SmolLM2 on the
core, each answer greedy until the end of its turn, streamed live as text
(whole characters only, the template's markers hidden) and recorded with
its real timing:

- chat.cast: asciicast v2 (https://docs.asciinema.org/manual/asciicast/v2/),
  written as the text appears; play with `asciinema play chat.cast`;
- chat.txt: the transcript;
- chat.jsonl: one line a turn (plan_log.py): tokens fed and generated, the
  prompt's and each answer token's wall time, CYCLES.

The conversation stays in the KV cache: each turn feeds only the tokens the
chat template adds after what the core already holds (an answer cut at
--max-new is closed by the template's end token, fed with the next turn).

Usage: python chat.py OUT_DIR [--golden] [--mhz MHZ] [--max-new N] [TURN ...]
  (--golden: the golden platform on the CPU, no card; slow, about 1.3 s a token)
Environment: S3 and PROVENANCE (plan_log.py); CABOSSE_WEIGHTS (model/paths.py).
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
for sub in ("model", "sw"):
    sys.path.insert(0, str(REPO / sub))

import paths  # noqa: E402
from commands import Layout  # noqa: E402
from golden import decoder  # noqa: E402
from plan_log import Log  # noqa: E402
from runtime import Runtime  # noqa: E402

TURNS = ["What is the capital of France?", "And of Italy?"]
MAX_NEW = 128  # answer tokens per turn, at most
CAP = 2048  # positions, as core_plan.py
WIDTH, HEIGHT = 100, 30  # the recording's terminal


class Cast:
    """An asciicast v2 file written as output happens, echoed to stdout."""

    def __init__(self, path: Path, title: str) -> None:
        self.f = path.open("w")
        self.t0 = time.monotonic()
        header = {"version": 2, "width": WIDTH, "height": HEIGHT}
        header |= {"timestamp": int(time.time()), "title": title}
        self.f.write(json.dumps(header) + "\n")

    def out(self, text: str) -> None:
        if not text:
            return
        t = time.monotonic() - self.t0
        self.f.write(json.dumps([round(t, 6), "o", text.replace("\n", "\r\n")]) + "\n")
        self.f.flush()
        print(text, end="", flush=True)

    def close(self) -> None:
        self.f.close()


def new_text(decode: Callable[[list[int]], str], tokens: list[int], shown: int) -> tuple[str, int]:
    """The answer's text past the `shown` characters already out, held back
    while it ends in a partial character (U+FFFD); and the new count."""
    text = decode(tokens)
    if text.endswith("�"):
        return "", shown
    return text[shown:], len(text)


def answer(
    rt: Runtime,
    feed: list[int],
    end_id: int,
    max_new: int,
    decode: Callable[[list[int]], str],
    cast: Cast,
) -> tuple[list[int], dict]:
    """Feed `feed` (the turn's new prompt tokens), then decode greedily until
    `end_id` or `max_new` tokens, streaming the text. Returns the answer's
    tokens (the end token included, if reached) and the turn's timings."""
    t0 = time.perf_counter()
    for t in feed:
        logits = rt.step(t)
    prompt_s = time.perf_counter() - t0
    tokens, wall_s, cycles, shown = [], [], [], 0
    while True:
        tokens.append(int(logits.argmax()))
        if tokens[-1] == end_id or len(tokens) == max_new:
            break
        text, shown = new_text(decode, tokens, shown)
        cast.out(text)
        t0 = time.perf_counter()
        logits = rt.step(tokens[-1])
        wall_s.append(time.perf_counter() - t0)
        cycles.append(rt.cycles)
    text, shown = new_text(decode, tokens, shown)
    cast.out(text)
    stats = {"prompt_tokens": len(feed), "prompt_s": prompt_s, "n_new": len(tokens)}
    stats |= {"ended": tokens[-1] == end_id, "wall_s": wall_s, "cycles": cycles}
    return tokens, stats


def chat(
    log: Log,
    rt: Runtime,
    tok,
    turns: list[str],
    max_new: int,
    out: Path,
) -> list[dict]:
    """The scripted conversation on `rt`, with the tokenizer `tok` (its chat
    template). Returns the messages."""
    end_id = tok.convert_tokens_to_ids("<|im_end|>")

    def decode(ids: list[int]) -> str:
        return tok.decode(ids, skip_special_tokens=True)

    messages: list[dict] = []
    fed: list[int] = []  # what the KV cache holds
    cast = Cast(out / "chat.cast", "Cabosse: SmolLM2-135M-Instruct on AWS F2")
    try:
        for k, user in enumerate(turns):
            messages.append({"role": "user", "content": user})
            ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True)
            ids = list(ids["input_ids"])
            if ids[: len(fed)] != fed:  # the template rewrote the history: start over
                rt.reset()
                fed = []
            feed = ids[len(fed) :]
            cast.out(("\n" if k else "") + f"> {user}\n\n")
            tokens, stats = answer(rt, feed, end_id, max_new, decode, cast)
            cast.out("\n")
            fed = ids + tokens
            messages.append({"role": "assistant", "content": decode(tokens)})
            params = {"turn": k, "user": user, "feed": feed}
            log.record("turn", params, stats | {"tokens": tokens, "text": messages[-1]["content"]})
    finally:
        cast.close()
    lines = [f"{m['role']}: {m['content']}" for m in messages]
    (out / "chat.txt").write_text("\n\n".join(lines) + "\n")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="the output directory")
    parser.add_argument("turns", nargs="*", default=TURNS, help="the user's turns")
    parser.add_argument("--golden", action="store_true", help="the golden platform, no card")
    parser.add_argument("--mhz", type=int, default=0, help="set the core clock first (F2)")
    parser.add_argument("--max-new", type=int, default=MAX_NEW, help="answer tokens a turn")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    log = Log(str(args.out / "chat.jsonl"))
    model = decoder.load(paths.SMOLLM2)
    if args.golden:
        from golden_platform import GoldenPlatform

        layout = Layout.of(model.config, CAP, ports=32)
        platform = nullcontext(GoldenPlatform(layout))
    else:
        from f2_platform import F2Platform

        if args.mhz:
            import core_plan

            if "error" in core_plan.set_clock(log, args.mhz):
                log.upload(now=True)
                sys.exit(1)
        platform = F2Platform()
    tok = AutoTokenizer.from_pretrained(str(paths.SMOLLM2))
    try:
        with platform as p:
            chat(log, Runtime.open(p, model, CAP), tok, args.turns, args.max_new, args.out)
    finally:
        log.upload(now=True)


if __name__ == "__main__":
    main()
