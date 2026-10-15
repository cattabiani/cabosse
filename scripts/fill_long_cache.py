# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Compute the long KV cache of the attention tests (model/long_cache.py)
into weights/kv/. 107 minutes for SmolLM2 on 16 cores (2026-10-14), saved every 512
positions: stop it with Ctrl-C at any time, and run it again to resume.

    nice -n 19 python scripts/fill_long_cache.py

Run it under the memory cap. --positions N fills a shorter prompt, to try
the script (its file is never used by the tests).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

import long_cache  # noqa: E402
import paths  # noqa: E402
from golden import decoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--weights", type=Path, default=paths.SMOLLM2)
    parser.add_argument("--positions", type=int, default=long_cache.N_POSITIONS)
    args = parser.parse_args()
    model = decoder.load(args.weights)
    tokens = long_cache.prompt_tokens(args.weights, args.positions)
    path = long_cache.cache_path(args.weights, tokens)
    if path.exists():
        sys.exit(f"already computed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"filling {len(tokens)} positions into {path}", flush=True)
    t0 = time.perf_counter()
    for n in long_cache.fill(model, tokens, path):
        minutes = (time.perf_counter() - t0) / 60
        print(
            f"{time.strftime('%H:%M')}  {n}/{len(tokens)} positions, {minutes:.0f} min", flush=True
        )
    print(f"done: {path}")


if __name__ == "__main__":
    main()
