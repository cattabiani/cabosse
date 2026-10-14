# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""A long KV cache of real text for the attention tests (PLAN.md, M5): the
golden model runs a prompt of N_POSITIONS tokens once (107 minutes for
SmolLM2 on 16 cores), and the cache is saved under the git-ignored weights/.

The prompt is our own docs at a fixed commit (Apache-2.0), so it never
changes with the repo. A cache's prefix is the cache of a shorter prompt
(attention is causal), so one file serves every position up to N_POSITIONS.
The file name hashes the weights and the prompt; load() recomputes a short
prefix with the current golden model and checks it bit for bit, so a file
older than a numerics change fails.
"""

import hashlib
import subprocess
from collections.abc import Iterator
from pathlib import Path

import paths
import torch
from golden import decoder
from transformers import AutoTokenizer

N_POSITIONS = 8191  # a decode step at the last position SmolLM2 supports (8192)
SOURCE_COMMIT = "4f0f44cb1410493562ee7f7ce717ed0b9bac4e1e"
SOURCE_FILES = (
    "PLAN.md",
    "docs/architecture.md",
    "docs/numerics.md",
    "docs/perf.md",
    "docs/f2.md",
    "docs/aws-setup.md",
)
CHUNK_POSITIONS = 512  # saved after each chunk; the last one took 12 minutes
CHECK_POSITIONS = 64


def prompt_tokens(weights: Path, n: int) -> list[int]:
    """The first n tokens of SOURCE_FILES at SOURCE_COMMIT, no special tokens."""
    show = [f"{SOURCE_COMMIT}:{f}" for f in SOURCE_FILES]
    text = subprocess.run(
        ["git", "-C", str(paths.REPO), "show", *show],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tokens = AutoTokenizer.from_pretrained(weights)(text, add_special_tokens=False).input_ids
    assert len(tokens) >= n, (len(tokens), n)
    return tokens[:n]


def cache_path(weights: Path, tokens: list[int]) -> Path:
    with (weights / "model.safetensors").open("rb") as f:
        h = hashlib.file_digest(f, "sha256")  # streamed: the file can be 1 GB
    h.update(torch.tensor(tokens, dtype=torch.int32).numpy().tobytes())
    return paths.WEIGHTS / "kv" / f"{weights.name}-{len(tokens)}-{h.hexdigest()[:16]}.pt"


def save(path: Path, tokens: list[int], cache: decoder.KVCache) -> None:
    """Written to a temporary file and renamed: an interrupted save leaves the
    previous one."""
    tmp = path.with_suffix(".tmp")
    torch.save({"tokens": tokens, "k": cache.k, "v": cache.v, "length": cache.length}, tmp)
    tmp.replace(path)


def read(path: Path, model: decoder.Model, tokens: list[int]) -> decoder.KVCache:
    """The cache saved at path (complete or partial), its prefix checked
    against the current golden model."""
    data = torch.load(path)
    assert data["tokens"] == tokens, path
    cache = decoder.KVCache(data["k"], data["v"], data["length"])
    n = min(CHECK_POSITIONS, cache.length)
    fresh = decoder.KVCache.empty(model, n)
    decoder.step(model, fresh, tokens[:n], last_only=True)
    same = torch.equal(fresh.k.view(torch.int16), cache.k[:, :, :n].view(torch.int16))
    same &= torch.equal(fresh.v.view(torch.int16), cache.v[:, :, :n].view(torch.int16))
    assert same, f"{path}: the first {n} positions differ from the golden model's; refill it"
    return cache


def fill(model: decoder.Model, tokens: list[int], path: Path) -> Iterator[int]:
    """Run tokens through the golden model in chunks, saving after each one
    to `path` + .partial and yielding the positions filled, and resume from
    that file if it exists. Running in parts gives the
    same bits (test_decoder.test_forward_equals_decode_steps)."""
    partial = path.with_suffix(".partial")
    if partial.exists():
        cache = read(partial, model, tokens)
        print(f"resuming at position {cache.length}", flush=True)
    else:
        cache = decoder.KVCache.empty(model, len(tokens))
    while cache.length < len(tokens):
        start = cache.length
        decoder.step(model, cache, tokens[start : start + CHUNK_POSITIONS], last_only=True)
        save(partial, tokens, cache)
        yield cache.length
    partial.replace(path)


def load(model: decoder.Model, weights: Path) -> tuple[list[int], decoder.KVCache] | None:
    """The saved cache for weights, checked, or None if it was not computed."""
    tokens = prompt_tokens(weights, N_POSITIONS)
    path = cache_path(weights, tokens)
    return (tokens, read(path, model, tokens)) if path.exists() else None
