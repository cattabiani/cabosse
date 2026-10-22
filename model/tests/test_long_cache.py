# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The long KV cache (model/long_cache.py) on a tiny model: filled in
chunks it is the cache of one run, bit for bit; an interrupted fill
resumes; a file that disagrees with the golden model, or holds another
prompt, is refused. The prompt itself needs SmolLM2's tokenizer (slow)."""

import long_cache
import paths
import pytest
import torch
from golden import arith, decoder, tiny

N_TOKENS = 40  # 3 chunks of 16 (the last one short)
CHUNK = 16
CHECK = 8


@pytest.fixture
def small(monkeypatch: pytest.MonkeyPatch) -> tuple[decoder.Model, list[int]]:
    monkeypatch.setattr(long_cache, "CHUNK_POSITIONS", CHUNK)
    monkeypatch.setattr(long_cache, "CHECK_POSITIONS", CHECK)
    c = tiny.tiny_config()
    model = decoder.from_state_dict(c, tiny.random_weights(c, seed=3))
    gen = torch.Generator().manual_seed(3)
    tokens = torch.randint(c.vocab_size, (N_TOKENS,), generator=gen).tolist()
    return model, tokens


def one_run(model: decoder.Model, tokens: list[int]) -> decoder.KVCache:
    cache = decoder.KVCache.empty(model, len(tokens))
    decoder.step(model, cache, tokens)
    return cache


def same_cache(a: decoder.KVCache, b: decoder.KVCache) -> bool:
    return (
        a.length == b.length
        and torch.equal(arith.bits_bf16(a.k), arith.bits_bf16(b.k))
        and torch.equal(arith.bits_bf16(a.v), arith.bits_bf16(b.v))
    )


def test_fill_in_chunks_is_one_run(small, tmp_path) -> None:
    model, tokens = small
    path = tmp_path / "cache.pt"
    assert list(long_cache.fill(model, tokens, path)) == [16, 32, 40]
    assert path.exists() and not path.with_suffix(".partial").exists()
    assert same_cache(long_cache.read(path, model, tokens), one_run(model, tokens))


def test_an_interrupted_fill_resumes(small, tmp_path) -> None:
    model, tokens = small
    path = tmp_path / "cache.pt"
    for n in long_cache.fill(model, tokens, path):
        if n == 16:
            break  # stopped after its first save
    assert not path.exists() and path.with_suffix(".partial").exists()
    assert list(long_cache.fill(model, tokens, path)) == [32, 40]
    assert same_cache(long_cache.read(path, model, tokens), one_run(model, tokens))


def test_a_stale_file_is_refused(small, tmp_path) -> None:
    """One value's sign flipped in the last checked position (a numerics
    change would change bits there): read() refuses the file."""
    model, tokens = small
    path = tmp_path / "cache.pt"
    long_cache.save(path, tokens, one_run(model, tokens))
    data = torch.load(path)
    data["k"][0, 0, CHECK - 1, 0] *= -1  # other bits, -0 included
    torch.save(data, path)
    with pytest.raises(AssertionError, match="refill it"):
        long_cache.read(path, model, tokens)


def test_another_prompt_is_refused(small, tmp_path) -> None:
    model, tokens = small
    path = tmp_path / "cache.pt"
    long_cache.save(path, tokens, one_run(model, tokens))
    with pytest.raises(AssertionError):
        long_cache.read(path, model, [*tokens[:-1], tokens[-1] ^ 1])


@pytest.mark.slow
@pytest.mark.skipif(not paths.SMOLLM2.exists(), reason=f"needs the checkpoint in {paths.SMOLLM2}")
def test_the_prompt_is_fixed() -> None:
    """The prompt comes from SOURCE_COMMIT, so it does not change with the
    repo, and a shorter prompt is its prefix."""
    tokens = long_cache.prompt_tokens(paths.SMOLLM2, long_cache.N_POSITIONS)
    assert len(tokens) == long_cache.N_POSITIONS
    assert long_cache.prompt_tokens(paths.SMOLLM2, 100) == tokens[:100]
