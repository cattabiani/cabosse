# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""A whole decode step on rtl/vec_unit.sv (verif/decode.py): the tiny config
(GQA) in CI, SmolLM2's at the real-data positions (slow, one at a time: its
memory is up to about 600 MiB).

Checked bit for bit against commands.run(): what each engine command reads
(so a divergence shows at the first engine command after it), every buffer
at the end, and the whole memory: the KV cache in the engine's order
(D-062), every other byte unchanged. An engine step waits for every earlier command,
so the engine never overlaps the vector unit here (on shared buffers: M7).
Both start from the same buffers, all NaN (commands.empty_buffers), so a
read of an element no command wrote shows. SmolLM2's commands also take
within 10% of the perf model's cycles each (D-047)."""

import decode
import engine as eng
import paths
import pytest
import torch
import vector
from golden import decoder, tiny

import rtl

SEED = 20261030
CONFIG = tiny.tiny_config(n_kv_heads=2)  # 4 query heads: 2 per KV head
LENGTH = 256  # elements of every buffer: the largest, LOGITS
CHAN_W = 17  # 128 KiB channels


# (position, cap, stride): position 0, and position 5 after a prompt, with
# a cache of L = 32 rows (vector.KV_PORTS), S and P's rows at 32 and 48.
CASES = [(0, 32, 32), (5, 32, 48)]


@rtl.needs_verilator
@pytest.mark.parametrize(("position", "cap", "stride"), CASES)
def test_decode_step(position: int, cap: int, stride: int) -> None:
    seed = SEED + position  # the tokens'
    gen = torch.Generator().manual_seed(seed)
    tokens = torch.randint(CONFIG.vocab_size, (position + 1,), generator=gen).tolist()
    model = decoder.from_state_dict(CONFIG, tiny.random_weights(CONFIG, SEED))
    cache = decoder.KVCache.empty(model, cap)
    if position:
        decoder.step(model, cache, tokens[:-1])
    label = f"position {position}, token seed {seed}"
    decode.decode_step(
        model, cache, tokens[-1], cap, stride, LENGTH, vector.KV_PORTS, CHAN_W, label
    )


@pytest.mark.slow
@rtl.needs_verilator
@pytest.mark.skipif(not paths.SMOLLM2.exists(), reason=f"needs the checkpoint in {paths.SMOLLM2}")
@pytest.mark.parametrize("position", decode.SMOLLM2_POSITIONS)
def test_smollm2(position: int) -> None:
    """SmolLM2's whole step at `position` of the long cache's text
    (scripts/fill_long_cache.py), at F2's memory latency: bit for bit, and
    each command within 10% of the perf model's cycles."""
    try:
        got, model, steps = decode.smollm2_step(position)
    except eng.NoLongCache as e:
        pytest.skip(str(e))
    i, worst = decode.worst(got, model)
    print(
        f"position {position}: {sum(got)} cycles on the unit, the perf model {sum(model):.0f}; "
        f"worst command {i} ({steps[i].kind}): {got[i]} against {model[i]:.0f}"
    )
    assert abs(worst) <= 0.10, (position, i, steps[i].kind, got[i], model[i])
