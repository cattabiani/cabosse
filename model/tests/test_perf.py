# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""First-order performance model (model/perf.py): its work list is the
golden model's decode step, its byte counts are the checkpoint's, and its
times scale as they should."""

import json
import math

import paths
import perf
import perf_report
import pytest
from golden import decoder, dot, tiny, vector
from safetensors import safe_open

STATUSES = {
    "quoted",
    "computed",
    "measured by AWS",
    "measured by us",
    "to verify",
    "guess",
    "choice",
}


def golden_events(monkeypatch, config, position: int) -> list[tuple]:
    """The operations the golden model runs for one decode step at
    `position`, recorded by wrapping its matrix-vector product, attention and
    vector operations."""
    model = decoder.from_state_dict(config, tiny.random_weights(config, 0))
    cache = decoder.KVCache.empty(model, position + 1)
    if position:
        decoder.step(model, cache, list(range(position)))
    events = []

    def record(module, name, event):
        original = getattr(module, name)

        def wrapper(*args, **kwargs):
            events.append(event(*args))
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)

    record(dot, "matvec", lambda w, x: ("matvec", *w.shape))
    record(
        decoder,
        "attention",
        lambda q, k, *_: ("attention", q.shape[1] * q.shape[2], k.shape[-2], q.shape[-1]),
    )
    record(vector, "rmsnorm", lambda x, *_: ("rmsnorm", x.shape[-1]))
    record(vector, "rope", lambda x, *_: ("rope", x[0].numel()))
    record(vector, "swiglu", lambda a, b: ("swiglu", a.shape[-1]))
    decoder.decode_step(model, cache, 1)
    return events


def perf_events(config, position: int) -> list[tuple]:
    """perf.ops() in the golden model's terms: one attention call for
    scores, softmax and p.V, and RoPE applied to k then q."""
    out = []
    for op in perf.ops(config, position):
        if op.unit == "engine" and op.name not in ("scores", "pv"):
            out.append(("matvec", *op.shape))
        elif op.name == "scores":
            out.append(("attention", *op.shape))
        elif op.name.endswith("norm"):
            out.append(("rmsnorm", *op.shape))
        elif op.name == "rope":
            q_dim = config.num_attention_heads * config.head_dim
            out += [("rope", op.shape[0] - q_dim), ("rope", q_dim)]
        elif op.name == "swiglu":
            out.append(("swiglu", *op.shape))
    return out


@pytest.mark.parametrize("position", [0, 5])
@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_ops_follow_the_golden_decode_step(monkeypatch, position: int, n_kv_heads: int) -> None:
    config = tiny.tiny_config(n_kv_heads=n_kv_heads, tied=False)
    assert perf_events(config, position) == golden_events(monkeypatch, config, position)


@pytest.mark.parametrize("tied", [True, False])
def test_parameter_bytes_match_the_weights(tied: bool) -> None:
    config = tiny.tiny_config(tied=tied)
    state = tiny.random_weights(config, 0)
    assert perf.parameter_bytes(config) == sum(t.numel() * t.element_size() for t in state.values())


@pytest.mark.parametrize("tied", [True, False])
def test_a_token_reads_every_weight_once(tied: bool) -> None:
    """All weights, plus the one embedding row; untied, minus the embedding
    table (only its row is read)."""
    c = tiny.tiny_config(tied=tied)
    read = sum(op.weight_bytes for op in perf.ops(c, 0))
    table = 0 if tied else c.vocab_size * c.hidden_size * perf.BF16_BYTES
    assert read == perf.parameter_bytes(c) + c.hidden_size * perf.BF16_BYTES - table


def test_kv_traffic_grows_with_the_position() -> None:
    """Each position adds one row of K and one of V per KV head and layer to
    what a token reads; the write is the new row."""
    c = tiny.tiny_config(n_kv_heads=2)
    row = c.num_hidden_layers * c.num_key_value_heads * c.head_dim * perf.BF16_BYTES
    a, b = (perf.ops(c, p) for p in (3, 4))
    assert sum(o.kv_read_bytes for o in b) - sum(o.kv_read_bytes for o in a) == 2 * row
    assert sum(o.kv_write_bytes for o in a) == 2 * row


PLATFORM = {
    "hbm_read_bytes_per_s": 1600.0,
    "hbm_port_max_clock_hz": 400.0,
    "hbm_ports": 2,
    "hbm_port_bytes": 1,
}
DESIGN = {
    "clock_hz": 100.0,
    "memory_clock_hz": 400.0,
    "macs_per_cycle": 10,
    "vector_elems_per_cycle": 4,
    "command_cycles": 5,
}


def test_seconds_by_hand() -> None:
    work = perf.Work(memory_bytes=1600, macs=1000, vector_elems=200, commands=4)
    # memory: the ports' raw width caps the measured 1600 B/s at 2 x 1 x 400 = 800 B/s
    assert perf.seconds(work, PLATFORM, DESIGN) == {
        "memory": 2.0,
        "engine": 1.0,
        "vector": 0.5,
        "commands": 0.2,
    }
    p = perf.Prediction(perf.seconds(work, PLATFORM, DESIGN))
    assert p.bound == "memory"
    assert p.tokens_per_s_overlapped == 0.5
    assert p.tokens_per_s_serial == pytest.approx(1 / 3.7)


def test_memory_time_scales_with_the_memory_clock() -> None:
    """Below the port's maximum clock the read rate scales with the clock;
    above it, it does not."""
    work = perf.Work(memory_bytes=100, macs=0, vector_elems=0, commands=0)
    platform = PLATFORM | {"hbm_ports": 100}  # no cap from the raw width
    time = {
        clock: perf.seconds(work, platform, DESIGN | {"memory_clock_hz": clock})["memory"]
        for clock in (100.0, 200.0, 400.0, 800.0)
    }
    assert time[100.0] == 2 * time[200.0] == 4 * time[400.0] == 4 * time[800.0]


@pytest.mark.parametrize(
    "path", sorted(perf.PLATFORMS.glob("*.json")) + sorted(perf.DESIGNS.glob("*.json"))
)
def test_inputs_are_complete(path) -> None:
    """Every parameter has a known status and a source; scenarios override
    only parameters that exist; the perf model gets every value it reads."""
    inputs = perf.load_inputs(path)
    assert inputs.params
    for name, p in inputs.params.items():
        assert p.status in STATUSES, (name, p.status)
        assert p.source, name
    for scenario in inputs.scenarios:
        inputs.values(scenario)


def test_smollm2_on_f2_runs() -> None:
    c = perf.load_config("SmolLM2-135M-Instruct")
    platform = perf.load_inputs(perf.PLATFORMS / "f2.json")
    design = perf.load_inputs(perf.DESIGNS / "v0.json")
    for scenario in design.scenarios:
        p = perf.predict(c, platform.values(), design.values(scenario), 0)
        assert set(p.terms) == set(perf.TERMS) and min(p.terms.values()) > 0


@pytest.mark.slow
@pytest.mark.skipif(not paths.SMOLLM2.exists(), reason=f"needs the checkpoint in {paths.SMOLLM2}")
def test_smollm2_config_and_size_match_the_checkpoint() -> None:
    committed = json.loads((perf.CONFIGS / "SmolLM2-135M-Instruct.json").read_text())
    assert committed == json.loads((paths.SMOLLM2 / "config.json").read_text())
    with safe_open(paths.SMOLLM2 / "model.safetensors", "pt") as f:
        n_bytes = sum(math.prod(f.get_slice(k).get_shape()) * 2 for k in f.keys())  # noqa: SIM118
    assert perf.parameter_bytes(perf.load_config("SmolLM2-135M-Instruct")) == n_bytes


def test_perf_doc_is_up_to_date() -> None:
    """docs/perf.md shows what the input files say; after changing one, run:
    python scripts/perf_report.py"""
    assert perf_report.render() == perf_report.PERF_DOC.read_text()
