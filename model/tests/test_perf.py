# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""First-order performance model (model/perf.py): its work list is the
golden model's decode step, its byte counts are the checkpoint's, and its
times scale as they should."""

import json
import math
import types
from itertools import product

import paths
import perf
import perf_report
import pytest
from golden import decoder, tiny
from safetensors import safe_open

# Small round inputs for times checked by hand.
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
# What the golden decoder calls, as (kind, *shape) per call, from the
# arguments: the same shapes perf.ops() gives (one decode position).
GOLDEN_CALLS = {
    "arith": {
        "up": lambda x: ("embed", x.shape[-1]),
        "add": lambda h, _: ("add", h.shape[-1]),
    },
    "dot": {"matvec": lambda w, _: ("matvec", *w.shape)},
    "vector": {
        "rmsnorm": lambda x, *_: ("rmsnorm", x.shape[-1]),
        "rope": lambda x, *_: ("rope", x[0].numel()),
        "softmax": lambda s, **_: ("softmax", s[0].numel()),
        "swiglu": lambda a, _: ("swiglu", a.shape[-1]),
    },
}
ATTENTION_CALLS = {  # module-level functions of decoder, called by attention()
    "attention_scores": lambda q, k, _: (
        "attention_scores",
        q[0].numel() // q.shape[-1],
        *k.shape[-2:],
    ),
    "attention_values": lambda p, v, *_: (
        "attention_values",
        p[0].numel() // p.shape[-1],
        *v.shape[-2:],
    ),
}


def wrap(f, event, events: list):
    def wrapper(*args, **kwargs):
        events.append(event(*args, **kwargs))
        return f(*args, **kwargs)

    return wrapper


def golden_events(monkeypatch, config, position: int) -> list[tuple]:
    """The calls the golden decoder makes for one decode step at `position`.
    The modules are wrapped as the decoder sees them, so their own inner
    calls (an arith.add inside the vector unit) are not counted."""
    model = decoder.from_state_dict(config, tiny.random_weights(config, 0))
    cache = decoder.KVCache.empty(model, position + 1)
    if position:
        decoder.step(model, cache, list(range(position)))
    events = []
    for module_name, calls in GOLDEN_CALLS.items():
        module = getattr(decoder, module_name)
        proxy = types.SimpleNamespace(**{k: getattr(module, k) for k in dir(module)})
        for name, event in calls.items():
            setattr(proxy, name, wrap(getattr(module, name), event, events))
        monkeypatch.setattr(decoder, module_name, proxy)
    for name, event in ATTENTION_CALLS.items():
        monkeypatch.setattr(decoder, name, wrap(getattr(decoder, name), event, events))
    decoder.decode_step(model, cache, 1)
    return events


@pytest.mark.parametrize("position", [0, 5])
@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_ops_are_the_golden_decode_step(monkeypatch, position: int, n_kv_heads: int) -> None:
    config = tiny.tiny_config(n_kv_heads=n_kv_heads, tied=False)
    ops = [(op.kind, *op.shape) for op in perf.ops(config, position)]
    assert ops == golden_events(monkeypatch, config, position)


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
    read = perf.Work.of(perf.ops(c, 0)).weight_bytes
    table = 0 if tied else c.vocab_size * c.hidden_size * perf.BF16_BYTES
    assert read == perf.parameter_bytes(c) + c.hidden_size * perf.BF16_BYTES - table


def test_kv_traffic_grows_with_the_position() -> None:
    """Each position adds one row of K and one of V per KV head and layer to
    what a token reads; the write is the new row."""
    c = tiny.tiny_config(n_kv_heads=2)
    row = c.num_hidden_layers * c.num_key_value_heads * c.head_dim * perf.BF16_BYTES
    a, b = (perf.Work.of(perf.ops(c, p)) for p in (3, 4))
    assert b.kv_read_bytes - a.kv_read_bytes == 2 * row
    assert a.kv_write_bytes == 2 * row


def test_seconds_by_hand() -> None:
    work = perf.Work(
        weight_bytes=1000,
        kv_read_bytes=500,
        kv_write_bytes=100,
        matvec_macs=900,
        attention_macs=100,
        vector_elems=200,
        commands=4,
    )
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
    work = perf.Work(100, 0, 0, 0, 0, 0, 0)
    platform = PLATFORM | {"hbm_ports": 100}  # no cap from the raw width
    time = {
        clock: perf.seconds(work, platform, DESIGN | {"memory_clock_hz": clock})["memory"]
        for clock in (100.0, 200.0, 400.0, 800.0)
    }
    assert time[100.0] == 2 * time[200.0] == 4 * time[400.0] == 4 * time[800.0]


def test_bad_inputs_are_rejected(tmp_path) -> None:
    raw = json.loads((paths.DESIGNS / "v0.json").read_text())
    clock = raw["parameters"]["clock_hz"]
    for bad in ({**clock, "status": "rumour"}, {**clock, "source": ""}):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(raw | {"parameters": raw["parameters"] | {"clock_hz": bad}}))
        with pytest.raises(AssertionError):
            perf.load_inputs(path)
    with pytest.raises(AssertionError):  # a scenario overriding a parameter that does not exist
        perf.Inputs("x", {}, {"s": {"clock_mhz": 1.0}}).values("s")


@pytest.mark.parametrize(
    ("config", "platform", "design"),
    list(
        product(
            sorted(p.stem for p in paths.CONFIGS.glob("*.json")),
            sorted(paths.PLATFORMS.glob("*.json")),
            sorted(paths.DESIGNS.glob("*.json")),
        )
    ),
)
def test_every_input_combination_predicts(config: str, platform, design) -> None:
    c = perf.load_config(config)
    plat, des = perf.load_inputs(platform), perf.load_inputs(design)
    for scenario in [None, *des.scenarios]:
        p = perf.predict(c, plat.values(), des.values(scenario), c.max_position_embeddings - 1)
        assert set(p.terms) == set(perf.TERMS) and min(p.terms.values()) > 0


@pytest.mark.slow
@pytest.mark.skipif(not paths.SMOLLM2.exists(), reason=f"needs the checkpoint in {paths.SMOLLM2}")
def test_smollm2_config_and_size_match_the_checkpoint() -> None:
    committed = json.loads((paths.CONFIGS / f"{perf_report.MODEL}.json").read_text())
    assert committed == json.loads((paths.SMOLLM2 / "config.json").read_text())
    with safe_open(paths.SMOLLM2 / "model.safetensors", "pt") as f:
        shapes = [f.get_slice(k).get_shape() for k in f.keys()]  # noqa: SIM118
    n_bytes = sum(math.prod(s) for s in shapes) * perf.BF16_BYTES
    assert perf.parameter_bytes(perf.load_config(perf_report.MODEL)) == n_bytes


def test_perf_doc_is_up_to_date() -> None:
    """docs/perf.md shows what the input files say; after changing one, run:
    python scripts/perf_report.py"""
    assert perf_report.render() == perf_report.PERF_DOC.read_text()
