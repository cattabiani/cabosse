# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""First-order performance model (model/perf.py): its work list is the
golden model's decode step, its byte counts are the checkpoint's, and its
times scale as they should."""

import json
import math
import shutil
import types
from itertools import product
from pathlib import Path

import paths
import perf
import perf_report
import pytest
from golden import decoder, tiny
from pydantic import ValidationError
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
# One of our measurements, as a measurement script would write it.
MEASUREMENT = {
    "value": 401.2e9,
    "unit": "B/s",
    "status": "measured by us",
    "source": "scripts/f2_hbm.sh, 2026-10-10, commit abc1234",
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


def platform_files(tmp_path, measured: dict | None) -> Path:
    """A copy of the F2 platform file in tmp_path, plus a measured file."""
    shutil.copy(paths.PLATFORMS / "f2.json", tmp_path / "f2.json")
    if measured is not None:
        (tmp_path / "measured").mkdir()
        (tmp_path / "measured" / "f2.json").write_text(json.dumps(measured))
    return tmp_path


def test_a_measurement_replaces_the_documented_entry(tmp_path) -> None:
    """In the platform, and in the report: the measurement with the
    documented value next to it, and one value fewer to measure."""
    directory = platform_files(tmp_path, {"hbm_read_bytes_per_s": MEASUREMENT})
    doc = perf.load_platform("f2", directory, measured=False)
    ours = perf.load_platform("f2", directory)
    assert ours.hbm_read_bytes_per_s == perf.Param(**MEASUREMENT)  # value, status and source
    others = {k: p for k, p in ours.params.items() if k != "hbm_read_bytes_per_s"}
    assert others == {k: p for k, p in doc.params.items() if k != "hbm_read_bytes_per_s"}
    row = next(r for r in perf_report.params_table(ours, doc).splitlines() if "hbm_read" in r)
    assert "401.2 GB/s | measured by us" in row and "426.28 GB/s (-5.9%)" in row
    assert "6 of 7" in perf_report.cleared(ours) and "7 of 7" in perf_report.cleared(doc)


def test_without_a_measured_file_the_documented_values_hold(tmp_path) -> None:
    directory = platform_files(tmp_path, None)
    assert perf.load_platform("f2", directory) == perf.load_platform(
        "f2", directory, measured=False
    )


def test_cleared_when_every_value_is_measured_or_not_measurable(tmp_path) -> None:
    doc = perf.load_platform("f2")
    measured = {k: {**p.model_dump(), "status": perf.MEASURED} for k, p in doc.params.items()}
    assert perf.load_platform("f2", platform_files(tmp_path, measured)).unmeasured() == []


@pytest.mark.parametrize(
    "measured",
    [
        {"hbm_read_bytes_per_s": {**MEASUREMENT, "status": "quoted"}},  # not a measurement
        {
            "hbm_read_bytes_per_s": {**MEASUREMENT, "status": "not measurable"}
        },  # goes in the doc file
        {"hbm_read_bytes_per_sec": MEASUREMENT},  # a typo in the name
        {"hbm_read_bytes_per_s": {**MEASUREMENT, "source": ""}},  # no source
        {"hbm_read_bytes_per_s": {**MEASUREMENT, "date": "x"}},  # an unknown field
    ],
)
def test_a_bad_measured_file_is_rejected(tmp_path, measured: dict) -> None:
    with pytest.raises((AssertionError, ValidationError)):
        perf.load_platform("f2", platform_files(tmp_path, measured))


def test_bad_design_inputs_are_rejected() -> None:
    raw = json.loads((paths.DESIGNS / "v0.json").read_text())
    for bad in ({**raw["clock_hz"], "status": "rumour"}, {**raw["clock_hz"], "source": ""}):
        with pytest.raises(ValidationError):
            perf.Design(**raw | {"clock_hz": bad})
    with pytest.raises(ValidationError, match="unknown"):
        perf.Design(**raw | {"scenarios": {"x": {"clock_mhz": 1}}})


def test_environment_variables_do_not_fill_in(tmp_path, monkeypatch) -> None:
    """A parameter missing from the files is an error, even if an environment
    variable of that name exists."""
    directory = platform_files(tmp_path, None)
    raw = json.loads((directory / "f2.json").read_text())
    del raw["hbm_ports"]
    (directory / "f2.json").write_text(json.dumps(raw))
    monkeypatch.setenv("HBM_PORTS", json.dumps({**MEASUREMENT, "value": 1}))
    with pytest.raises(ValidationError, match="hbm_ports"):
        perf.load_platform("f2", directory)


@pytest.mark.parametrize(
    ("config", "platform", "design"),
    list(
        product(
            sorted(p.stem for p in paths.CONFIGS.glob("*.json")),
            sorted(p.stem for p in paths.PLATFORMS.glob("*.json")),
            sorted(p.stem for p in paths.DESIGNS.glob("*.json")),
        )
    ),
)
def test_every_input_combination_predicts(config: str, platform: str, design: str) -> None:
    c = perf.load_config(config)
    plat, des = perf.load_platform(platform), perf.load_design(design)
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
