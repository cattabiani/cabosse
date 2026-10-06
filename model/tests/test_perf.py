# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""First-order performance model (model/perf.py): its work list is the
golden model's decode step, its byte counts are the checkpoint's, and its
times scale as they should."""

import json
import math
import shutil
from itertools import product
from pathlib import Path

import commands
import paths
import perf
import pytest
from golden import tiny
from pydantic import ValidationError
from reporting import perf_doc
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
    "lanes": 10,
    "macs_per_lane": 1,
    "accumulators": 1,
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


@pytest.mark.parametrize("position", [0, 5])
def test_ops_are_the_command_list(position: int) -> None:
    """One entry per command, in order: the command list is the decode step
    (test_commands checks it bit for bit against the golden model)."""
    config = tiny.tiny_config(n_kv_heads=2)
    cmds = commands.build(commands.Layout.of(config, position + 1))
    names = {"scores": "attention_scores", "values": "attention_values"}
    kinds = [names.get(c.op.name.lower(), c.op.name.lower()) for c in cmds]
    assert [o.kind for o in perf.ops(config, position)] == kinds


@pytest.mark.parametrize("tied", [True, False])
def test_parameter_bytes_match_the_weights(tied: bool) -> None:
    config = tiny.tiny_config(tied=tied)
    state = tiny.random_weights(config, 0)
    assert perf.parameter_bytes(config) == sum(t.numel() * t.element_size() for t in state.values())


@pytest.mark.parametrize("tied", [True, False])
def test_a_token_reads_every_weight_once(tied: bool) -> None:
    """All weights, plus the one embedding row and the position's FP32 RoPE
    rows; untied, minus the embedding table (only its row is read)."""
    c = tiny.tiny_config(tied=tied)
    read = perf.Work.of(perf.ops(c, 0)).weight_bytes
    table = 0 if tied else c.vocab_size * c.hidden_size * perf.BF16_BYTES
    rope = 2 * c.head_dim * perf.F32_BYTES
    assert read == perf.parameter_bytes(c) + c.hidden_size * perf.BF16_BYTES + rope - table


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
        engine_shapes=((10, 100),),  # one pass of the 10 lanes, 100 cycles
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


# (rows, cols) -> engine cycles with ENGINE_DESIGN.
ENGINE_DESIGN = {"lanes": 4, "macs_per_lane": 2, "accumulators": 4}
ENGINE_CASES = [
    ((4, 10), 5),  # one pass, 10 / 2 cycles per row
    ((5, 10), 10),  # a fifth row needs a second pass, with 3 lanes idle
    ((4, 11), 6),  # a partial last group still takes a cycle
    ((4, 2), 3),  # a short row waits for its final sum: 4 - 1 adds
    ((8, 6), 6),  # 6 / 2 = 3 cycles, as long as the final sum
]


@pytest.mark.parametrize(("shape", "cycles"), ENGINE_CASES)
def test_engine_cycles_by_hand(shape: tuple[int, int], cycles: int) -> None:
    assert perf.engine_cycles((shape,), ENGINE_DESIGN) == cycles


def test_engine_cycles_add_up_over_ops() -> None:
    shapes = tuple(shape for shape, _ in ENGINE_CASES)
    assert perf.engine_cycles(shapes, ENGINE_DESIGN) == sum(cycles for _, cycles in ENGINE_CASES)


@pytest.mark.parametrize("position", [0, 5])
def test_engine_shapes_cover_every_multiply_add(position: int) -> None:
    """The engine's rows times cols are exactly the ops' multiply-adds."""
    c = tiny.tiny_config(n_kv_heads=2)
    op_list = perf.ops(c, position)
    on_engine = [o for o in op_list if o.engine_shape is not None]
    assert all(math.prod(o.engine_shape) == o.macs for o in on_engine)
    assert {o.kind for o in on_engine} == {"matvec", *perf.ATTENTION_KINDS}
    assert all(o.macs == 0 for o in op_list if o.engine_shape is None)


def test_engine_use() -> None:
    """Full when every pass fills the lanes and rows are longer than the
    final sum; idle lanes and short rows show up as lost slots."""
    assert perf.engine_use(((8, 10),), ENGINE_DESIGN) == 1.0
    assert perf.engine_use(((5, 10),), ENGINE_DESIGN) == 5 / 8  # second pass: 1 of 4 lanes busy
    assert perf.engine_use(((4, 2),), ENGINE_DESIGN) == 1 / 3  # 1 cycle of work, 3 of final sum


def test_memory_time_scales_with_the_memory_clock() -> None:
    """Below the port's maximum clock the read rate scales with the clock;
    above it, it does not."""
    work = perf.Work(100, 0, 0, 0, 0, 0, 0, ())
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
    row = next(r for r in perf_doc.params_table(ours, doc).splitlines() if "hbm_read" in r)
    assert "401.2 GB/s | measured by us" in row and "426.28 GB/s (-5.9%)" in row
    assert "6 of 7" in perf_doc.cleared(ours) and "7 of 7" in perf_doc.cleared(doc)


@pytest.mark.parametrize("measured", [None, {}])  # no file, an empty file
def test_without_measurements_the_documented_values_hold(tmp_path, measured) -> None:
    directory = platform_files(tmp_path, measured)
    assert perf.load_platform("f2", directory) == perf.load_platform(
        "f2", directory, measured=False
    )


def test_cleared_when_every_value_is_measured_or_not_measurable(tmp_path) -> None:
    doc = perf.load_platform("f2")
    measured = {k: {**p.model_dump(), "status": perf.MEASURED} for k, p in doc.params.items()}
    platform = perf.load_platform("f2", platform_files(tmp_path, measured))
    assert platform.unmeasured() == []
    assert perf_doc.cleared(platform).startswith("All platform values are measured by us")


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
        {"hbm_read_bytes_per_s": {**MEASUREMENT, "unit": ""}},  # no unit
        {"hbm_read_bytes_per_s": None},  # looks like an entry, measures nothing
    ],
)
def test_a_bad_measured_file_is_rejected(tmp_path, measured: dict) -> None:
    with pytest.raises(ValidationError):
        perf.load_platform("f2", platform_files(tmp_path, measured))


def test_bad_design_inputs_are_rejected() -> None:
    raw = json.loads((paths.DESIGNS / "v0.json").read_text())
    for bad in ({**raw["clock_hz"], "status": "rumour"}, {**raw["clock_hz"], "source": ""}):
        with pytest.raises(ValidationError):
            perf.Design(**raw | {"clock_hz": bad})
    with pytest.raises(ValidationError, match="unknown"):
        perf.Design(**raw | {"scenarios": {"x": {"clock_mhz": 1}}})
    for bad in ({"lanes": 2.5}, {"macs_per_lane": 0}, {"accumulators": 12}):
        with pytest.raises(ValidationError, match="x: "):
            perf.Design(**raw | {"scenarios": {"x": bad}})


def test_a_measured_file_that_is_not_json_is_rejected(tmp_path) -> None:
    directory = platform_files(tmp_path, {})
    (directory / "measured" / "f2.json").write_text('{"hbm_ports": ')
    with pytest.raises(ValidationError, match="json"):
        perf.load_platform("f2", directory)


def test_a_measurement_in_another_unit_is_rejected(tmp_path) -> None:
    """No conversion: 401.2 GB/s written as {"value": 401.2, "unit": "GB/s"}
    would otherwise be read as 401.2 B/s."""
    measured = {"hbm_read_bytes_per_s": {**MEASUREMENT, "value": 401.2, "unit": "GB/s"}}
    with pytest.raises(ValueError, match="hbm_read_bytes_per_s in 'GB/s', documented in 'B/s'"):
        perf.load_platform("f2", platform_files(tmp_path, measured))


def test_a_measurement_does_not_complete_the_documented_file(tmp_path) -> None:
    directory = platform_files(tmp_path, {"hbm_read_bytes_per_s": MEASUREMENT})
    raw = json.loads((directory / "f2.json").read_text())
    del raw["hbm_read_bytes_per_s"]
    (directory / "f2.json").write_text(json.dumps(raw))
    with pytest.raises(ValidationError, match="hbm_read_bytes_per_s"):
        perf.load_platform("f2", directory)


def test_a_documented_zero_cannot_be_compared() -> None:
    zero = perf.Param(**{**MEASUREMENT, "status": "quoted", "value": 0})
    with pytest.raises(ValueError, match="documented as 0"):
        perf.Param(**MEASUREMENT).change_from(zero)


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
    committed = json.loads((paths.CONFIGS / f"{perf_doc.MODEL}.json").read_text())
    assert committed == json.loads((paths.SMOLLM2 / "config.json").read_text())
    with safe_open(paths.SMOLLM2 / "model.safetensors", "pt") as f:
        shapes = [f.get_slice(k).get_shape() for k in f.keys()]  # noqa: SIM118
    n_bytes = sum(math.prod(s) for s in shapes) * perf.BF16_BYTES
    assert perf.parameter_bytes(perf.load_config(perf_doc.MODEL)) == n_bytes


def test_perf_doc_is_up_to_date() -> None:
    """docs/perf.md shows what the input files say; after changing one, run:
    python scripts/report_perf.py"""
    assert perf_doc.render() == perf_doc.PATH.read_text()
