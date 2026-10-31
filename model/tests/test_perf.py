# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""First-order performance model (model/perf.py): its work list is the
golden model's decode step, its byte counts are the checkpoint's, and its
times scale as they should."""

import dataclasses
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
    "issue_cycles": 5,
    "engine_startup_cycles": 25,
    "vector_startup_cycles": 10,
    "reduction_tail_cycles": 20,
    "reduction_row_cycles": 6,
    "load_latency_cycles": 30,
    "kv_store_latency_cycles": 40,
    "softmax_fused": 0,
    "max_in_drain": 0,
    "kv_tiles_shared": 0,
}
LANES = 128  # the engine the work tests count on
# One of our measurements, as a measurement script would write it.
MEASUREMENT = {
    "value": 401.2e9,
    "unit": "B/s",
    "status": "measured by us",
    "source": "scripts/f2_hbm.sh, 2026-10-10, commit abc1234",
}


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
    read = perf.Work.of(perf.ops(c, 0, LANES)).weight_bytes
    table = 0 if tied else c.vocab_size * c.hidden_size * perf.BF16_BYTES
    rope = 2 * c.head_dim * 4  # FP32 cos and sin
    assert read == perf.parameter_bytes(c) + c.hidden_size * perf.BF16_BYTES + rope - table


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize(("position", "tiles"), [(0, 1), (7, 1), (8, 2), (20, 3)])
def test_kv_traffic_in_whole_tiles(position: int, tiles: int, shared: bool) -> None:
    """A token reads K and V in whole tiles of 8 positions on 8 lanes, per
    query head and layer (no tile shared between heads), or per KV head with
    kv_tiles_shared (Q-27), which changes nothing else; it writes one row of
    K and one of V per KV head and layer."""
    c = tiny.tiny_config(n_kv_heads=2)
    row = c.num_hidden_layers * c.head_dim * perf.BF16_BYTES
    work = perf.Work.of(perf.ops(c, position, 8, kv_tiles_shared=shared))
    readers = c.num_key_value_heads if shared else c.num_attention_heads
    assert work.kv_read_bytes == 2 * readers * tiles * 8 * row
    assert work.kv_write_bytes == 2 * c.num_key_value_heads * row
    alone = perf.Work.of(perf.ops(c, position, 8))
    assert dataclasses.replace(work, kv_read_bytes=alone.kv_read_bytes) == alone


@pytest.mark.parametrize(
    ("flags", "passes"),
    [
        ({}, 5),
        ({"softmax_fused": True}, 3),
        ({"max_in_drain": True}, 4),
        ({"softmax_fused": True, "max_in_drain": True}, 2),
    ],
)
def test_softmax_flags_remove_passes(flags: dict[str, bool], passes: int) -> None:
    """Softmax makes 5 passes over each head's t scores (MAX, ADD, EXP,
    SUM, MUL); fusing ADD, EXP and SUM leaves 3, the engine's max 4, both 2.
    Each pass removed is one command per layer; the engine's work stays."""
    c, position = tiny.tiny_config(n_kv_heads=2), 20
    t, layers, heads = position + 1, c.num_hidden_layers, c.num_attention_heads
    base = perf.Work.of(perf.ops(c, position, 8))
    work = perf.Work.of(perf.ops(c, position, 8, **flags))
    removed = 5 - passes
    assert base.vector_elems - work.vector_elems == removed * layers * heads * t
    assert base.commands - work.commands == removed * layers
    assert (work.engine_shapes, work.engine_waits) == (base.engine_shapes, base.engine_waits)


def test_scores_pass_per_head_and_tile() -> None:
    """SCORES' engine cycles: a pass per head and tile, not packed across
    heads (5 positions, 4 heads on 8 lanes: 4 passes, not 3)."""
    design = {"lanes": 8, "macs_per_lane": 4, "accumulators": 16, "engine_startup_cycles": 0}
    cmd = commands.Command(commands.Op.SCORES, n=4, m=64, kv_heads=2)
    shapes = perf.attention(cmd, 5, 8).engine_shapes
    assert perf.engine_cycles(shapes, design) == 4 * 16


def test_values_passes_of_two_heads() -> None:
    """p.V's engine cycles: passes of at most 2 heads, as the engine runs
    them, even when the lanes would hold more (SmolLM2's 9 heads of 64 on
    256 lanes: 5 passes, not 3); one head a pass when two do not fit."""
    design = {"lanes": 256, "macs_per_lane": 4, "accumulators": 16, "engine_startup_cycles": 0}
    cmd = commands.Command(commands.Op.VALUES, n=9, m=64, kv_heads=3)
    one_pass = perf.engine_cycles(((128, 100),), design)
    assert perf.engine_cycles(perf.attention(cmd, 100, 256).engine_shapes, design) == 5 * one_pass
    assert len(perf.attention(cmd, 100, 64).engine_shapes) == 9


def test_tile_wait_by_hand() -> None:
    """Three slots, the first three full at the go: a unit after them waits
    when its fill and landing (24 cols) are longer than the steps since its
    slot was freed. SmolLM2's p.V at 64 positions on 128 lanes: 4 passes of
    2 heads and 1 of 1 (fills of 128, 128, 128, 128 and 64 cols, steps of
    64): the fourth fills from 64 to 192 and steps at 216, 24 cols late; the
    fifth fills by 256 and steps at 280, on time. Balanced units, and SCORES
    (d to fill, d to step), never wait."""
    units = [(128, 64)] * 4 + [(64, 64)]
    assert perf.tile_wait_cols(units) == 24
    assert perf.tile_wait_cols([(128, 128)] * 9) == 0
    values = commands.Command(commands.Op.VALUES, n=9, m=64, kv_heads=3)
    assert perf.attention(values, 64, 128).fill_cols == 24
    assert perf.attention(values, 1024, 128).fill_cols == 0
    scores = commands.Command(commands.Op.SCORES, n=9, m=64, kv_heads=3)
    assert perf.attention(scores, 64, 128).fill_cols == 0


def test_seconds_by_hand() -> None:
    work = perf.Work(
        weight_bytes=1000,
        kv_read_bytes=500,
        kv_write_bytes=100,
        matvec_macs=900,
        attention_macs=100,
        vector_elems=160,
        commands=4,
        engine_shapes=((10, 100),),  # one pass of the 10 lanes, 100 cycles
        engine_waits=2,  # 2 x 25 cycles of start-up
        vector_steps=(perf.Step("add", (1, 160), walk=(1, 160)),),  # 40 groups + 10
    )
    # memory: the ports' raw width caps the measured 1600 B/s at 2 x 1 x 400 = 800 B/s
    assert perf.seconds(work, PLATFORM, DESIGN) == {
        "memory": 2.0,
        "engine": 1.5,
        "vector": 0.5,
        "commands": 0.2,
    }
    p = perf.Prediction(perf.seconds(work, PLATFORM, DESIGN))
    assert p.bound == "memory"
    assert p.tokens_per_s_overlapped == 0.5
    assert p.tokens_per_s_serial == pytest.approx(1 / 4.2)
    assert p.tokens_per_s_in_order == pytest.approx(1 / 2.7)  # max(2, 1.5) + 0.5 + 0.2


def vector_step(kind: str, walk: tuple[int, int], stops: bool = False, **kw: int) -> perf.Step:
    return perf.Step(kind, walk, walk=walk, stops=stops, **kw)


# Steps -> each one's cycles with DESIGN (4 elements a group, start-up 10,
# a reduction's tail 20 and rows 6 apart, LOAD 30 and KV_STORE 40 later).
VECTOR_CASES = [
    # Back to back, the latency once; a stop waits for the done before it.
    ([vector_step("add", (1, 16)), vector_step("mul", (1, 8))], [14, 2]),
    ([vector_step("add", (1, 16)), vector_step("mul", (1, 8), stops=True)], [14, 12]),
    # Rows of a reduction 6 apart at least (rows of 1 group) or their groups.
    ([vector_step("sum", (3, 4))], [1 + 2 * 6 + 10 + 20]),
    ([vector_step("max", (3, 40))], [30 + 10 + 20]),
    # The next step starts at a reduction's done.
    ([vector_step("sumsq", (1, 4)), vector_step("add", (1, 4))], [31, 11]),
    # A LOAD a beat (32 bytes) a cycle after its latency.
    ([vector_step("load", (1, 64), weight_bytes=128)], [30 + 4 + 10]),
    # A KV_STORE's writes answered later; a step after it waits for its done.
    ([vector_step("kv_store", (2, 8)), vector_step("add", (1, 8))], [4 + 10 + 40, 1]),
]


@pytest.mark.parametrize(("steps", "want"), VECTOR_CASES)
def test_vector_cycles_by_hand(steps: list[perf.Step], want: list[int]) -> None:
    assert perf.vector_cycles(tuple(steps), DESIGN) == want


@pytest.mark.parametrize(
    ("op", "flags", "n", "length", "walk"),
    [
        (commands.Op.ADD, commands.Flag.NONE, 3, 64, (1, 192)),  # packed
        (commands.Op.MUL, commands.Flag.B_PER_ROW, 3, 64, (3, 64)),
        (commands.Op.EXP, commands.Flag.LEN_T, 9, 5, (9, 5)),
        (commands.Op.ROTATE_HALF, commands.Flag.NONE, 3, 64, (6, 32)),
        (commands.Op.SUM, commands.Flag.LEN_T, 9, 5, (9, 5)),
    ],
)
def test_vector_walk(
    op: commands.Op, flags: commands.Flag, n: int, length: int, walk: tuple[int, int]
) -> None:
    cmd = commands.Command(op, commands.Buf.X, commands.Buf.H, n=n, m=length, flags=flags)
    assert perf.vector_walk(cmd, length) == walk


# (rows, cols) -> engine cycles with ENGINE_DESIGN.
ENGINE_DESIGN = {"lanes": 4, "macs_per_lane": 2, "accumulators": 4, "engine_startup_cycles": 9}
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
    op_list = perf.ops(c, position, LANES)
    on_engine = [o for o in op_list if o.engine_shapes]
    assert all(sum(map(math.prod, o.engine_shapes)) == o.macs for o in on_engine)
    assert {o.kind for o in on_engine} == {"matvec", *perf.ATTENTION_KINDS}
    assert all(o.macs == 0 for o in op_list if not o.engine_shapes)


def test_engine_use() -> None:
    """Full when every pass fills the lanes and rows are longer than the
    final sum; idle lanes and short rows show up as lost slots."""
    assert perf.engine_use(((8, 10),), ENGINE_DESIGN) == 1.0
    assert perf.engine_use(((5, 10),), ENGINE_DESIGN) == 5 / 8  # second pass: 1 of 4 lanes busy
    assert perf.engine_use(((4, 2),), ENGINE_DESIGN) == 1 / 3  # 1 cycle of work, 3 of final sum


def test_memory_time_scales_with_the_memory_clock() -> None:
    """Below the port's maximum clock the read rate scales with the clock;
    above it, it does not."""
    work = perf.Work(100, 0, 0, 0, 0, 0, 0, (), 0)
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
    for bad in ({"lanes": 2.5}, {"macs_per_lane": 0}, {"accumulators": 12}, {"max_in_drain": 2}):
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
