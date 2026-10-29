# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""First-order performance model (M2): how long one decode token takes on
the accelerator, and which part limits it.

Generic over three inputs:
- the model's shape (a `transformers` config): what a token computes;
- the platform (model/platforms/*.json): what the hardware gives, each value
  with a status (quoted, measured, guess, ...) and a source; our own
  measurements (platforms/measured/*.json) replace the documented values;
- the design (model/designs/*.json): what we choose (clock, lanes, ...),
  with named scenarios that override it.

ops() lists the work of one decode step, one entry per command of the
command list (model/commands.py), which a test checks bit for bit against the
golden model's decode step. predict() turns the work
into four times: memory traffic, the matrix-vector engine, the vector unit,
and the fixed cost per command. The token takes at least the largest of them
(perfect overlap) and at most their sum (no overlap).

Assumptions: weights and the KV cache live in HBM and are read once per token
(K and V once per KV head, shared by its query heads); activations stay on
chip; every command costs a fixed time, and every vector-unit command is one
pass over its elements; the HBM read rate scales with the memory
clock from the measured rate at the port's maximum clock. The engine is
`lanes` lanes of `macs_per_lane` multiply-adds each; a lane computes one row
at a time with `accumulators` partial sums (docs/numerics.md, section 3),
and adds them on its own adder while it runs the next row (engine_cycles).
Back-to-back engine commands leave no gap (D-041); one that steps only after
an earlier command's result (D-042) adds `engine_startup_cycles`, its
pipeline's fill and drain (its weights were read ahead).

Three design flags change the work, not the bits (D-054, Q-27):
`softmax_fused` runs softmax's subtraction, exp and sum in one vector pass,
`max_in_drain` has the engine keep each score row's max as it writes it (no
MAX pass), and `kv_tiles_shared` reads a K or V tile once per KV head
instead of once per query head. Each is assumed to cost no cycles.
"""

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import commands
import paths
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from transformers import LlamaConfig

BF16_BYTES = 2
TERMS = ("memory", "engine", "vector", "commands")
# How we know an input value -> whether it is settled (no sensitivity row in
# the report). "measured by us" replaces every other status (D-025); "not
# measurable" needs a reason in the source.
STATUSES = {
    "quoted": True,
    "computed": True,
    "measured by AWS": False,
    "measured by us": True,
    "not measurable": True,
    "to verify": False,
    "guess": False,
    "choice": True,
}
MEASURED = "measured by us"
CLEARED = {MEASURED, "not measurable"}  # the platform is cleared when all are
# Units shown scaled: unit -> (divisor, shown unit).
UNITS = {"Hz": (1e6, "MHz"), "B/s": (1e9, "GB/s"), "B": (2**30, "GiB")}
# Ops on the matrix-vector engine that are attention, not weights.
ATTENTION_KINDS = ("scores", "values")
# Design values that are counts: whole numbers (Design checks).
ENGINE_COUNTS = ("lanes", "macs_per_lane", "accumulators")
# Design values that are on (1) or off (0); see ops().
FLAGS = ("softmax_fused", "max_in_drain", "kv_tiles_shared")
VALUES_HEADS = 2  # heads per p.V pass, at most (rtl/engine.sv's Heads)


@dataclass(frozen=True)
class Step:
    """The work of one command. kind is the command in lower case, and shape
    what it works on:
    (rows, cols) of a weight matrix, (heads, positions, head_dim) for
    attention, (rows, length) on the vector unit, (elements,) for a load."""

    kind: str
    shape: tuple[int, ...]
    macs: int = 0
    weight_bytes: int = 0
    kv_read_bytes: int = 0
    kv_write_bytes: int = 0
    vector_elems: int = 0  # elements on the vector unit (one pass)
    # (rows, cols) of each block on the engine: a row per lane, cols
    # multiply-adds per row, rows in passes of the lanes within a block;
    # empty off the engine.
    engine_shapes: tuple[tuple[int, int], ...] = ()
    waits: bool = False  # on the engine, steps only after an earlier result (D-042)
    # Attention: cols' worth of cycles the engine waits for its tiles
    # (tile_wait_cols).
    fill_cols: int = 0


class Param(BaseModel):
    """One input value, with how we know it and where from."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    value: float
    unit: str = Field(min_length=1)
    status: Literal[tuple(STATUSES)]
    source: str = Field(min_length=1)

    def __str__(self) -> str:
        """The value in its unit, scaled for reading: "426.28 GB/s"."""
        scale, unit = UNITS.get(self.unit, (1, self.unit))
        return f"{self.value / scale:g} {unit}"

    def change_from(self, doc: Param) -> str:
        """The documented value and how far this one is from it, e.g.
        "426.28 GB/s (-5.9%)"; empty when they are the same entry."""
        if self == doc:
            return ""
        if doc.value == 0:
            raise ValueError(f"documented as 0 ({doc.source}), so no relative difference")
        return f"{doc} ({self.value / doc.value - 1:+.1%})"


class Measurement(Param):
    """One of our own measurements (D-025)."""

    status: Literal["measured by us"]


class Inputs(BaseModel):
    """A set of Params (the class's Param fields) plus a name."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str

    @property
    def params(self) -> dict[str, Param]:
        return {k: v for k, v in self if isinstance(v, Param)}

    def values(self) -> dict[str, float]:
        return {k: p.value for k, p in self.params.items()}


class Platform(Inputs):
    """What the hardware gives. Documented values come from
    platforms/<name>.json; our measurements, in platforms/measured/<name>.json,
    replace them entry by entry (load_platform)."""

    hbm_bytes: Param
    hbm_ports: Param
    hbm_port_bytes: Param
    hbm_port_max_clock_hz: Param
    hbm_read_bytes_per_s: Param
    shell_clock_hz: Param
    ddr_peak_bytes_per_s: Param

    def unmeasured(self) -> list[str]:
        """Parameters still waiting for our own measurement (D-025)."""
        return [k for k, p in self.params.items() if p.status not in CLEARED]


class Design(Inputs):
    """What we choose, with named scenarios that override some values."""

    clock_hz: Param
    memory_clock_hz: Param
    lanes: Param
    macs_per_lane: Param
    accumulators: Param
    vector_elems_per_cycle: Param
    command_cycles: Param
    engine_startup_cycles: Param
    softmax_fused: Param
    max_in_drain: Param
    kv_tiles_shared: Param
    scenarios: dict[str, dict[str, float]] = {}

    @model_validator(mode="after")
    def _scenarios_override_parameters(self) -> Design:
        for name, overrides in self.scenarios.items():
            unknown = set(overrides) - set(self.params)
            assert not unknown, f"scenario {name!r} overrides unknown {sorted(unknown)}"
        return self

    @model_validator(mode="after")
    def _engine_counts_are_whole(self) -> Design:
        """Lanes, multiply-adds per lane and accumulators are counts, and the
        accumulators' final tree needs a power of two (numerics.md, section 3)."""
        for scenario in [None, *self.scenarios]:
            v = self.values(scenario)
            for k in ENGINE_COUNTS:
                assert v[k] >= 1 and v[k] == int(v[k]), f"{scenario or 'base'}: {k} = {v[k]}"
            a = int(v["accumulators"])
            assert a & (a - 1) == 0, f"{scenario or 'base'}: accumulators = {a}, not a power of 2"
        return self

    @model_validator(mode="after")
    def _flags_are_on_or_off(self) -> Design:
        for scenario in [None, *self.scenarios]:
            v = self.values(scenario)
            for k in FLAGS:
                assert v[k] in (0, 1), f"{scenario or 'base'}: {k} = {v[k]}, not 0 or 1"
        return self

    def values(self, scenario: str | None = None) -> dict[str, float]:
        return super().values() | (self.scenarios[scenario] if scenario else {})


# A measured-values file: platform parameter name -> Measurement.
PLATFORM_PARAMS = tuple(k for k, f in Platform.model_fields.items() if f.annotation is Param)
Measured = TypeAdapter(dict[Literal[PLATFORM_PARAMS], Measurement])


def load_platform(name: str, directory: Path = paths.PLATFORMS, measured: bool = True) -> Platform:
    """The documented values, each replaced by our measurement when there is
    one (measured=False: the documented values alone). A measurement replaces
    the whole entry, so its status and source come with it."""
    raw = json.loads((directory / f"{name}.json").read_text())
    documented = Platform(**raw)  # complete on its own, whatever is measured
    ours = directory / "measured" / f"{name}.json"
    if not measured or not ours.exists():
        return documented
    found = {k: m.model_dump() for k, m in Measured.validate_json(ours.read_text()).items()}
    for k, m in found.items():  # no conversion: a measurement is in the documented unit
        unit = documented.params[k].unit
        if m["unit"] != unit:
            raise ValueError(f"{ours}: {k} in {m['unit']!r}, documented in {unit!r}")
    return Platform(**raw | found)


def load_design(name: str, directory: Path = paths.DESIGNS) -> Design:
    return Design.model_validate_json((directory / f"{name}.json").read_text())


def load_config(name: str) -> LlamaConfig:
    return LlamaConfig.from_json_file(paths.CONFIGS / f"{name}.json")


def tile_wait_cols(units: list[tuple[int, int]]) -> int:
    """How long the engine waits for its tile buffer over these units (a
    pass and a tile each: cols to fill, cols to step; a cycle is
    `macs_per_lane` cols of either). Fill reads at the engine's rate, one
    unit at a time, into two slots, a slot refilled once its unit is
    stepped; read ahead (D-042, D-044), the first two are full at the go."""
    fill_end = step_end = 0
    ends: list[int] = []  # each unit's step end
    for u, (fill, step) in enumerate(units):
        if u >= 2:
            fill_end = max(fill_end, ends[u - 2]) + fill
        step_end = max(step_end, fill_end) + step
        ends.append(step_end)
    return step_end - sum(step for _, step in units)


def attention(
    cmd: commands.Command, positions: int, lanes: int, tiles_shared: bool = False
) -> Step:
    """q.K (SCORES) or p.V (VALUES) over `positions` cache rows, as the engine
    runs it (architecture.md, KV cache layout): K or V read in whole tiles of
    `lanes` positions, once per query head (no tile shared between heads;
    Q-27), or once per KV head with `tiles_shared`: the best case, as p.V's
    passes of two heads cross groups of an odd size, and the tile waits
    are kept as unshared. Scores have a row per position, a pass
    per head and tile; p.V a row per head and output dimension, in passes
    of up to VALUES_HEADS heads (one if two do not fit the lanes)."""
    heads, d = cmd.n, cmd.m
    tiles = math.ceil(positions / lanes)
    if cmd.op == commands.Op.SCORES:  # a unit fills d cols and steps d
        shapes = ((positions, d),) * heads
        units = [(d, d)] * heads * tiles
    else:  # a unit fills its heads' d values per position, steps its positions
        per_pass = max(1, min(VALUES_HEADS, lanes // d))
        passes = [min(per_pass, heads - h) for h in range(0, heads, per_pass)]
        shapes = tuple((n * d, positions) for n in passes)
        in_tile = [min(lanes, positions - j * lanes) for j in range(tiles)]
        units = [(n * d, p) for n in passes for p in in_tile]
    readers = cmd.kv_heads if tiles_shared else heads
    return Step(
        cmd.op.name.lower(),
        (heads, positions, d),
        heads * positions * d,
        kv_read_bytes=readers * tiles * lanes * d * BF16_BYTES,
        engine_shapes=shapes,
        fill_cols=tile_wait_cols(units),
    )


def step(cmd: commands.Command, t: int, lanes: int, tiles_shared: bool) -> Step:
    """The work of one command when attention reads t positions, on an
    engine of `lanes` lanes (tiles_shared: attention())."""
    match cmd.op:
        case commands.Op.MATVEC:
            n = cmd.n * cmd.m
            return Step(
                "matvec", (cmd.n, cmd.m), n, n * BF16_BYTES, engine_shapes=((cmd.n, cmd.m),)
            )
        case commands.Op.LOAD:
            dtype = commands.BF16 if cmd.flags & commands.Flag.SRC_BF16 else commands.F32
            return Step("load", (cmd.m,), weight_bytes=commands.n_bytes(cmd.m, dtype))
        case commands.Op.KV_STORE:
            return Step("kv_store", (cmd.n, cmd.m), kv_write_bytes=cmd.n * cmd.m * BF16_BYTES)
        case commands.Op.SCORES | commands.Op.VALUES:
            return attention(cmd, t, lanes, tiles_shared)
        case commands.Op.OUTPUT | commands.Op.END:  # OUTPUT goes over PCIe, not HBM
            return Step(cmd.op.name.lower(), ())
        case _:  # the vector unit
            length = cmd.row_length(t)
            return Step(cmd.op.name.lower(), (cmd.n, length), vector_elems=cmd.n * length)


def ops(
    config: LlamaConfig,
    position: int,
    lanes: int,
    softmax_fused: bool = False,
    max_in_drain: bool = False,
    kv_tiles_shared: bool = False,
) -> list[Step]:
    """The work of one decode step at `position` (0-based: it attends to
    position + 1 cache rows), one entry per command, on an engine of
    `lanes` lanes (attention reads whole tiles of that many positions),
    with the design flags (module docstring). The softmax flags drop
    softmax's commands here: commands.build() has neither the fused
    command nor the engine's max (D-054, deferred by D-057)."""
    layout = commands.Layout.of(config, cap=position + 1)
    cmds = commands.build(layout)
    dropped = {commands.Op.MAX} if max_in_drain else set()
    dropped |= {commands.Op.EXP, commands.Op.SUM} if softmax_fused else set()
    return [
        dataclasses.replace(step(cmd, position + 1, lanes, kv_tiles_shared), waits=w)
        for cmd, w in zip(cmds, commands.engine_waits(cmds), strict=True)
        if not (cmd.flags & commands.Flag.LEN_T and cmd.op in dropped)
    ]


def design_ops(config: LlamaConfig, position: int, design: dict[str, float]) -> list[Step]:
    """ops() on the design's engine, with its flags."""
    return ops(config, position, int(design["lanes"]), **{k: bool(design[k]) for k in FLAGS})


def parameter_bytes(config: LlamaConfig) -> int:
    """Bytes of the checkpoint's BF16 tensors (the embedding counted once when
    the output projection is tied to it). Independent of ops(), so the tests
    can check one against the other."""
    c = config
    d, hidden, inter = c.head_dim, c.hidden_size, c.intermediate_size
    q_dim, kv_dim = c.num_attention_heads * d, c.num_key_value_heads * d
    layer = 2 * hidden * q_dim + 2 * hidden * kv_dim + 3 * hidden * inter + 2 * hidden
    head = (1 if c.tie_word_embeddings else 2) * c.vocab_size * hidden
    return (c.num_hidden_layers * layer + head + hidden) * BF16_BYTES


@dataclass(frozen=True)
class Work:
    """Totals of a list of ops: what a token moves and computes."""

    weight_bytes: int
    kv_read_bytes: int
    kv_write_bytes: int
    matvec_macs: int
    attention_macs: int
    vector_elems: int
    commands: int
    engine_shapes: tuple[tuple[int, int], ...]  # (rows, cols) per engine op
    engine_waits: int  # engine ops that wait for an earlier result
    engine_fill_cols: tuple[int, ...] = ()  # Step.fill_cols of each engine op

    @property
    def memory_bytes(self) -> int:
        return self.weight_bytes + self.kv_read_bytes + self.kv_write_bytes

    @property
    def macs(self) -> int:
        return self.matvec_macs + self.attention_macs

    @staticmethod
    def of(op_list: list[Step]) -> Work:
        def total(attr: str, keep=lambda o: True) -> int:
            return sum(getattr(o, attr) for o in op_list if keep(o))

        return Work(
            weight_bytes=total("weight_bytes"),
            kv_read_bytes=total("kv_read_bytes"),
            kv_write_bytes=total("kv_write_bytes"),
            matvec_macs=total("macs", lambda o: o.kind not in ATTENTION_KINDS),
            attention_macs=total("macs", lambda o: o.kind in ATTENTION_KINDS),
            vector_elems=total("vector_elems"),
            commands=len(op_list),
            engine_shapes=tuple(shape for o in op_list for shape in o.engine_shapes),
            engine_waits=total("waits"),
            engine_fill_cols=tuple(o.fill_cols for o in op_list if o.engine_shapes),
        )


def hbm_bytes_per_s(platform: dict[str, float], design: dict[str, float]) -> float:
    """The measured HBM read rate, scaled to the memory clock and capped by
    the ports' raw width."""
    clock = min(design["memory_clock_hz"], platform["hbm_port_max_clock_hz"])
    measured = platform["hbm_read_bytes_per_s"] * clock / platform["hbm_port_max_clock_hz"]
    return min(measured, platform["hbm_ports"] * platform["hbm_port_bytes"] * clock)


def engine_cycles(
    shapes: tuple[tuple[int, int], ...],
    design: dict[str, float],
    waits: int = 0,
    fill_cols: tuple[int, ...] = (),
) -> float:
    """Cycles of the engine for these (rows, cols), of which `waits` commands
    wait for an earlier result: rows in passes of `lanes` (a partial pass
    leaves lanes idle); a row takes cols / `macs_per_lane` cycles, but at
    least the `accumulators` - 1 adds of its final sum, which run during the
    next row; each waiting command adds `engine_startup_cycles` (D-042), and
    each command's tile wait its fill_cols / `macs_per_lane` (Step.fill_cols)."""
    lanes, per_lane = int(design["lanes"]), int(design["macs_per_lane"])
    tree = int(design["accumulators"]) - 1
    return (
        waits * design["engine_startup_cycles"]
        + sum(math.ceil(f / per_lane) for f in fill_cols)
        + sum(
            math.ceil(rows / lanes) * max(math.ceil(cols / per_lane), tree) for rows, cols in shapes
        )
    )


def engine_use(
    shapes: tuple[tuple[int, int], ...], design: dict[str, float], fill_cols: tuple[int, ...] = ()
) -> float:
    """Share of the engine's multiply-add slots that do work (tile waits
    count as idle)."""
    slots = design["lanes"] * design["macs_per_lane"] * engine_cycles(shapes, design, 0, fill_cols)
    return sum(rows * cols for rows, cols in shapes) / slots


def seconds(work: Work, platform: dict[str, float], design: dict[str, float]) -> dict[str, float]:
    """Time of each term for one token, in seconds."""
    clock = design["clock_hz"]
    return {
        "memory": work.memory_bytes / hbm_bytes_per_s(platform, design),
        "engine": engine_cycles(
            work.engine_shapes, design, work.engine_waits, work.engine_fill_cols
        )
        / clock,
        "vector": work.vector_elems / (design["vector_elems_per_cycle"] * clock),
        "commands": work.commands * design["command_cycles"] / clock,
    }


@dataclass(frozen=True)
class Prediction:
    terms: dict[str, float]  # seconds per token

    @property
    def bound(self) -> str:
        return max(self.terms, key=self.terms.get)

    @property
    def tokens_per_s_overlapped(self) -> float:
        return 1 / max(self.terms.values())

    @property
    def tokens_per_s_serial(self) -> float:
        return 1 / sum(self.terms.values())

    @property
    def tokens_per_s_in_order(self) -> float:
        """Memory and engine overlapped, the vector unit and commands after
        them: the controller stops at each command until its inputs are
        written (D-045), and a decode step's commands depend on each other."""
        t = self.terms
        return 1 / (max(t["memory"], t["engine"]) + t["vector"] + t["commands"])


def predict(
    config: LlamaConfig, platform: dict[str, float], design: dict[str, float], position: int
) -> Prediction:
    work = Work.of(design_ops(config, position, design))
    return Prediction(seconds(work, platform, design))
