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
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import commands
import paths
from commands import Flag, Op
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from transformers import LlamaConfig

BF16_BYTES = 2
F32_BYTES = 4
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
ATTENTION_KINDS = ("attention_scores", "attention_values")
# Design values that are counts: whole numbers (Design checks).
ENGINE_COUNTS = ("lanes", "macs_per_lane", "accumulators")


@dataclass(frozen=True)
class Step:
    """The work of one command. kind is the command (lower case; attention_scores
    and attention_values for SCORES and VALUES), and shape what it works on:
    (rows, cols) of a weight matrix, (heads, positions, head_dim) for
    attention, (rows, length) on the vector unit, (elements,) for a load."""

    kind: str
    shape: tuple[int, ...]
    macs: int = 0
    weight_bytes: int = 0
    kv_read_bytes: int = 0
    kv_write_bytes: int = 0
    vector_elems: int = 0  # elements on the vector unit (one pass)
    # (rows, cols) on the engine: a row per lane, cols multiply-adds per row;
    # None off the engine.
    engine_shape: tuple[int, int] | None = None


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


def attention(kind: str, heads: int, positions: int, d: int, kv_heads: int) -> Step:
    """q.K or p.V over `positions` cache rows: reads K (or V) once per KV
    head, shared by its query heads. On the engine, scores have a row per
    head and position, p.V a row per head and output dimension."""
    kv_bytes = kv_heads * positions * d * BF16_BYTES
    rows = {"attention_scores": (heads * positions, d), "attention_values": (heads * d, positions)}
    return Step(
        kind,
        (heads, positions, d),
        heads * positions * d,
        kv_read_bytes=kv_bytes,
        engine_shape=rows[kind],
    )


def step(cmd: commands.Command, t: int) -> Step:
    """The work of one command when attention reads t positions."""
    match cmd.op:
        case Op.MATVEC:
            n = cmd.n * cmd.m
            return Step("matvec", (cmd.n, cmd.m), n, n * BF16_BYTES, engine_shape=(cmd.n, cmd.m))
        case Op.LOAD:
            size = BF16_BYTES if cmd.flags & Flag.SRC_BF16 else F32_BYTES
            return Step("load", (cmd.m,), weight_bytes=cmd.m * size)
        case Op.KV_STORE:
            return Step("kv_store", (cmd.n, cmd.m), kv_write_bytes=cmd.n * cmd.m * BF16_BYTES)
        case Op.SCORES:
            return attention("attention_scores", cmd.n, t, cmd.m, cmd.kv_heads)
        case Op.VALUES:
            return attention("attention_values", cmd.n, t, cmd.m, cmd.kv_heads)
        case Op.OUTPUT | Op.END:  # OUTPUT goes over PCIe, not HBM
            return Step(cmd.op.name.lower(), ())
        case _:  # the vector unit
            length = t if cmd.flags & Flag.LEN_T else cmd.m
            return Step(cmd.op.name.lower(), (cmd.n, length), vector_elems=cmd.n * length)


def ops(config: LlamaConfig, position: int) -> list[Step]:
    """The work of one decode step at `position` (0-based: it attends to
    position + 1 cache rows), one entry per command."""
    layout = commands.Layout.of(config, cap=position + 1)
    return [step(cmd, position + 1) for cmd in commands.build(layout)]


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
            engine_shapes=tuple(o.engine_shape for o in op_list if o.engine_shape is not None),
        )


def hbm_bytes_per_s(platform: dict[str, float], design: dict[str, float]) -> float:
    """The measured HBM read rate, scaled to the memory clock and capped by
    the ports' raw width."""
    clock = min(design["memory_clock_hz"], platform["hbm_port_max_clock_hz"])
    measured = platform["hbm_read_bytes_per_s"] * clock / platform["hbm_port_max_clock_hz"]
    return min(measured, platform["hbm_ports"] * platform["hbm_port_bytes"] * clock)


def engine_cycles(shapes: tuple[tuple[int, int], ...], design: dict[str, float]) -> int:
    """Cycles of the engine for these (rows, cols): rows in passes of `lanes`
    (a partial pass leaves lanes idle); a row takes cols / `macs_per_lane`
    cycles, but at least the `accumulators` - 1 adds of its final sum, which
    run during the next row."""
    lanes, per_lane = int(design["lanes"]), int(design["macs_per_lane"])
    tree = int(design["accumulators"]) - 1
    return sum(
        math.ceil(rows / lanes) * max(math.ceil(cols / per_lane), tree) for rows, cols in shapes
    )


def engine_use(shapes: tuple[tuple[int, int], ...], design: dict[str, float]) -> float:
    """Share of the engine's multiply-add slots that do work."""
    slots = design["lanes"] * design["macs_per_lane"] * engine_cycles(shapes, design)
    return sum(rows * cols for rows, cols in shapes) / slots


def seconds(work: Work, platform: dict[str, float], design: dict[str, float]) -> dict[str, float]:
    """Time of each term for one token, in seconds."""
    clock = design["clock_hz"]
    return {
        "memory": work.memory_bytes / hbm_bytes_per_s(platform, design),
        "engine": engine_cycles(work.engine_shapes, design) / clock,
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


def predict(
    config: LlamaConfig, platform: dict[str, float], design: dict[str, float], position: int
) -> Prediction:
    return Prediction(seconds(Work.of(ops(config, position)), platform, design))
