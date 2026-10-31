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
chip; every command costs a fixed time to issue; the vector unit runs its
commands in order, each a pass over its elements plus the latencies that
show where a command waits for an earlier result (vector_cycles, D-053); the
HBM read rate scales with the memory
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
REDUCTION_KINDS = tuple(op.name.lower() for op in commands.REDUCE)
# Design values that are counts: whole numbers (Design checks).
ENGINE_COUNTS = ("lanes", "macs_per_lane", "accumulators")
# Design values that are on (1) or off (0); see ops().
FLAGS = ("softmax_fused", "max_in_drain", "kv_tiles_shared")
VALUES_HEADS = commands.VALUES_HEADS


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
    # On the vector unit (its commands, LOAD, KV_STORE): (rows, length) as
    # it walks them (vector_walk), and whether the controller stops at it
    # (commands.stops).
    walk: tuple[int, int] | None = None
    stops: bool = False


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
    issue_cycles: Param
    engine_startup_cycles: Param
    vector_startup_cycles: Param
    reduction_tail_cycles: Param
    reduction_row_cycles: Param
    load_latency_cycles: Param
    kv_store_latency_cycles: Param
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


def attention(
    cmd: commands.Command, positions: int, lanes: int, tiles_shared: bool = False
) -> Step:
    """q.K (SCORES) or p.V (VALUES) over `positions` cache rows, as the engine
    runs it (architecture.md, KV cache layout): K and V stream from HBM in
    the lanes' order (D-062), the positions read rounded up to 4,
    once per query head (Q-27), or once per KV head with `tiles_shared`: the
    best case, as p.V's passes of two heads cross groups of an odd size.
    Scores have a row per position; p.V a row per head and output
    dimension, in passes of up to VALUES_HEADS heads (one if two do not fit
    the lanes)."""
    heads, d = cmd.n, cmd.m
    if cmd.op == commands.Op.SCORES:
        shapes = ((positions, d),) * heads
    else:
        per_pass = max(1, min(VALUES_HEADS, lanes // d))
        passes = [min(per_pass, heads - h) for h in range(0, heads, per_pass)]
        shapes = tuple((n * d, positions) for n in passes)
    readers = cmd.kv_heads if tiles_shared else heads
    # Each port reads its 4 positions per beat (D-062): v0's lanes per port
    # and E; other scenarios reuse the rounding (estimate, off by under a
    # beat per port and head).
    read_positions = math.ceil(positions / commands.LANES_PER_PORT) * commands.LANES_PER_PORT
    return Step(
        cmd.op.name.lower(),
        (heads, positions, d),
        heads * positions * d,
        kv_read_bytes=readers * read_positions * d * BF16_BYTES,
        engine_shapes=shapes,
    )


def vector_walk(cmd: commands.Command, length: int) -> tuple[int, int]:
    """(rows, length) of a vector-unit command as rtl/vec_unit.sv walks it:
    an elementwise command without B_PER_ROW, B_ROW or LEN_T is one row of
    n·m, ROTATE_HALF walks each row's two halves."""
    per_row = commands.Flag.B_PER_ROW | commands.Flag.B_ROW | commands.Flag.LEN_T
    if cmd.op == commands.Op.ROTATE_HALF:
        return 2 * cmd.n, length // 2
    if cmd.op in commands.REDUCE or cmd.flags & per_row:
        return cmd.n, length
    return 1, cmd.n * length


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
            n_bytes = commands.n_bytes(cmd.m, dtype)
            return Step("load", (cmd.m,), weight_bytes=n_bytes, walk=(1, cmd.m))
        case commands.Op.KV_STORE:
            n_bytes = cmd.n * cmd.m * BF16_BYTES
            return Step("kv_store", (cmd.n, cmd.m), kv_write_bytes=n_bytes, walk=(cmd.n, cmd.m))
        case commands.Op.SCORES | commands.Op.VALUES:
            return attention(cmd, t, lanes, tiles_shared)
        case commands.Op.OUTPUT | commands.Op.END:  # OUTPUT goes over PCIe, not HBM
            return Step(cmd.op.name.lower(), ())
        case _:  # the vector unit
            length = cmd.row_length(t)
            return Step(
                cmd.op.name.lower(),
                (cmd.n, length),
                vector_elems=cmd.n * length,
                walk=vector_walk(cmd, length),
            )


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
    `lanes` lanes,
    with the design flags (module docstring). The softmax flags drop
    softmax's commands here: commands.build() has neither the fused
    command nor the engine's max (D-054, deferred by D-057). Waits and
    stops are the full list's: the commands next to the dropped ones stop
    either way."""
    layout = commands.Layout.of(config, cap=position + 1)
    cmds = commands.build(layout)
    dropped = {commands.Op.MAX} if max_in_drain else set()
    dropped |= {commands.Op.EXP, commands.Op.SUM} if softmax_fused else set()
    return [
        dataclasses.replace(step(cmd, position + 1, lanes, kv_tiles_shared), waits=w, stops=s)
        for cmd, w, s in zip(cmds, commands.engine_waits(cmds), commands.stops(cmds), strict=True)
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
    vector_steps: tuple[Step, ...] = ()  # the vector unit's, in order (vector_cycles)

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
            vector_steps=tuple(o for o in op_list if o.walk is not None),
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
) -> float:
    """Cycles of the engine for these (rows, cols), of which `waits` commands
    wait for an earlier result: rows in passes of `lanes` (a partial pass
    leaves lanes idle); a row takes cols / `macs_per_lane` cycles, but at
    least the `accumulators` - 1 adds of its final sum, which run during the
    next row; each waiting command adds `engine_startup_cycles` (D-042)."""
    lanes, per_lane = int(design["lanes"]), int(design["macs_per_lane"])
    tree = int(design["accumulators"]) - 1
    return waits * design["engine_startup_cycles"] + sum(
        math.ceil(rows / lanes) * max(math.ceil(cols / per_lane), tree) for rows, cols in shapes
    )


def vector_issue(s: Step, lanes: float) -> tuple[int, int]:
    """A vector-unit step's groups in a row (`lanes` elements each), and the
    cycles it issues for, latencies apart: its groups, a LOAD its beats."""
    rows, length = s.walk
    row_groups = math.ceil(length / lanes)
    if s.kind == "load":
        return row_groups, math.ceil(s.weight_bytes / commands.BEAT_BYTES)
    return row_groups, rows * row_groups


def vector_cycles(steps: tuple[Step, ...], design: dict[str, float]) -> list[float]:
    """Each vector-unit step's cycles from the done before it to its own
    (D-053): rtl/vec_unit.sv in order, as measured (reports/data/M6.json).
    A step starts at the done before it if the controller stops at it, else
    once the unit has issued the previous step's groups, so their latencies
    overlap. The unit issues a group (`vector_elems_per_cycle` elements of a
    row) a cycle; a LOAD a beat a cycle, `load_latency_cycles` after it
    starts. A step is done `vector_startup_cycles` after its last group, a
    KV_STORE `kv_store_latency_cycles` later (its writes answered), a
    reduction `reduction_tail_cycles` later (its last row's tree), whose rows
    end at least `reduction_row_cycles` apart; the next step starts once a
    reduction is done. Dones come in order, one a cycle at most."""
    lanes = design["vector_elems_per_cycle"]
    issued = done = 0.0  # the cycle the unit has issued the last group by; the last done
    out = []
    for s in steps:
        row_groups, issue = vector_issue(s, lanes)
        start = done if s.stops else issued
        latency = design["vector_startup_cycles"]
        if s.kind == "load":
            issued = start + design["load_latency_cycles"] + issue
        elif s.kind in REDUCTION_KINDS:
            row = max(row_groups, design["reduction_row_cycles"])
            issued = start + row_groups + (s.walk[0] - 1) * row
            latency += design["reduction_tail_cycles"]
        else:
            issued = start + issue
            if s.kind == "kv_store":
                latency += design["kv_store_latency_cycles"]
        finish = max(done + 1, issued + latency)
        out.append(finish - done)
        done = finish
        if s.kind in REDUCTION_KINDS:
            issued = done
    return out


def engine_use(shapes: tuple[tuple[int, int], ...], design: dict[str, float]) -> float:
    """Share of the engine's multiply-add slots that do work."""
    slots = design["lanes"] * design["macs_per_lane"] * engine_cycles(shapes, design)
    return sum(rows * cols for rows, cols in shapes) / slots


def seconds(work: Work, platform: dict[str, float], design: dict[str, float]) -> dict[str, float]:
    """Time of each term for one token, in seconds."""
    clock = design["clock_hz"]
    return {
        "memory": work.memory_bytes / hbm_bytes_per_s(platform, design),
        "engine": engine_cycles(work.engine_shapes, design, work.engine_waits) / clock,
        "vector": sum(vector_cycles(work.vector_steps, design)) / clock,
        "commands": work.commands * design["issue_cycles"] / clock,
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
