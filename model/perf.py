# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""First-order performance model (M2): how long one decode token takes on
the accelerator, and which part limits it.

Generic over three inputs:
- the model's shape (a `transformers` config): what a token computes;
- the platform (model/platforms/*.json): what the hardware gives, each value
  with a status (quoted, measured, guess, ...) and a source; our own
  measurements (platforms/measured/*.json) replace the documented values;
- the design (model/designs/*.json): what we choose (clock, multiply-adds
  per cycle, ...), with named scenarios that override it.

ops() lists the work of one decode step in the order of golden/decoder.py's
step(); a test checks that against the golden model. predict() turns the work
into four times: memory traffic, the matrix-vector engine, the vector unit,
and the fixed cost per command. The token takes at least the largest of them
(perfect overlap) and at most their sum (no overlap).

Assumptions: weights and the KV cache live in HBM and are read once per token
(K and V once per KV head, shared by its query heads); activations stay on
chip; one command per operation; the HBM read rate scales with the memory
clock from the measured rate at the port's maximum clock.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import paths
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from transformers import LlamaConfig

BF16_BYTES = 2
# Passes over the data per element in the vector unit (docs/numerics.md,
# section 4): RMSNorm sums squares then scales; softmax finds the max, sums
# the exponentials, then normalizes.
RMSNORM_PASSES = 2
SOFTMAX_PASSES = 3
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


@dataclass(frozen=True)
class Op:
    """One operation of a decode step. kind is the golden-model function it
    stands for, and shape what that function sees: (rows, cols) of a weight
    matrix, (heads, positions, head_dim) for attention, or (n,) for a vector
    operation (elements per token)."""

    name: str
    kind: str
    shape: tuple[int, ...]
    macs: int = 0
    weight_bytes: int = 0
    kv_read_bytes: int = 0
    kv_write_bytes: int = 0
    vector_elems: int = 0  # elements times passes


class Param(BaseModel):
    """One input value, with how we know it and where from."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    value: float
    unit: str = Field(min_length=1)
    status: Literal[tuple(STATUSES)]
    source: str = Field(min_length=1)

    def __str__(self) -> str:
        return show(self.value, self.unit)

    def change_from(self, doc: Param) -> str:
        """The documented value and how far this one is from it, e.g.
        "426.28 GB/s (-5.9%)"; empty when they are the same entry."""
        if self == doc:
            return ""
        if doc.value == 0:
            raise ValueError(f"documented as 0 ({doc.source}), so no relative difference")
        return f"{doc} ({self.value / doc.value - 1:+.1%})"


def show(value: float, unit: str) -> str:
    """A value in its unit, scaled for reading: 426280000000.0 B/s -> "426.28 GB/s"."""
    scale, name = UNITS.get(unit, (1, unit))
    return f"{value / scale:g} {name}".strip()


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
    macs_per_cycle: Param
    vector_elems_per_cycle: Param
    command_cycles: Param
    scenarios: dict[str, dict[str, float]] = {}

    @model_validator(mode="after")
    def _scenarios_override_parameters(self) -> Design:
        for name, overrides in self.scenarios.items():
            unknown = set(overrides) - set(self.params)
            assert not unknown, f"scenario {name!r} overrides unknown {sorted(unknown)}"
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


def matvec(name: str, rows: int, cols: int, kv_write_bytes: int = 0) -> Op:
    n = rows * cols
    return Op(name, "matvec", (rows, cols), n, n * BF16_BYTES, kv_write_bytes=kv_write_bytes)


def vector(name: str, kind: str, n: int, passes: int = 1, **kw) -> Op:
    return Op(name, kind, (n,), vector_elems=n * passes, **kw)


def rmsnorm(name: str, n: int) -> Op:
    """Reads its gains (n BF16 weights) from memory."""
    return vector(name, "rmsnorm", n, RMSNORM_PASSES, weight_bytes=n * BF16_BYTES)


def attention(kind: str, heads: int, positions: int, d: int, kv_heads: int) -> Op:
    """q.K or p.V over `positions` cache rows: reads K (or V) once per KV
    head, shared by its query heads."""
    kv_bytes = kv_heads * positions * d * BF16_BYTES
    return Op(kind, kind, (heads, positions, d), heads * positions * d, kv_read_bytes=kv_bytes)


def ops(config: LlamaConfig, position: int) -> list[Op]:
    """The work of one decode step at `position` (0-based: it attends to
    position + 1 cache rows), in the order of decoder.step()."""
    c = config
    d, hidden, inter = c.head_dim, c.hidden_size, c.intermediate_size
    n_heads, n_kv = c.num_attention_heads, c.num_key_value_heads
    q_dim, kv_dim, t = n_heads * d, n_kv * d, position + 1
    row = kv_dim * BF16_BYTES  # one position's K (or V) in one layer
    out = [Op("embed", "embed", (hidden,), weight_bytes=hidden * BF16_BYTES)]
    for _ in range(c.num_hidden_layers):
        out += [
            rmsnorm("attn_norm", hidden),
            matvec("q", q_dim, hidden),
            matvec("k", kv_dim, hidden),
            matvec("v", kv_dim, hidden, kv_write_bytes=row),
            vector("rope_k", "rope", kv_dim, kv_write_bytes=row),
            vector("rope_q", "rope", q_dim),
            attention("attention_scores", n_heads, t, d, n_kv),
            vector("softmax", "softmax", n_heads * t, SOFTMAX_PASSES),
            attention("attention_values", n_heads, t, d, n_kv),
            matvec("o", hidden, q_dim),
            vector("residual", "add", hidden),
            rmsnorm("mlp_norm", hidden),
            matvec("gate", inter, hidden),
            matvec("up", inter, hidden),
            vector("swiglu", "swiglu", inter),
            matvec("down", hidden, inter),
            vector("residual", "add", hidden),
        ]
    return out + [rmsnorm("final_norm", hidden), matvec("lm_head", c.vocab_size, hidden)]


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

    @property
    def memory_bytes(self) -> int:
        return self.weight_bytes + self.kv_read_bytes + self.kv_write_bytes

    @property
    def macs(self) -> int:
        return self.matvec_macs + self.attention_macs

    @staticmethod
    def of(op_list: list[Op]) -> Work:
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
        )


def hbm_bytes_per_s(platform: dict[str, float], design: dict[str, float]) -> float:
    """The measured HBM read rate, scaled to the memory clock and capped by
    the ports' raw width."""
    clock = min(design["memory_clock_hz"], platform["hbm_port_max_clock_hz"])
    measured = platform["hbm_read_bytes_per_s"] * clock / platform["hbm_port_max_clock_hz"]
    return min(measured, platform["hbm_ports"] * platform["hbm_port_bytes"] * clock)


def seconds(work: Work, platform: dict[str, float], design: dict[str, float]) -> dict[str, float]:
    """Time of each term for one token, in seconds."""
    clock = design["clock_hz"]
    return {
        "memory": work.memory_bytes / hbm_bytes_per_s(platform, design),
        "engine": work.macs / (design["macs_per_cycle"] * clock),
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
