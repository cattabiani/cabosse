# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Each token's cycles on the simulated core (rtl/core.sv) by command and
unit, against the perf model's (PLAN.md, M7 step 5).

The driver traces the controller's handshakes with the units
(SimPlatform.trace). A unit runs its commands in order, so a command's
cycles run from its start, the latest of its handover, the previous one's
done on its unit and (an engine command that waits) its go, to its done.
The perf model gives the same per command: perf.engine_cycles for an engine
command (with its start-up when it waits), perf.vector_cycles (done to
done, D-053) for the vector unit's. The simulation's memory answers a beat
a cycle on every port after a cycle (axi_mem.h's defaults): SIM_PLATFORM
is the perf model's platform for that memory."""

from dataclasses import dataclass, field, replace

import commands
import numpy as np
import perf
from commands import ENGINE_OPS, Command, Op
from sim_platform import SimPlatform

DESIGN = perf.load_design("v0").values()
F2 = perf.load_platform("f2").values()
# The simulation's memory as a perf model platform: every port a beat a
# core cycle.
SIM_PLATFORM = F2 | {
    "hbm_port_max_clock_hz": DESIGN["clock_hz"],
    "hbm_read_bytes_per_s": F2["hbm_ports"] * F2["hbm_port_bytes"] * DESIGN["clock_hz"],
}
CONTROLLER_OPS = frozenset({Op.OUTPUT, Op.END})  # done by the controller itself


@dataclass
class TracedPlatform:
    """A SimPlatform that keeps each run's trace and CYCLES."""

    sim: SimPlatform
    runs: list[tuple[dict[str, np.ndarray], int]] = field(default_factory=list)

    def __getattr__(self, name: str):
        return getattr(self.sim, name)

    def run_until_done(self) -> None:
        self.sim.run_until_done()
        lo, hi = (self.sim.reg_read(commands.REG[r]) for r in ("CYCLES_LO", "CYCLES_HI"))
        self.runs.append((self.sim.trace(), lo | hi << 32))


def unit_spans(
    take: np.ndarray, done: np.ndarray, go: dict[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Each command's (start, done) on a unit that runs them in order: start
    is the latest of its handover, the previous done and its go (go: the
    command's index on the unit -> the go's cycle)."""
    assert len(take) == len(done), (len(take), len(done))
    start = np.empty(len(take), dtype=np.int64)
    before = 0
    for k, (t, d) in enumerate(zip(take.tolist(), done.tolist(), strict=True)):
        start[k] = max(t, before, (go or {}).get(k, 0))
        before = d
    return start, done.astype(np.int64)


def busy(spans: tuple[np.ndarray, np.ndarray], first: int, n: int) -> np.ndarray:
    """A bool per cycle of [first, first + n): the unit runs a command."""
    on = np.zeros(n + 1, dtype=np.int64)
    for s, d in zip(*spans, strict=True):
        on[s - first] += 1
        on[d - first] -= 1
    return np.cumsum(on[:n]) > 0


def model_steps(cmds: list[Command], t: int) -> list[perf.Step]:
    """perf.Step of each command when attention reads t positions, with the
    list's waits and stops."""
    lanes = int(DESIGN["lanes"])
    waits, stops = commands.engine_waits(cmds), commands.stops(cmds)
    return [
        replace(perf.step(c, t, lanes, False), waits=w, stops=s)
        for c, w, s in zip(cmds, waits, stops, strict=True)
    ]


def token(cmds: list[Command], trace: dict[str, np.ndarray], cycles: int, position: int) -> dict:
    """One token's breakdown: each unit command's cycles and the model's,
    the units' busy cycles, and the perf model's terms (in cycles)."""
    eng = [i for i, c in enumerate(cmds) if c.op in ENGINE_OPS]
    vec = [i for i, c in enumerate(cmds) if c.op not in ENGINE_OPS | CONTROLLER_OPS]
    waits = commands.engine_waits(cmds)
    waiting = [k for k, i in enumerate(eng) if waits[i]]
    assert len(trace["eng_go"]) == len(waiting), (len(trace["eng_go"]), len(waiting))
    go = dict(zip(waiting, trace["eng_go"].tolist(), strict=True))
    eng_spans = unit_spans(trace["eng_take"], trace["eng_done"], go)
    vec_spans = unit_spans(trace["vec_take"], trace["vec_done"])
    assert len(eng_spans[0]) == len(eng) and len(vec_spans[0]) == len(vec)

    steps = model_steps(cmds, position + 1)
    eng_model = [perf.engine_cycles(steps[i].engine_shapes, DESIGN, int(waits[i])) for i in eng]
    vec_model = perf.vector_cycles(tuple(steps[i] for i in vec), DESIGN)
    rows = []
    for unit, idx, spans, model in (
        ("engine", eng, eng_spans, eng_model),
        ("vector", vec, vec_spans, vec_model),
    ):
        for i, s, d, m in zip(idx, *spans, model, strict=True):
            rows.append(
                {
                    "index": i,
                    "unit": unit,
                    "kind": steps[i].kind,
                    "waits": bool(waits[i]),
                    "stops": steps[i].stops,
                    "cycles": int(d - s),
                    "model_cycles": float(m),
                }
            )
    rows.sort(key=lambda r: r["index"])

    first = int(min(eng_spans[0].min(), vec_spans[0].min()))
    last = int(max(eng_spans[1].max(), vec_spans[1].max()))
    e, v = busy(eng_spans, first, last - first), busy(vec_spans, first, last - first)
    terms = perf.seconds(perf.Work.of(steps), SIM_PLATFORM, DESIGN)
    in_order = 1 / perf.Prediction(terms).tokens_per_s_in_order
    clock = DESIGN["clock_hz"]
    return {
        "position": position,
        "cycles": cycles,
        "units": {
            "engine only": int((e & ~v).sum()),
            "vector only": int((v & ~e).sum()),
            "both": int((e & v).sum()),
            "neither": cycles - int((e | v).sum()),
        },
        "model": {k: s * clock for k, s in terms.items()} | {"in order": in_order * clock},
        "commands": rows,
    }
