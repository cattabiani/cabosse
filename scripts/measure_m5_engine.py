# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Measurements of the 128-lane engine (PLAN.md, M5) into
reports/data/M5-engine.json, each part replacing its own key:

    python scripts/measure_m5_engine.py depth   # the ports' FIFO depth against pauses
    python scripts/measure_m5_engine.py real    # SmolLM2's engine commands against the perf model
    python scripts/measure_m5_engine.py bandwidth  # bytes per cycle on large matrices
    python scripts/measure_m5_engine.py speed   # Verilator: build time and cycles per second
    python scripts/measure_m5_engine.py show    # print the tables from the saved data

The memory has F2's read latency (verif/engine.py, F2_LATENCY_CYCLES). `real`
and `speed` need the SmolLM2 checkpoint, `real` the long cache too
(scripts/fill_long_cache.py). `speed` rebuilds the model three times (about
2.5 minutes each on 16 cores); run it on a quiet machine. Run everything
under the memory cap; commit first, the data records the commit.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for sub in ("model", "verif"):
    sys.path.insert(0, str(REPO / sub))

import bulk  # noqa: E402
import engine as eng  # noqa: E402
import numpy as np  # noqa: E402
import paths  # noqa: E402
import perf  # noqa: E402
import torch  # noqa: E402
from engine import LANES_PER_PORT, Matvec  # noqa: E402
from golden import decoder  # noqa: E402
from reporting import blocks, measure  # noqa: E402

DATA = REPO / "reports" / "data" / "M5-engine.json"
PORTS = 32
MEMORY = eng.F2_MEMORY
DEPTHS = (64, 96, 128, 192, 256, 1024)  # 1024: deep enough that only the pauses limit
REFERENCE_DEPTH = DEPTHS[-1]
PAUSES = {  # per port, independently
    "none": {},
    "5 per mille, up to 20 cycles": dict(pause=5, pause_max=20),
    "20 per mille, up to 40 cycles": dict(pause=20, pause_max=40),
}
SEEDS = range(1, 6)
DEPTH_SHAPE = (8 * PORTS * LANES_PER_PORT, 512)  # 8 passes of 128 beats
DEPTH_MATRICES = 4
SPEED_RUNS = 3
POSITIONS = (63, 8191)  # a short and the longest position, for the per-token estimate


def depth() -> dict:
    """Rate (beats per cycle per port) by depth, pauses and memory seed."""
    gen = torch.Generator().manual_seed(1)
    mats = [Matvec(torch.randn(*DEPTH_SHAPE, generator=gen).to(torch.bfloat16))]
    mats *= DEPTH_MATRICES
    xs = {0: torch.randn(DEPTH_SHAPE[1], generator=gen).to(torch.bfloat16)}
    beats = sum(eng.n_beats(m, PORTS) for m in mats)
    for d in DEPTHS:  # one build at a time; the runs below go in parallel
        bulk.build("engine", (("Ports", PORTS), ("Depth", d)))
    rate: dict[tuple[str, int, int], float] = {}

    def one(job: tuple[str, int, int]) -> None:
        pauses, d, seed = job
        params = (("Depth", d),)
        r = eng.fetch(mats, xs, PORTS, params=params, seed=seed, **MEMORY, **PAUSES[pauses])
        for m, got in zip(mats, r.out, strict=True):
            eng.assert_bits(got, eng.expected(m, xs[0]), str(job))
        rate[job] = beats / int(r.done[-1] - r.taken[0])

    bulk.in_parallel(one, [(p, d, s) for p in PAUSES for d in DEPTHS for s in SEEDS])
    return {
        "ports": PORTS,
        "memory": MEMORY,
        "shape": DEPTH_SHAPE,
        "matrices": DEPTH_MATRICES,
        "seeds": list(SEEDS),
        "pauses": PAUSES,
        "rates": {p: {str(d): [rate[p, d, s] for s in SEEDS] for d in DEPTHS} for p in PAUSES},
    }


def real() -> dict:
    """SmolLM2's engine commands (the first and last layers' and the
    classifier), bit-exact, with the controller's waits, at each of
    eng.SMOLLM2_POSITIONS: each command's cycles next to the perf model's,
    with and without the start-up of a waiting command (design v0's
    engine_startup_cycles: the most a waiting command after the first takes
    over its engine cycles, from this measurement)."""
    model = decoder.load(paths.SMOLLM2)
    positions = {}
    for position in eng.SMOLLM2_POSITIONS:
        found, result = eng.run_smollm2(model, PORTS, position, **MEMORY)
        positions[str(position)] = [
            {
                "name": f.name,
                "op": op(f.command),
                "shape": shape(f.command),
                "wait": f.command.wait,
                "cycles": c,
                "model": eng.model_cycles(f.command, PORTS),
                "engine": eng.model_cycles(f.command, PORTS, startup=False),
            }
            for f, c in zip(found, eng.command_cycles(result), strict=True)
        ]
    return {"ports": PORTS, "memory": MEMORY, "positions": positions}


def op(m: eng.Command) -> str:
    return "matvec" if isinstance(m, Matvec) else eng.attention_command(m).op.name.lower()


def shape(m: eng.Command) -> list[int]:
    """rows × cols for a MATVEC; heads, positions and d for attention."""
    if isinstance(m, Matvec):
        return list(m.w.shape)
    return [m.heads, m.t, m.cache.shape[2]]


def startup(commands: list[dict]) -> int:
    """The most a waiting command after the first took over its engine
    cycles: the first has nothing before it to read its weights behind."""
    return max(c["cycles"] - c["engine"] for c in commands[1:] if c["wait"])


def bandwidth() -> dict:
    """Each large matrix alone, from its start to its done: bytes per cycle,
    as a share of min(the memory's bytes per cycle, 2·E·L), and the perf
    model's engine use (the most a partial last pass allows)."""
    jobs = [(rate, shape) for rate in eng.RATES for shape in eng.LARGE]
    found = {}

    def one(job: tuple[int, tuple[int, int]]) -> None:
        rate, shape = job
        found[job] = {"shape": list(shape), "rate": rate} | eng.bandwidth(
            *shape, PORTS, rate, seed=1
        )

    bulk.in_parallel(one, jobs)
    matrices = [found[job] for job in jobs]
    return {"ports": PORTS, "memory": MEMORY, "matrices": matrices}


def speed() -> dict:
    """Build times from scratch, and simulated cycles per second of the
    driver's loop on SmolLM2's MATVECs; with the perf model's cycles per
    token, an estimate of M7's wall time per simulated token."""
    params = (("Ports", PORTS),)
    build_s = []
    for _ in range(SPEED_RUNS):
        bulk.forget_build("engine", params)
        start = time.monotonic()
        bulk.build("engine", params)
        build_s.append(time.monotonic() - start)
    found, xs = eng.smollm2_commands(decoder.load(paths.SMOLLM2), PORTS)
    runs = [eng.fetch([f.command for f in found], xs, PORTS, **MEMORY) for _ in range(SPEED_RUNS)]
    config, design = perf.load_config("SmolLM2-135M-Instruct"), eng.DESIGN
    platform = perf.load_platform("f2").values()
    token_cycles = {
        str(pos): {
            term: s * design["clock_hz"]
            for term, s in perf.predict(config, platform, design, pos).terms.items()
        }
        for pos in POSITIONS
    }
    return {
        "ports": PORTS,
        "build_s": build_s,
        "run_cycles": [r.cycles for r in runs],
        "run_s": [r.wall_us / 1e6 for r in runs],
        "token_cycles": token_cycles,
    }


def show(data: dict) -> None:
    def span(values: list[float]) -> str:
        return f"{min(values):.3f}-{max(values):.3f}"

    for name, part in data.items():
        print(f"{name}: {blocks.provenance_text(part['provenance'])}")
    if "depth" in data:
        d = data["depth"]
        rates = d["rates"]
        ref = {p: rates[p][str(REFERENCE_DEPTH)] for p in d["pauses"]}
        print(f"\nRate as a share of a {REFERENCE_DEPTH}-beat FIFO's, {d['ports']} ports, ", end="")
        print(d["memory"])
        rows = [
            [depth_] + [span(list(np.array(rates[p][str(depth_)]) / ref[p])) for p in d["pauses"]]
            for depth_ in DEPTHS[:-1]
        ]
        print(blocks.table(["depth", *d["pauses"]], rows))
        for p in d["pauses"]:
            print(f"{p}: {REFERENCE_DEPTH}-beat FIFO, {span(ref[p])} beats/cycle")
    if "real" in data:
        r = data["real"]
        for position, cmds in r["positions"].items():
            print(f"\nSmolLM2, position {position}, {r['memory']}")
            rows = [
                [c["name"], c["op"], " × ".join(map(str, c["shape"])), "yes" if c["wait"] else ""]
                + [c["cycles"], c["model"], f"{(c['cycles'] - c['model']) / c['model']:+.1%}"]
                for c in cmds
            ]
            header = ["command", "op", "shape", "waits", "cycles", "perf model", "difference"]
            print(blocks.table(header, rows))
            print(f"start-up (the most over the engine cycles, after the first): {startup(cmds)}")
        print("The first command has nothing before it to hide its weights' reads behind.")
    if "bandwidth" in data:
        rows = [
            [f"{m['shape'][0]} × {m['shape'][1]}", f"{m['rate'] / 1000:.0%}"]
            + [f"{m['share']:.3f}", f"{m['engine_use']:.3f}"]
            for m in data["bandwidth"]["matrices"]
        ]
        header = ["rows × cols", "memory rate", "share of min(memory, 2·E·L)", "engine use"]
        print("\n" + blocks.table(header, rows))
    if "speed" in data:
        s = data["speed"]
        rates = [c / t for c, t in zip(s["run_cycles"], s["run_s"], strict=True)]
        print(f"\nBuild {min(s['build_s']):.0f}-{max(s['build_s']):.0f} s; ", end="")
        print(f"{min(rates):,.0f}-{max(rates):,.0f} simulated cycles/s")
        for pos, terms in s["token_cycles"].items():
            serial = sum(terms.values())
            print(
                f"position {pos}: {serial:,.0f} cycles per token (perf model, serial), "
                f"{serial / max(rates):,.0f}-{serial / min(rates):,.0f} s at the engine's rate"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=("depth", "real", "bandwidth", "speed", "show"))
    part = parser.parse_args().part
    data = json.loads(DATA.read_text()) if DATA.exists() else {}
    if part != "show":
        data[part] = {"provenance": measure.provenance()} | globals()[part]()
        DATA.write_text(json.dumps(data, indent=1) + "\n")
    show(data)


if __name__ == "__main__":
    main()
