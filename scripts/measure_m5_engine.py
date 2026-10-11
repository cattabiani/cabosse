# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Measurements of the 128-lane engine (PLAN.md, M5) into
reports/data/M5-engine.json, each part replacing its own key:

    python scripts/measure_m5_engine.py depth   # the ports' FIFO depth against pauses
    python scripts/measure_m5_engine.py real    # SmolLM2's MATVECs: cycles against the perf model
    python scripts/measure_m5_engine.py bandwidth  # bytes per cycle on large matrices
    python scripts/measure_m5_engine.py speed   # Verilator: build time and cycles per second
    python scripts/measure_m5_engine.py show    # print the tables from the saved data

The memory has F2's read latency (verif/engine.py, F2_LATENCY_CYCLES). `real`
and `speed` need the SmolLM2 checkpoint. `speed` rebuilds the model three
times (about 2.5 minutes each on 16 cores); run it on a quiet machine. Run
everything under the memory cap; commit first, the data records the commit.
"""

import argparse
import json
import shutil
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
from reporting import measure  # noqa: E402

import rtl  # noqa: E402

DATA = REPO / "reports" / "data" / "M5-engine.json"
PORTS = 32
MEMORY = dict(latency=eng.F2_LATENCY_CYCLES, outstanding=8)
DEPTHS = (64, 96, 128, 192, 256, 1024)  # 1024: deep enough that only the pauses limit
REFERENCE_DEPTH = 1024
PAUSES = {  # per port, independently
    "none": {},
    "5 per mille, up to 20 cycles": dict(pause=5, pause_max=20),
    "20 per mille, up to 40 cycles": dict(pause=20, pause_max=40),
}
SEEDS = range(1, 6)
DEPTH_SHAPE = (8 * PORTS * LANES_PER_PORT, 512)  # 8 passes of 128 beats
DEPTH_MATRICES = 4
SPEED_RUNS = 3
BANDWIDTH_SHAPES = ((1536, 576), (8192, 576), (576, 1536))  # as verif's test_bandwidth
RATES = (1000, 500)  # permille of cycles each port may send a beat
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
            assert np.array_equal(got, eng.expected(m, xs[0])), job
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
    """SmolLM2's MATVECs, bit-exact, with the controller's waits: each
    command's cycles next to the perf model's (with the start-up of a
    waiting command, design v0's engine_startup_cycles, which comes from the
    first run of this measurement: the most a waiting command after the
    first took over its engine cycles)."""
    found = eng.smollm2_matvecs(decoder.load(paths.SMOLLM2))
    mats = [f.matvec for f in found]
    result = eng.fetch(mats, {i: f.x for i, f in enumerate(found)}, PORTS, **MEMORY)
    for f, got in zip(found, result.out, strict=True):
        assert np.array_equal(got, f.want), f.name
    design = perf.load_design("v0").values()
    return {
        "ports": PORTS,
        "memory": MEMORY,
        "position": len(eng.SMOLLM2_PROMPT),
        "commands": [
            {
                "name": f.name,
                "shape": list(f.matvec.w.shape),
                "wait": f.matvec.wait,
                "cycles": c,
                "model": perf.engine_cycles((tuple(f.matvec.w.shape),), design)
                + f.matvec.wait * design["engine_startup_cycles"],
            }
            for f, c in zip(found, eng.command_cycles(result), strict=True)
        ],
    }


def bandwidth() -> dict:
    """Each large matrix alone, from its start to its done: bytes per cycle,
    as a share of min(the memory's bytes per cycle, 2·E·L), and the perf
    model's engine use (the most a partial last pass allows)."""
    gen = torch.Generator().manual_seed(1)
    design = perf.load_design("v0").values()
    found = []
    for rate in RATES:
        peak = min(PORTS * eng.BEAT_BYTES * rate / 1000, 2 * eng.E * PORTS * LANES_PER_PORT)
        for rows, cols in BANDWIDTH_SHAPES:
            m = Matvec(torch.randn(rows, cols, generator=gen).to(torch.bfloat16))
            xs = {0: torch.randn(cols, generator=gen).to(torch.bfloat16)}
            r = eng.fetch([m], xs, PORTS, **MEMORY, rate=rate, seed=1)
            assert np.array_equal(r.out[0], eng.expected(m, xs[0])), (rows, cols, rate)
            found.append(
                {
                    "shape": [rows, cols],
                    "rate": rate,
                    "share": 2 * rows * cols / int(r.done[0] - r.taken[0]) / peak,
                    "engine_use": perf.engine_use(((rows, cols),), design),
                }
            )
    return {"ports": PORTS, "memory": MEMORY, "matrices": found}


def speed() -> dict:
    """Build times from scratch, and simulated cycles per second of the
    driver's loop on SmolLM2's MATVECs; with the perf model's cycles per
    token, an estimate of M7's wall time per simulated token."""
    build_dir = rtl.BUILD / f"bulk_engine_Ports{PORTS}"
    build_s = []
    for _ in range(SPEED_RUNS):
        shutil.rmtree(build_dir, ignore_errors=True)
        bulk._build.cache_clear()
        start = time.monotonic()
        bulk.build("engine", (("Ports", PORTS),))
        build_s.append(time.monotonic() - start)
    found = eng.smollm2_matvecs(decoder.load(paths.SMOLLM2))
    xs = {i: f.x for i, f in enumerate(found)}
    runs = [eng.fetch([f.matvec for f in found], xs, PORTS, **MEMORY) for _ in range(SPEED_RUNS)]
    config, design = perf.load_config("SmolLM2-135M-Instruct"), perf.load_design("v0").values()
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
    if "depth" in data:
        d = data["depth"]
        print(
            f"Rate as a share of a {REFERENCE_DEPTH}-beat FIFO's, {d['ports']} ports, {d['memory']}"
        )
        print("| depth | " + " | ".join(d["pauses"]) + " |")
        print("|---|" + "---|" * len(d["pauses"]))
        for depth_ in DEPTHS[:-1]:
            cells = []
            for p in d["pauses"]:
                share = np.array(d["rates"][p][str(depth_)]) / d["rates"][p][str(REFERENCE_DEPTH)]
                cells.append(f"{share.min():.3f}-{share.max():.3f}")
            print(f"| {depth_} | " + " | ".join(cells) + " |")
        for p in d["pauses"]:
            ref = d["rates"][p][str(REFERENCE_DEPTH)]
            print(f"{p}: {REFERENCE_DEPTH}-beat FIFO, {min(ref):.3f}-{max(ref):.3f} beats/cycle")
    if "real" in data:
        r = data["real"]
        print(f"\nSmolLM2 MATVECs, position {r['position']}, {r['memory']}")
        print("| command | rows × cols | waits | cycles | perf model | difference |")
        print("|---|---|---|---|---|---|")
        for c in r["commands"]:
            diff = (c["cycles"] - c["model"]) / c["model"]
            print(
                f"| {c['name']} | {c['shape'][0]} × {c['shape'][1]} | "
                f"{'yes' if c['wait'] else ''} | {c['cycles']} | {c['model']} | {diff:+.1%} |"
            )
        print("The first command has nothing before it to hide its weights' reads behind.")
    if "bandwidth" in data:
        print("\n| rows × cols | memory rate | share of min(memory, 2·E·L) | engine use |")
        print("|---|---|---|---|")
        for m in data["bandwidth"]["matrices"]:
            print(
                f"| {m['shape'][0]} × {m['shape'][1]} | {m['rate'] / 1000:.0%} | "
                f"{m['share']:.3f} | {m['engine_use']:.3f} |"
            )
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
