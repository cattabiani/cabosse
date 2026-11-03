# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Measurements of the whole core in simulation (PLAN.md, M7) into
reports/data/M7.json, each part replacing its own key:

    python scripts/measure_m7.py smollm2   # SmolLM2 on rtl/core.sv, 32 ports
    python scripts/measure_m7.py show      # print the tables from the saved data

`smollm2` builds the 32-port core (about 6.5 minutes and 10 GB) and decodes
a short prompt and N_NEW tokens greedily through the runtime, checking every
token, logit and KV cache row bit for bit against the golden model; each
token's cycles go into the data by command and unit against the perf model
(verif/breakdown.py). About 2 minutes a token (estimate): run it under the
memory cap (MemoryMax=12G). It needs the checkpoint in weights/. Commit
first: the data records the commit.
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for sub in ("model", "model/tests", "verif", "sw"):
    sys.path.insert(0, str(REPO / sub))

import paths  # noqa: E402
from reporting import measure  # noqa: E402

DATA = REPO / "reports" / "data" / "M7.json"
N_NEW = 8


def smollm2() -> dict:
    """Each token's breakdown (breakdown.token; the commands' rows for the
    last token only) and the run's totals."""
    import breakdown  # the verification stack, only here
    import commands
    from generation import (
        SMOLLM2_CAP,
        SMOLLM2_CHAN_W,
        SMOLLM2_LENGTH,
        SMOLLM2_PORTS,
        SMOLLM2_PROMPT,
        check_generation,
    )
    from golden import decoder
    from runtime import Runtime
    from sim_platform import SimPlatform
    from transformers import AutoTokenizer

    model = decoder.load(paths.SMOLLM2)
    prompt = AutoTokenizer.from_pretrained(paths.SMOLLM2)(SMOLLM2_PROMPT)["input_ids"]
    start = time.monotonic()
    with SimPlatform(SMOLLM2_PORTS, SMOLLM2_CHAN_W, SMOLLM2_LENGTH) as sim:
        built = time.monotonic()
        p = breakdown.TracedPlatform(sim)
        rt = Runtime.open(p, model, SMOLLM2_CAP)
        check_generation(rt, model, prompt, N_NEW, f"SmolLM2, {SMOLLM2_PORTS} ports")
        cmds = commands.build(rt.layout)
        tokens = [breakdown.token(cmds, tr, cyc, k) for k, (tr, cyc) in enumerate(p.runs)]
    for t in tokens[:-1]:
        del t["commands"]
    return {
        "ports": SMOLLM2_PORTS,
        "cap": SMOLLM2_CAP,
        "prompt": prompt,
        "n_new": N_NEW,
        "bit_exact": True,  # check_generation raises otherwise
        "build_seconds": round(built - start, 1),
        "run_seconds": round(time.monotonic() - built, 1),
        "tokens": tokens,
    }


def summary(run: dict) -> str:
    """Per token: cycles, the units' split and the model's in-order cycles;
    for the last token, cycles by unit and kind against the model, and the
    waiting engine commands' start-up."""
    lines = [
        f"SmolLM2 at {run['ports']} ports, {len(run['tokens'])} tokens, bit-exact:"
        f" {run['bit_exact']}; build {run['build_seconds']} s, run {run['run_seconds']} s",
        "position  cycles  model in order  engine only  vector only  both  neither",
    ]
    for t in run["tokens"]:
        u = t["units"]
        lines.append(
            f"{t['position']:8}  {t['cycles']:6}  {t['model']['in order']:14.0f}"
            f"  {u['engine only']:11}  {u['vector only']:11}  {u['both']:4}  {u['neither']:7}"
        )
    last = run["tokens"][-1]
    by = collections.defaultdict(lambda: [0, 0.0, 0])
    for r in last["commands"]:
        b = by[(r["unit"], r["kind"])]
        b[0] += r["cycles"]
        b[1] += r["model_cycles"]
        b[2] += 1
    lines += ["", f"position {last['position']}: unit  kind  commands  cycles  model"]
    for (unit, kind), (c, m, n) in sorted(by.items()):
        lines.append(f"{unit:6}  {kind:12}  {n:4}  {c:8}  {m:10.0f}")
    from breakdown import DESIGN

    startup = DESIGN["engine_startup_cycles"]
    waits = [r for r in last["commands"] if r["waits"]]
    if waits:
        found = collections.Counter(int(r["cycles"] - r["model_cycles"] + startup) for r in waits)
        lines.append(f"\nwaiting engine commands' start-up (model {startup:.0f}): {dict(found)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=("smollm2", "show"))
    args = parser.parse_args()
    data = json.loads(DATA.read_text()) if DATA.exists() else {}
    if args.part != "show":
        data[args.part] = {"provenance": measure.provenance()} | globals()[args.part]()
        DATA.write_text(json.dumps(data, indent=1) + "\n")
    if "smollm2" in data:
        print(summary(data["smollm2"]))
    if not data:
        sys.exit(f"no data in {DATA}: run the `smollm2` part first")


if __name__ == "__main__":
    main()
