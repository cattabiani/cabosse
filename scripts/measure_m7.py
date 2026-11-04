# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Measurements of the whole core in simulation (PLAN.md, M7) into
reports/data/M7.json, each part replacing its own key:

    python scripts/measure_m7.py smollm2   # SmolLM2 on rtl/core.sv, 32 ports
    python scripts/measure_m7.py show      # print the tables from the saved data

`smollm2` builds the 32-port core (about 6.5 minutes and 10 GB) and decodes
a short prompt and 8 tokens greedily through the runtime
(generation.smollm2_on_sim), checking every token, logit and KV cache row
bit for bit against the golden model; each token's cycles go into the data
by command and unit against the perf model (verif/breakdown.py). About
100 s a token: run it under the memory cap (MemoryMax=12G). It needs the
checkpoint in weights/. Commit first: the data records the commit.
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
from reporting import blocks, measure  # noqa: E402

DATA = REPO / "reports" / "data" / "M7.json"


def smollm2() -> dict:
    """Each token's breakdown (breakdown.token; the commands' rows for the
    last token only) and the run's time, build included."""
    import breakdown  # the verification stack, only here
    import commands
    from generation import SMOLLM2_CAP, SMOLLM2_N_NEW, SMOLLM2_PORTS, SMOLLM2_PROMPT, smollm2_on_sim
    from golden import decoder

    start = time.monotonic()
    rt, runs = smollm2_on_sim(decoder.load(paths.SMOLLM2))
    seconds = time.monotonic() - start
    cmds = commands.build(rt.layout)
    tokens = [breakdown.token(cmds, tr, cyc, k, SMOLLM2_PORTS) for k, tr, cyc in runs]
    for t in tokens[:-1]:
        del t["commands"]
    return {
        "ports": SMOLLM2_PORTS,
        "cap": SMOLLM2_CAP,
        "prompt": SMOLLM2_PROMPT,
        "n_new": SMOLLM2_N_NEW,
        "bit_exact": True,  # smollm2_on_sim raises otherwise
        "seconds": round(seconds, 1),
        "engine_startup_cycles": breakdown.DESIGN["engine_startup_cycles"],
        "tokens": tokens,
    }


def summary(run: dict) -> str:
    """Per token: cycles, the units' split and the model's in-order cycles;
    for the last token, cycles by unit and kind against the model, and the
    waiting engine commands' start-up."""
    lines = [
        blocks.provenance_text(run["provenance"]),
        f"SmolLM2 at {run['ports']} ports, {len(run['tokens'])} runs, bit-exact:"
        f" {run['bit_exact']}; {run['seconds']} s, build included",
        "position  cycles  model in order  engine only  vector only  both  neither",
    ]
    for t in run["tokens"]:
        u = t["units"]
        lines.append(
            f"{t['position']:8}  {t['cycles']:6}  {t['model']['in order']:14.0f}"
            f"  {u['engine only']:11}  {u['vector only']:11}  {u['both']:4}  {u['neither']:7}"
        )
    last = run["tokens"][-1]
    cycles, model, n = collections.Counter(), collections.Counter(), collections.Counter()
    for r in last["commands"]:
        key = (r["unit"], r["kind"])
        cycles[key] += r["cycles"]
        model[key] += r["model_cycles"]
        n[key] += 1
    lines += ["", f"position {last['position']}: unit  kind  commands  cycles  model"]
    for unit, kind in sorted(n):
        key = (unit, kind)
        lines.append(f"{unit:6}  {kind:12}  {n[key]:4}  {cycles[key]:8}  {model[key]:10.0f}")
    startup = run["engine_startup_cycles"]
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
