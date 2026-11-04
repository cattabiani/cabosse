# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Measurements of the whole core in simulation (PLAN.md, M7) into
reports/data/M7.json, each part replacing its own key:

    python scripts/measure_m7.py smollm2   # SmolLM2 on rtl/core.sv, 32 ports
    python scripts/measure_m7.py cpu       # D-067: transformers on this PC's CPU
    python scripts/measure_m7.py cuda      # D-067: on its GPU (D-070)
    python scripts/measure_m7.py show      # print the tables from the saved data

`smollm2` builds the 32-port core (about 6.5 minutes and 10 GB) and decodes
a short prompt and 8 tokens greedily through the runtime
(generation.smollm2_on_sim), checking every token, logit and KV cache row
bit for bit against the golden model; each token's cycles go into the data
by command and unit against the perf model (verif/breakdown.py). About
100 s a token: run it under the memory cap (MemoryMax=12G). It needs the
checkpoint in weights/. Commit first: the data records the commit.

`cpu` and `cuda` time D-067's baseline: SmolLM2's published BF16 weights
in `transformers` (no compile, its default attention, recorded), batch 1,
the same prompt, then N_DECODE greedy tokens through the KV cache, prefill excluded; the
median of REPEATS runs after WARMUP. `cuda` needs torch's CUDA build in
.venv (requirements-cuda.txt, D-070).
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for sub in ("model", "model/tests", "verif", "sw"):
    sys.path.insert(0, str(REPO / sub))

import paths  # noqa: E402
from reporting import m7, measure  # noqa: E402

DATA = m7.DATA
N_DECODE = 128
WARMUP, REPEATS = 1, 5


def smollm2() -> dict:
    """Each token's breakdown (breakdown.token; the commands' rows for the
    last token only) and the run's time, build included."""
    import breakdown  # the verification stack, only here
    import commands
    from decode import F2_MEMORY
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
        "memory": F2_MEMORY,  # the simulated memory's settings (axi_bench.h)
        "seconds": round(seconds, 1),
        "engine_startup_cycles": breakdown.DESIGN["engine_startup_cycles"],
        "tokens": tokens,
    }


def decode_tokens_per_s(device: str) -> dict:
    """D-067's decode speed on `device`, and what it ran on."""
    import torch
    from generation import SMOLLM2_PROMPT
    from transformers import AutoModelForCausalLM, AutoTokenizer

    before = measure.host()  # its load average before the timing
    model = AutoModelForCausalLM.from_pretrained(paths.SMOLLM2, dtype=torch.bfloat16)
    model = model.to(device).eval()
    prompt = AutoTokenizer.from_pretrained(paths.SMOLLM2)(SMOLLM2_PROMPT)["input_ids"]
    ids = torch.tensor([prompt], device=device)
    sync = torch.get_device_module(device).synchronize
    rates = []
    with torch.inference_mode():
        for _ in range(WARMUP + REPEATS):
            out = model(ids, use_cache=True)  # prefill, not timed
            sync()
            start = time.perf_counter()
            for _ in range(N_DECODE):
                token = out.logits[:, -1].argmax(-1, keepdim=True)
                out = model(token, past_key_values=out.past_key_values, use_cache=True)
            sync()
            rates.append(N_DECODE / (time.perf_counter() - start))
    gpu = (
        {"gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda}
        if device == "cuda"
        else {}
    )
    return (
        before
        | gpu
        | {
            "attention": model.config._attn_implementation,
            "prompt": SMOLLM2_PROMPT,
            "n_decode": N_DECODE,
            "warmup": WARMUP,
            "tokens_per_s": [round(r, 1) for r in rates[WARMUP:]],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=("smollm2", *m7.DEVICES, "show"))
    args = parser.parse_args()
    data = json.loads(DATA.read_text()) if DATA.exists() else {}
    if args.part != "show":
        provenance = measure.provenance()  # of the code before it runs
        run = decode_tokens_per_s(args.part) if args.part in m7.DEVICES else smollm2()
        data[args.part] = {"provenance": provenance} | run
        DATA.write_text(json.dumps(data, indent=1) + "\n")
    if "smollm2" in data:
        print(m7.smollm2_text(data["smollm2"]))
    if any(d in data for d in m7.DEVICES):
        print(m7.baseline_text(data))
    if not data:
        sys.exit(f"no data in {DATA}: run a part first")


if __name__ == "__main__":
    main()
