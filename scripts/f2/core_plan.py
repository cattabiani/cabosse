# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""M9's measurements of the core on F2 (PLAN.md; platforms/f2/README.md),
one JSON line a step (plan_log.py), run by scripts/f2/run_core.sh with the
AFI loaded:

- clock: platforms/f2/host/core_clock sets clk_hbm_axi (D-065) and reports
  the wrapper's ID, BUILD and parameters;
- pcis_write, pcis_read: the runtime's memory calls through the driver
  (sw/f2_platform.py), REPEATS times each, every read checked;
- open: SmolLM2 laid out for CAP positions and written to the card;
- fixture: the three 64-token SmolLM2 fixtures (smollm2_greedy.json),
  tokens and logit bits, each step's wall time and CYCLES;
- long: the long fixture (smollm2_long_greedy.json), 46 + 1000 positions,
  each position's logit bits against its own hash.

Usage: python core_plan.py OUT.jsonl [--quick]   (--quick: no long fixture)
Environment: S3 and PROVENANCE (plan_log.py); CABOSSE_WEIGHTS (model/paths.py).
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
for sub in ("model", "sw"):
    sys.path.insert(0, str(REPO / sub))

import paths  # noqa: E402
from golden import arith, decoder  # noqa: E402
from plan_log import Log  # noqa: E402
from runtime import Runtime  # noqa: E402

from platforms import Platform  # noqa: E402

CLOCK = REPO / "platforms" / "f2" / "host" / "core_clock"
CLOCK_MHZ = 250  # D-065: the build is timed for 250, never above
CAP = 2048  # positions: the long fixture's 1046 and room; a multiple of 128 lanes, Len allows 5376
REPEATS = 5
PCIS_WRITE_BYTES = 64 << 20
PCIS_READ_BYTES = 4 << 20
PCIS_SEED = 0x5EED


def pcis(log: Log, p: Platform, write_bytes: int, read_bytes: int, repeats: int) -> None:
    """Write and read card memory through the platform, `repeats` times each,
    from address 0 (before the model is loaded), every read checked."""
    gen = torch.Generator().manual_seed(PCIS_SEED)
    data = torch.randint(0, 256, (write_bytes,), dtype=torch.uint8, generator=gen)
    for k in range(repeats):
        t0 = time.perf_counter()
        p.mem_write(0, data)
        s = time.perf_counter() - t0
        log.record("pcis_write", {"bytes": write_bytes, "repeat": k}, rate(write_bytes, s))
    for k in range(repeats):
        t0 = time.perf_counter()
        got = p.mem_read(0, read_bytes)
        s = time.perf_counter() - t0
        result = rate(read_bytes, s)
        if not torch.equal(got.view(torch.uint8), data[:read_bytes]):
            result["error"] = "read data differs from what was written"
        log.record("pcis_read", {"bytes": read_bytes, "repeat": k}, result)


def rate(n_bytes: int, s: float) -> dict:
    return {"s": s, "bytes_per_s": n_bytes / s}


def timed_greedy(rt: Runtime, prompt: list[int], n_new: int) -> dict:
    """Runtime.generate's greedy decoding, each step timed: the new tokens,
    the logits' hash over all positions (arith.bits_sha256 of generate's
    logits) and each position's own, and each step's wall time and CYCLES."""
    whole, rows, wall_s, cycles = hashlib.sha256(), [], [], []

    def step(token: int) -> torch.Tensor:
        t0 = time.perf_counter()
        logits = rt.step(token)
        wall_s.append(time.perf_counter() - t0)
        cycles.append(rt.cycles)
        bits = arith.bits_f32(logits).numpy().astype("<u4").tobytes()
        whole.update(bits)
        rows.append(hashlib.sha256(bits).hexdigest())
        return logits

    rt.reset()
    for t in prompt:
        logits = step(t)
    tokens = [int(logits.argmax())]
    while len(tokens) < n_new:
        tokens.append(int(step(tokens[-1]).argmax()))
    return {
        "tokens": tokens,
        "logits_sha256": whole.hexdigest(),
        "rows_sha256": rows,
        "wall_s": wall_s,
        "cycles": cycles,
    }


def first_difference(a: list, b: list) -> int | None:
    """The first index where a and b differ (or where the shorter ends)."""
    i = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
    return i if i is not None or len(a) == len(b) else min(len(a), len(b))


def summary(run: dict, n_prompt: int, mhz: float) -> dict:
    """Timing of the decode steps (after the prompt's): medians, tokens/s."""
    wall, cycles = run["wall_s"][n_prompt:], run["cycles"][n_prompt:]
    if not wall:
        return {}
    wall_med, cycles_med = statistics.median(wall), statistics.median(cycles)
    return {
        "decode_steps": len(wall),
        "wall_s_median": wall_med,
        "tokens_per_s_wall": 1 / wall_med,
        "cycles_median": cycles_med,
        "tokens_per_s_cycles": mhz * 1e6 / cycles_med if cycles_med else None,  # golden: 0
    }


def check_run(log: Log, step: str, params: dict, rt: Runtime, want: dict, mhz: float) -> None:
    """One fixture run: tokens and logit bits against `want`, with timing.
    `want` has tokens and either logits_sha256 (one hash) or a list of them
    (one a position)."""
    prompt, tokens = want["prompt"], want["tokens"]
    run = timed_greedy(rt, prompt, len(tokens))
    result = {"tokens_ok": run["tokens"] == tokens} | summary(run, len(prompt), mhz)
    if isinstance(want["logits_sha256"], list):
        bad = first_difference(run["rows_sha256"], want["logits_sha256"])
        result |= {"logits_ok": bad is None, "first_bad_position": bad}
    else:
        result["logits_ok"] = run["logits_sha256"] == want["logits_sha256"]
    if not result["tokens_ok"]:
        result["first_bad_token"] = first_difference(run["tokens"], tokens)
    if not (result["tokens_ok"] and result["logits_ok"]):
        result["error"] = "not bit-exact with the fixture"
    result |= {"tokens": run["tokens"], "wall_s": run["wall_s"], "cycles": run["cycles"]}
    log.record(step, params, result)


def fixture_runs(log: Log, rt: Runtime, fixtures: dict[str, list[dict]], mhz: float) -> None:
    """Every run of every fixture file (name -> runs), in order."""
    for name, runs in fixtures.items():
        for k, want in enumerate(runs):
            step = "long" if "long" in name else "fixture"
            check_run(
                log, step, {"fixture": name, "run": k, "n_new": len(want["tokens"])}, rt, want, mhz
            )


def stored(name: str) -> dict:
    return json.loads((paths.FIXTURES / name).read_text())


def set_clock(log: Log) -> dict:
    """core_clock CLOCK_MHZ, recorded: its JSON, or {"error": ...}."""
    r = subprocess.run([str(CLOCK), str(CLOCK_MHZ)], capture_output=True, text=True)
    try:
        result = json.loads(r.stdout.strip().splitlines()[-1])
    except IndexError, json.JSONDecodeError:
        result = {"error": f"no JSON (exit {r.returncode})", "stderr": r.stderr[-2000:]}
    return log.record("clock", {"mhz": CLOCK_MHZ}, result)


def run_plan(
    log: Log,
    platform: Callable[[], Platform],
    model: decoder.Model,
    fixtures: dict[str, list[dict]],
    mhz: float,
    pcis_bytes: tuple[int, int] = (PCIS_WRITE_BYTES, PCIS_READ_BYTES),
    cap: int = CAP,
) -> None:
    """The steps after the clock, on the platform `platform()` opens (a
    context manager)."""
    try:
        with platform() as p:
            pcis(log, p, *pcis_bytes, REPEATS)
            t0 = time.perf_counter()
            rt = Runtime.open(p, model, cap)
            result = {"s": time.perf_counter() - t0, "stride": rt.layout.stride}
            log.record("open", {"cap": cap}, result)
            fixture_runs(log, rt, fixtures, mhz)
    except Exception as e:
        log.record("abort", {}, {"error": repr(e)})
        raise
    finally:
        log.upload(now=True)
    log.record("done", {}, {})
    log.upload(now=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", help="the JSON-lines file")
    parser.add_argument("--quick", action="store_true", help="no long fixture")
    args = parser.parse_args()
    from f2_platform import F2Platform

    log = Log(args.out)
    clock = set_clock(log)
    if "error" in clock:
        log.upload(now=True)
        sys.exit(1)
    mhz = clock["clocks_mhz"]["hbm"][0]  # the generator's, for CYCLES -> seconds
    fixtures = {"smollm2_greedy.json": stored("smollm2_greedy.json")["runs"]}
    if not args.quick:
        fixtures["smollm2_long_greedy.json"] = [stored("smollm2_long_greedy.json")]
    run_plan(log, F2Platform, decoder.load(paths.SMOLLM2), fixtures, mhz)


if __name__ == "__main__":
    main()
