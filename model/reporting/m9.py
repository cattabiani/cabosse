# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The generated tables of the M9 checkpoint report (reports/M9.md): core
build 1 on F2 at 100 MHz (D-072), from the runs' JSON lines
(scripts/f2/core_plan.py), the build's summary, the CPU baselines
(scripts/f2/cpu_baseline.sh) and reports/data/M9.json, against the perf
model. Filled by scripts/report_m9.py; needs no weights, no RTL tools and
no AWS access."""

import functools
import json
import re
import statistics
from pathlib import Path

import paths
import perf
from transformers import LlamaConfig

from reporting import blocks, m6
from reporting.measure import QUIET_LOAD

PATH = paths.REPORTS / "M9.md"
DATA = paths.REPORTS / "data" / "M9.json"
F2 = paths.REPORTS / "data" / "f2"
RUNS = {
    1: F2 / "2026-11-05-core-run1.jsonl",
    2: F2 / "2026-11-05-core-run2" / "core.jsonl",
    3: F2 / "2026-11-05-core-run3" / "core.jsonl",
}
BASELINE_RUN = 3  # the CPU baselines ran in run 3's session
BASELINE = F2 / "2026-11-05-core-run3" / "baseline"
CHAT = F2 / "2026-11-05-core-run2" / "chat"
BUILD = F2 / "2026-11-04-core1.txt"
MODEL = "SmolLM2-135M-Instruct"
MIN_LONG_TOKENS = 1000  # PLAN.md, M9: tokens/s over at least 1000 tokens
LOGITS_BYTES = 49152 * 4  # OUTPUT's FP32 logits, read by the host each token
# Windows of the long run's positions: the CPU baseline's (results/README.md;
# llama.cpp's last one starts at 1023, past the run's end) and the 1000 new
# tokens (prompt 46 tokens: position 45 makes the first new token).
WINDOWS = ((0, 127), (46, 173), (917, 1044), (45, 1044))
BENCH_DEPTHS = (0, 46, 1023)
# D-008's baseline: llama2.c's stories260K (its model card: dim 64, 5 layers,
# 8 heads, 4 KV heads, vocabulary 512; hidden 172 by llama2.c's rounding),
# 9.0 tokens/s at 29.8 MHz. Gemmini's weight format was not recorded (int8
# likely, to verify), so its bytes are given per weight.
GEMMINI = {"tokens_per_s": 9.0, "clock_hz": 29.8e6, "position": 0}
STORIES260K = LlamaConfig(
    hidden_size=64,
    intermediate_size=172,
    num_hidden_layers=5,
    num_attention_heads=8,
    num_key_value_heads=4,
    vocab_size=512,
    tie_word_embeddings=True,
)
UTILIZATION = re.compile(r"^\| (CLB LUTs|CLB Registers|Block RAM Tile|URAM|DSPs)\s*\|(.*)\|$", re.M)
STAMPS = re.compile(r"build (\d\d):(\d\d):(\d\d)Z, build end (\d\d):(\d\d):(\d\d)Z")
TIMING = re.compile(r"Timing failed at (\d+) MHz \(\d+ ns\): WNS (-?[\d.]+) ns")


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@functools.cache
def run(n: int) -> dict[str, list[dict]]:
    """A run's steps by name, each a list of its lines in order."""
    steps: dict[str, list[dict]] = {}
    for line in lines(RUNS[n]):
        steps.setdefault(line["step"], []).append(line)
    return steps


def clock_hz(n: int) -> float:
    return run(n)["clock"][0]["result"]["set_mhz"] * 1e6


def long_result(n: int) -> dict:
    return run(n)["long"][0]["result"]


@functools.cache
def design() -> dict[str, float]:
    """The perf model's design at the run's clock: the core and its HBM ports
    on one clock (clk_hbm_axi, D-065)."""
    d = dict(m6.design())
    d["clock_hz"] = d["memory_clock_hz"] = clock_hz(BASELINE_RUN)
    return d


@functools.cache
def config() -> LlamaConfig:
    return perf.load_config(MODEL)


@functools.cache
def model_cycles(position: int) -> float:
    """The perf model's in-order cycles of one decode step (estimate)."""
    p = perf.predict(config(), m6.platform(), design(), position)
    return design()["clock_hz"] / p.tokens_per_s_in_order


@functools.cache
def work(position: int) -> perf.Work:
    return perf.Work.of(perf.design_ops(config(), position, design()))


def port_bytes_per_s() -> float:
    """The 32 ports' raw rate at the run's clock: one 32-byte beat a cycle,
    what M8 measured each port reach up to 250 MHz."""
    p = m6.platform()
    return p["hbm_ports"] * p["hbm_port_bytes"] * design()["memory_clock_hz"]


def window_rows() -> list[list]:
    cycles = long_result(BASELINE_RUN)["cycles"]
    clock = design()["clock_hz"]
    rows = []
    for first, last in WINDOWS:
        span = range(first, last + 1)
        measured = len(span) * clock / sum(cycles[p] for p in span)
        model = len(span) * clock / sum(model_cycles(p) for p in span)
        bound = len(span) * port_bytes_per_s() / sum(work(p).memory_bytes for p in span)
        rows.append(
            [
                f"{first}–{last}",
                f"{measured:.1f}",
                f"{model:.1f}",
                f"{measured / model - 1:+.1%}",
                f"{bound:.1f}",
                f"{measured / bound:.0%}",
                f"{model / bound:.0%}",
            ]
        )
    return rows


def window_table() -> str:
    return blocks.table(
        [
            "positions",
            "measured (cycles)",
            "perf model (estimate)",
            "measured vs model",
            "bandwidth bound",
            "measured / bound",
            "model / bound",
        ],
        window_rows(),
    )


def position_spread() -> str:
    """Measured cycles against the model at every position of the long run."""
    cycles = long_result(BASELINE_RUN)["cycles"]
    shares = [c / model_cycles(p) - 1 for p, c in enumerate(cycles)]
    return (
        f"At each of the long run's {len(cycles)} positions the measured cycles are"
        f" {min(shares):+.1%} to {max(shares):+.1%} off the model"
        f" (median {statistics.median(shares):+.1%})."
    )


def terms_table() -> str:
    """The model's terms of one step at a few positions, in cycles."""
    clock = design()["clock_hz"]
    cycles = long_result(BASELINE_RUN)["cycles"]
    rows = []
    for p in (0, 1023):
        t = perf.predict(config(), m6.platform(), design(), p).terms
        rows.append(
            [p, f"{cycles[p]:,}", f"{model_cycles(p):,.0f}"]
            + [f"{t[k] * clock:,.0f}" for k in ("memory", "engine", "vector", "commands")]
        )
    return blocks.table(
        ["position", "measured", "model, in order", "memory", "engine", "vector", "commands"],
        rows,
    )


def runs_table() -> str:
    rows = []
    for n in RUNS:
        fixtures = run(n)["fixture"]
        long = long_result(n)
        ok = fixtures_ok(n)
        rows.append(
            [
                n,
                f"{len(fixtures)} × {fixtures[0]['params']['n_new']}: {blocks.met(ok)}",
                f"{long_new(n)}: {blocks.met(long_ok(n))}",
                f"{statistics.median(f['result']['cycles_median'] for f in fixtures):,.0f}",
                f"{long['cycles_median']:,}",
                f"{long['tokens_per_s_cycles']:.1f}",
                f"{long['tokens_per_s_wall']:.1f}",
            ]
        )
    return blocks.table(
        [
            "run",
            "64-token fixtures bit-exact",
            "long fixture bit-exact",
            "fixtures, median cycles a step",
            "long, median cycles a step",
            "long, tokens/s (cycles)",
            "long, tokens/s (wall)",
        ],
        rows,
    )


def long_new(n: int) -> int:
    return run(n)["long"][0]["params"]["n_new"]


def long_ok(n: int) -> bool:
    r = long_result(n)
    return r["tokens_ok"] and r["logits_ok"] and r["first_bad_position"] is None


def fixtures_ok(n: int) -> bool:
    return all(f["result"]["tokens_ok"] and f["result"]["logits_ok"] for f in run(n)["fixture"])


def all_bit_exact() -> bool:
    return all(long_ok(n) and fixtures_ok(n) for n in RUNS)


def repeat_text() -> str:
    """How far the long run's cycles moved between runs."""
    cycles = [long_result(n)["cycles"] for n in RUNS]
    diff = max(max(per_position) - min(per_position) for per_position in zip(*cycles, strict=True))
    return (
        f"Runs 2 and 3 repeat run 1's tokens and bits; a step's cycles differ"
        f" between runs by at most {diff} at any position."
    )


def pcis_read_bytes_per_s(n: int) -> float:
    return statistics.median(r["result"]["bytes_per_s"] for r in run(n)["pcis_read"])


def pcis_write_bytes_per_s(n: int) -> float:
    return statistics.median(r["result"]["bytes_per_s"] for r in run(n)["pcis_write"])


def pcis_table() -> str:
    """Each run's five 64 MiB writes and 4 MiB reads, in MB/s, in order."""
    rows = [
        [n]
        + [
            " ".join(f"{r['result']['bytes_per_s'] / 1e6:.0f}" for r in run(n)[k])
            for k in ("pcis_write", "pcis_read")
        ]
        for n in RUNS
    ]
    return blocks.table(["run", "PCIS writes, MB/s", "PCIS reads, MB/s"], rows)


def wall_table() -> str:
    """A decode step's wall time on the long run: the core's cycles, the
    logits' read at the run's measured PCIS rate (estimate), the rest."""
    n = BASELINE_RUN
    r = long_result(n)
    wall = r["wall_s_median"]
    core = r["cycles_median"] / design()["clock_hz"]
    read = LOGITS_BYTES / pcis_read_bytes_per_s(n)
    rest = wall - core - read
    rows = [
        ["the core (CYCLES, median)", f"{core * 1e3:.2f}", f"{core / wall:.0%}"],
        [
            f"logits read, {LOGITS_BYTES:,} B at {pcis_read_bytes_per_s(n) / 1e6:.1f} MB/s"
            " (estimate)",
            f"{read * 1e3:.2f}",
            f"{read / wall:.0%}",
        ],
        ["the rest (host, driver, commands)", f"{rest * 1e3:.2f}", f"{rest / wall:.0%}"],
        ["wall clock (median)", f"{wall * 1e3:.2f}", "100%"],
    ]
    return blocks.table(["part of a step", "ms", "share"], rows)


def per_s(x: float) -> str:
    """A rate with its prefix: 2.34 M, 23.4 G."""
    return f"{x / 1e9:.1f} G" if x >= 1e9 else f"{x / 1e6:.2f} M"


def weight_macs(c: LlamaConfig, position: int) -> tuple[int, int]:
    """(weight bytes, weight multiply-adds) of one token, BF16, from the perf
    model's op list."""
    w = perf.Work.of(perf.design_ops(c, position, design()))
    return w.weight_bytes, w.matvec_macs


def normalized_table() -> str:
    """D-008's baseline against the core, per second and per cycle."""
    clock = design()["clock_hz"]
    r = long_result(BASELINE_RUN)
    ours_tps = r["tokens_per_s_cycles"]
    ours_bytes, ours_macs = weight_macs(config(), 0)
    g_bytes, g_macs = weight_macs(STORIES260K, GEMMINI["position"])
    g_tps, g_clock = GEMMINI["tokens_per_s"], GEMMINI["clock_hz"]
    g_weights = g_bytes // 2  # BF16 bytes in the op list: one weight is 2

    def row(name, tps, n_weights, n_bytes, macs, hz, model):
        return [
            name,
            model,
            f"{hz / 1e6:.1f}",
            f"{tps:.1f}",
            per_s(n_weights * tps),
            n_bytes,
            per_s(macs * tps),
            f"{macs * tps / hz:.3g}",
        ]

    rows = [
        row(
            "Gemmini + Rocket (D-008)",
            g_tps,
            g_weights,
            "not recorded (int8 likely, to verify)",
            g_macs,
            g_clock,
            f"stories260K, {g_weights / 1e3:.0f}k weights",
        ),
        row(
            "Cabosse core, build 1",
            ours_tps,
            ours_bytes // 2,
            f"{ours_bytes * ours_tps / 1e9:.1f} GB/s (BF16)",
            ours_macs,
            clock,
            f"{MODEL}, {ours_bytes // 2 / 1e6:.0f}M weights",
        ),
    ]
    text = blocks.table(
        [
            "",
            "model",
            "MHz",
            "tokens/s",
            "weights read/s",
            "weight bytes/s",
            "weight MACs/s",
            "MACs a cycle",
        ],
        rows,
    )
    ratio = (ours_macs * ours_tps / clock) / (g_macs * g_tps / g_clock)
    return (
        text + "\n\nThe core's row is the long run's median step. Weights read once a"
        " token; MACs of the weight matrices only, attention left out. Per cycle the"
        f" core does {ratio:,.0f} times Gemmini's weight multiply-adds."
    )


def cpu_rows() -> list[list]:
    bench = {b["n_depth"]: b for b in json.loads((BASELINE / "llama_bench.json").read_text())}
    cycles = long_result(BASELINE_RUN)["cycles"]
    clock = design()["clock_hz"]
    rows = []
    for depth, (first, last) in zip(BENCH_DEPTHS, WINDOWS[:3], strict=True):
        b = bench[depth]
        span = range(first, last + 1)
        core = len(span) * clock / sum(cycles[p] for p in span)
        rows.append(
            [
                f"{first}–{last}",
                f"{core:.1f}",
                f"{depth}–{depth + b['n_gen'] - 1}",
                f"{b['avg_ts']:.1f} ± {b['stddev_ts']:.1f}",
                f"{core / b['avg_ts'] - 1:+.0%}",
            ]
        )
    return rows


def cpu_table() -> str:
    t = json.loads((BASELINE / "transformers.json").read_text())
    bench = json.loads((BASELINE / "llama_bench.json").read_text())[0]
    busy = t["load_average"][0] > QUIET_LOAD
    flag = (
        f" The 1-minute load average was {t['load_average'][0]:.1f} when it started"
        " (llama.cpp had just finished): timing suspect, likely low."
        if busy
        else ""
    )
    return (
        blocks.table(
            ["core positions", "core, tokens/s", "llama.cpp positions", "llama.cpp", "core vs CPU"],
            cpu_rows(),
        )
        + f"\n\nllama.cpp {bench['build_commit']}, BF16 GGUF, {bench['n_threads']} threads,"
        f" `llama-bench` mean ± standard deviation of {bench['n_gen']} generated tokens."
        f" `transformers` {t['transformers']} ({t['attention']}, BF16, {t['threads']}"
        f" threads, {t['n_decode']} tokens): {statistics.median(t['tokens_per_s']):.1f}"
        f" tokens/s, median of {len(t['tokens_per_s'])}.{flag}"
    )


def utilization() -> dict[str, tuple[int, int, float]]:
    """(used, available, %) of each resource in the build's summary."""
    out = {}
    part1 = BUILD.read_text().split("## Part 2")[0]
    for name, cells in UTILIZATION.findall(part1):
        used, _, _, available, pct = (c.strip() for c in cells.split("|"))
        out[name] = (int(used), int(available), float(pct))
    return out


def build_table() -> str:
    target_mhz, wns = TIMING.search(BUILD.read_text()).groups()
    period_ns = 1e3 / float(target_mhz)
    reached = 1e3 / (period_ns - float(wns))
    rows = [
        [
            "clock",
            f"target {target_mhz} MHz missed: WNS {wns} ns, so about {reached:.0f} MHz;"
            f" run at {design()['clock_hz'] / 1e6:.0f} MHz",
        ]
    ]
    h0, m0, s0, h1, m1, s1 = (int(x) for x in STAMPS.search(BUILD.read_text()).groups())
    seconds = (h1 - h0) * 3600 + (m1 - m0) * 60 + (s1 - s0)
    hours = (seconds % 86400) / 3600  # a build shorter than a day, past midnight or not
    rows.append(["build time", f"{hours:.1f} h (log stamps, synthesis to route)"])
    rows += [
        [name, f"{used:,} of {available:,} ({pct:.1f}%)"]
        for name, (used, available, pct) in utilization().items()
    ]
    return blocks.table(["", "core build 1"], rows)


def chat_text() -> str:
    turns = [line for line in lines(CHAT / "chat.jsonl") if line["step"] == "turn"]
    steps = [s for t in turns for s in t["result"]["wall_s"]]
    return (
        f"{len(turns)} turns, {sum(t['result']['n_new'] for t in turns)} tokens,"
        f" {statistics.median(steps) * 1e3:.1f} ms a token (median wall clock)."
    )


def cost_text() -> str:
    days = json.loads(DATA.read_text())["cost"]["usd_by_day"]
    lines_ = [
        f"{day}: " + (", ".join(f"{k} ${v:.2f}" for k, v in costs.items() if v) or "nothing yet")
        for day, costs in sorted(days.items())
    ]
    return "Billed (AWS Cost Explorer): " + "; ".join(lines_) + "."


def exit_rows() -> list[list]:
    long_tokens = long_new(BASELINE_RUN)
    window = window_rows()[-1]
    clock = design()["clock_hz"] / 1e6
    return [
        [
            "Bit-exact against the golden model, at the build's clock (D-072)",
            f"runs 1–3 at {clock:.0f} MHz: three 64-token fixtures and the"
            f" {long_tokens}-token long fixture, tokens and every logit's bits",
            blocks.met(all_bit_exact()),
        ],
        [
            "Tokens/s over ≥ 1000 tokens, against the perf model and the bandwidth bound",
            f"{window[1]} tokens/s over positions {window[0]}; model {window[2]}"
            f" ({window[3]}); bound {window[4]} ({window[5]} of it)",
            blocks.met(long_tokens >= MIN_LONG_TOKENS),
        ],
        ["Clock and resource use reported", "below (core build 1)", "yes"],
        ["Against Gemmini, normalized (weight bytes/s, MACs/s)", "below", "yes"],
        ["Against llama.cpp on the F2 host's CPU, same positions (D-056)", "below", "yes"],
        [
            "A recorded 1-2 turn chat, command and log saved",
            f"run 2: {chat_text()}",
            blocks.met((CHAT / "chat.cast").exists() and (CHAT / "chat.jsonl").exists()),
        ],
    ]


def generated() -> dict[str, str]:
    rows = exit_rows()
    return {
        "verdict": blocks.verdict(rows, "M9"),
        "exit-criteria": blocks.table(["criterion", "measured", "met"], rows),
        "runs": runs_table() + "\n\n" + repeat_text(),
        "windows": window_table(),
        "spread": position_spread(),
        "terms": terms_table(),
        "wall": wall_table(),
        "pcis": pcis_table(),
        "normalized": normalized_table(),
        "cpu": cpu_table(),
        "build": build_table(),
        "cost": cost_text(),
    }


def render() -> str:
    return blocks.fill(PATH.read_text(), generated())
