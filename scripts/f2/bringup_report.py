# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Summary of an F2 bring-up run (scripts/f2/bringup_plan.py's JSON lines):
the clock, OCL latency, host access through PCIS, and the test engines'
runs, in the run's order: HBM bandwidth in total and per port, against
burst length and bursts in flight, the switch, partial writes and read
latency. A run's reads and writes follow its recorded mode, its swept
setting its recorded registers; rates come from the ports' own cycle
counters at the clock the last clock step set (platforms/f2/README.md).

Usage: python3 bringup_report.py RUN.jsonl
"""

import json
import statistics
import sys

from bringup_plan import BEAT_BYTES, READ, WRITE

ERROR_FLAG = {"rd": 0b0100, "wr": 0b1000}  # FLAGS: a read, a write error response (README.md)


def gb_per_s(beats: int, cycles: int, mhz: float) -> float:
    """Beats over cycles at the core clock."""
    return beats * BEAT_BYTES * mhz * 1e6 / cycles / 1e9 if cycles else 0.0


def rates(result: dict, mhz: float, kind: str) -> list[float]:
    """Every busy port's read ("rd") or write ("wr") rate."""
    return [
        gb_per_s(p[f"{kind}_beats"], p[f"{kind}_cycles"], mhz)
        for p in result["ports"]
        if p[f"{kind}_beats"]
    ]


def total_gb_per_s(result: dict, mhz: float, kind: str) -> float:
    """All ports' beats over the slowest port's cycles (ports start together)."""
    ports = result["ports"]
    beats = sum(p[f"{kind}_beats"] for p in ports)
    return gb_per_s(beats, max(p[f"{kind}_cycles"] for p in ports), mhz)


def mean_wait(result: dict, burst: int, kind: str) -> float:
    """Mean cycles a burst waits (Little's law over the occupancy sum)."""
    ports = result["ports"]
    occ = sum(p[f"{kind}_occ_lo"] + (p[f"{kind}_occ_hi"] << 32) for p in ports)
    bursts = sum(p[f"{kind}_beats"] for p in ports) // burst
    return occ / bursts if bursts else 0.0


def label(step: str, params: dict, kind: str) -> str:
    """The step's name, the setting it sweeps if its name ends in one
    (read_burst 4, write_limit 16), and its base address if not 0."""
    setting = f"{kind}_{step.rpartition('_')[2]}"
    text = f"{step} {params[setting]}" if setting in params else step
    base = params["base_hi"] << 32 | params["base_lo"]
    return f"{text} @{base:#x}" if base else text


def line(step: str, params: dict, result: dict, mhz: float, kind: str) -> str:
    """Total, port min and max of one direction of a run and its ports with
    an error response; for reads the bad beats, for writes the most bursts
    unanswered at once and the mean wait."""
    r = rates(result, mhz, kind)
    ports = result["ports"]
    errors = sum(1 for p in ports if p["flags"] & ERROR_FLAG[kind])
    text = (
        f"  {label(step, params, kind):<22} {kind} {total_gb_per_s(result, mhz, kind):6.1f} GB/s "
        f"total, {len(r):2} ports {min(r, default=0):5.2f}-{max(r, default=0):5.2f} GB/s, "
        f"error ports {errors}"
    )
    if kind == "rd":
        return text + f", bad {sum(p['rd_bad'] for p in ports)}"
    return text + (
        f", max out {max(p['wr_maxout'] for p in ports)}, "
        f"mean wait {mean_wait(result, params['wr_burst'], 'wr'):.1f} cycles"
    )


def latency_line(params: dict, result: dict, mhz: float) -> str:
    """First-beat cycles of one-beat reads on one port."""
    first = result["rd_first"]
    port = params["port_en"].bit_length() - 1
    q = statistics.quantiles(first, n=100)
    return (
        f"  read_latency port {port:<2}     first beat {min(first)}/{q[49]:g}/{q[98]:g}/"
        f"{max(first)} cycles (min/p50/p99/max, n {len(first)}), p50 {q[49] * 1e3 / mhz:.0f} ns"
    )


def alone_line(alone: list[dict], mhz: float) -> str:
    """The read_one runs (each port by itself) in one line; a port that read
    nothing counts as 0 GB/s."""
    each = [max(rates(r, mhz, "rd"), default=0.0) for r in alone]
    ports = [p for r in alone for p in r["ports"]]
    errors = sum(1 for p in ports if p["flags"] & ERROR_FLAG["rd"])
    return (
        f"  read_one (each alone)  rd {min(each):5.2f}-{max(each):5.2f} GB/s, {len(each)} ports, "
        f"error ports {errors}, bad {sum(p['rd_bad'] for p in ports)}"
    )


def engine_lines(row: dict, mhz: float) -> list[str]:
    """One engine run: a line per direction its mode has."""
    step, params, result = row["step"], row["params"], row["result"]
    if "error" in result:
        return [f"  {step}: error {result['error']}"]
    if step == "read_latency":
        return [latency_line(params, result, mhz)]
    return [
        line(step, params, result, mhz, kind)
        for bit, kind in ((READ, "rd"), (WRITE, "wr"))
        if params["mode"] & bit
    ]


def host_line(step: str, params: dict, result: dict) -> str:
    """One host-side step (clock, OCL, PCIS), short."""
    if "error" in result:
        return f"{step} {json.dumps(params)}: error {result['error']}"
    if step == "clock":
        return (
            f"clock {params['mhz']} MHz (measured {result['measured_mhz']:.3f}; "
            f"a port's limit {gb_per_s(1, 1, params['mhz']):.1f} GB/s)"
        )
    if step == "ocl":
        ns = result["read_ns"]
        return (
            f"OCL read {ns['min']}/{ns['p50']}/{ns['p99']}/{ns['max']} ns "
            f"(min/p50/p99/max, n {params['n']})"
        )
    words = f", bad words {result['bad_words']}" if "bad_words" in result else ""
    return f"{step} offset {params['offset']:#x}: {result['gb_per_s'] * 1e3:.1f} MB/s{words}"


def report(rows: list[dict]) -> list[str]:
    """The run's summary in its order. Engine runs take the rate of the
    last clock step; after a failed one, the clock is unknown and they are
    not reported. The read_one runs make one line, at the next clock step."""
    first = rows[0]
    out = [
        f"commit {first['commit']}, {first['agfi']} ({first['afi']}), "
        f"{first['instance']} {first['instance_type']}"
    ]
    mhz = None  # the core clock, once a clock step set it
    alone: list[dict] = []
    for row in [*rows, {"step": "end", "params": {}, "result": {}}]:
        step, params, result = row["step"], row["params"], row["result"]
        if step in ("clock", "end") and alone:
            out.append(alone_line(alone, mhz))
            alone = []
        if step == "clock":
            mhz = None if "error" in result else params["mhz"]
        if "mode" not in params:
            if step not in ("info", "done", "end"):
                out.append(host_line(step, params, result))
        elif mhz is None:
            out.append(f"  {step}: not reported, the core clock is not known")
        elif step == "read_one" and "error" not in result:
            alone.append(result)
        else:
            out += engine_lines(row, mhz)
    return out


def main() -> None:
    with open(sys.argv[1]) as f:
        rows = [json.loads(text) for text in f]
    print("\n".join(report(rows)))


if __name__ == "__main__":
    main()
