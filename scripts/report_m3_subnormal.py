# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The M3 interim report on the cost of subnormal support (D-016),
reports/M3-interim-subnormal.md: its numbers are generated, never typed
(model/reporting/m3_subnormal.py).

    python scripts/report_m3_subnormal.py measure          # Yosys runs -> reports/data/
    python scripts/report_m3_subnormal.py render           # fills the report from the log
    python scripts/report_m3_subnormal.py render --check   # fails if the report is stale

`measure` needs Yosys with the slang plugin (.tools/, scripts/get_rtl_tools.sh)
and takes about a minute. Commit the RTL first: the log records the commit.
"""

import argparse
import datetime
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for sub in ("model", "model/tests", "verif", "verif/tests"):
    sys.path.insert(0, str(REPO / sub))

import paths  # noqa: E402
import test_bf16_mac  # noqa: E402
import test_fp32_fma  # noqa: E402
from reporting import blocks  # noqa: E402
from reporting import m3_subnormal as report  # noqa: E402

import rtl  # noqa: E402  (puts .tools/ on PATH)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=paths.REPO, capture_output=True, text=True, check=True
    ).stdout


def yosys(sources: Path, top: str, params: tuple) -> str:
    """Yosys's output for a generic flattened synthesis of `top` and `stat`."""
    overrides = " ".join(f"-G {k}={v}" for k, v in params)
    script = f"read_slang -F {sources} --top {top} {overrides}; synth -flatten -top {top}; stat"
    run = subprocess.run(
        ["yosys", "-m", "slang", "-p", script], capture_output=True, text=True, check=True
    )
    return run.stdout[run.stdout.rindex("Printing statistics.") :]


def section(rev: str, sources: Path, top: str, params: tuple) -> str:
    label = report.run_label(top, params, rev)
    header = (
        f"CABOSSE-YOSYS label={label} rev={rev} top={top} "
        + "params="
        + ",".join(f"{k}={v}" for k, v in params)
    )
    print(label, flush=True)
    return f"{header}\n{yosys(sources, top, params)}\n"


def flush_hits() -> list[str]:
    """CABOSSE-FLUSH-HITS lines: per unit and operation, the test rows each
    flush changes (the inputs of the flush-to-zero tests)."""

    def line(unit: str, op: str, hits: dict[str, int]) -> str:
        fields = " ".join(f"{k.replace(' ', '_')}={v}" for k, v in hits.items())
        return f"CABOSSE-FLUSH-HITS unit={unit} op={op} {fields}"

    lines = [line("bf16_mac", "mac", test_bf16_mac.flush_hits(test_bf16_mac.ftz_inputs()))]
    abc = test_fp32_fma.ftz_inputs()
    lines += [line("fp32_fma", op, test_fp32_fma.flush_hits(op, abc)) for op in test_fp32_fma.OPS]
    return lines


def measure() -> None:
    version = subprocess.run(["yosys", "-V"], capture_output=True, text=True).stdout.strip()
    commit, dirty = git("rev-parse", "HEAD").strip(), bool(git("status", "--porcelain").strip())
    if dirty:
        print("warning: uncommitted changes; the report will say so")
    out = [
        f"CABOSSE-PROVENANCE commit={commit} dirty={dirty} "
        f"date={datetime.date.today().isoformat()} yosys={version.split()[1]}"
    ]
    out += flush_hits()
    for top, params in report.UNITS.values():
        for ftz in (0, 1):
            out.append(section(report.HEAD, rtl.SOURCES, top, (*params, ("Ftz", ftz))))
    for top, params in report.PARTS.values():
        out.append(section(report.HEAD, rtl.SOURCES, top, params))
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(
            ["git", "archive", report.BASELINE_REV, "rtl"],
            cwd=paths.REPO,
            capture_output=True,
            check=True,
        ).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)
        for top in report.BASELINE_TOPS:
            out.append(section(report.BASELINE_REV, Path(tmp) / "rtl" / "sources.f", top, ()))
    report.LOG.write_text("\n".join(out) + "\n")
    print(f"written: {report.LOG}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("measure")
    sub.add_parser("render").add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.command == "measure":
        measure()
        return
    blocks.write_or_check(report.PATH, report.render(), args.check)


if __name__ == "__main__":
    main()
