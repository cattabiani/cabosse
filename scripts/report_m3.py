# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The M3 checkpoint report, reports/M3.md: its numbers are generated, never
typed (model/reporting/m3.py). The Vivado logs come from scripts/f2/.

    python scripts/report_m3.py measure-cells   # Yosys: cost of subnormals -> reports/data/
    python scripts/report_m3.py measure-tests   # the test suites -> reports/data/M3.json
    python scripts/report_m3.py render          # fills reports/M3.md from the data
    python scripts/report_m3.py render --check  # fails if M3.md is stale

`measure-cells` needs Yosys with the slang plugin (.tools/,
scripts/get_rtl_tools.sh) and takes about a minute. `measure-tests` runs the
fast and slow suites (the slow one needs the SmolLM2 checkpoint and the RTL
tools) and takes about half an hour; run it under the memory cap. Commit
first: the data records the commit it was measured on.
"""

import argparse
import datetime
import json
import platform
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
import test_fp32_max  # noqa: E402
from reporting import blocks, m3  # noqa: E402
from reporting import m3_subnormal as cells_log  # noqa: E402

import rtl  # noqa: E402  (puts .tools/ on PATH)

# pytest arguments. The report's own up-to-date test waits for this data.
UP_TO_DATE = "model/tests/test_reporting.py::test_m3_report_is_up_to_date"
SUITES = {"fast": ["--deselect", UP_TO_DATE], "slow": ["-m", "slow"]}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=paths.REPO, capture_output=True, text=True, check=True
    ).stdout


def commit_and_dirty() -> tuple[str, bool]:
    commit, dirty = git("rev-parse", "HEAD").strip(), bool(git("status", "--porcelain").strip())
    if dirty:
        print("warning: uncommitted changes; the report will say so")
    return commit, dirty


# --- measure-cells: the cost of subnormal support (D-016) ------------------------


def yosys(sources: Path, top: str, params: tuple) -> str:
    """Yosys's output for a generic flattened synthesis of `top` and `stat`."""
    overrides = " ".join(f"-G {k}={v}" for k, v in params)
    script = f"read_slang -F {sources} --top {top} {overrides}; synth -flatten -top {top}; stat"
    run = subprocess.run(
        ["yosys", "-m", "slang", "-p", script], capture_output=True, text=True, check=True
    )
    return run.stdout[run.stdout.rindex("Printing statistics.") :]


def section(rev: str, sources: Path, top: str, params: tuple) -> str:
    label = cells_log.run_label(top, params, rev)
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


def measure_cells() -> None:
    version = subprocess.run(["yosys", "-V"], capture_output=True, text=True).stdout.strip()
    commit, dirty = commit_and_dirty()
    out = [
        f"CABOSSE-PROVENANCE commit={commit} dirty={dirty} "
        f"date={datetime.date.today().isoformat()} yosys={version.split()[1]}"
    ]
    out += flush_hits()
    for top, params in cells_log.UNITS.values():
        for ftz in (0, 1):
            out.append(section(cells_log.HEAD, rtl.SOURCES, top, (*params, ("Ftz", ftz))))
    for top, params in cells_log.PARTS.values():
        out.append(section(cells_log.HEAD, rtl.SOURCES, top, params))
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(
            ["git", "archive", cells_log.BASELINE_REV, "rtl"],
            cwd=paths.REPO,
            capture_output=True,
            check=True,
        ).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)
        for top in cells_log.BASELINE_TOPS:
            out.append(section(cells_log.BASELINE_REV, Path(tmp) / "rtl" / "sources.f", top, ()))
    cells_log.LOG.write_text("\n".join(out) + "\n")
    print(f"written: {cells_log.LOG}")


# --- measure-tests: the suites and what the slow one covers ------------------------


def pytest_summary(extra: list[str]) -> str:
    """The last line of a pytest run, e.g. '166 passed, 5 deselected in 38.8s'."""
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *extra],
        cwd=paths.REPO,
        capture_output=True,
        text=True,
    )
    return run.stdout.strip().splitlines()[-1].strip("= ")


def measure_tests() -> None:
    commit, dirty = commit_and_dirty()
    data = {
        "provenance": {
            "commit": commit,
            "dirty": dirty,
            "date": datetime.date.today().isoformat(),
            "machine": f"{platform.system()} {platform.machine()}",
        },
        # What the slow suite's bit-exactness tests run, from their own constants.
        "inputs": {
            "fma_add_mul": test_fp32_fma.N_SLOW,
            "max": test_fp32_max.N_SLOW,
            # test_bf16_mac's exhaustive run: chunks of W_PER_CHUNK values of w, all x.
            "bf16_pairs": (2**16 // test_bf16_mac.W_PER_CHUNK) * test_bf16_mac.W_PER_CHUNK * 2**16,
        },
        "tests": {},
    }
    for name, extra in SUITES.items():
        data["tests"][name] = pytest_summary(extra)
        print(name, data["tests"][name], flush=True)
    m3.DATA.write_text(json.dumps(data, indent=1) + "\n")
    print(f"written: {m3.DATA}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("measure-cells")
    sub.add_parser("measure-tests")
    sub.add_parser("render").add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.command == "measure-cells":
        measure_cells()
    elif args.command == "measure-tests":
        measure_tests()
    else:
        blocks.write_or_check(m3.PATH, m3.render(), args.check)


if __name__ == "__main__":
    main()
