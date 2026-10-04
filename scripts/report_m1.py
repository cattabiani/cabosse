# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The M1 checkpoint report, reports/M1.md: its numbers are generated, never
typed (model/report.py).

    python scripts/report_m1.py measure          # writes reports/data/M1.json
    python scripts/report_m1.py render           # fills reports/M1.md from it
    python scripts/report_m1.py render --check   # fails if M1.md is stale

`measure` needs the SmolLM2 checkpoint and takes about 25 minutes: the full
comparison, decode timing, the logits where the SmolLM2 fixture diverges
from `transformers`, and both test suites. Commit the code first: the data
records the commit it was measured on. Run it under the memory cap.
"""

import argparse
import datetime
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

import compare  # noqa: E402
import fixtures  # noqa: E402
import paths  # noqa: E402
import report  # noqa: E402
from golden import decoder  # noqa: E402

POSITIONS = 256  # per comparison sequence
DECODE_STEPS = 16  # timed decode steps, after the prompt
# pytest arguments. The report's own up-to-date test waits for this data.
UP_TO_DATE = "model/tests/test_report.py::test_m1_report_is_up_to_date"
SUITES = {"fast": ["--deselect", UP_TO_DATE], "slow": ["-m", "slow"]}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=paths.REPO, capture_output=True, text=True, check=True
    ).stdout


def provenance() -> dict:
    cpuinfo = Path("/proc/cpuinfo").read_text().splitlines()  # Linux: the reference platform
    cpu = next(line.split(":", 1)[1].strip() for line in cpuinfo if line.startswith("model name"))
    return {
        "commit": git("rev-parse", "HEAD").strip(),
        "dirty": bool(git("status", "--porcelain").strip()),
        "date": datetime.date.today().isoformat(),
        "machine": f"{platform.system()} {platform.machine()}",
        "cpu": cpu,
        "threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def comparison(tokenizer, reference, bf16, golden: decoder.Model) -> dict:
    start = time.perf_counter()
    sequences = compare.chat_sequences(tokenizer, reference, compare.PROMPTS, POSITIONS)
    raw = compare.as_json(compare.compare(golden, reference, bf16, sequences))
    return raw | {
        "prompts": compare.PROMPTS,
        "sequences": sequences,
        "minutes": (time.perf_counter() - start) / 60,
    }


def decode_seconds(tokenizer, golden: decoder.Model) -> list[float]:
    """Wall time of each decode_step after the first prompt."""
    prompt = compare.chat_prompt(tokenizer, compare.PROMPTS[0])
    cache = decoder.KVCache.empty(golden, len(prompt) + DECODE_STEPS)
    token = int(decoder.step(golden, cache, prompt)[-1].argmax())
    seconds = []
    for _ in range(DECODE_STEPS):
        start = time.perf_counter()
        token = int(decoder.decode_step(golden, cache, token).argmax())
        seconds.append(time.perf_counter() - start)
    return seconds


def divergences(tokenizer, reference, golden: decoder.Model) -> list[dict]:
    """For each SmolLM2 fixture run: where the golden model's greedy tokens
    first differ from FP32 `transformers`', and both tokens' logits there in
    each model."""
    fixture = json.loads((paths.FIXTURES / "smollm2_greedy.json").read_text())
    out = []
    for text, run in zip(fixtures.SMOLLM2_PROMPTS, fixture["runs"], strict=True):
        i = run["first_difference"]
        entry = {"prompt": text, "first_difference": i, "golden": None, "transformers_fp32": None}
        if i is not None:
            prefix = run["prompt"] + run["tokens"][:i]  # the same in both models
            choices = [run["tokens"][i], run["transformers_fp32_tokens"][i]]
            rows = {
                "golden": decoder.forward(golden, prefix)[-1],
                "transformers_fp32": compare.transformers_logits(reference, prefix)[-1],
            }
            for name, row in rows.items():
                entry[name] = [[tokenizer.decode([t]), float(row[t])] for t in choices]
        out.append(entry)
    return out


def pytest_summary(extra: list[str]) -> str:
    """The last line of a pytest run, e.g. '166 passed, 5 deselected in 38.8s'."""
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *extra],
        cwd=paths.REPO,
        capture_output=True,
        text=True,
    )
    return run.stdout.strip().splitlines()[-1].strip("= ")


def measure(weights: Path) -> None:
    data = {"provenance": provenance()}
    if data["provenance"]["dirty"]:
        print("warning: uncommitted changes; the report will say so")
    tokenizer = AutoTokenizer.from_pretrained(weights)
    reference = LlamaForCausalLM.from_pretrained(weights, dtype=torch.float32).eval()
    bf16 = LlamaForCausalLM.from_pretrained(weights, dtype=torch.bfloat16).eval()
    golden = decoder.load(weights)

    data["decode"] = {"steps": DECODE_STEPS, "seconds_per_token": decode_seconds(tokenizer, golden)}
    print(f"decode: median {statistics.median(data['decode']['seconds_per_token']):.2f} s/token")
    data["divergences"] = divergences(tokenizer, reference, golden)
    data["comparison"] = comparison(tokenizer, reference, bf16, golden)
    data["tests"] = {name: pytest_summary(extra) for name, extra in SUITES.items()}
    print(data["tests"])

    report.M1_DATA.parent.mkdir(parents=True, exist_ok=True)
    report.M1_DATA.write_text(json.dumps(data, indent=1) + "\n")
    print(f"written: {report.M1_DATA}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("measure").add_argument("--weights", type=Path, default=paths.SMOLLM2)
    sub.add_parser("render").add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.command == "measure":
        measure(args.weights)
        return
    text = report.render_m1()
    if args.check:
        sys.exit(
            0 if text == report.M1_REPORT.read_text() else f"{report.M1_REPORT} is stale: render it"
        )
    report.M1_REPORT.write_text(text)
    print(f"written: {report.M1_REPORT}")


if __name__ == "__main__":
    main()
