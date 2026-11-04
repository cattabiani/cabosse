# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The M7 checkpoint report's text (reports/M7.md) from reports/data/M7.json
(scripts/measure_m7.py): D-067's CPU and GPU baseline."""

import statistics

from reporting.measure import QUIET_LOAD

DEVICES = ("cpu", "cuda")  # D-067's parts of the data


def baseline_text(data: dict) -> str:
    """D-067's rows: device, median decode tokens/s, what it ran on; a row
    timed on a busy machine (QUIET_LOAD) says so."""
    lines = ["D-067 baseline, SmolLM2 BF16, batch 1, greedy, decode tokens/s (median):"]
    for device in DEVICES:
        if device in data:
            b = data[device]
            on = f"{b['gpu']}, host " if "gpu" in b else ""
            load = b["load_average"][0]
            busy = " (busy: timing suspect)" if load > QUIET_LOAD else ""
            lines.append(
                f"{device:4}  {statistics.median(b['tokens_per_s']):8.1f}  {on}{b['cpu']},"
                f" {b['threads']} threads, load {load:.2f}{busy}; torch {b['torch']},"
                f" transformers {b['transformers']}, {b['attention']} attention;"
                f" {b['provenance']['date']}"
            )
    return "\n".join(lines)
