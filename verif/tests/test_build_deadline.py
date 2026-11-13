# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""scripts/f2/build_cl.sh's watchdog: it finishes at DEADLINE_MIN, a number
in $S3/deadline_min moves the deadline (capped at MAX_MIN), and anything
else there is ignored. Run on the function's own text, with aws and finish
stubbed and a minute shortened to MIN_S seconds."""

import re
import subprocess
import time
from pathlib import Path

import pytest

BUILD_CL = Path(__file__).parents[2] / "scripts/f2/build_cl.sh"
WATCHDOG = re.compile(r"^watchdog\(\) \{.*?^\}$", re.S | re.M)
MIN_S = 1  # one "minute" of the test, in seconds (bash arithmetic: an integer)


def run_watchdog(tmp_path: Path, deadline_min: int, s3_text: str | None) -> tuple[str, float]:
    body = WATCHDOG.search(BUILD_CL.read_text())
    assert body, "watchdog() not found"
    s3 = tmp_path / "deadline_min"
    if s3_text is not None:
        s3.write_text(s3_text)
    script = f"""
DEADLINE_MIN={deadline_min}; MAX_MIN=4; WATCH_S=0.05; MIN_S={MIN_S}; S3=unused
aws() {{ cat {s3} 2>/dev/null; }}
finish() {{ echo FINISHED; exit; }}
{body[0]}
SECONDS=0
watchdog
"""
    t0 = time.monotonic()
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=20)
    return out.stdout, time.monotonic() - t0


@pytest.mark.parametrize(
    ("s3_text", "minutes", "moved"),
    [
        (None, 2, False),  # no file: DEADLINE_MIN
        ("junk\n", 2, False),  # not a number: ignored
        ("3\n", 3, True),  # extended
        ("09", 4, True),  # leading zero (not octal), capped at MAX_MIN
    ],
)
def test_watchdog_deadline(tmp_path: Path, s3_text: str | None, minutes: int, moved: bool) -> None:
    out, elapsed_s = run_watchdog(tmp_path, 2, s3_text)
    assert out.rstrip().endswith(f"CABOSSE-RESULT fail deadline: {minutes} min\nFINISHED"), out
    assert ("deadline now" in out) == moved, out
    assert out.count("deadline now") <= 1, out  # reported once, not every poll
    # SECONDS has 1 s resolution, so allow a second either way.
    assert minutes * MIN_S - 1 <= elapsed_s <= minutes * MIN_S + 1.5, elapsed_s
