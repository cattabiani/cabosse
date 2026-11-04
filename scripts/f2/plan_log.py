# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The JSON-lines log of an F2 plan (bringup_plan.py, core_plan.py): each
step's result with its provenance, appended to a file and copied to S3 at
most every UPLOAD_S when S3 is set. Standard library only: bring-up runs it
with the system's python3.

Environment: S3 (s3://bucket/prefix), PROVENANCE (JSON: commit, AFI, AGFI,
instance, host CPU model, ...), merged into every line.
"""

import json
import os
import subprocess
import time

UPLOAD_S = 30  # results to S3 at most this often, and at the end


class Log:
    """The output file and the provenance of every line."""

    def __init__(self, out: str) -> None:
        self.out = out
        self.provenance = json.loads(os.environ.get("PROVENANCE", "{}"))
        self.s3 = os.environ.get("S3", "")
        self.uploaded = 0.0  # when the file last went to S3

    def upload(self, now: bool = False) -> None:
        """The file to S3, at most every UPLOAD_S unless now."""
        if self.s3 and (now or time.monotonic() - self.uploaded > UPLOAD_S):
            dst = f"{self.s3}/{os.path.basename(self.out)}"
            subprocess.run(["aws", "s3", "cp", "--only-show-errors", self.out, dst])
            self.uploaded = time.monotonic()

    def record(self, step: str, params: dict, result: dict) -> dict:
        """Append a step's result to the file (and S3); return it."""
        line = {"step": step, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        line |= self.provenance | {"params": params, "result": result}
        with open(self.out, "a") as f:
            f.write(json.dumps(line) + "\n")
        self.upload()
        print(step, json.dumps(params), "error" if "error" in result else "ok", flush=True)
        return result
