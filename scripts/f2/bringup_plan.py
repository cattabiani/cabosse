# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""M8's measurements on F2 (PLAN.md; platforms/f2/README.md): runs the host
tool (platforms/f2/host/bringup) step by step and appends each result, with
its provenance, to a JSON-lines file, copied to S3 after every step when S3
is set. Runs on the F2 instance with the AFI loaded (scripts/f2/run_bringup.sh),
with the system's python3 and no packages.

Usage: python3 bringup_plan.py TOOL OUT.jsonl [--quick]
Environment: S3 (s3://bucket/prefix), PROVENANCE (JSON: commit, AFI, AGFI,
instance, ...), merged into every line.
"""

import json
import os
import subprocess
import sys
import time

PORTS = 32
REGION_BYTES = 512 << 20  # each port's own pseudo channel: 16 GiB / 32
STACK_BYTES = 8 << 30  # one HBM stack
BEAT_BYTES = 32
ALL = (1 << PORTS) - 1
BIG_BEATS = 1 << 21  # 64 MiB a port: a run of milliseconds
SEED = 0x5EED_0000
READ, WRITE, CHECK = 1, 2, 4
CLOCKS_MHZ = [250, 200, 125]  # timed for 250 (D-065): never above
BURSTS = [1, 2, 4, 8, 16]
LIMITS = [1, 2, 4, 8, 16, 32, 64, 128]
LATENCY_SAMPLES = 1000
HOST_BYTES = 64 << 20  # PCIS writes
HOST_READ_BYTES = 4 << 20  # PCIS reads: slow, a word at a time
OCL_SAMPLES = 10000
UPLOAD_S = 30  # results to S3 at most this often, and at the end


class Plan:
    """The tool, the output file and the provenance of every line."""

    def __init__(self, tool: str, out: str, quick: bool) -> None:
        self.tool, self.out, self.quick = tool, out, quick
        self.provenance = json.loads(os.environ.get("PROVENANCE", "{}"))
        self.s3 = os.environ.get("S3", "")
        self.uploaded = 0.0  # when the file last went to S3

    def call(self, *args: object) -> dict:
        """One tool command; its JSON, or {"error": ...}."""
        result = subprocess.run([self.tool, *map(str, args)], capture_output=True, text=True)
        lines = result.stdout.strip().splitlines()
        try:
            return json.loads(lines[-1]) if lines else self.no_json(result)
        except json.JSONDecodeError:
            return self.no_json(result)

    @staticmethod
    def no_json(result: subprocess.CompletedProcess) -> dict:
        """The error of a command that printed no JSON."""
        return {"error": f"no JSON (exit {result.returncode})", "stderr": result.stderr[-2000:]}

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

    @staticmethod
    def settings(**regs: int) -> dict:
        """A run's registers (the tool's names): the defaults, then regs."""
        return {
            "host_port": 0,
            "port_en": ALL,
            "base_lo": 0,
            "base_hi": 0,
            "stride": REGION_BYTES,
            "seed": SEED,
            "rd_burst": 16,
            "wr_burst": 16,
            "rd_limit": 64,
            "wr_limit": 64,
            "strb": 0xFFFF_FFFF,
        } | regs

    def run(self, step: str, **regs: int) -> dict:
        """A run of the test engines with these registers."""
        settings = self.settings(**regs)
        args = [f"{k}={v}" for k, v in settings.items()]
        return self.record(step, settings, self.call("run", *args))

    def latency(self, step: str, n: int, **regs: int) -> dict:
        """n runs of one beat, one burst, each port's first beat's cycles."""
        settings = self.settings(mode=READ, beats=1, rd_burst=1, wr_burst=1, rd_limit=1, **regs)
        args = [f"{k}={v}" for k, v in settings.items()]
        return self.record(step, settings | {"n": n}, self.call("latency", n, *args))


def base(addr: int) -> dict:
    """BASE_LO and BASE_HI for a byte address."""
    return {"base_lo": addr & 0xFFFF_FFFF, "base_hi": addr >> 32}


def at_clock(plan: Plan, mhz: int) -> None:
    """Every memory measurement at one core clock."""
    beats = BIG_BEATS >> 4 if plan.quick else BIG_BEATS
    plan.record("clock", {"mhz": mhz}, plan.call("clock", mhz))
    plan.run("fill", mode=WRITE, beats=beats)
    plan.run("read_check_all", mode=READ | CHECK, beats=beats)
    plan.run("write_all", mode=WRITE, beats=beats)
    plan.run("read_write_all", mode=READ | WRITE, beats=beats)
    for p in range(PORTS):  # each port alone
        plan.run("read_one", mode=READ | CHECK, beats=beats, port_en=1 << p)
    for b in BURSTS:
        plan.run("read_burst", mode=READ, beats=beats, rd_burst=b, port_en=ALL)
        plan.run("write_burst", mode=WRITE, beats=beats, wr_burst=b, port_en=ALL)
    for lim in LIMITS:
        plan.run("read_limit", mode=READ, beats=beats >> 2, rd_burst=1, rd_limit=lim)
        plan.run("write_limit", mode=WRITE, beats=beats >> 2, wr_burst=1, wr_limit=lim)
    # The switch: every port at its own pseudo channel's address, then the
    # next one's, then the other stack's, then all on port 0's.
    for name, addr, stride in [
        ("local", 0, REGION_BYTES),
        ("next_pc", REGION_BYTES, REGION_BYTES),
        ("other_stack", STACK_BYTES, REGION_BYTES),
        ("same_pc", 0, 0),
    ]:
        plan.run(f"switch_{name}", mode=READ, beats=beats >> 2, stride=stride, **base(addr))
    # A one-beat partial write (D-062's V stores) against a full one.
    plan.run("write_partial_1beat", mode=WRITE, beats=beats >> 2, wr_burst=1, strb=0x0000_FFFF)
    plan.run("write_full_1beat", mode=WRITE, beats=beats >> 2, wr_burst=1)
    # Latency: one beat, one burst, ports 0 and 31 (the stacks' far ends).
    n = LATENCY_SAMPLES // 10 if plan.quick else LATENCY_SAMPLES
    for p in (0, 31):
        plan.latency("read_latency", n, port_en=1 << p)


def main() -> None:
    tool, out = sys.argv[1], sys.argv[2]
    plan = Plan(tool, out, "--quick" in sys.argv)
    # First the clock: after the AFI's load the clock generator holds the
    # design in reset until a clock is set (AWS_CLK_GEN_spec.md), and an
    # access before that would not be answered.
    plan.record("clock", {"mhz": CLOCKS_MHZ[0]}, plan.call("clock", CLOCKS_MHZ[0]))
    plan.record("info", {}, plan.call("info"))
    plan.record("ocl", {"n": OCL_SAMPLES}, plan.call("ocl", OCL_SAMPLES))
    # Host access through PCIS, then the engine on port 0 checks what it wrote.
    for offset in (0, STACK_BYTES):
        plan.record(
            "pcis_write",
            {"bytes": HOST_BYTES, "offset": offset},
            plan.call("pcis-write", HOST_BYTES, offset, SEED + 1),
        )
        plan.run(
            "pcis_check",
            mode=READ | CHECK,
            beats=HOST_BYTES // BEAT_BYTES,
            port_en=1,
            seed=SEED + 1,
            **base(offset),
        )
        plan.record(
            "pcis_read",
            {"bytes": HOST_READ_BYTES, "offset": offset},
            plan.call("pcis-read", HOST_READ_BYTES, offset, SEED + 1),
        )
    for mhz in CLOCKS_MHZ:
        at_clock(plan, mhz)
    plan.record("clock", {"mhz": CLOCKS_MHZ[0]}, plan.call("clock", CLOCKS_MHZ[0]))
    plan.record("done", {}, {})
    plan.upload(now=True)


if __name__ == "__main__":
    main()
