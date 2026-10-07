# Cabosse

Cabosse is an open-source hardware accelerator for LLM inference. The chip design
(SystemVerilog) and all software (golden model, testbenches, host runtime) are
published in this repository.

The name: Swiss chips → choco chips → *cabosse*, the cocoa pod where chocolate
starts. Same as chocolate: everyone loves the chips, nobody remembers the pod.
We're making the pod open source.

## Goal

Show that open AI hardware can be built and reproduced from public sources:
anyone with an AWS account can rebuild it and run it on F2. Tools are open
source wherever possible; Vivado, F2's build flow, is the exception. Two
steps:

1. **Simulation:** run a small Llama-style model token by token in RTL
   simulation and match a bit-exact Python reference.
2. **FPGA:** run the same design on an AWS F2 FPGA instance and measure
   tokens/s.

An ASIC on an open-source process is a possible later development, out of
scope for now.

The first target workload is single-stream decode (one token at a time) of
small models, starting with SmolLM2-135M-Instruct (a small chat model). Numerics are BF16
multiplies with FP32 accumulation, and models load directly from their published
checkpoints.

## Status

**M3 in progress** (toolchain and FP units). M2 is done:
[reports/M2.md](reports/M2.md). See [PLAN.md](PLAN.md) for goals, decisions,
milestones, and open questions.

## Development setup

Linux, nothing installed system-wide; everything lives in the repo,
git-ignored.

- **Python** (3.14): `python3.14 -m venv .venv && .venv/bin/pip install -r
  requirements.txt`.
- **RTL tools:** the [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build)
  (Verilator, Yosys with the slang SystemVerilog plugin), unpacked in
  `.tools/` by `scripts/get_rtl_tools.sh` (the pinned release, about 750 MB).
  The tests find it there by themselves; for the tools by hand, add its
  `bin/` to `PATH` (direnv: `PATH_add .tools/oss-cad-suite/bin`) and do not
  source its `environment` script. The suite also bundles cocotb, in its own
  Python (3.11, a development snapshot); the testbenches use the pinned
  cocotb in `.venv`, which runs with torch and the golden model. The release,
  not distribution packages: cocotb 2.1 needs Verilator 5.036 or newer, which
  Ubuntu 26.04 does not have, and the slang plugin is not packaged. It sits
  in the repo because nothing else uses it.
- **Tests:** `pytest` runs the golden-model tests and the RTL tests
  (`pytest -m slow` the long ones). RTL tests skip without the suite;
  `CABOSSE_REQUIRE_RTL=1` turns that into a failure.
- **Vivado** is not installed locally: F2 builds run on AWS (D-032,
  [docs/aws-setup.md](docs/aws-setup.md)).
- **Model weights** go in `weights/` (git-ignored), for example
  `.venv/bin/hf download HuggingFaceTB/SmolLM2-135M-Instruct --local-dir
  weights/SmolLM2-135M-Instruct`. Tests that need them are marked `slow`
  and skip without them.

## Repository layout

| Path         | Contents                                                   |
|--------------|------------------------------------------------------------|
| `docs/`      | Architecture, numerics spec                                |
| `model/`     | Python golden model and performance model                  |
| `rtl/`       | Synthesizable SystemVerilog                                |
| `verif/`     | cocotb testbenches                                         |
| `sw/`        | Host runtime, driver, PyTorch integration                  |
| `platforms/` | Thin wrapper for AWS F2                                    |
| `scripts/`   | Developer tooling                                          |

## License

- Hardware (`rtl/`, `platforms/`): Solderpad Hardware License v2.1
  ([LICENSE-HARDWARE](LICENSE-HARDWARE)), which wraps Apache-2.0 and extends
  it to hardware rights such as chip layouts. Recipients may instead use the
  files under plain Apache-2.0.
- Everything else (software, testbenches, docs): Apache-2.0 ([LICENSE](LICENSE)).

Each source file states its license with an SPDX identifier.
