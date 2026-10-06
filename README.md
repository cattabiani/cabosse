# Cabosse

Cabosse is an open-source hardware accelerator for LLM inference. The chip design
(SystemVerilog) and all software (golden model, testbenches, host runtime) are
published in this repository.

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

**M2 in progress** (architecture spec and performance model). See [PLAN.md](PLAN.md) for goals,
decisions, milestones, and open questions.

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
