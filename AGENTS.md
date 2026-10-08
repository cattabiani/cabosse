# Instructions for AI agents working on Cabosse

Read this file fully before doing anything. Then read [PLAN.md](PLAN.md). Both
take precedence over your defaults.

## Project summary

Cabosse is an open-source hardware accelerator for LLM inference. The chip
design and all software are open source. The aim is to show that open AI
hardware is real and reproducible. The path is simulation → FPGA (AWS F2);
an ASIC on an open-source process is out of scope for now (D-036). The
project owner is a software engineer and numerical scientist who is new to
hardware. Explain hardware concepts briefly when you use them.

## Decisions already made

The full text is in the PLAN.md decision log. Do not reopen these. If you think
one is wrong, say so and explain why, and let the owner decide.

- **Numerics:** BF16 × BF16 multiply, FP32 accumulate, FP32 vector unit.
  Models load from official published checkpoints with no conversion or
  calibration. Weight-only quantization (int8/int4 → BF16) may come later. The
  design must not rule it out.
- **Workload:** single-stream LLM decode, which is memory-bandwidth bound.
  Blocks: streaming matrix-vector engine, vector unit (RMSNorm, softmax,
  SiLU/SwiGLU, RoPE, residuals), DMA engines, on-chip buffers, and a small
  controller running a command stream. Activations and the KV cache stay in
  accelerator memory. The host is not involved in individual operations.
- **System:** host CPU + accelerator as a PCIe device, AWS F2 first. The core
  exposes AXI-Lite (control) + AXI (memory). Thin wrappers in `platforms/`
  target F2. Other FPGAs (ECP5) and an ASIC are out of scope (D-034, D-036).
- **Languages/tools:** SystemVerilog (no Chisel), Verilator, cocotb. The golden
  model is written in PyTorch (torch tensors, explicit operations).
  Development happens on Linux.
- **Verification:** the Python golden model reproduces the hardware's exact
  formats and summation order and is the bit-exact reference. Comparisons
  against PyTorch use stated tolerances.
- **Model ladder:** tiny random-weight Llama configs (fast RTL tests) →
  SmolLM2-135M-Instruct (first real model) → Qwen2.5-0.5B. Reference: Hugging Face
  `transformers`.
- **Baseline:** Gemmini + Rocket on F2, 29.8 MHz, 9.0 tokens/s on
  stories260K. Data movement and orchestration dominated, not the matmul.
  Gemmini is not reused.

## Hard rules

1. **Commit freely; push only to branches, never to `main`.** Commits are
   fine whenever they make sense: small, one logical change each, with a
   clear message. Pushing a branch other than `main` needs no OK; `main`
   changes only through pull requests the owner merges. Creating or changing
   remotes needs explicit consent. Do not rewrite history that has already
   been pushed.
2. **Never run `sudo`.** If something needs it, print the exact command and
   ask the owner to run it.
3. **No cloud resources** (AWS or other) unless the owner has authorized it
   in the current session, with a budget. Reading public documentation is
   fine.
4. **Stay inside the current milestone.** Do not implement work that belongs
   to a later milestone. If you have an idea for later, add it to PLAN.md
   (open questions or the relevant milestone's notes).
5. **Stop at checkpoints.** When a milestone's exit criteria are met, report
   what was done and what was measured, list open issues, and stop. Do not
   start the next milestone.
6. **Do not add dependencies** (Python packages, tools, vendored IP) without
   asking. Every dependency needs a license compatible with the project.
   Python packages may be installed into the project's `.venv` and must be
   pinned in `requirements.txt`. **Never install anything outside `.venv`**
   (apt, system-wide, tarballs). If a program is missing, say what it is and
   why it is needed, and suggest the install command. The owner installs it.
7. **Never commit model weights, waveforms, or build outputs.** Weights go in
   the git-ignored `weights/` directory.
8. **Numerics changes go through the spec.** Any change to formats, rounding,
   or summation order updates `docs/numerics.md`, the golden model, and the
   affected RTL tests together. A change to a D-xxx decision needs a new
   decision-log entry approved by the owner.
9. **Report results honestly.** If a test fails, show the output. If a
   number is an estimate, call it an estimate. Never tune a test or a tolerance
   just to make it pass without saying so.

## Working style

- Work proceeds in milestones (see PLAN.md). Each has deliverables, exit
  criteria, and a checkpoint where the owner reviews.
- Prefer small, reviewable changes. One logical change per commit.
- Each step of a milestone goes on its own branch and ends in a pull request
  that the owner reviews and merges. Merging does not wait for CI: check the
  result on the next push, and fix any failure in a new pull request.
- CI is a broad, shallow check across platforms. It never runs the real model
  or downloads weights. Deep checks (SmolLM2, fixtures, full comparisons) are
  `slow` tests that skip without the weights; run them locally before a pull
  request.
- Any review of changes (a PR, a branch, a diff) follows
  [REVIEW.md](REVIEW.md), and so does the final check before declaring a step
  done.
- A pull request that changes behaviour comes with tests for it. A PR without
  tests is incomplete unless tests make no sense for it (docs, a one-line
  config change). Use judgement: test what could break, not trivia.
- When you make a decision that is not already in PLAN.md, write it down:
  either as a proposed decision for the owner to confirm, or as an open
  question.
- Keep PLAN.md current: mark finished deliverables, update open questions, and
  add decision-log entries (dated, never rewritten).
- Keep docs plain and direct. No hype. No claims we cannot back with a
  reproducible measurement.
- When unsure about intent, ask. When unsure about a fact (a datasheet number,
  a tool behaviour), check it or mark it "to verify".

## Repository layout

- `docs/`: architecture, numerics spec, perf model notes.
- `model/`: Python golden model and performance model.
- `rtl/`: synthesizable SystemVerilog only (no testbenches, no platform code).
- `verif/`: cocotb testbenches and shared test utilities.
- `sw/`: host runtime, driver/platform glue, PyTorch integration.
- `platforms/f2/`: thin wrapper and build flow for F2.
- `scripts/`: developer tooling.

## Licensing

- Hardware (`rtl/`, `platforms/`): Solderpad Hardware License v2.1. Every
  file starts with `SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1`.
- Everything else: Apache-2.0. Every source file starts with
  `SPDX-License-Identifier: Apache-2.0`.
- Copyright line: `Copyright <year> The Cabosse Authors`.
- Never copy code from elsewhere without checking that its license is
  compatible, and record where it came from in a README next to the copied
  files, which keep their own headers.

## Coding conventions

These are proposed in M0 and will be confirmed when the first code is written
(M1 for Python, M3 for SystemVerilog).

### SystemVerilog

- Base style: the lowRISC Verilog style guide, unless this file says
  otherwise.
- One module per file. The file name matches the module name (`snake_case`).
  A package goes in its own file, `<name>_pkg.sv`.
- No `` `default_nettype none ``: with it, the standard requires `input
  wire logic` ports, and Vivado enforces that. Verilator's `-Wall` lint
  (`IMPLICIT`) catches nets created by mistake instead.
- Clock `clk_i`, active-low synchronous reset `rst_ni`. Port suffixes `_i` /
  `_o`. Use `_q` / `_d` for register output / next value.
- `always_ff` for flops, `always_comb` for combinational logic. No latches.
  No `initial` blocks in synthesizable code.
- Valid/ready handshakes on all streaming interfaces. Data must not change
  while `valid && !ready`. Exception: a fixed-latency arithmetic pipeline
  (`fp_add`, `fp_mul_add`, ...) has `valid` only and never stalls. A stall
  there would need an enable on every register, including the accumulator
  loop; the block that uses it handles backpressure at its own boundary.
- Parameters for widths and counts (`L` lanes, etc.). No magic numbers.
- Synthesizable by Verilator, Yosys, and Vivado. No vendor primitives in
  `rtl/`. They belong in `platforms/`.
- Must pass `verilator --lint-only -Wall` with no warnings, or with
  waivers that each have a comment.

### Python (golden model, testbenches, runtime)

- Python 3.14, type hints on public functions.
- Formatting and linting with `ruff` (settings in `pyproject.toml`). Run
  `ruff format . && ruff check .` before committing.
- Golden model: explicit dtypes everywhere. Never let float64 slip into a
  computation that models hardware. Bit-level operations on `uint16` /
  `uint32` views are preferred over implicit casts.
- Keep functions small and named after the hardware operation they model.
- Module-level constants go at the top of the file, after the imports, not
  between functions. Exceptions: a value that needs a class or function
  defined in the same file, and a test's parameter table, which sits right
  above the test that uses it.
- Put units in names when they are not obvious (`n_bytes`, `n_cycles`,
  `bw_bytes_per_s`).
- Tests with `pytest` (model) and cocotb (RTL). Fixed random seeds, printed on
  failure.

### C++ (Verilator test drivers, `verif/bulk/`)

- C++23, built with `-std=c++23`. Prefer the standard library's modern
  pieces (`std::span`, ranges, `std::print`, concepts) to C-style code.
- Doxygen comments (`///`, `@brief`, `@tparam`, `@param`, `@return`) on
  every file, type and function.

### Docs

- Markdown. Plain sentences.
