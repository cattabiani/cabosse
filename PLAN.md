# Cabosse plan

This is a living document. Change it when we learn something. Record each
decision in the [decision log](#decision-log) and keep each open question in
[open questions](#open-questions) until it is resolved. Hardware terms are
defined in [docs/glossary.md](docs/glossary.md).

Last updated: 2026-10-01 (initial draft, M0).

---

## 1. Goal

Build an open-source accelerator for LLM inference: chip design, verification,
and host software. Anyone with the listed tools should be able to rebuild it
and reproduce our numbers. The path:

1. **Simulation:** a full decode step of a small Llama-style model runs in RTL
   simulation and matches a bit-exact Python golden model.
2. **FPGA:** the same RTL runs on AWS F2 as a PCIe device. We measure it
   against the hackathon baseline (Gemmini, 9.0 tokens/s on stories260K) and
   against our own memory-bandwidth bound. Then it runs on a cheap FPGA with
   a fully open toolchain (Lattice ECP5).
3. **ASIC:** a small slice of the design is taken through an open PDK flow to
   a clean, tapeout-ready layout. If a shuttle is affordable and available,
   we submit it.

Every published number must come with the commands that reproduce it.

## 2. Non-goals

These are out of scope for now. Revisit them only through the decision log.

- Training or backpropagation.
- Batched or multi-user serving. The design target is one sequence at a time.
- Fast prefill. To begin with, the prompt is processed one token at a time
  using the decode path.
- Competing with commercial GPUs on absolute speed. The comparison points are
  our own baseline and our own bandwidth bound.
- General programmability (SIMT, graphics, arbitrary kernels). "Open GPU/TPU"
  describes the role this chip plays, not its architecture. It is a
  fixed-function LLM decode engine driven by a command stream.
- Arbitrary model architectures. Llama-style decoder-only transformers only
  (RMSNorm, RoPE, GQA, SwiGLU).
- Low-precision activations or KV cache (FP8, int8 activations). Weight-only
  quantization is a possible extension (M10).
- A fully open toolchain on F2. The F2 flow needs AMD Vivado, which is
  proprietary. The fully open flow is the ECP5 path (M11) and the ASIC path
  (M12).

## 3. Decision log

Format: `D-NNN (date) — decision. Why. Consequences.` Each decision is written
once and not edited afterwards. To reverse one, add a new entry that says it
supersedes the old one.

**D-001 (2026-10-01) — Project scope and path.** Open-source LLM inference
accelerator. Path: simulation → AWS F2 FPGA → small chip on an open-source
process via a shuttle run. *Why:* each step builds confidence for the next one,
and each produces something others can reproduce.

**D-002 (2026-10-01) — Numerics: BF16 × BF16 multiply, FP32 accumulate, FP32
vector unit.** Models load from their official published checkpoints with no
conversion or calibration step. Weight-only quantization (int8/int4 weights,
dequantized to BF16) is a possible later extension. The architecture must not
rule it out. *Why:* BF16 has the same exponent range as FP32, so most published
checkpoints run in it without calibration. *Consequence:* a BF16×BF16 product
has at most 16 significant bits and fits exactly in FP32 (the only exception is
overflow or underflow at the extremes of the exponent range). So the only
rounding in a dot product happens in the FP32 additions, and the summation
order fully determines the result. The golden model must reproduce that order
exactly.

**D-003 (2026-10-01) — Workload: single-stream LLM decode, which is
memory-bandwidth bound.** Core blocks: a streaming matrix-vector engine (many
dot-product lanes fed from memory), a vector unit (RMSNorm, softmax,
SiLU/SwiGLU, RoPE, residual adds), DMA engines, on-chip buffers, and a small
controller that executes a command stream. All activations and the KV cache
stay in accelerator memory. The host is not involved in individual operations.
*Why:* in decode, every weight is read once per token and used for only one
multiply-accumulate, so memory bandwidth limits speed. The baseline (D-008)
also showed that orchestration and data movement dominate.

**D-004 (2026-10-01) — System: host CPU + accelerator as a PCIe device, AWS F2
first.** The accelerator core exposes one standard interface: AXI-Lite for
control and AXI for memory. Thin platform wrappers target F2, a Lattice ECP5
board (Yosys/nextpnr), and an ASIC. *Why:* one core and several cheap wrappers
is less work than separate designs, and it lets the same RTL run on every
target.

**D-005 (2026-10-01) — Languages and tools.** SystemVerilog for hardware (no
Chisel). Verilator for simulation. cocotb (Python) for testbenches. numpy for
the golden model. Development happens on Linux.

**D-006 (2026-10-01) — Verification philosophy.** A Python golden model that
reproduces the hardware's exact formats and summation order is the bit-exact
reference for RTL. Comparisons against PyTorch use stated tolerances.

**D-007 (2026-10-01) — Model ladder.** stories260K (Karpathy, llama2.c) →
stories15M → SmolLM2-135M / Qwen2.5-0.5B.

**D-008 (2026-10-01) — Baseline and prior work.** At a hackathon we ran
stories260K on Gemmini + Rocket on AWS F2 at 29.8 MHz and got 9.0 tokens/s.
Time per token broke down as: 4% matmul, 13% copying data around each matmul,
49% copying and quantizing the KV cache, ~17% softmax/SwiGLU/RoPE on the CPU.
*Lesson:* orchestration and data movement dominate, not the matmul. Gemmini is
not reused.

**D-009 (2026-10-01) — Licenses.** Hardware (`rtl/`, `platforms/`): Solderpad
Hardware License v2.1 (`Apache-2.0 WITH SHL-2.1`). Everything else: Apache-2.0.
*Why:* the goal is wide reuse and reproducibility, not stopping closed forks.
Permissive licenses dominate open processor/accelerator RTL (Gemmini and the
Berkeley stack: BSD-3; lowRISC: Apache-2.0; OpenHW: Solderpad), so reusing
or contributing IP is easy. SHL-2.1 is Apache-2.0 plus explicit coverage of
hardware rights (mask works, design rights), which matters for the ASIC step.
Recipients may treat SHL-2.1 files as plain Apache-2.0. Reciprocal licenses
(CERN-OHL-W/S) were considered and rejected because they could put off
industry and academic contributors.

**D-010 (2026-10-01) — D-005 reconfirmed: SystemVerilog, not Amaranth or
Chisel.** Amaranth and Chisel are still RTL (same abstraction level). What they
add is better metaprogramming. *Why we keep SystemVerilog:* every tool in our
flows reads it directly, so we debug the code we wrote, not generated Verilog.
It lets us reuse open IP (CVFPU, lowRISC blocks) directly, and it has the
largest community. Its verification half (UVM) is not needed because cocotb
does that job. HLS was ruled out because we need cycle-exact control of
pipelines and memory traffic.

**D-011 (2026-10-01) — Rounding points (resolves Q-02).** Values are rounded
from FP32 to BF16 (round to nearest, ties to even) in these places only: the
input vector `x` of every matrix-vector product, the query `q` before q·Kᵀ, the
attention probabilities `p` before p·V, and the K/V values written to the KV
cache. Weights are BF16 as loaded. The residual stream, everything inside the
vector unit, and the logits stay FP32. *Why:* the multipliers take only BF16
inputs. Keeping everything else FP32 limits the precision loss to the places
where it is required. A BF16 KV cache halves its memory.

**D-012 (2026-10-01) — Special values (resolves Q-03).** Subnormals are
flushed to zero on inputs and outputs of every FP operation (FTZ), keeping the
sign. Infinities propagate per IEEE 754. Every NaN result is the single
canonical NaN (BF16 `0x7FC0`, FP32 `0x7FC00000`). *Why:* subnormal support is
expensive in hardware, and real weights and activations almost never contain
subnormals. The golden model emulates FTZ explicitly, because numpy keeps
subnormals. M1 measures whether FTZ changes any result.

**D-013 (2026-10-01) — Python environment (resolves Q-07).** Python 3.14
(installed on the dev machine; torch, cocotb and ml_dtypes all ship 3.14
wheels, torch does not yet ship 3.15 wheels). One virtual environment in
`.venv/` created with the standard `venv` module. It is activated
automatically by `direnv` through a committed `.envrc`, which also sets
project environment variables. Dependencies are pinned in a requirements
file.

## 4. Milestones

Every milestone ends at a **checkpoint**: work stops, and the owner reviews it
before the next milestone begins. Exit criteria must be shown with
commands that can be rerun, not just described.

| #   | Milestone                                          | Runs on              |
|-----|----------------------------------------------------|----------------------|
| M0  | Repo foundation                                    | —                    |
| M1  | Numerics spec + golden model (stories260K)         | Python               |
| M2  | Architecture spec + performance model              | docs + Python        |
| M3  | Toolchain + floating-point units                   | Verilator            |
| M4  | Dot-product lane                                   | Verilator            |
| M5  | Matrix-vector engine + simulated memory            | Verilator            |
| M6  | Vector unit                                        | Verilator            |
| M7  | Controller + command stream: full token in sim     | Verilator            |
| M8  | F2 platform bring-up (no model)                    | AWS F2               |
| M9  | stories260K on F2, measured                        | AWS F2               |
| M10 | Larger models (+ optional weight-only quant)       | Sim + AWS F2         |
| M11 | Fully open FPGA flow (ECP5)                        | ECP5 board           |
| M12 | ASIC slice on an open PDK                          | ASIC flow            |

**Changes from the original outline, and why:**

- **M1 now produces a written numerics spec (`docs/numerics.md`) alongside the
  golden model.** "Bit-exact" only means something against a written
  definition of rounding, special values, and summation order.
- **M1 and M2 are coupled. The golden model is parameterized for this
  reason.** The summation order depends on the lane structure (number of
  lanes, adder pipeline depth), and that is decided in M2. So M1 builds the
  golden model with summation order as a parameter, and M2 fixes the values.
  The alternative was to do M2 first, but then we would be designing hardware
  before we have a working reference.
- **M3 is split into "toolchain + FP units" (M3) and "dot-product lane" (M4).**
  An IEEE-correct FP32 adder is the hardest numerics RTL in the project and
  needs its own review. Accumulating with a pipelined adder also creates a
  feedback-loop problem that changes the summation order (see M4). That
  deserves its own checkpoint.
- **F2 is split into bring-up (M8) and model run (M9).** F2 has a cost per hour
  and long build times. Proving that the host can talk to the board (register
  read/write, DMA, memory access) before adding the accelerator keeps
  platform problems separate from design problems.

---

### M0 — Repo foundation

**Goal:** a repository others can read, and a plan we agree on.

**Deliverables:** `git init`, `.gitignore`, `README.md`, folder skeleton with
per-folder READMEs, `AGENTS.md` + `CLAUDE.md`, `docs/glossary.md`, this
`PLAN.md`, `LICENSE` (Apache-2.0) and `LICENSE-HARDWARE` (SHL-2.1).

**Exit criteria:** owner has reviewed the plan; license chosen (done: D-009);
first commit made by the owner or with their explicit OK.

**Checkpoint:** review PLAN.md, especially the milestone split and the open
questions marked **[needed for M1]**.

**Risks / open questions:** none remaining for M0.

---

### M1 — Numerics spec and golden model (stories260K)

**Goal:** a numpy model of stories260K decode in hardware formats. Its error
against PyTorch is measured, and every rounding point is written down.

**Deliverables:**

- `docs/numerics.md` v0. It defines:
  - the formats at every tensor boundary (weights, activations, KV cache,
    logits);
  - where FP32 values are rounded to BF16, and with which rounding mode;
  - subnormal handling (keep subnormals, or flush to zero);
  - NaN/Inf behaviour;
  - summation order for dot products, as a function of parameters (lanes `L`,
    interleaved accumulators `A`, final reduction tree shape);
  - how each nonlinear function is computed (exp, reciprocal, rsqrt, SiLU,
    sin/cos for RoPE). The golden model must use the same algorithm as the
    hardware, not `np.exp`.
- `model/` golden model: loads the stories260K checkpoint as published (FP32
  weights, cast to BF16 on load as the numerics decision requires). It runs
  decode one token at a time with a KV cache.
- A comparison harness against a PyTorch FP32 reference (llama2.c's
  `model.py` or an equivalent written in plain PyTorch), plus PyTorch in
  BF16 as a second reference point.
- A short results note: per-layer error statistics and the end-to-end metrics
  below.

**Exit criteria (proposed numbers, to confirm at the checkpoint):**

- Using teacher forcing (both models are fed the same reference token
  sequence, so one early mismatch cannot snowball), on ≥ 10 prompts × 256
  positions:
  - top-1 token agreement with PyTorch FP32 ≥ 99%;
  - max |logit error| and relative error per layer are reported. Thresholds are
    set from the observed PyTorch-BF16 vs FP32 error: the golden model should
    not be meaningfully worse than PyTorch's own BF16 run.
- Greedy decode of 64 tokens from 3 fixed prompts produces readable text, and
  the output is recorded as a regression fixture.
- The golden model is deterministic: two runs are bit-identical, and changing
  only the summation-order parameters changes the results the way the spec
  predicts.
- Runtime for one stories260K token: seconds, not minutes. This model will be
  run constantly.

**Checkpoint:** review `docs/numerics.md`, the error report, and the list of
rounding points.

**Risks / open questions:**

- "No conversion": stories260K ships as an FP32 llama2.c checkpoint, not as an
  HF-transformers model. We treat reading that format plus an FP32→BF16 cast
  as "loading", not conversion (Q-05).
- Nonlinear functions: approximations that are cheap in hardware might hurt
  accuracy, while accurate ones might be expensive in area. The spec should
  allow swapping the algorithm without touching the rest (Q-08).
- numpy has no native BF16. Options: bit manipulation on `uint16`/`uint32`
  views, or the `ml_dtypes` package (Q-09).
- numpy float32 arithmetic is IEEE round-to-nearest-even and keeps subnormals.
  If the hardware flushes subnormals to zero, the golden model must emulate
  that explicitly.

---

### M2 — Architecture spec and performance model

**Goal:** a written design detailed enough that M3–M7 can be implemented
without major redesign, plus a performance model that says what tokens/s to
expect and why.

**Deliverables:**

- `docs/architecture.md`:
  - block diagram: matrix-vector engine, vector unit, DMA engines, on-chip
    buffers, controller, AXI-Lite register file, AXI memory port(s);
  - interfaces: AXI-Lite register map (ID/version, control, status, doorbell,
    command-stream base address, error codes); AXI data width(s) and burst
    behaviour;
  - memory map: weights, KV cache, activations, command buffers, scratch;
  - data layouts: weight matrix tiling for streaming, KV cache layout per
    layer/head, and how attention's two products (q·Kᵀ and p·V) map onto the
    engine (see Q-11);
  - command format: opcodes, fields, addressing (absolute vs
    base-register-relative), how the per-token position `pos` gets into
    addresses and RoPE, completion/interrupt;
  - one stories260K token written out as a command sequence (on paper);
  - parameter choices: lanes `L`, accumulators per lane `A`, buffer sizes,
    clock target per platform. These fix the summation-order parameters left
    open in M1.
- `model/` performance model (Python), first order:
  - `bytes_per_token ≈ 2·N_weights_streamed + KV_bytes_read(pos) +
    activation traffic`
  - `t_mem = bytes_per_token / BW_effective`
  - `t_compute = MACs_per_token / (L · f_clk)`
  - `t_overhead = n_commands · per_command_overhead`
  - `tokens/s ≈ 1 / (max(t_mem, t_compute) + t_overhead)`, with and without
    overlap between commands.
  - Tables for each model in the ladder × each platform (F2, ECP5).
- `docs/perf-model.md`: assumptions, and the numbers M9 will be judged
  against.

**Exit criteria:**

- The command sequence for one stories260K token is complete: every operation
  in the golden model maps to a command. Nothing is handled by the host
  mid-token.
- The perf model gives a predicted tokens/s for stories260K on F2 at the chosen
  clock, and states which term dominates.
- The golden model is updated to use the fixed summation parameters and still
  passes the M1 criteria.
- The M9 target (tokens/s on F2) is written down. The provisional floor is
  ≥ 90 tokens/s (10× the baseline). The real target comes from the perf model.

**Checkpoint:** design review of architecture.md and the perf tables. This is
the most important checkpoint in the plan: redesigning later costs much more.

**Risks / open questions:**

- Model size vs on-chip memory: stories260K is ~0.5 MB in BF16 and fits easily
  on-chip. stories15M (~30 MB) is borderline on F2. SmolLM2-135M (~270 MB) is
  the first model that really needs HBM bandwidth. We should stream weights
  from external memory from the start, so that the real path is exercised
  even when the model would fit on-chip (Q-10).
- What the F2 shell exposes for HBM: number of AXI ports, width, and clock.
  This limits `BW_effective`. To verify from AWS documentation, without
  creating cloud resources (Q-13).
- Controller design: fixed-function sequencer vs a small RISC-V core (Q-12).
- Leaving room for weight-only quantization: the weight path should have a
  place for a dequant stage between memory and the lanes. Group-scale layout
  is a later decision.

---

### M3 — Toolchain and floating-point units

**Goal:** the hardware toolchain works locally and in CI, and the two
arithmetic primitives are proven correct.

**Deliverables:**

- Setup instructions for Linux: Verilator, cocotb, Python env.
  Pinned versions.
- A test runner (one command runs all cocotb tests) and CI on every push.
  The CI provider is decided at the checkpoint (Q-15).
- A lint gate: Verilator `--lint-only -Wall` clean.
- RTL: BF16×BF16 → FP32 multiplier; FP32 adder (round-to-nearest-even,
  subnormal policy as in the spec, NaN/Inf as in the spec).
- cocotb tests comparing both units against the golden model's bit-level
  functions.

**Exit criteria:**

- Multiplier: bit-exact on all 2³² BF16×BF16 input pairs if this finishes in
  reasonable time in Verilator (estimate first), otherwise ≥ 10⁸ random pairs
  plus every special-value class (±0, subnormals, ±Inf, NaN, max/min normal).
- Adder: bit-exact on ≥ 10⁸ random pairs plus directed corner cases:
  cancellation, rounding ties, overflow to Inf, results that become
  subnormal, operands with very different exponents.
- Pipeline latency and throughput (one result per cycle) documented.
- Yosys synthesis of each unit succeeds as a smoke test, with LUT/DSP
  counts reported. This is a portability check, not an optimization target.

**Checkpoint:** review the FP unit design, the test coverage, and the
toolchain setup on the owner's machine.

**Risks / open questions:**

- Write our own FP units or use an existing open one (e.g. PULP's
  FPnew/CVFPU, SystemVerilog, Solderpad-licensed) (Q-14). Our own is more
  work but easier to explain and match bit for bit. An existing one is
  proven, but brings its license and its features.
- Exhaustive multiplier testing may be too slow in simulation. Fallback:
  exhaustive testing in a C++ Verilator harness without cocotb, documented.

---

### M4 — Dot-product lane

**Goal:** one lane that streams BF16 weight/activation pairs and produces an
FP32 dot product bit-identical to the golden model.

**Deliverables:** lane RTL (multiplier → accumulator(s) → final reduction),
streaming input interface with valid/ready handshake, cocotb tests.

**The key design issue:** a pipelined FP32 adder takes several cycles (`A`)
to produce a sum. To accept one product per cycle, a lane must keep `A`
partial sums in rotation and combine them at the end. This fixes the summation
order. The golden model must use exactly the same scheme (parameter `A` from
M1/M2).

**Exit criteria:**

- Bit-exact against the golden model on: random vectors of length 1…4096
  (including lengths not divisible by `A`), the actual row lengths of
  stories260K, and adversarial inputs (large cancellation, mixed
  magnitudes).
- Sustains 1 MAC/cycle per lane in simulation, with stalls on the input
  handled correctly (randomized backpressure tests).
- Report the error of the lane's summation order versus float64
  dot products, as an accuracy sanity check (not a pass/fail gate).

**Checkpoint:** review lane microarchitecture and the summation-order
documentation.

**Risks / open questions:** whether to use a different accumulation scheme
(e.g. wider internal accumulator, or Kulisch-style exact accumulation). That
would change D-002 and need a new decision entry.

---

### M5 — Matrix-vector engine with simulated memory

**Goal:** `y = W·x` for BF16 `W` in memory, BF16 `x` on-chip, FP32 `y`.
Weights stream from an AXI memory model at the rate the lanes consume them.

**Deliverables:**

- Engine RTL: `L` lanes, input-vector buffer, output handling, weight DMA (AXI
  read master with bursts), tiling for matrices larger than one pass.
- AXI memory model for simulation with configurable latency and bandwidth
  (cocotb or SV).
- Tests over the real matrix shapes of stories260K and stories15M.

**Exit criteria:**

- Bit-exact against the golden model for every stories260K weight matrix
  (including the classifier) with real weights and real activations taken
  from golden-model runs.
- Measured utilization: with the memory model set to `BW` bytes/cycle, the
  engine's achieved bytes/cycle is ≥ 90% of `min(BW, 2·L)` for large
  matrices. Overhead per matrix (startup and drain cycles) is reported and
  fed back into the perf model.

**Checkpoint:** review engine design and utilization numbers vs the perf
model.

**Risks / open questions:** AXI data width vs lanes (e.g. a 512-bit bus carries
32 BF16 values per beat); burst sizes and alignment rules; whether the
input vector `x` is BF16 (rounded from FP32) or the engine accepts FP32 `x`
and rounds internally. This is decided in the M1 spec.

---

### M6 — Vector unit

**Goal:** FP32 elementwise and reduction operations needed by decode.

**Deliverables:** RTL for RMSNorm (sum of squares, rsqrt, scale), softmax
(max, exp, sum, reciprocal, scale), SiLU and the SwiGLU product, RoPE
(rotation using sin/cos per the spec), residual add, FP32→BF16 rounding, and
writing results to the KV cache. cocotb tests per operation.

**Exit criteria:**

- Each operation is bit-exact against the golden model on random inputs and
  on real activations from stories260K.
- Throughput per operation documented (elements/cycle). With the M5 numbers,
  the perf model shows the vector unit takes < 10% of token time for
  stories15M (if not, record why and decide).

**Checkpoint:** review the nonlinear-function implementations (area vs
accuracy) and the accuracy numbers.

**Risks / open questions:** exp/rsqrt/reciprocal implementations (Q-08);
whether RoPE sin/cos values are precomputed tables in memory or computed on
chip; reductions (sum, max) add another summation order to specify.

---

### M7 — Controller and command stream: a full token in simulation

**Goal:** the host writes a command buffer once. Each token, it writes the
token ID and starts the accelerator, and reads back the result. Everything in
between runs on the accelerator.

**Deliverables:**

- Controller RTL: fetches commands over AXI, dispatches them to engine,
  vector unit, and DMA, tracks dependencies, reports completion and errors.
- AXI-Lite register file matching architecture.md.
- Top level with one AXI-Lite slave and the AXI master port(s).
- `sw/` runtime for simulation: lays out weights, builds the command stream,
  runs decode through the same register interface the hardware will use.
- End-to-end cocotb test: stories260K greedy decode.

**Exit criteria:**

- Bit-exact logits against the golden model for every position of greedy
  decode, 3 prompts × 64 tokens. The generated text equals the M1 fixture.
- No host interaction during a token beyond: write token ID / position, start,
  wait for done, read logits (or argmax).
- Simulated cycles per token are reported and compared with the perf model,
  with differences explained. Projected tokens/s at the planned F2 clock.
- Lint clean. Synthesizes with Yosys (generic) as a portability smoke test.

**Checkpoint:** demo of end-to-end simulation; review the cycle breakdown
(where do the cycles go: matvec, vector, DMA, controller stalls). Compare
the breakdown with the baseline's profile.

**Risks / open questions:** stories15M in simulation is ~60× more work per
token than stories260K. Check whether running it at M7 is practical, or
whether it waits until M10.

---

### M8 — AWS F2 platform bring-up (no model)

**Goal:** prove the F2 path with no accelerator logic: host ↔ PCIe ↔ AXI-Lite
registers, host → device DMA, device logic ↔ HBM/DDR.

**Deliverables:** `platforms/f2/` wrapper with a trivial test design (register
file + memory tester), Vivado build scripts, host-side test program using the
`sw/` driver interface, cost and time log (build hours, instance hours, $).

**Exit criteria:**

- Register read/write from the host works.
- Host can DMA ≥ 256 MB to and from device memory, verified by checksum, and
  the throughput is measured.
- Device-side memory read bandwidth measured per HBM/DDR port and in total,
  then compared with the datasheet and fed into the perf model.
- The whole flow from a clean checkout is documented step by step.

**Checkpoint:** review cost log and measured bandwidth. **Before any cloud
resource is created, the owner approves the budget and the instance types.**

**Risks / open questions:** Vivado version required by the AWS F2 development
kit; where builds run (cloud build instance vs the local Linux machine)
(Q-04); AFI creation turnaround; cost
control (Q-16).

---

### M9 — stories260K on F2, measured

**Goal:** the M7 design runs on F2, gives the same answers as simulation, and
is measured.

**Deliverables:** F2 build of the accelerator, runtime port, benchmark
script, results write-up in `docs/` with commands to reproduce.

**Exit criteria:**

- Bit-exact against the golden model on the same 3 × 64-token runs as M7.
- Tokens/s measured over ≥ 1000 tokens and reported against: (a) the 9.0
  tokens/s Gemmini baseline, (b) the perf-model prediction, (c) the
  memory-bandwidth bound. Provisional floor: ≥ 90 tokens/s. The real target
  is set in M2.
- Clock frequency achieved (timing closure) and resource use (LUT, FF, DSP,
  BRAM, URAM) reported.
- A time breakdown per token from on-device cycle counters, compared with the
  baseline profile.

**Checkpoint:** review results; decide between scaling up (M10), going open
(M11), or improving performance.

**Risks / open questions:** timing closure at the target clock; PCIe round
trip per token may become significant at high tokens/s for tiny models. If
so, document it as host overhead rather than hide it.

---

### M10 — Larger models and optional weight-only quantization

**Goal:** climb the model ladder. stories15M first, then SmolLM2-135M and/or
Qwen2.5-0.5B. At these sizes HBM bandwidth becomes the limit, which is the
regime this design exists for.

**Deliverables:** golden-model support for the new models (tied embeddings,
QKV biases in Qwen2.5, different vocab sizes and head counts, longer
contexts). RTL/runtime changes if needed. F2 measurements. Optional:
int8/int4 weight-only quantization with BF16 dequant, as a separate
sub-milestone with its own checkpoint.

**Exit criteria:**

- Each model meets the M1-style PyTorch tolerance criteria in the golden model
  and is bit-exact against the golden model on F2.
- For SmolLM2-135M: measured tokens/s ≥ 50% of the bandwidth bound for the
  memory bandwidth the design actually uses. The percentage is to be
  confirmed after M8 measurements.

**Checkpoint:** after each model, and before starting quantization.

**Risks / open questions:** whether `docs/numerics.md` changes are needed for
larger hidden sizes (longer dot products → more accumulation error);
tokenizers for the HF models live on the host side; KV cache size at long
contexts.

---

### M11 — Fully open FPGA flow (Lattice ECP5)

**Goal:** the same core, built only with open tools, runs stories260K on a
low-cost board.

**Deliverables:** `platforms/ecp5/` wrapper with a host link (likely
USB/UART-to-AXI, since typical ECP5 boards have no PCIe; Q-17), board memory
controller (SDRAM or similar), Yosys + nextpnr build, a smaller
configuration (fewer lanes) if needed to fit.

**Exit criteria:**

- stories260K bit-exact against the golden model on the board.
- tokens/s measured and compared with the perf model for that board's
  memory bandwidth.
- The full build from source uses only open-source tools, and is documented.

**Checkpoint:** review; decide on the ASIC slice scope.

**Risks / open questions:** board choice (Q-17); fitting the design into the
board's LUTs and DSP blocks; memory controller IP availability and license.
This milestone could move earlier, before F2, as a cheap first hardware target
(Q-18).

---

### M12 — ASIC slice on an open PDK

**Goal:** take a small, self-contained part of the design (e.g. a few
dot-product lanes plus a small buffer and a simple test interface) through an
open-source ASIC flow to a layout that passes signoff checks.

**Deliverables:** `platforms/asic/` flow config, a test wrapper (e.g. SPI or
a simple serial interface), gate-level simulation, signoff reports.

**Exit criteria:**

- DRC and LVS clean; static timing passes at the target clock in the PDK's
  corners.
- Gate-level simulation of the slice is bit-exact against the golden model on
  the M4 test vectors.
- Area and power estimates reported.
- Submitting to a shuttle is a separate decision (cost, schedule) and is made
  at this checkpoint.

**Checkpoint:** review signoff results; go/no-go on shuttle submission.

**Risks / open questions:** PDK choice and shuttle availability (Q-19); FP
unit area in an older process node; I/O-limited test interface.

---

## 5. Open questions

Questions marked **[needed for M1]** block the next milestone.

| ID   | Question | Notes / current leaning |
|------|----------|--------------------------|
| Q-01 | ~~License for hardware and software.~~ | **Resolved: D-009.** |
| Q-02 | ~~Where are FP32 values rounded to BF16?~~ | **Resolved: D-011.** |
| Q-03 | ~~Subnormals, NaN/Inf behaviour.~~ | **Resolved: D-012.** |
| Q-04 | Where do Vivado builds for F2 run: the local Linux machine or an AWS build instance? | Local needs the Vivado version required by the AWS F2 kit and a license that covers the F2 device. Decide before M8. |
| Q-05 | **[needed for M1]** Is "read llama2.c `.bin` + FP32→BF16 cast" acceptable as "no conversion" for the stories models? | Leaning yes. Treated as loading, not a separate conversion step. |
| Q-06 | **[needed for M1]** PyTorch reference: llama2.c's `model.py`, HF transformers, or our own minimal PyTorch? | Leaning: our own minimal PyTorch reference, cross-checked against llama2.c's `run.c` output. |
| Q-07 | ~~Python version and environment.~~ | **Resolved: D-013.** |
| Q-08 | Algorithms for exp, reciprocal, rsqrt, sigmoid, sin/cos. | Decide in M1 (spec) with area estimates in M2/M6. |
| Q-09 | BF16 in numpy: own bit manipulation vs `ml_dtypes` dependency. | Leaning: own bit-level helpers (small, explicit), cross-checked with `ml_dtypes` in tests. |
| Q-10 | Always stream weights from external memory, or cache small models on-chip? | Leaning: always stream. On-chip caching can be an optimization later. |
| Q-11 | Attention mapping: store V transposed (q·Kᵀ and p·V are both row-dot products, but appending a token becomes a strided write), or give the engine a second mode that computes `Σ pᵢ·vᵢ`? | Decide in M2. |
| Q-12 | Controller: fixed-function command sequencer or a small RISC-V core? | Leaning: fixed-function sequencer, with base registers so one command list serves every token. |
| Q-13 | F2 shell specifics: HBM/DDR ports exposed to custom logic, widths, clocks, PCIe DMA mechanism. | Check AWS docs during M2 (reading only, no cloud resources). |
| Q-14 | Write our own FP units or use an existing open IP (e.g. CVFPU/FPnew)? | Decide at M3 start. |
| Q-15 | CI provider (GitHub Actions or other) and where the repo is hosted. | Decide at M3. |
| Q-16 | F2 budget per month and cost controls. | Owner decision before M8. |
| Q-17 | ECP5 board (e.g. ULX3S, OrangeCrab, or other) and host link. | Decide before M11. Depends on memory size/bandwidth and price. |
| Q-18 | Move ECP5 earlier (before F2) as the first hardware target? | It costs no cloud time and the flow is fully open, but the board has no PCIe and much less bandwidth. Decide at M7 checkpoint. |
| Q-19 | Open PDK and shuttle (e.g. SkyWater SKY130, GF180MCU, IHP SG13G2; Tiny Tapeout or other MPW programs). | Shuttle availability changes often. Survey at M11 checkpoint. |
| Q-20 | Prefill: is single-token prefill acceptable long-term? | Fine for now (non-goal). Revisit after M10. |
| Q-21 | Sampling: argmax on device, or return logits to the host? | Leaning: both supported. Argmax on device keeps PCIe traffic tiny. |
