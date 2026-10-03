# Cabosse plan

A living document: what we are going to do and why. Details belong in `docs/`
once they exist. IDs: **M** = milestone, **D** = decision, **Q** = open
question.

Last updated: 2026-10-02.

## Goal

An open-source accelerator for LLM inference (chip design, verification, host
software) that anyone can rebuild, with every published number reproducible.
The path:

1. **Simulation:** a full decode step of a real model runs in RTL simulation
   and matches a bit-exact Python golden model.
2. **FPGA:** the same RTL runs on AWS F2 as a PCIe device, measured against
   our memory-bandwidth bound and the Gemmini hackathon baseline. Then it runs
   on a cheap FPGA with a fully open toolchain (Lattice ECP5).
3. **ASIC:** a small slice of the design passes signoff on an open PDK. If
   a shuttle is affordable, we submit it.

## Non-goals (for now)

- Training.
- Batched or multi-user serving: one sequence at a time.
- Fast prefill: the prompt goes through the decode path one token at a time.
- Beating commercial GPUs. We compare against our own bandwidth bound and our
  baseline.
- General programmability. It is a fixed-function decode engine driven by
  a command stream, for Llama-style models (RMSNorm, RoPE, GQA, SwiGLU).
- Low-precision activations or KV cache. Weight-only quantization is a
  possible extension.
- A fully open flow on F2, because Vivado is proprietary. The open flow is
  ECP5 and the ASIC.

## Decision log

Entries are added, not rewritten. A new entry supersedes an old one.

**D-001 (2026-10-01) — Scope.** Open-source LLM inference accelerator. Path: simulation →
AWS F2 → small chip on an open process via a shuttle. *Why:* each step is
reproducible and builds confidence for the next.

**D-002 (2026-10-01) — Numerics: BF16 × BF16 multiply, FP32 accumulate, FP32 vector
unit.** Models load from official checkpoints without calibration.
Weight-only int8/int4 (dequantized to BF16) must remain possible. *Why:*
BF16 has FP32's range, so checkpoints run without calibration. A BF16×BF16
product is exact in FP32, so all rounding happens in the additions, and the
summation order fully determines the result.

**D-003 (2026-10-01) — Workload: single-stream decode, which is bandwidth-bound.** Blocks:
streaming matrix-vector engine, vector unit, DMA, on-chip buffers, a
controller running a command stream. Activations and KV cache stay in
accelerator memory. The host is not involved in individual operations.
*Why:* every weight is read once per token for one MAC. The baseline also
showed that data movement and orchestration dominate.

**D-004 (2026-10-01) — System: host + PCIe accelerator, AWS F2 first.** The core exposes
AXI-Lite (control) and AXI (memory). Thin wrappers target F2, ECP5, and the
ASIC. *Why:* one core runs on every target.

**D-005 (2026-10-01) — Languages and tools.** SystemVerilog, Verilator, cocotb, numpy for
the golden model. Development on Linux.

**D-006 (2026-10-01) — Verification.** The golden model reproduces the hardware's formats
and summation order and is the bit-exact reference. Comparisons with
PyTorch use stated tolerances.

**D-007 (2026-10-01) — Model ladder** (superseded by D-014).

**D-008 (2026-10-01) — Baseline.** Hackathon: stories260K on Gemmini + Rocket, F2, 29.8
MHz, 9.0 tokens/s. Per token: 4% matmul, 13% copying around matmuls, 49%
copying/quantizing the KV cache, ~17% softmax/SwiGLU/RoPE on the CPU.
*Lesson:* orchestration and data movement dominate. Gemmini is not reused.

**D-009 (2026-10-01) — Licenses.** Hardware (`rtl/`, `platforms/`): Solderpad Hardware
License 2.1. Everything else: Apache-2.0. *Why:* the goal is wide reuse, not
blocking closed forks. Permissive licenses are the norm in open processor
RTL, and SHL-2.1 adds explicit coverage of chip layouts.

**D-010 (2026-10-01) — SystemVerilog reconfirmed** (over Amaranth, Chisel, HLS). *Why:*
generators are still RTL and add mainly metaprogramming, at the cost of
debugging generated code. SystemVerilog is read natively by every tool, lets
us reuse open IP, and has the largest community. HLS gives up the
cycle-level control we need.

**D-011 (2026-10-01) — Rounding points.** FP32 values are rounded to BF16 (RNE) only at:
matrix-vector inputs `x`, `q` before q·Kᵀ, `p` before p·V, and KV cache
writes. The residual stream, the vector unit, and the logits stay FP32.
*Why:* the multipliers need BF16 inputs. Everything else keeps full
precision, and a BF16 KV cache halves its memory.

**D-012 (2026-10-01) — Special values** (subnormal part superseded by D-016). Infinities
propagate per IEEE 754. Every NaN becomes the canonical NaN (`0x7FC0` /
`0x7FC00000`).

**D-013 (2026-10-01) — Python environment.** Python 3.14, `.venv/` via `venv`,
pinned `requirements.txt`. (The owner's local `direnv` setup is not part of
the repo.) *Why:* newest Python
with wheels for torch, cocotb, and ml_dtypes.

**D-014 (2026-10-01) — Model ladder** (supersedes D-007; absorbs D-018). Tiny random-weight
Llama configs (fast RTL tests) → **SmolLM2-135M-Instruct** (our model) →
Qwen2.5-0.5B. The reference is Hugging Face `transformers`. *Why:*
SmolLM2-135M-Instruct is an official Llama-architecture chat model, and people
expect a chatbot. Its weights are already in BF16 (269 MB), so it loads with
zero conversion. It is bandwidth-bound, which is what we are designing for.
*Cost:* full-token RTL simulation is slow (~135M MACs per token), and the
model does not fit ECP5 boards.

**D-015 (2026-10-01) — "No conversion" = at most a cast at load time**, as
`transformers` and vLLM do. No offline conversion step. SmolLM2-135M-Instruct
needs no cast.

**D-016 (2026-10-01) — Subnormals are kept (IEEE 754), as in PyTorch.** *Why:* PyTorch
follows IEEE 754 by default on CPU and GPU. Flush-to-zero is an opt-in speed
setting (`torch.set_flush_denormal`). Matching the reference removes one
source of divergence. *Cost:* extra logic in each FP unit, measured in M3.
The golden model keeps a flush-to-zero switch in case that cost turns out to
be too high.

**D-017 (2026-10-01) — Golden model in torch** (supersedes "numpy" in D-005).
*Why:* torch is already needed for the `transformers` reference. That
gives one tensor library and the same dtype semantics as the reference. The
golden model still does every addition explicitly in hardware order. It never
uses `matmul`/`sum`, whose order is unspecified. BF16 rounding is our own
small helper, checked against torch's conversion.

**D-019 (2026-10-01) — Fused multiply-add (FMA) everywhere.** Every
multiply followed by an add is one FMA with a single rounding: in the
dot-product lanes (BF16×BF16 + FP32) and in the vector unit (FP32). *Why:*
it is never less accurate than a separate multiply and add, and one unit does
the work of two. In the lanes it costs nothing, because the BF16×BF16 product
is exact anyway.

**D-020 (2026-10-01) — Nonlinear functions.** RoPE sin/cos are FP32 tables
computed by the host as `transformers` does. 1/√x and 1/x use a bit-trick
first guess (Quake 3 style: magic constant minus the shifted bit pattern)
plus Newton steps. exp uses range reduction: 2ⁱ written into the exponent
field, and a short polynomial for 2ᶠ. Constants, number of Newton steps, and
polynomial degree are parameters fixed by measurement in M1. *Why:* no
lookup tables, only integer ops and FP32 FMAs that the vector unit already
has. Results are rounded to BF16 before every multiplier input, so
BF16-level accuracy is enough.

**D-021 (2026-10-02) — Hosting and CI: GitHub (`cattabiani/cabosse`, public)
and GitHub Actions** (resolves Q-15). Lint and fast tests run on every pull
request and push to `main`. The exhaustive tests run on `main` and on demand.
*Why:* that is where the code is, and Actions is free for public repos.

**D-022 (2026-10-03) — Softmax: max, sum, then normalize, without
FlashAttention-style online rescaling.** The spec keeps the three steps of
`docs/numerics.md` section 4. Hardware may recompute `exp` instead of storing
it (same bits), so it needs `n` FP32 scores per head. *Why:* fewer roundings.
Online softmax rescales the running sum and output each time the max grows,
which adds roundings and moves the BF16 rounding of `p` (D-011) before the
max is final. It stays a possible optimization if the perf model or hardware
shows score storage or the extra pass is a bottleneck (M2, M6).

## Milestones

Each milestone ends at a **checkpoint**: work stops for the owner's review.

### M0 — Repo foundation ✅
Repository, plan, agent instructions, licenses.

### M1 — Numerics spec and golden model
**What:** `docs/numerics.md` (formats, rounding points, special values,
summation order as parameters, algorithms for exp/rsqrt/reciprocal/SiLU/RoPE)
and a golden model of SmolLM2-135M-Instruct decode with a KV cache. It also runs the
tiny configs. Compared against `transformers` in FP32 and in BF16.
**Why:** everything later is verified against this model. The summation
order depends on the lane design (M2), so it is a parameter here.
**Done when:**
- Teacher-forced over ≥ 10 prompts × 256 positions, the golden model is no
  worse than `transformers`' own BF16 run, measured against FP32
  `transformers` on both top-1 agreement and logit error. (No fixed
  percentage: a first check showed BF16 `transformers` itself at 98.1% top-1.)
- Greedy decode text is saved as a regression fixture.
- The model is deterministic. One SmolLM2 token takes ≤ 10 s.

**Progress:**
- [x] Model runs as published: `scripts/run_smollm.py` (chat, streaming).
- [x] Day-one check: torch's FP32→BF16 conversion matches round-to-nearest-even
  on all 2³² inputs, subnormals included. NaNs come out as `0xFFFF`, not
  `0x7FC0`, so tests compare NaNs by class.
- [x] `docs/numerics.md` v0 draft, with D-019 (FMA) and D-020 (function
  approximations).
- [x] Arithmetic primitives (`bf16`, `up`, `add`, `mul`, `fma`, `mac`) with
  exact-oracle tests, and CI on x86 Linux, ARM Linux, and ARM macOS
  (PR #1). The exhaustive `bf16` test passes on all three.
- [x] Function approximations (rsqrt, recip, exp), parameters chosen by
  measurement (Q-08): 2 Newton steps each, exp degree 4. Errors below 2⁻¹⁷
  (exp: 2⁻¹⁶·⁶ for subnormal results, which are now kept as in IEEE). Tests:
  exhaustive accuracy, flush-to-zero behaviour, and bit-exact agreement with
  an independent restatement of the spec (PR #2).
- Golden model of SmolLM2 decode, in steps (one PR each):
  - [x] Dot product (section 3): `A` accumulators + pairwise tree, fast exact
    `mac`. About 3 s of matrix-vector work per SmolLM2 token (measured on the
    dev machine, Ryzen 7 7800X3D, 8 threads; machine-dependent).
  - [x] Reductions (sum, sum of squares, max), RMSNorm, softmax (section 4):
    one `interleaved_sum` shared with the dot product, `S` provisional 8.
    Bit-exact on edge-case inputs (overflow, subnormal variance and `exp`
    results, ties); errors against float64 2⁻¹⁷·⁷ (RMSNorm) and 2⁻¹⁶·³
    (softmax). Softmax without online rescaling (D-022). Review checklist
    for agents in `REVIEW.md` (PRs #4, #5).
  - [ ] **Next:** RoPE, SwiGLU, residual add.
  - [ ] Decoder: weights, KV cache, decode step, tiny configs.
- Comparison against `transformers`:
  - [ ] Harness and error report (teacher forcing, vs FP32 and BF16).
  - [ ] Greedy-decode fixtures and regression tests.
- [ ] M1 checkpoint review.

### M2 — Architecture spec and performance model
**What:** `docs/architecture.md` (block diagram, register map, memory map,
data layouts, command format, one token written out as commands, lanes `L`,
accumulators `A`, clock targets) and a first-order perf model
(tokens/s ≈ effective bandwidth / bytes per token, plus compute and
per-command overhead).
**Why:** this is the most expensive thing to change later, and the perf
model sets the targets for M9.
**Done when:** every golden-model operation maps to a command, with no host
work mid-token. Predicted tokens/s for SmolLM2 on F2 is written down, with
the dominant term identified. The summation parameters are fixed and the
golden model still passes M1.
**Starting points (from M1 discussion, to confirm with measurements):**
- `A` = adder latency rounded up to a power of two (likely 4 or 8). A larger
  `A` gives no speed, slightly better accuracy, and a longer final tree.
- `L` = 64 divides every SmolLM2 row count (576, 192, 1536, 49152), so no lane
  idles. `L` = 128 wastes 10–25% on some matrices. 64 lanes are a few percent
  of the F2 FPGA (to verify); memory bandwidth, not area, limits `L`.
- The final tree can overlap the next row with one small extra adder per
  lane (0% overhead), or reuse the lane's adder (about 2%). The bits are the
  same either way.

### M3 — Toolchain and FP units
**What:** OSS CAD Suite (Verilator, Yosys with the slang SystemVerilog
plugin, nextpnr, cocotb; owner installs), one-command test runner, CI, lint.
RTL for the BF16×BF16 multiplier and the FP32 adder.
**Why:** the FP adder is the hardest numerics RTL. Proving the toolchain on
something small first.
**Done when:** both units are bit-exact against the golden model on ≥ 10⁸
random inputs plus every special-value class. The multiplier is tested
exhaustively if that is fast enough. Lint is clean, and both synthesize in
Yosys. The area cost of subnormal support is reported (D-016).

### M4 — Dot-product lane
**What:** a lane that streams BF16 pairs into an FP32 dot product.
**Why:** a pipelined adder needs `A` rotating partial sums to take one input
per cycle. That fixes the summation order, which the golden model must match.
**Done when:** bit-exact on random lengths, SmolLM2 row lengths, and
adversarial inputs. 1 MAC/cycle sustained under randomized backpressure.

### M5 — Matrix-vector engine with simulated memory
**What:** `L` lanes, weight DMA over AXI, tiling, and an AXI memory model
with configurable bandwidth.
**Why:** this is where bandwidth is won or lost.
**Done when:** bit-exact on one full SmolLM2 layer and the classifier with
real data. Achieved bytes/cycle is ≥ 90% of `min(BW, 2·L)` on large
matrices. Verilator speed is measured, to plan M7. Row edge cases tested:
rows fewer than `L`, rows not a multiple of `L` (masked last pass, e.g.
SmolLM2's 192-row k/v with `L` = 128), a single row. Idle lanes must not
write results.

### M6 — Vector unit
**What:** RMSNorm, softmax, SiLU/SwiGLU, RoPE, residual add, BF16 rounding,
KV cache writes.
**Done when:** each operation is bit-exact on random and real activations,
and the perf model shows the vector unit at < 10% of token time.
**Possible optimization:** online softmax (D-022), only if score storage or
the extra pass shows up as a bottleneck. It needs a spec change.

### M7 — Controller and a full token in simulation
**What:** a controller that fetches and runs the command stream, a register
file, the top level, and the `sw/` runtime driving simulation through the
same interface as the hardware.
**Why:** this is the main lesson from the baseline. The host only says
"go", and everything else stays on the device.
**Done when:** tiny configs are bit-exact for 3 prompts × 64 tokens.
SmolLM2 is bit-exact for ≥ 8 tokens. The cycle breakdown per token is
compared with the perf model and with the baseline profile.

### M8 — F2 bring-up (no model)
**What:** F2 wrapper with a trivial test design: registers, host DMA, and
device memory bandwidth.
**Why:** keeps platform problems apart from design problems. Cloud time costs
money.
**Done when:** register access and DMA work, and HBM bandwidth per port and
in total is measured. **Cloud budget and instances are approved by the
owner before anything is created.**

### M9 — SmolLM2-135M-Instruct on F2, measured
**Done when:** bit-exact against the golden model. Tokens/s is measured over
≥ 1000 tokens and compared with the perf model and the bandwidth bound
(provisional floor: ≥ 50% of it). Clock and resource use are reported.
Compared with the Gemmini baseline (a different model) using normalized
figures: weight bytes/s and MACs/s.

### M10 — Larger models, optional weight-only quantization
Qwen2.5-0.5B (and/or SmolLM2-360M). Quantization is a separate
sub-milestone.

### M11 — Fully open FPGA flow (ECP5)
The same core, built with open tools only, runs a tiny config on a
cheap board. It may move earlier (Q-18).

### M12 — ASIC slice on an open PDK
A few lanes with a simple test interface. DRC/LVS clean, timing met,
gate-level simulation bit-exact. Shuttle submission is decided at this
checkpoint.

## Open questions

| ID   | Question | Leaning / when |
|------|----------|----------------|
| Q-04 | Vivado builds for F2: local machine or AWS build instance? | Before M8. Local needs the right Vivado version and a license for the F2 device. |
| Q-08 | Do the D-020 parameters (2 Newton steps, exp degree 4) hold end to end? | Measured per function in M1; confirm in the end-to-end comparison; area in M6. |
| Q-10 | How many HBM ports the engine reads in parallel, and how weights are spread across them. | M2. This sets the bandwidth bound. |
| Q-11 | Attention p·V: store V transposed, or add an engine mode for `Σ pᵢ·vᵢ`? | M2. |
| Q-22 | Hide the adder latency with `A` partial sums per row (current spec), or by rotating `A` rows per lane (one running sum per row, no tree)? | M2, together with the weight memory layout. Rotating rows gives plain sequential sums; it equals `dot` with `A` = 1. |
| Q-12 | Controller: fixed-function sequencer or small RISC-V core? | Leaning sequencer. M2. |
| Q-13 | What the F2 shell exposes (HBM ports, widths, clocks, DMA). | M2, from AWS docs only. |
| Q-14 | Own FP units or existing open IP (e.g. CVFPU)? | M3 start. |
| Q-16 | F2 budget and cost controls. | Before M8. |
| Q-17 | ECP5 board and host link. | Before M11. |
| Q-18 | Do ECP5 before F2? | M7 checkpoint. |
| Q-19 | Open PDK and shuttle (SKY130, GF180MCU, IHP SG13G2; Tiny Tapeout, …). | M11 checkpoint. |
| Q-20 | Is one-token-at-a-time prefill acceptable long-term? | After M10. |
| Q-21 | Sampling: argmax on device or logits to host? | Leaning both. |
