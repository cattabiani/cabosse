# Cabosse plan

A living document: what we are going to do and why. Details belong in `docs/`
once they exist. IDs: **M** = milestone, **D** = decision, **Q** = open
question.

Last updated: 2026-10-08.

## Goal

An open-source accelerator for LLM inference (hardware design, verification,
host software) that anyone with an AWS account can rebuild and run on F2,
with every published number reproducible. Tools are open source wherever
possible; Vivado, F2's only build flow, is the exception (D-032). The path:

1. **Simulation:** a full decode step of a real model runs in RTL simulation
   and matches a bit-exact Python golden model.
2. **FPGA:** the same RTL runs on AWS F2 as a PCIe device, measured against
   our memory-bandwidth bound and the Gemmini hackathon baseline.
3. **ASIC:** out of scope (D-036); a possible later development.

## Non-goals (for now)

- Training.
- Batched or multi-user serving: one sequence at a time.
- Fast prefill: the prompt goes through the decode path one token at a time
  (about 0.5 s for 500 tokens of SmolLM2, estimate). Revisit if M9 shows
  prompt latency matters (was Q-20).
- Contexts beyond the model's trained window (attention sinks with a circular
  cache, as in StreamingLLM). At the window's end the host runtime stops the
  conversation with a message that longer contexts are not supported yet. A
  possible later improvement; it changes the spec (was Q-24).
- Beating commercial GPUs. We compare against our own bandwidth bound and our
  baseline.
- General programmability. It is a fixed-function decode engine driven by
  a command stream, for Llama-style models (RMSNorm, RoPE, GQA, SwiGLU).
- Low-precision activations or KV cache. Weight-only quantization is a
  possible extension.
- A fully open flow on F2, because Vivado is proprietary. Every other tool
  stays open source.
- Other FPGAs, such as a cheap Lattice ECP5 board with a fully open flow
  (D-034). The RTL stays vendor-neutral, so others can port it.
- An ASIC: neither signoff on an open PDK nor a manufactured chip (D-036). A
  possible later development; the RTL stays vendor-neutral and synthesizable
  with Yosys.

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

**D-023 (2026-10-04) — Reference platform: x86-64 Linux.** Values pinned to
exact bits (RoPE table hashes, decode fixtures) are generated and checked on
x86-64 Linux only. ARM Linux and macOS run every test that does not depend
on the platform. *Why:* the golden model's arithmetic gives the same bits
everywhere by construction, but the RoPE tables come from the platform's
float32 cos/sin, which are not correctly rounded (D-020). Other platforms
are there to catch errors, not to make every pinned value portable.

**D-024 (2026-10-04) — KV cache: one flat array, allocated once.** K and V
for every layer, KV head and position are preallocated up to the model's
`max_position_embeddings` (8192 for SmolLM2). Token `pos` writes row `pos`,
and a new conversation resets the length to 0. When the cache is full,
decoding stops. *Why:* the model was trained on at most that many
positions, so a cache that runs longer (circular buffer, sliding window)
would only feed it contexts it was not trained for: a longer cache is
useless without a model trained for it. Block paging (as in vLLM) only
helps when many sequences share memory, and the workload is single-stream
(D-003). A flat array is also the simplest hardware layout: the address of
a row is arithmetic, and one head's K or V streams as one block. Q-24
keeps the alternatives for later.

**D-025 (2026-10-04) — Measure F2 first-hand in M2, with AWS's example
designs.** A new M2 step runs AWS's HDK examples (no Cabosse RTL) on one
f2.6xlarge in `eu-central-1` to measure what the docs leave open: HBM
bandwidth in total and per port, through the switch, at 250 and 450 MHz;
DDR4 bandwidth; host-to-card throughput; and the build flow itself. Own AWS
account (setup in `docs/aws-setup.md`); nothing is created without the
owner's approval and a budget (rule 3). M8 keeps the Cabosse wrapper.
*Why:* the architecture and the perf model rest on these numbers, and some
are not in any public document.

**D-026 (2026-10-05) — No example-design build in M2; HBM and DDR are
measured with our own design in M8.** The F2 check ran AWS's public
`cl_sde` image only (no build). No public image measures HBM or DDR, and
`cl_mem_perf` needs a full Vivado build of hours. Its numbers would not
change the design: engine width, lanes and ports are RTL parameters, the
first build uses our best guess from the documented values, and a rebuild
fixes it if the F2 says otherwise. M8 measures HBM per port and in total
with the Cabosse wrapper; until then the perf model keeps the documented
values as "quoted". *Why:* a build of AWS's design costs as much as a build
of ours and only answers a question about theirs.

**D-027 (2026-10-06) — Engine: 128 lanes × 4 multiply-adds, `A` = 16.**
512 multiply-adds per cycle use what the 32 HBM ports deliver at 250 MHz.
4 per lane is the most a 64-long attention row allows: its 16 partial sums
(4 per multiply-add unit, for a 4-cycle accumulate loop) take a 15-add final
sum, which fits in the next row's 16 cycles. Fewer, wider lanes stall on
attention; more, narrower lanes leave lanes idle on small matrices
(`docs/architecture.md`, `docs/perf.md`). `A` = 16 replaces the provisional
8 in `docs/numerics.md`. *Why:* the best engine use across positions of the
shapes compared; `L` can grow later without changing the bits.

**D-028 (2026-10-06) — Memory and clocks for v0.** Each of the 32 HBM
ports reads only its own channel (no switch), and port `p` feeds lanes `4p`
to `4p+3`; weights are stored in the lanes' read order, a reordering at load
time. K and V are stored as written (one row per position) and staged
through an on-chip tile buffer holding two KV heads, which the engine reads
across positions for p·V, so V has no transposed copy. Command list and
activations stay on chip; DDR4 is unused. Everything runs on the shell's
250 MHz clock, to verify in M3 and the first full build. AWS fixes that
clock for the shell's interfaces only; the HBM ports could run at 450 MHz,
so 250 MHz there is our choice for the first build (M9 raises it). Resolves Q-10 and
Q-11. *Why:* every port streams in parallel with sequential reads; no
second copy of the cache; no clock crossings in the first build
(`docs/architecture.md`).

**D-029 (2026-10-06) — Controller, commands and registers for v0.** The
controller is a fixed-function sequencer: it runs one unrolled command list
per model in order, with the token id and the position in registers; no
processor (resolves Q-12). Commands have one 32-byte encoding; results are
FP32 and round to BF16 where the destination is BF16 (the D-011 points); END
is the last command and an unknown command is an error. On-chip buffers are
sized at build time for the largest ladder model at full context; the
command buffer holds 64 KiB. The host interface is plain registers on the
OCL port: TOKEN, POSITION, CONTROL start, STATUS done or the interrupt, and
OUTPUT copies the logits to a host address. Whether the vector-unit commands
stay fused is open (Q-26). *Why:* a decode step has no data-dependent
control, so a processor would add a core, firmware and slower command issue
for nothing used; a queue of token descriptors, a completion record in host
memory or on-device argmax save microseconds per millisecond token, so they
wait for Q-20 and Q-21 (`docs/architecture.md`, `model/commands.py`).

**D-030 (2026-10-06) — Vector unit: 16 FP32 elements per cycle, `S` = 128.**
`S` fits every vector unit planned: up to 32 elements per cycle with a
4-cycle add loop, or 16 with an 8-cycle loop; v0 builds 16. `S` = 128
replaces the provisional 8 in `docs/numerics.md`. *Why:* at 16 wide the
vector unit stays below the engine's time at every position of both ladder
models (`docs/perf.md`); a wider unit or online softmax (D-022) can come
later, and widening does not change the bits (`docs/architecture.md`).

**D-031 (2026-10-06) — Vector-unit commands are primitives.** The vector
unit takes elementwise operations (add, multiply, FMA, exp, recip, rsqrt)
and reductions (sum, sum of squares, max); the command list chains them
into RMSNorm, RoPE, softmax and SwiGLU. The engine keeps its fused commands
(MATVEC, SCORES, VALUES). The opcode space leaves room for fused vector
commands, added only if M9 measures command overhead costing tokens/s; a
fused command runs the same operations in the same order, so the bits do
not change. Resolves Q-26; to be applied to `model/commands.py`,
`docs/architecture.md` and the perf model's command count. *Why:* a model
needing a new step (Qwen2.5's q/k/v biases, Qwen3's q/k norm) needs a new
command list, not new hardware; the vector unit is one generic pipeline.

**D-032 (2026-10-06) — F2 builds run on AWS.** Vivado builds for F2 run on
an EC2 instance from the FPGA Developer AMI, which includes the Vivado
license; nothing is built locally. Each build instance, like every launch,
needs the owner's OK with a time limit. Resolves Q-04. *Why:* no local
Vivado install or VU47P license to manage, the HDK's Vivado version comes
with the AMI, and turning the build into an FPGA image (AFI) happens on
AWS anyway. The build's result does not depend on the host machine.

**D-033 (2026-10-06) — FP units from CVFPU.** The lanes' and the vector
unit's floating-point operations use CVFPU (OpenHW Group / ETH Zurich,
open-source, Solderpad license; license compatibility to verify when it is
vendored), pinned to one version. M3 first tests it for bit exactness
against `docs/numerics.md` and the golden model. A mismatch goes to the
owner: a spec change (rule 8) or our own unit for that operation. Already
known to check: RISC-V's `fmax` returns the other operand when one is NaN,
where the spec returns NaN. Resolves Q-14. *Why:* a mature, tested design
saves the hardest numerics RTL; the bit-exactness tests keep the spec in
charge.

**D-034 (2026-10-06) — F2 is the only FPGA; ECP5 is out of scope.** The
project has no FPGA but F2, so the ECP5 port (M11) is left as future work
for others; wrappers target F2 and the ASIC (narrows D-004). The RTL keeps
no vendor primitives, and tools stay open source wherever F2 allows.
Resolves Q-17 and Q-18. *Why:* we can only measure what we can run.

**D-035 (2026-10-06) — Logits go to the host; sampling runs on the host.**
OUTPUT copies the FP32 logits to host memory every token (D-029), and the
host picks the token (argmax or sampling). Argmax on the device comes only
if M9 measures the copy costing tokens/s. Resolves Q-21. *Why:* for SmolLM2
the copy is 192 KiB, about 20 µs at the PCIe rate measured on F2 against a
token of about 1 ms (estimate); the host gets every sampling method for free.

**D-036 (2026-10-06) — The ASIC is out of scope.** M12 (a slice of the
design through signoff on an open PDK, and a shuttle run) is left as a
possible later development; the project ends at F2. Narrows D-001, and
D-004 and D-034 (wrappers target F2 only). Resolves Q-19. *Why:* signoff
and a chip change nothing on F2, and each costs a milestone or money; the
RTL stays vendor-neutral and synthesizable with Yosys, so the ASIC remains
possible.

**D-037 (2026-10-07) — `max` is IEEE 754-2019 `maximumNumber`.** A NaN
operand gives the other operand; two NaNs give the canonical NaN; `max(-0,
+0) = +0` as before. This is RISC-V's `fmax`, which CVFPU implements, so the
vector unit uses CVFPU's unit unchanged (settles the mismatch D-033 named).
Softmax's output does not change: a NaN score's own `exp` is NaN, so the sum
and every `p` are NaN, as before. *Why:* the spec follows a standard
operation and a tested unit, at no cost to any result the model produces.

**D-038 (2026-10-07) — The lanes' multiply-add is our own BF16 unit.** An
exact BF16 × BF16 product (an 8 × 8-bit multiply) outside the accumulate
loop, and inside it only that product plus the FP32 accumulator with one
rounding: the same bits as `mac` (docs/numerics.md), so no numerics change.
The vector unit keeps CVFPU (narrows D-033). If the unit does not close a
4-cycle loop at 250 MHz on F2's part, the fallback is CVFPU in 256 lanes ×
2 with an 8-cycle loop: the same `A` = 16 and bits, 5-9% fewer tokens/s
(perf model, estimate). *Why:* CVFPU's FMA needs about 4.5 cycles at
250 MHz for the loop even with retiming (measured, M3), and spends about
1,100 LUTs on a general 24 × 24 multiply that BF16 inputs do not need.

**D-039 (2026-10-07) — All FP units are our own; CVFPU is removed.** The
vector unit's FP32 add, multiply, fma and max become our own RTL, built on
the lanes' adder (`mac_add`, D-038), and must pass the same bit-exactness
tests CVFPU passed (the golden functions, 10⁸ inputs, every special-value
combination). Then `rtl/vendor/` goes, CVFPU and the common_cells files
with it (our own leading-zero count replaces `lzc`). Supersedes D-033;
D-037 (`max` as IEEE `maximumNumber`) stays. *Why:* CVFPU could serve only
the vector unit (D-038), and needs Vivado's retiming to reach 250 MHz
(measured, M3); two implementations of the same operations would have to
be kept equal in different places. One set of units, written and tested
for exactly what we need, is simpler to own.

## Milestones

Each milestone ends at a **checkpoint**: work stops for the owner's review.

### M0 — Repo foundation ✅
Repository, plan, agent instructions, licenses.

### M1 — Numerics spec and golden model ✅
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
  - [x] RoPE, SiLU/SwiGLU, residual add (`arith.add`). RoPE tables come from
    `transformers`' own rotary embedding, in FP32; they are within 1 ulp of
    correct rounding on every platform, and pinned by hash on the reference
    platform (D-023). Errors against float64: RoPE 2⁻²³·¹ (relative to
    |xC| + |rS|), SiLU 2⁻¹⁶·⁴ (PR #6).
  - [x] Decoder: weights (checkpoint BF16 tensors, `transformers` names), flat
    KV cache (D-024), decode step (spec section 4), tiny random-weight configs
    (`golden/tiny.py`). Bit-exact against a per-head restatement; within 1%
    of `transformers` FP32 as a wiring check. SmolLM2: 3 s per token on the
    dev machine, same greedy tokens as `transformers` on a short prompt
    (PR #7).
- Comparison against `transformers`:
  - [x] Harness and error report (`model/compare.py`,
    `scripts/compare_transformers.py`). **Exit criterion met:** teacher-forced
    on 10 chat prompts × 256 positions against FP32 `transformers`, the
    golden model has top-1 agreement 99.30% (BF16 `transformers`: 98.75%),
    mean relative logit error 6.3e-3 (1.5e-2), mean KL 1.2e-4 nats (8.6e-4).
    Golden is no worse on every metric. Speed: see `reports/M1.md`
    (measured, with the machine it ran on) (PR #8).
  - [x] Greedy-decode fixtures (`decoder.generate`, `model/fixtures.py`,
    `scripts/make_fixtures.py`): tiny untied config, 3 prompts × 64 tokens
    with logit hashes (fast suite); SmolLM2, 3 chat prompts × 64 tokens with
    FP32 `transformers`' greedy tokens alongside (slow test). Golden and FP32
    `transformers` agree on all 64 tokens for two prompts; the third differs
    at token 62, a near-tie (29.883 vs 29.869). The fixtures do not catch a
    change of `S` (BF16 outputs absorb it); per-op tests do (PR #9).
- [x] M1 checkpoint (2026-10-04): all exit criteria met; report in
  `reports/M1.md`, numbers generated by `scripts/report_m1.py` (PRs #10,
  #11). Resolves Q-08: the D-020 parameters hold end to end (the comparison
  uses them); their area is checked in M6.

### M2 — Architecture spec and performance model ✅
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
  **Revised by the perf model:** 64 multiply-adds per cycle at 250 MHz
  leaves the engine, not memory, as the limit; see Q-25 (resolved by D-027).
- The final tree can overlap the next row with one small extra adder per
  lane (0% overhead), or reuse the lane's adder (about 2%). The bits are the
  same either way.

**Progress:**
- [x] F2 platform facts (`docs/f2.md`), from public AWS and AMD docs; resolves
  Q-13. HBM: 32 AXI3 ports up to 450 MHz, any port reaches any address;
  shell interfaces run at 250 MHz, so full HBM bandwidth needs a faster
  memory-side clock. DDR4 runs at 2133 MT/s, 17.1 GB/s peak (from AWS's
  controller configuration). Still to verify: two AMD values read only from
  search excerpts.
- [x] First-order performance model (`model/perf.py`, `docs/perf.md`):
  generic over model config, platform file (`model/platforms/f2.json`, each
  value with a status and a source) and design file with scenarios
  (`model/designs/v0.json`). Its work list is checked against the golden
  decode step, its byte count against the checkpoint. SmolLM2 on F2, position
  0: 64 MAC/cycle at 250 MHz is engine-bound at 119 tokens/s; with HBM at
  450 MHz and 1024 MAC/cycle it becomes memory-bound at 1584 (751 with no
  overlap). At the full 8192-token window, attention's multiply-adds exceed
  the weights' and the vector unit becomes the limit. All numbers in
  `docs/perf.md`; they change with the inputs.
- [x] Inputs validated with pydantic; our measurements
  (`model/platforms/measured/f2.json`) replace documented values entry by
  entry, with the documented value kept alongside. The
  report says "provisional" until every platform value is measured by us or
  marked not measurable.
- [x] `docs/architecture.md`, draft: how a token runs, the blocks, the
  engine, memory and clocks. Decided: the engine, 128 lanes × 4 multiply-adds,
  `A` = 16 (D-027; resolves Q-25 and Q-22); memory and clocks (D-028;
  resolves Q-10 and Q-11). The perf model now counts idle lanes and the final sum of short
  rows. Vector unit: 16 FP32 elements per cycle, `S` = 128 (D-030).
- [x] Controller, commands and registers, D-029 (`docs/architecture.md`,
  tables generated from `model/commands.py`): a fixed-function sequencer
  (Q-12) running one unrolled command list per model; token and position in
  registers; 12 commands of 32 bytes; 575 commands for SmolLM2. A reference
  controller (`commands.run()`) on the golden functions gives the golden
  decode step's logits and KV cache bit for bit (tiny configs, 5 tokens):
  every operation maps to a command, with no host work mid-token (M2 exit
  criterion).
- [x] Primitive vector-unit commands (D-031): 17 commands, 44 per layer,
  1333 for SmolLM2 (42 KB of the 64 KiB buffer); RMSNorm, RoPE, softmax and
  SwiGLU are chains of primitives in the spec's order, still bit-exact
  against the golden decode step. The perf model now takes its work from
  the command list (one entry per command, one vector pass each), so its
  command count is the real one. Against the fused commands: vector-unit
  passes ×1.7 (softmax: five passes over the scores instead of three) and
  commands ×2.6. In v0 neither limits the token (SmolLM2 at position 0:
  880 tokens/s overlapped, unchanged; 369 with no overlap, was 411); at
  Qwen2.5-0.5B's full window the vector unit reaches 95% of the engine's
  time, and with M9's faster engine and memory it becomes the limit at
  long positions (fused softmax, D-031, or a wider unit then).
- [x] `A` = 16 (D-027) in `docs/numerics.md` and the golden model; greedy
  fixtures regenerated. Teacher-forced comparison rerun on the M1 set (10
  prompts × 256 positions, `reports/data/M2-accumulators-16-compare.json`):
  still well ahead of BF16 `transformers` on every measure; against `A` = 8,
  top-1 0.9965 vs 0.9930 and mean KL lower, mean logit error 1.6% higher.
  `S` (vector-unit reductions) stays provisional 8 until the vector unit is
  designed.
- [x] `S` = 128 (D-030) in `docs/numerics.md` and the golden model; SmolLM2
  greedy fixture regenerated (the tiny one keeps its bits: its 64-long sums
  differ, but not after the BF16 roundings). Comparison rerun on the M1 set
  (`reports/data/M2-reduce-width-128-compare.json`): against `S` = 8, top-1
  (0.9965) and mean logit error unchanged; max logit error 1.91e-2 vs
  2.20e-2, max KL 3.5e-3 vs 4.4e-3.
- [x] Report-generation code in its own package, `model/reporting/`:
  `blocks.py` (generated blocks), `m1.py` (`reports/M1.md`), `perf_doc.py`
  (`docs/perf.md`), each with `PATH` and `render()`; scripts
  `report_m1.py` and `report_perf.py`. Value formatting and the measured-vs-documented
  comparison live on `perf.Param` (`str(p)`, `p.change_from(doc)`); the
  table code only lays out rows. No change in output.
- [x] Checkpoint report: [reports/M2.md](reports/M2.md), tables generated by
  `scripts/report_m2.py` (exit criteria, predictions, the summation
  comparisons, open issues).
- [x] F2 platform check (D-025, D-026): account set up (`docs/aws-setup.md`,
  profile `cabosse`, `eu-central-1`, budgets on, F quota 24 vCPUs). On
  2026-10-05 one f2.6xlarge loaded AWS's public `cl_sde` image
  (`scripts/f2/measure_sde.sh`; raw output in `reports/data/f2/`): PCIe link
  16 GT/s x8; host-to-card streaming up to 10.55 GB/s and card-to-host up to
  14.21 GB/s (64 KiB packets); FPGA core power (Vccint) 6–9 W; no clock
  readout (the image has no clock generator). No perf-model value is
  measured by us yet: HBM and DDR wait for M8 (D-026).
  Resolves Q-16: a 30 USD monthly budget and a zero-spend alert; each
  launch needs the owner's OK with a time limit; instances terminate
  themselves at the limit, and every session ends with a check that
  nothing is left.

### M3 — Toolchain and FP units ✅
**What:** in this order:
1. OSS CAD Suite (Verilator, Yosys with the slang SystemVerilog plugin,
   cocotb; owner installs) and a one-command test runner.
2. The FP units: BF16 × BF16 → FP32 multiply-add and the FP32 operations
   of the vector unit, tested for bit exactness against the golden model.
   First CVFPU (D-033), then our own (D-038, D-039). This comes first
   because a mismatch can change the numerics, and every later milestone
   builds on them.
3. The accumulate loop's timing at 250 MHz: a small, targeted Vivado run on
   AWS (D-032), with the owner's OK when it is needed.
4. CI and lint for the RTL.
**Why:** the FP units carry the numerics. Proving the toolchain and the
units' bit exactness on something small first.
**Done when:** a bit-exactness step tests each FP operation we use
against the golden model on ≥ 10⁸ random inputs plus every special-value
class (the BF16 multiply exhaustively if that is fast enough); every
mismatch is reported and settled by the owner (D-033). The multiply-add's
accumulate loop closes in 4 cycles at 250 MHz, or `A` changes (D-027). Lint
is clean, and the units synthesize in Yosys (a vendor-neutral check). The
cost of subnormal support is reported in Yosys cells (D-016); its F2
resource cost comes with the first build.

**Progress:**
- [x] CVFPU vendored (`rtl/vendor/`, D-033; pins and reasons in its
  README; checksums tested). `rtl/sources.f` is the one compile list for
  every tool. `rtl/fp32_fma.sv` wraps the FMA (fma, add, mul, round to
  nearest even); smoke test bit-exact on every triple of 18 special values
  and 3,000 random inputs per operation. The BF16 lane needs no
  multi-format unit: BF16 inputs widen to FP32 exactly, and a BF16 × BF16
  product is exact in FP32.
- [x] Bulk harness (`verif/bulk.py`; drivers in `verif/bulk/` share
  `stream.h`): Python makes inputs and expected bits with the golden model,
  Verilator streams them one per cycle, Python compares whole arrays. One
  driver runs about 7 million FMA inputs per second (measured; cocotb about
  0.2 million), and the slow run checks 8 chunks at a time. FMA, add and mul
  are bit-exact against the golden model on 10⁸ inputs each (the golden
  tests' rounding families plus random bits, seeded per chunk of 10⁶;
  `slow`, about 4 s per operation), and on every triple of the special
  values, also with a 3-register pipeline.
- [x] The lane's arithmetic on every BF16 pair: all 2³² products and all
  2³² multiply-adds (a random accumulator per pair, against `mac_f32`) are
  bit-exact (`slow`, about 4.5 minutes). The lane is `bf16_to_fp32` into
  `fp32_fma`, each now checked over its whole input domain; M4 checks the
  wiring.
- [x] `max` (D-037): `rtl/fp32_max.sv` wraps CVFPU's `fpnew_noncomp`
  (MINMAX, MAX). Bit-exact against the golden `max` on every pair of
  special values and on 10⁸ inputs (`slow`).
- [x] CI runs the RTL tests (lint, Yosys synthesis, simulation) on x86
  Linux with the pinned suite (`scripts/get_rtl_tools.sh`), about 2
  minutes; the slow RTL tests run locally.
- Accumulate loop at 250 MHz, first Vivado run (2026-10-07, Vivado 2025.2,
  `xcvu47p-fsvh2892-2-e`, out of context, no retiming;
  `scripts/f2/time_loops.sh`, log in `reports/data/f2/`): CVFPU's FP32
  FMA with BF16 inputs fails 4.0 ns at every loop length tried. Slack
  (computed max clock): 2 cycles -1.115 ns (196 MHz), 3 cycles -0.777 ns
  (213 MHz), 4 cycles -0.694 ns (213 MHz), 5 cycles -0.527 ns (222 MHz);
  about 1,100 LUTs and 2 DSPs. The worst paths are inside the FMA's fixed
  stages (operands to the 76-bit sum: up to 27 logic levels, 13 of them
  carry chains; then normalize and round), so more registers barely help.
  With retiming (`synth_design -global_retiming on`, `phys_opt_design
  -retime`; second log): 4 cycles -0.639 ns (216 MHz), 5 cycles +0.300 ns,
  6 cycles +0.183 ns, 8 cycles +0.340 ns. At 4 cycles the worst path is the
  loop's own normalize and round: CVFPU's loop needs about 4.5 cycles at
  250 MHz, so 8 with `A` a power of two. Options (perf model, SmolLM2,
  estimates) for the owner: our own BF16 unit for a 4-cycle loop; CVFPU in
  256 lanes x 2 with an 8-cycle loop, the same `A` = 16 and bits, 5-9%
  fewer tokens/s; a 200 MHz core, 20% fewer.
- [x] Our own BF16 multiply-add (D-038): `rtl/bf16_mul.sv` (the exact
  product, outside the loop) and `rtl/mac_add.sv` (product + accumulator,
  one rounding, 3 stages). Bit-exact against `mac_f32` on every BF16 pair,
  every special-value combination, and cancelling, scaled and range-edge
  accumulators (`slow`, 2 minutes). On F2's part at 250 MHz (third log,
  `platforms/f2/timing/mac_loop.sv`): the 4-cycle loop closes with +0.146 ns
  without retiming, a 3-cycle loop with +0.057 ns (retiming); the worst
  path is the product, outside the loop. About 625 LUTs, 247 registers and
  no DSPs per unit (CVFPU's FMA: about 1,100 LUTs and 2 DSPs). The
  accumulate-loop exit criterion is met with `A` = 16 unchanged.
- Done (D-039, branch `m3/own-fp`): our own FP units replace CVFPU.
  `fp_add` (generic one-rounding adder, W-bit significands: 27 for the
  lanes, 51 for FP32 fma), `fp_product` (exact product, M-bit significands:
  8 for BF16, 24 for FP32; replaces `bf16_mul`), `fp32_fma` (add = fma(a,
  1, b), mul = fma(a, b, -0)), `fp32_max`, `leading_zeros`, `sticky_shift`,
  `fp_pkg`; `rtl/vendor/` and `scripts/vendor_rtl.sh` are gone, and the docs
  no longer mention them. The old tests pass unchanged except for CVFPU's
  pipeline parameter: fast and slow (10⁸ per op, all BF16 pairs). Mutation
  check (19 hand mutations of the units, fast tests): 15 caught; the 4 left
  give the same bits (two were redundant logic, now removed). Yosys
  (generic `synth`, flattened): `bf16_mac` 2,513 cells, `fp32_fma` 7,798,
  `fp_add` (W = 27) 1,509. On F2's part at 250 MHz (2026-10-07, fourth
  log, `platforms/f2/timing/fma_path.sv`): `fp32_fma` with its product in
  one cycle fails, -2.700 ns without retiming (about 149 MHz), -2.023 ns
  with one more input register and retiming (166 MHz). The worst path was
  op mux, the 24 x 24 multiply (2 DSPs), normalize and the shift below
  exponent 1 (6.7 ns, 24 logic levels); retiming cannot split the DSP
  multiply. So `fp_product` now splits multiply | normalize | shift below
  exponent 1 with `Regs` registers, and `fp32_fma` has `MulRegs` (fifth log,
  no retiming unless noted): 1 register +0.155 ns (+0.180 ns retimed), 2
  registers +0.588 ns with 1,183 LUTs, 543 registers and 2 DSPs (1: 1,243
  LUTs, 541 registers). `MulRegs` = 2 is the default: a 6-cycle fma, not
  in a loop. The lanes' product (outside the loop) sat at the edge of
  4 ns: the same function, written three ways in this branch, gave +0.146,
  +0.016 and -0.053 ns, its worst path always the product (about 4.0 ns, 19
  logic levels). Vivado repeats a result for the same RTL (the fma's
  +0.588 ns twice), so the spread came from the rewrites. `bf16_mac` now
  has one register inside the product too (`MulRegs` = 1, a 5-cycle unit;
  `A` and the bits unchanged). With it (sixth log): the 4-cycle loop
  +0.657 ns without retiming (596 LUTs, 275 registers, no DSPs), +0.770 ns
  retimed; a 3-cycle loop +0.813 ns retimed. The worst path is now the
  loop's own (accumulator to alignment, 3.3 ns). The fifth log's last run
  was read from the live console, its replay being cut by the shutdown.
  Review cleanup (PR #36): `fp_mul_add` (product, register, adder) is the
  one body of `bf16_mac` and `fp32_fma`; `fp_add` takes its second operand
  as FP32 bits; `fp_pkg` holds the canonical NaN, the NaN test and the
  product's width; `fp_product` refuses `Regs` > 2 (it has places for two
  registers only); the exponent order comes from the subtractions'
  borrows; `fp32_max` uses one magnitude compare. Yosys: `bf16_mac` 2,565
  → 2,569 cells, `fp32_fma` 7,976 → 7,919. Not done, no gain in Yosys
  cells: the significand compare beside the alignment shifter instead of
  in front of it (+142 cells; it shortens `fp_add`'s first stage, now the
  lanes' worst path, so worth a Vivado run only if that path limits), and
  the product's zero flag from the operands (same cells, one more flop per
  register). Second review: `fp_product` and `fp_mul_add` take a and b in
  an M + 8-bit format (BF16 for M = 8), so a BF16 product cannot drop
  input bits; the timing harness's valid follows the product's registers;
  register names follow `_q`; FP32's significand width is one constant
  (`fp_pkg::F32SigW`); `bf16_mac` is tested with `MulRegs` = 2 too.
  Fixed-latency arithmetic pipelines have `valid` only (AGENTS.md,
  SystemVerilog conventions; approved by the owner). Yosys: `bf16_mac`
  2,569 → 2,552 cells, `fp32_fma` 7,919 → 7,964 (no logic change for
  M = 24: Yosys gives the same count on every run, but rewrites with the
  same logic move it; see reports/M3.md).
  Seventh log (2026-10-08, the same four runs on this RTL, commit
  4897dad): the 4-cycle loop +0.877 ns without retiming (633 LUTs, 275
  registers, no DSPs), +0.953 ns retimed; the 3-cycle loop +0.841 ns
  retimed; `fp32_fma` +0.514 ns (1,262 LUTs, 535 registers, 2 DSPs). All
  four pass. The loop's worst path is now inside `fp_add`'s round stage
  (3.1 ns), the fma's still the product's normalize stage (3.4 ns).
- [x] Cost of subnormal support (D-016). `Ftz` = 1 (default 0) builds
  `bf16_mac`'s and `fp32_fma`'s flush-to-zero variant, bit-exact against
  the golden model's ftz switch at every pipeline depth; the cost is the
  Yosys cells between the two builds: a few percent of each unit, more for
  the fma. Numbers, method and how far to trust them:
  [reports/M3.md](reports/M3.md),
  generated from Yosys's log by `scripts/report_m3.py`. D-016
  stands unless the owner decides otherwise; the F2 resource cost comes
  with the first build. `Ftz` stays in the RTL (owner, 2026-10-08), to
  measure that cost and to re-measure later.
- [x] Checkpoint report: [reports/M3.md](reports/M3.md), every number
  generated by `scripts/report_m3.py` from the Vivado logs, the Yosys log
  and the test runs; it replaces the two interim reports.
- Later: the FMA's op codes (`OpAdd`, `OpMul`) move to a package when a
  second module uses them. Synthesizing every module as its own top
  repeats the FMA under each parent, but all 13 Yosys checks take 7.6 s
  together (measured 2026-10-08, dev machine), so every module stays a
  top; revisit only if a larger top (M5's engine) makes it slow.

### M4 — Dot-product lane
**What:** a lane that streams BF16 pairs into an FP32 dot product.
**Why:** a pipelined adder needs `A` rotating partial sums to take one input
per cycle. That fixes the summation order, which the golden model must match.
**Done when:** bit-exact on random lengths, SmolLM2 row lengths, and
adversarial inputs. `E` = 4 pairs/cycle sustained under randomized
backpressure, for rows of 64 or more elements.

**Answers to the start-of-milestone questions (owner, 2026-10-08):**
- Shape (D-027): parameters `E` (pairs per cycle, 4) and `A` (16); the
  lane fails to elaborate unless `A / E` equals the accumulate loop's
  length (4 cycles). Unit `e` takes element `4c + e` and owns partial
  sums `e, e+4, e+8, e+12`.
- Bubbles: each unit keeps its 4 partial sums in an accumulator file read
  and written by group index, so a cycle with no input costs that cycle
  only. Fallback if the loop's multiplexer misses 4 ns: the group waits for
  its sum to come round (same bits, up to 3 more cycles per bubble).
- Final sum: inside the lane, the 15 adds of the fixed tree on one extra
  `fp_add` (W = 27), overlapping the next row. It keeps up for rows of
  64 elements or more; shorter rows lower `ready`, with the same bits. A
  row's first group adds to +0 instead of reading the file.
- Interface: input beats {4 w, 4 x, `last`, 4-bit element mask on the
  last beat} with valid/ready; a masked element leaves its partial sum
  unchanged (numerics.md, section 3). One FP32 result per row through a
  small FIFO with valid/ready. Control is one block, so M5 can drive many
  lanes' datapaths from one control.
- Tests, three layers: cocotb (handshake; single bubbles, runs of bubbles
  and random patterns on input valid and output ready, each checked for
  bits and for cycles = groups + bubbles + latency; lengths 1 to 20
  exhaustively and random to 300; special values; throughput); a bulk
  Verilator driver (SmolLM2 row lengths 64, 576, 1536, p·V rows up to
  8192, random lengths, adversarial values; fast subset in CI, about 10⁸
  pairs `slow`); real SmolLM2 rows and activations (`slow`, skips without
  weights).
- Loop timing: a Yosys check counts the logic levels on the loop path,
  with a limit in the test (a proxy). One Vivado run on F2's part
  (`scripts/f2/time_loops.sh`) once the RTL and cocotb tests pass: timing
  at 250 MHz and one lane's resources (×128 for the engine's first area
  estimate). The launch needs the owner's OK with a time limit.
- Synthesis checks: unchanged, the lane added as one more top.

**Steps (one branch and PR each):** lane RTL with cocotb tests; the
Vivado run; bulk driver and slow tests; checkpoint report.

**Progress:**
- [x] Lane RTL, `rtl/dot_lane.sv`, with cocotb tests
  (`verif/tests/test_dot_lane.py`, `MulRegs` 0, 1 and 2): every length 1
  to 20, random lengths to 300, single bubbles at each beat of a row, runs
  of 1 to 5 and 40, random gaps, random output ready, masks anywhere,
  signed zeros, special values; bit-exact against `golden.dot.dot`. With
  the output ready, rows of 60 or more ((`A` - 1) × `E`, the tree's adds) never see ready drop; rows of 56 do. Mutation check
  (11 hand mutations): 10 caught; the one left (the root waiting for the
  oldest context) gives the same order anyway, kept as a guarantee.
  Changed from the answers above: 3 final-sum contexts, not 2. A final
  sum holds its context about 30 cycles from the row's last beat
  (measured in simulation: with 2, 64-long rows stalled about once a
  row); bypasses could save about 4 cycles and keep 2, with little margin.
  A missing element also reads +0 where its slot has no sum yet in the
  row (written flags), not only in the row's first round. Yosys depth
  proxy (`test_lane_logic_depth`): 19 cells, `mac_loop` 17. Yosys's
  UltraScale+ mapping, an estimate until the Vivado run: about 6,250
  LUTs, 3,400 flip-flops and 4 DSPs per lane; ×128 it would be well above
  the 25-30% guessed for Q6, so the Vivado run checks area as much as
  timing. `leading_zeros` gets an `UNOPTFLAT` waiver: Verilator saw its
  step array as a loop once the lane inlines four of them.
- [x] Vivado run of the whole lane (2026-10-08, Vivado 2025.2,
  `xcvu47p-fsvh2892-2-e`, out of context; harness
  `platforms/f2/timing/lane_path.sv` registers every port; log
  `reports/data/f2/2026-10-08-lane.txt`, cabosse 3347b22). All three runs
  meet 250 MHz: `MulRegs` = 1 +0.173 ns, retimed +0.070 ns, `MulRegs` = 2
  +0.166 ns. The worst paths are the final sum's scheduler (context state,
  the choice of the next add, the operand multiplexer: 14 to 16 logic
  levels), not the accumulate loop. A negative slack now fails the run
  (`timing.tcl`). Area, `MulRegs` = 1: 5,191 LUTs, 3,395 flip-flops, no
  DSPs per lane (Yosys estimated 6,250 LUTs). The four units take 2,790
  LUTs (54%); the final sum the rest: its adder's instance shows 1,325
  LUTs (a unit's adder about 470, so the operand multiplexers are likely
  counted there) and the lane's own logic (contexts, copy, control) 1,077.
  ×128 lanes: 664,448 LUTs, 51.0% of the part, and 434,560 flip-flops,
  16.7% (Q6 guessed 25-30%). Open for the owner: whether to make the final
  sum cheaper before M5.

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
**Check:** the area of the D-020 function units (rsqrt, recip, exp); fewer
Newton steps or a lower exp degree only through a numerics change (was Q-08).
**Possible optimization:** online softmax (D-022), only if score storage or
the extra pass shows up as a bottleneck. It needs a spec change.

### M7 — Controller and a full token in simulation
**What:** a controller that fetches and runs the command stream, a register
file, the top level, and the `sw/` runtime driving simulation through the
same interface as the hardware. At the end of the model's context window
the runtime stops the conversation with a clear message (see Non-goals).
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
Every Vivado run (timing harnesses, full builds) checks that the worst
slack at 250 MHz (4 ns) is ≥ 0, and its script fails otherwise; the slack
goes into the report next to the perf numbers (owner, 2026-10-08).

### M9 — SmolLM2-135M-Instruct on F2, measured
**Done when:** bit-exact against the golden model. Tokens/s is measured over
≥ 1000 tokens and compared with the perf model and the bandwidth bound
(provisional floor: ≥ 50% of it). Clock and resource use are reported.
Compared with the Gemmini baseline (a different model) using normalized
figures: weight bytes/s and MACs/s.

### M10 — Larger models, optional weight-only quantization
Qwen2.5-0.5B (and/or SmolLM2-360M). Quantization is a separate
sub-milestone.

### M11 — Fully open FPGA flow (ECP5): out of scope (D-034)
Future work for others: the same core, built with open tools only, running
a tiny config on a cheap board.

### M12 — ASIC slice on an open PDK: out of scope (D-036)
A possible later development: a few lanes with a simple test interface. DRC/LVS clean, timing met,
gate-level simulation bit-exact; then, perhaps, a shuttle run.

## Open questions

None open. New ones are added as a table:

| ID   | Question | Leaning / when |
|------|----------|----------------|
