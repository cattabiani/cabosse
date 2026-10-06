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
   our memory-bandwidth bound and the Gemmini hackathon baseline.
3. **ASIC:** a small slice of the design passes signoff on an open PDK. If
   a shuttle is affordable, we submit it.

## Non-goals (for now)

- Training.
- Batched or multi-user serving: one sequence at a time.
- Fast prefill: the prompt goes through the decode path one token at a time.
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
- A fully open flow on F2, because Vivado is proprietary. The open flow is
  the ASIC.
- Other FPGAs, such as a cheap Lattice ECP5 board with a fully open flow
  (D-034). The RTL stays vendor-neutral, so others can port it.

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
- [ ] `docs/architecture.md`, draft: how a token runs, the blocks, the
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
  criterion). The perf model still counts one command per operation (513
  for SmolLM2, not 575); its command time is a guess either way.
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

### M3 — Toolchain and FP units
**What:** OSS CAD Suite (Verilator, Yosys with the slang SystemVerilog
plugin, nextpnr, cocotb; owner installs), one-command test runner, CI, lint.
CVFPU (D-033), vendored at a pinned version and configured for BF16 × BF16
→ FP32 multiply-add and the FP32 operations of the vector unit.
**Why:** the FP units carry the numerics. Proving the toolchain and CVFPU's
bit exactness on something small first.
**Done when:** a bit-exactness step tests each CVFPU operation we use
against the golden model on ≥ 10⁸ random inputs plus every special-value
class (the BF16 multiply exhaustively if that is fast enough); every
mismatch is reported and settled by the owner (D-033). The multiply-add's
accumulate loop closes in 4 cycles at 250 MHz, or `A` changes (D-027). Lint
is clean, and the units synthesize in Yosys. The area cost of subnormal
support is reported (D-016).

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

### M12 — ASIC slice on an open PDK
A few lanes with a simple test interface. DRC/LVS clean, timing met,
gate-level simulation bit-exact. Shuttle submission is decided at this
checkpoint.

## Open questions

| ID   | Question | Leaning / when |
|------|----------|----------------|
| Q-19 | Open PDK and shuttle (SKY130, GF180MCU, IHP SG13G2; Tiny Tapeout, …). | Before M12. |
| Q-20 | Is one-token-at-a-time prefill acceptable long-term? | After M10. |
| Q-21 | Sampling: argmax on device or logits to host? | Leaning both. |
