# Cabosse plan

A living document: what we are going to do and why. Details belong in `docs/`
once they exist. IDs: **M** = milestone, **D** = decision, **Q** = open
question.

Last updated: 2026-10-23.

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

**D-040 (2026-10-11) — A matrix sits at the same offset in every channel.**
An engine command's `addr` is an offset within a channel: port `p` reads its
share of the matrix from channel `p` at `addr`, and every channel reserves
the largest share, `ceil(n / L) × ceil(m / E)` beats (a port with fewer rows
in the last pass leaves the rest unused). The KV cache follows the same
rule (M5, attention). *Why:* a command has one address (32-byte encoding);
32 per-port addresses do not fit and buy nothing, since the ports' shares
differ only in the last pass. The model's `Layout` still uses one flat
address space; it moves to channel offsets when the command list runs on
the engine (M7).

**D-041 (2026-10-11) — Engine commands overlap; the controller orders
dependent ones.** The engine queues the commands it takes in three stages
(issue to the DMAs, step the lanes, drain the results), so the next
matrix's reads start while the current one computes, and back-to-back
commands leave no gap. The engine does not check whether a command reads
what an earlier, unfinished one writes: the controller waits for the
earlier command's done before starting it (o after attention, down after
SwiGLU; M7). *Why:* a command's start-up (memory latency and the lanes'
pipeline, 50-150 cycles, estimate) is 20-50% of SmolLM2's k and v
matrices at 128 lanes, above the 10% the M5 answers allow against the perf
model. Back-to-back engine commands that share an input (q, k and v; gate
and up) do not depend on each other.

**D-042 (2026-10-11) — A dependent engine command reads its weights ahead
of its go** (refines D-041; owner, 2026-10-11). The controller gives the
engine a command whose input is not ready yet with `wait` set, and pulses
`go` once the command it depends on is done. The engine's issue stage
reads the command's weights as soon as it takes it; only the step stage
waits for the go. Rules: a go is a credit for the oldest waiting command
without one, sent only after the engine took that command; the controller
never makes a go depend on the engine taking a later command (that would
deadlock with the queues full). Read-ahead covers memory no command writes
(weights); attention's KV tiles, which KV_STORE writes just before, are
settled with the tile buffer (M5, attention). The perf model counts
`engine_startup_cycles` per waiting command (`commands.engine_waits`: the
engine commands whose input was written at or after the previous engine
command). *Why:* reading after the go adds the memory's latency to every
dependent command: 118 cycles at F2's latency, 16% of SmolLM2's q and o
(measured in an exploratory run, not saved), above the 10% the M5 answers allow; read ahead, 47-55 cycles
remain (the lanes' pipeline, the final sum and the last pass's drain;
reports/data/M5-engine.json).

**D-043 (2026-10-13) — KV cache positions are contiguous within a
channel** (refines D-040 for the KV cache; owner, 2026-10-13). Position
`p` of a KV head goes on channel `p mod Ports`, as row `p div Ports` of
that head's region; the region sits at the same offset in every channel
and reserves `ceil(cap / Ports)` rows. A tile of `L` = 4 × `Ports`
positions is then 4 adjacent rows in every channel: one read of
`4 × head_dim × 2 / 32` beats per port (16 for SmolLM2). `KV_STORE` writes
one row to one channel either way. *Why:* architecture.md spread the
positions over the channels but left their place within a channel open;
with rows at their flat addresses a tile is 4 separate short reads per
port, and short DMA commands run below full rate (M5, memory model and
DMA: a 1-beat command at half rate).

**D-044 (2026-10-14) — An engine command reads the KV cache only after
the KV_STORE that writes it is done** (refines D-041 and D-042; owner,
2026-10-14). The controller gives the engine a `SCORES` or `VALUES` only
once every `KV_STORE` writing rows it reads is done: every write response
received, since the write path and the 32 read ports are separate AXI
ports with no order between them. Both come with `wait` and read the cache
ahead as D-042 allows: `SCORES` after `KV_STORE` K, its go once q after
RoPE is written (in the command list RoPE of q follows `KV_STORE`, so its
K reads overlap it); `VALUES` after `KV_STORE` V, its go once P is
written. Besides the read-after-write rule of D-041, a command that writes
an on-chip buffer is started only after every earlier command that reads
that buffer is done (the current command list already orders every such
case through its data; M7's controller must not rely on that). *Why:*
read ahead, the engine could read the newest K or V row before
`KV_STORE` writes it. The first draft of this entry gave `SCORES` without
`wait` and claimed q was ready after `KV_STORE`; it is not (review of the
command list, 2026-10-14).

**D-045 (2026-10-22) — The controller hands commands over in order and
stops at a non-engine command until the buffers it reads are written**
(refines D-041; owner, 2026-10-22). A vector-unit (or other non-engine)
command whose input is still being written holds every later command back;
engine commands are handed over at once and wait inside the engine
(D-042). Under this rule the current command lists order every overwrite
after the reads it would clobber through their data (D-044's remark), and
`commands.hazards()` finds none with or without the write-after-read rule;
D-044's rule stays. *Why:* the simplest controller. Handing over past a
waiting command (the vector unit waiting by itself) would let the next
engine command read its weights earlier: at most about 70 cycles for each
of SmolLM2's 181 waiting engine commands, up to 1.8% of a token at
position 63 and 0.6% at 8191 (estimate, from M5's start-up with and
without read-ahead), for a controller that tracks each buffer's readers.
Premature (owner). M7 measures the loss.

**D-046 (2026-10-22) — M5's bandwidth criterion is 90% of min(BW, 2·E·L)
times the perf model's engine use** (owner, 2026-10-22, at the M5
checkpoint). A matrix whose rows are not a multiple of `L` leaves ports idle
in its last pass; the perf model's engine use is the most that allows (0.90
for `down`, 576 × 1536, at 128 lanes). *Why:* the shortfall is the matrix's
shape, not the engine: `down` reaches 0.845 of the bound as written and every
large matrix at least 0.929 of the restated one (reports/M5.md). Using the
idle ports is Q-28.

**D-047 (2026-10-22) — M6's speed criterion: each vector-unit command
within 10% of the perf model, and < 10% of token time at position 0**
(owner, 2026-10-22, at M6's start). The report gives the share at 0, 1023
and the full window. *Why:* the perf model already puts the vector unit at
6.7% of SmolLM2's token at position 0 and 42.6% at 8191 (in order,
estimate): softmax's passes over the scores grow with the position, and no
option without a numerics change brings 8191 below about 18%. The long
positions are Q-29.

**D-048 (2026-10-22) — The vector unit's slots are one chain of 7 FP
stages** (owner, 2026-10-22). Every operation flows through the chain:
`exp` uses 7 stages, `rsqrt` 6, `recip` 5, plain `ADD`, `MUL`, `FMA` and
`MAX` one; integer logic between the stages does the bit tricks (first
guesses, rounding to an integer, exponent edits, clamps), and the opcode
picks each stage's use. A scaling numerics.md writes as a multiply by a
power of two becomes an exponent edit when the result is exact (same bits,
tested); multiplies that can round stay multiplies. *Why:* every operation
at 16 elements per cycle for about 112 FP units (about 140 k LUTs and 224
DSPs from M3's per-unit numbers, estimate), against about 300 for a
pipeline per function, and one pipeline to verify; looping through one FP
unit per slot would make softmax about 2.2× slower at long positions. Cost:
about 45-50 cycles of latency on every command (measured in M6). The chain's
length is set by `exp`, so one Newton step fewer saves no area.

**D-049 (2026-10-22) — S and P's row stride is a register, STRIDE,
independent of `cap`** (owner, 2026-10-22). Head h's scores and
probabilities sit at `h·stride` in S and P; `SCORES` writes there, `VALUES`
and the `LEN_T` vector commands read there. The host writes STRIDE once
with the command list; the controller passes it to the engine and the
vector unit. It needs `t` ≤ stride and a multiple of 16, the vector
unit's width (so every row starts on one of its 16-element groups; the
engine alone needs a multiple of E).
Tests use strides different from `cap`. *Why:* the two mean different
things (the KV cache's rows per head; the on-chip buffers' layout), and a
valid image may differ (S and P sized for the full context, the host
running a shorter cache). An equality built in would hide a mix-up of the
two. The 32-byte commands have no room for a second per-command field
(`SCORES` uses all but 2 bytes), and the stride is one value for the whole
list.

**D-050 (2026-10-23) — A reduction's final tree runs in a separate tree
unit** (owner, 2026-10-23). Each slot keeps 8 of a row's 128 partial sums
and accumulates them with its chain's first FMA (a 7-cycle loop for 8
partials). At the end of a row, a small tree unit (`TreeAdders` FP adders
and a max, 1 for now) takes its 128 partials and adds them in the spec's
fixed tree while the slots accumulate the next row; only the last row's
tree is exposed (about 170 cycles with one adder, estimate). *Why:* the
alternative, the slots doing the tree through a routing network, costs
about 55 cycles a row that never overlap, about 500 for softmax's 9 rows
(estimate). The tree unit is simpler, hides the trees at long positions
(where the vector unit is the limit, Q-29), and can take more adders for
a model with many heads at short positions without touching the slots.

**D-051 (2026-10-23) — The engine's step control is pipelined, and a
placed build of the engine comes before M6 ends** (owner, 2026-10-23,
design review). Today `fire` (`rtl/engine.sv`) is one cycle's AND of the
128 lanes' ready and the 32 ports' valid, fanned out to every lane and
port FIFO, and x reaches all 128 lanes in the same cycle. The lanes stop
pushing back (a lane's ready drops only for rows below 60 elements, which
a command's shape tells in advance), and the stall and the x broadcast go
through register stages with a small skid buffer per group of lanes: same
bits, cycle counts re-measured. Then one placed-and-routed Vivado build of
the 128-lane engine (moved from M8), its launch with the owner's OK and a
time limit. Max in the drain (D-054) waits (D-057). *Why:* the
engine is about 480k LUTs (M4's lane × 128, estimate), more than one of
the VU47P's SLRs (the dies it is built from; 3, to verify) holds, and a
crossing costs about 1 ns of the 4. M3 and M4 timed one lane out of
context, which says nothing about wires across the chip; a global path
found failing at the first full build would reopen a closed milestone.

**D-052 (2026-10-23) — F2 bring-up runs beside M6 and M7** (owner,
2026-10-23, design review; changes when D-026's measurements happen, not
what they are). M8's skeleton (the wrapper, registers, host DMA, HBM
bandwidth per port and in total) starts now instead of after M7; the RTL
work does not wait for it. Every launch needs the owner's OK with a time
limit. *Why:* all 7 platform values of the perf model are still
documented ones, the FIFO depth of 128 rests on an assumed 70-cycle
latency, and problems with the shell, the HBM IP, the AFI flow or the
clocks are cheaper found early and apart from design bugs.

**D-053 (2026-10-23) — The perf model counts each dependent command's
latency** (owner, 2026-10-23, design review). A vector command that waits
for an earlier result adds a measured `vector_startup_cycles` (the chain:
50 cycles at `MulRegs` = 2), a reduction its exposed tree (D-050; about
170 cycles, estimate), and `command_cycles` splits into the controller's
issue cost and its stops (D-045). Done in M6's real-data step; M6's speed
criterion (D-047) is judged on that model. *Why:* at SmolLM2's width a
vector pass is 36 cycles, less than the chain's latency, and most vector
commands wait for the one before: about 2,000-3,000 cycles a layer that
the model leaves out (estimate). `command_cycles`, still a guess, is
0.341 ms of a 1.642 ms token at position 0 (docs/perf.md, in order). The
fixes that latency points to are Q-30.

**D-054 (2026-10-23) — Softmax keeps D-022's order, with a fused pass and
the max from the engine** (owner, 2026-10-23, design review; amends D-031
and D-048). A new vector command computes `exp(a - b)` per row (b one
value per row) and the row's sum of the results, the operations and order
of numerics.md section 4, so the same bits; each slot gets an FP32 adder
before its chain and one after it (D-048's 7 stages unchanged; about
30-45k LUTs, and 14 cycles of latency more for two FMA stages' 7, estimate). It is built in M6
with the reductions. The engine's drain keeps the max of each `SCORES`
row as it writes it (`max` is exact, so its order is free) and writes it
to R, so softmax has no `MAX` command; built with D-051. Softmax then
reads the scores twice instead of 5 times. Perf model (docs/perf.md,
"fused softmax, max in drain", in order, estimate): SmolLM2 148 → 198
tokens/s at position 8191, 446 → 497 at 1023, 609 → 618 at 0; on M9's
256 lanes with HBM at 450 MHz, 177 → 253 at 8191. Each assumes the fused
pass and the max cost no cycles. *Why:* S and P fit on chip at the full
context, so online softmax's main gain (no stored scores) does not apply
here; its other gain, fewer passes, comes without a numerics change,
while online softmax would rescale the engine's partial sums in the middle
of a `VALUES` row (an engine redesign) and add commands per tile.

**D-055 (2026-10-23) — M9's engine keeps a margin over memory** (owner,
2026-10-23, design review). M9 sizes the engine with 10-25% more
multiply-adds per cycle than the memory delivers weights. *Why:* v0
matches them exactly (512 multiply-adds, 1024 B per cycle at 250 MHz), so
a design meant to be memory-bound is engine-bound at position 0 (1.190 ms
engine, 1.173 ms memory): idle lanes, start-ups and final sums all cost
tokens/s.

**D-056 (2026-10-23) — The baseline includes the F2 host's own CPU**
(owner, 2026-10-23, design review; extends D-008). M9 measures llama.cpp
decoding SmolLM2-135M-Instruct in BF16 on the f2.6xlarge's host CPU, at
the same positions as the accelerator, next to the Gemmini figures. It
runs in the same AWS session as an F2 measurement. *Why:* the same
instance at the same price is the comparison a reader asks for first;
Gemmini ran a model 500 times smaller on a 29.8 MHz soft core.

**D-057 (2026-10-29) — The fused softmax pass and the max in the drain wait
for a working core** (owner, 2026-10-29; defers D-054's fused pass and the
max in the drain, which D-051 was to build). Softmax stays six primitive commands (`MAX`,
`ADD`, `EXP`, `SUM`, `RECIP`, `MUL`), which run and pass their tests today.
Both changes are kept as a possible improvement (Q-31). *Why:* a simpler
structure first. The gain is small at short positions (609 → 618 tokens/s
at position 0, perf model, in order, estimate) and only large at long ones
(148 → 198 at 8191), and its hardware costs more than D-054 assumed: the
fused sum is exp's, after the chain, while `SUMSQ` must keep its single
rounding (`fma(x, x, p)`), so either a second accumulation point beside the
first FMA or an FMA, not an adder, after each slot's chain.

**D-058 (2026-10-30) — Speed waits for the real thing; the path to a token
on F2 comes first** (owner, 2026-10-30; changes D-047). M6's share of
token time at position 0 is a recorded number, not an exit criterion.
Speed work (Q-30, Q-32, Q-31) is judged after M9 on F2's measured numbers.
Bit-exactness stays a criterion everywhere; no new cycle-matching criteria
until there is hardware to match. *Why:* the share is not met (36.8% for
SmolLM2, perf model, estimate), but what can still stop the project is the
placed build, the controller and the F2 platform, not the vector unit's
speed. The cycles that miss it are inside the vector unit and are already
measured, so the finding does not depend on the modelled memory or engine.

**D-059 (2026-10-30) — The placed build comes before the engine's
pipelined step control** (owner, 2026-10-30; reorders D-051). The engine
as it is gets its placed-and-routed Vivado build first. The pipelined step
control is built only if that build fails timing, on the paths its report
names. *Why:* measure before building the fix; if `fire`'s fan-out meets
250 MHz the pipelining is not needed.

**D-060 (2026-10-30) — The engine's timing loop, with a stopping rule**
(owner, 2026-10-30; follows D-059). The placed build failed (worst slack
-4.698 ns at 250 MHz, D-051's step path), so D-051's pipelined step control
is built, then at most 3 placed builds of the engine, each with the
owner's go. If the third is still worse than about -0.5 ns, the target
clock becomes 200 MHz. *Why:* the datapath already meets 250 MHz (the lane
+0.17 ns, M4); what fails is one-cycle control across the chip, a known
pattern fixed by register stages. 200 MHz alone does not help: the worst
path is 8.68 ns, about 115 MHz. The cap keeps the tinkering from eating
the path to F2 (D-058).

**D-061 (2026-10-31) — Three tile slots** (owner, 2026-10-31; for
D-051). The engine's tile buffer has three slots, not two: its banks are
96 entries deep instead of 64, about double the tile buffer's LUT RAM
(roughly 20,000 to 30,000 LUTs, about 2% of the part; estimate). *Why:*
with the step decided ahead, a slot's last reads come a few cycles after
the decision that frees it, and a filled tile reaches step a few cycles
after its last write; with two slots fill would fall behind by those
cycles on every unit, about 25% on SmolLM2's attention (estimate). Two
slots and the bubbles was the alternative.

**D-062 (2026-10-31) — The KV cache in the lanes' order; no tile buffer**
(owner, 2026-10-31; revises D-028 and D-043, replaces D-061; for D-060).
K and V are stored the way weights are, so port `p` feeds its own lanes
`4p` to `4p+3` on attention too, and `SCORES` and `VALUES` stream from
HBM like a `MATVEC`:
- K: in the tile of positions `128j` to `128j+127` of a KV head, channel
  `c` holds positions `128j+4c` to `128j+4c+3`; for `k` = 0, 4, …, it
  holds the beat `K[q][k…k+3]` for those 4 positions. `SCORES`: lane `q`
  takes position `q`, as now. A tile is still `d/4` sequential beats per
  port (16 for SmolLM2).
- V: stored as `Vᵀ`, rows the head's dimensions, columns the positions.
  Channel `p` holds dimensions `4p` to `4p+3`, beat by beat over the
  positions. `VALUES`: lane `j` takes output dimension `j`, as now. A pass
  of two heads (`d` ≤ `L/2`) puts its second head on ports `L/8` and up,
  and with GQA a KV head can be either, so V is stored once per such half
  (2 copies for SmolLM2: 94 → 189 MB at 8192 positions, estimate; reads
  unchanged).
- `KV_STORE` writes a K row as one burst of `d/4` beats, and a V row as
  one beat per channel and copy, each with 8 bytes enabled (AXI write
  strobes). For SmolLM2 about 7,200 cycles per token against about 700
  now, about 2% at position 1023 if not hidden (estimate; a one-beat
  write at half rate, as the memory model's reads).
- Gone: the tile buffer, its rotations, fill, the tile slots (D-061) and
  the bank mapping. Each port's 256-bit FIFO becomes 4 FIFOs of 64 bits,
  one per lane, each with its own pointers.
- Same numerics: `SCORES` sums over `d`, `VALUES` over the positions, 4 at
  a time in the same order as now (to confirm with the tests).
- To verify: the HBM controller's handling of partial writes (strobes);
  the first full build (M8) measures their rate.

*Why:* the first placed build after D-051 (-3.096 ns) fails on wires, not
logic: about 145 of the 177 worst paths are the tile buffer's rotations
and addresses, two 8,192-bit crossbars that exist only because the cache
was stored by position. With K and V in the lanes' order no data crosses
the engine, and about 85,000 to 100,000 LUTs go (estimate: the tile's LUT
RAM and the rotations), mostly from the fullest SLR. D-028's reason for
the tile buffer was V without a second copy; a copy costs HBM space only,
which is plentiful. A one-head `VALUES` pass instead of the copy halves
`VALUES`' rate (about 10% more engine time at 1023, estimate); pairing
passes by KV head does not balance with SmolLM2's 3 KV heads.
*Correction (2026-10-31):* the second head of a pass reads ports `d / 4`
and up (its lanes `d` and up), not `L/8`: the same for SmolLM2 (d = 64),
different for d < L/2 (rtl/engine.sv, `commands.kv_lane_order`).
*Implementation note (2026-10-31, proposed; owner to confirm):*
- A V cache is written by a new opcode, `V_STORE` (17), not a flag:
  the FLAGS byte's 8 bits are all taken. `KV_STORE` now writes K only.
- The writer sends a burst per group of 16 values for K (4 beats) and
  a one-beat burst per channel and copy for V, a beat a cycle (port_wdma's
  address channel now takes a burst a cycle, two deep). SmolLM2 at
  position 0 on the RTL: 179,893 cycles for the vector unit's whole step,
  the perf model's exactly; the stores add about 15 µs a token at 250 MHz
  (perf model, 0.705 → 0.720 ms of vector time; 523 → 519 tokens/s in
  order).
- The vector unit keeps up to 128 write bursts in flight
  (`WrOutstanding`): a beat a cycle at a write latency of 70 cycles needs
  more than 70, and V's bursts are one beat each. F2's limit on
  outstanding writes, and how its HBM controller handles partial writes,
  are to verify in M8.
- A K row may be any power of two from 16 values up (a burst per group, so
  no AXI3 limit on `d`); `V_STORE` needs `d` <= L.

**D-063 (2026-11-01) — A few placed builds, not at most 3** (owner,
2026-11-01; revises D-060). The timing loop is no longer capped at 3
placed builds: it takes a few, each with the owner's go and a time
limit, while each one closes a named family of paths. The owner decides
when to stop and fall back to 200 MHz. *Why:* build 2 is at about
-0.53 ns with three small, named families left (the timing loop, step
3); a hard cap would force the 200 MHz fallback with fixes in hand.

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

**Progress** (full notes: [PLAN.md at 008a76c](https://github.com/cattabiani/cabosse/blob/008a76ce2cacc93815b940d3ea27d359325c9c78/PLAN.md)):
- [x] Spec and primitives: `docs/numerics.md` v0 (D-019, D-020); `bf16`,
  `add`, `mul`, `fma`, `mac` with exact-oracle tests; rsqrt, recip, exp with
  parameters chosen by measurement (Q-08: 2 Newton steps, exp degree 4)
  (PRs #1, #2).
- [x] Golden decode step of SmolLM2: dot product, reductions, RMSNorm,
  softmax (D-022), RoPE, SwiGLU, flat KV cache (D-024), tiny configs
  (PRs #4-#7).
- [x] Against `transformers` (PR #8) and greedy fixtures (PR #9): top-1
  99.30% against FP32, where BF16 `transformers` gets 98.75%.
- [x] Checkpoint, 2026-10-04: [reports/M1.md](reports/M1.md) (PRs #10, #11).

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

**Progress** (full notes: [PLAN.md at 008a76c](https://github.com/cattabiani/cabosse/blob/008a76ce2cacc93815b940d3ea27d359325c9c78/PLAN.md)):
- [x] F2 platform facts (`docs/f2.md`, Q-13) and the perf model
  (`model/perf.py`, `docs/perf.md`): inputs with status and source, our
  measurements replacing documented values.
- [x] `docs/architecture.md`: engine (D-027), memory and clocks (D-028),
  controller, commands and registers (D-029), vector unit (D-030),
  primitive vector commands (D-031); `commands.run()` gives the golden
  decode step bit for bit.
- [x] `A` = 16 and `S` = 128 in the spec and the golden model, comparison
  rerun (`reports/data/M2-*-compare.json`).
- [x] F2 check with AWS's `cl_sde` image (D-025, D-026): PCIe and
  host-card rates in `reports/data/f2/`; the AWS budget (Q-16).
- [x] Checkpoint: [reports/M2.md](reports/M2.md).

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

**Progress** (full notes: [PLAN.md at 008a76c](https://github.com/cattabiani/cabosse/blob/008a76ce2cacc93815b940d3ea27d359325c9c78/PLAN.md)):
- [x] Toolchain, `rtl/sources.f`, the bulk harness (`verif/bulk.py`) and CI
  for the RTL.
- [x] FP units, first CVFPU, then our own (D-038, D-039): bit-exact on
  10⁸ inputs per operation, every BF16 pair and every special-value
  combination.
- [x] The 4-cycle accumulate loop closes at 250 MHz on F2's part, and
  `fp32_fma` too (Vivado logs in `reports/data/f2/`).
- [x] Cost of subnormals in Yosys cells (D-016); `Ftz` kept.
- [x] Checkpoint: [reports/M3.md](reports/M3.md).

### M4 — Dot-product lane ✅
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

**Progress** (full notes: [PLAN.md at 008a76c](https://github.com/cattabiani/cabosse/blob/008a76ce2cacc93815b940d3ea27d359325c9c78/PLAN.md)):
- [x] `rtl/dot_lane.sv` with cocotb tests; the final sum as a conveyor belt
  (owner, 2026-10-08).
- [x] Vivado: 250 MHz met, 3,762 LUTs and no DSPs a lane
  (`reports/data/f2/2026-10-08-lane-belt.txt`).
- [x] Bulk driver and slow tests: 10⁸ pairs and a SmolLM2 decode step's
  rows, bit-exact.
- [x] Checkpoint: [reports/M4.md](reports/M4.md).

### M5 — Matrix-vector engine with simulated memory ✅
**What:** `L` lanes, weight DMA over AXI, tiling, the KV tile buffer for
attention, and an AXI memory model with configurable bandwidth.
**Why:** this is where bandwidth is won or lost.
**Done when:** bit-exact on one full SmolLM2 layer (`MATVEC`, `SCORES`,
`VALUES`) and the classifier with real data. Achieved bytes/cycle is ≥ 90%
of `min(BW, 2·E·L)` on large matrices (1024 B/cycle at full bandwidth; the
criterion predated D-027 and read `2·L`). On `SCORES` and `VALUES` at long
positions, engine use is ≥ 90% of the perf model's prediction. Verilator
speed is measured, to plan M7. Row edge cases tested: rows fewer than `L`,
rows not a multiple of `L` (masked last pass, e.g. SmolLM2's 192-row k/v
with `L` = 128), a single row. Idle lanes must not write results.

**Answers to the start-of-milestone questions (owner, 2026-10-09):**
- Scope: `MATVEC`, `SCORES` and `VALUES`, with the KV tile buffer (its
  banking serves both read directions, and the next tile loads while the
  engine reads the current one). Attention is the engine's limit at long
  positions (perf.md), so it is not left for later. The testbench stands in
  for the controller (one command at a time) and the vector unit (softmax
  from the golden model; the KV cache preloaded in the memory model).
  `KV_STORE` stays with M6/M7.
- Speed: besides the two criteria above, every engine command of the layer
  test reports its measured cycles next to the perf model's; a difference
  above 10% means fixing the hardware or the model, and the report says
  which. All speed figures are counted in simulated cycles.
- Ports: the lanes run in lockstep on one control (as planned in M4). A
  FIFO between each port and its 4 lanes absorbs pauses; the engine steps
  when every FIFO has a beat. The FIFO depth is a parameter, sized in M5
  against random per-port pauses; provisional until M8 measures F2's HBM.
- Buffers: M5 builds the engine's side: a read port for `x` (4 values per
  cycle, broadcast; 2 × 4 for `VALUES`, whose lanes serve two heads), and a
  write port sized for `SCORES` (128 results every 16 cycles) with the
  FP32 → BF16 rounder for BF16 destinations. Behind them, plain memories
  the testbench fills and reads. The full buffer arrangement comes with the
  vector unit (M6, M7).
- Memory: 32 read-only AXI ports of 256 bits, in the subset common to AXI3
  and AXI4 (bursts of at most 16 beats, none crossing 4 KB), so the F2
  wrapper connects them to the HBM ports directly. The model, in C++ in the
  Verilator driver, sets latency, requests in flight, bandwidth per port and
  random pauses per port (independent seeds), and checks itself (every
  request answered once, in order, with the right beats). It also brings the
  valid/ready loop shared by the bulk drivers (`stream.h`). All engine tests
  use it; cocotb stays for small blocks without memory. `cocotbext-axi`
  (MIT) may be added temporarily to debug, said in the PR and removed after.
- Lanes in tests: `L` = 4 × ports, since a port's 32-byte beat holds 4
  weights for each of 4 lanes (a free ratio needs a crossbar the chip does
  not have). Lanes per port is a named parameter derived from the beat
  width and `E` (it becomes 8 in M9). Configurations: 1 port / 4 lanes
  (degenerate), 2 / 8 (plain), 5 / 20 (prime ports, `L` divides no SmolLM2
  size) in CI; 32 / 128 for real data and speed (`slow`). The attention
  mapping works for any `L`, including `L` below the head dimension. Small
  configurations check bits and completion, not speed. Stress: row counts
  around `L`, columns around beats, bursts and 4 KB boundaries, per-port
  pauses and latency, one slow port, positions around `L`. CI builds the
  128-lane model and runs one short `MATVEC` if the build time allows,
  otherwise lints it.
- Real data from the command list: a recording hook in `commands.run()`
  gives each engine command's fields, input buffer and output bit for bit
  (replacing the M4 test's call counting, which moves onto it). New: the
  reordering of weights into the lanes' layout (architecture.md, "Weight
  layout"), part of the host's model loading, which the model left out.
  Data: the first and last layers' engine commands and the classifier, at
  positions 63, 1023 and 8191 if affordable (closes M4's open issue 4), at
  `L` = 128. A long position is reached by filling the KV cache with
  `decoder.step` and copying it into `run()`'s memory image (to verify).
- Verilator speed: build time and simulated cycles per second of the
  128-lane model, three runs on a quiet machine, reported as a range; with
  the perf model's cycles per token, an estimated wall time per simulated
  token in M7, at a short and a long position.
- No AWS: the engine's Vivado run moves to M8.

**Steps (one branch and PR each):** these answers; command-list recording
and weight layout; memory model and DMA; engine `MATVEC` (two PRs, owner
2026-10-11: the engine with small configurations, then real data and speed
at 128 lanes); engine attention (two PRs, then a small one, owner
2026-10-22: one walker for `MATVEC` passes and attention units in issue,
step and drain, so a sequencing fix is made once); checkpoint report.

**Progress** (full notes: [PLAN.md at 008a76c](https://github.com/cattabiani/cabosse/blob/008a76ce2cacc93815b940d3ea27d359325c9c78/PLAN.md)):
- [x] Answers (PR #50); command-list recording and the lanes' weight
  layout (`commands.record_step`, `lane_order`).
- [x] Memory model and DMA: `rtl/port_dma.sv`, `rtl/stream_fifo.sv`,
  `verif/bulk/axi_mem.h` (PR #52).
- [x] `MATVEC` (PRs #53, #54): three stages (D-041), read-ahead (D-042),
  FIFO depth 128.
- [x] Attention (PRs #55, #56 and the walker PR): tile buffer, `SCORES`,
  `VALUES` (D-043, D-044); real data at positions 63, 1023 and 8191
  bit-exact and within 10% of the perf model.
- [x] Checkpoint: [reports/M5.md](reports/M5.md); the bandwidth criterion
  restated (D-046).

### M6 — Vector unit
**What:** RMSNorm, softmax, SiLU/SwiGLU, RoPE, residual add, BF16 rounding,
KV cache writes.
**Done when:** each operation is bit-exact on random and real activations.
Every vector-unit command of the real-data test takes within 10% of the
perf model's cycles. The perf model's share of token time at position 0 for
SmolLM2 and Qwen2.5-0.5B is recorded, not a criterion (D-058; it was < 10%,
D-047).
**Check:** the area of the D-020 function units (rsqrt, recip, exp); fewer
Newton steps or a lower exp degree only through a numerics change (was Q-08).
**Possible optimization:** online softmax (D-022), only if score storage or
the extra pass shows up as a bottleneck. It needs a spec change.

**Answers to the start-of-milestone questions (owner, 2026-10-22):**
- Scope: the vector unit (`ADD`, `MUL`, `FMA`, `EXP`, `RECIP`, `RSQRT`,
  `SUM`, `SUMSQ`, `MAX`, `ROTATE_HALF`, with their flags; a BF16
  destination rounds on write), `KV_STORE` (the core's first AXI write
  port) and `LOAD` (embedding row, RoPE rows, norm weights). `OUTPUT` waits
  for M8: the host address exists only on F2, and M7's runtime reads
  `LOGITS` from the simulation. The testbench stands in for the controller
  (one command at a time) and the engine (its results from the golden
  model).
- Criterion: restated (D-047). The perf model puts the vector unit at 6.7%
  of token time at position 0 and 42.6% at 8191 for SmolLM2 (in order:
  memory and engine overlapped, vector and commands after them; estimate),
  because softmax's five passes over the scores grow with the position.
  The report gives the share at 0, 1023 and the full window; the long
  positions are Q-29.
- Function units: one chain of 7 FP stages per slot that every operation
  flows through (D-048).
- Buffers: one memory per buffer, one read and one write port of 16
  elements per cycle each, the format fixed per buffer (a BF16 read converts
  up exactly, a write rounds). No vector command reads two operands from
  the same buffer (checked on both ladder models' lists). Sizes are
  parameters; the vector unit generates the addresses (rows of `m`, or of
  `t` at stride `cap` with `LEN_T`, now the STRIDE register, D-049;
  `B_PER_ROW`, `B_ROW`). A port is muxed
  to the command that owns it; `commands.hazards()` gains a check that no
  two running commands use the same port of a buffer, so the hardware needs
  no arbitration. The engine's ports (x read, result write) are added in M7
  with the top level; M5's engine tests keep their testbench memories.
- S and P at stride `cap`: the golden command list moves to rows of `t` at
  `h·cap` (the first PR after these answers), for any `cap`; the engine's
  need for `cap` a multiple of `L` stays checked in its tests.
  `real_attention` drops its padding and re-striding. Layout only, not a
  numerics change. *Why stride `cap`:* a head's row starts at a fixed,
  aligned address whatever the position; with packed rows it moves every
  token and is misaligned for `VALUES`' 4-wide and the vector unit's
  16-wide reads. The buffer is sized `heads × cap` anyway. (Superseded
  the same day by D-049: the stride is the STRIDE register, not `cap`.)
- Tests: a C++ Verilator bulk driver per level. One slot's chain per
  operation against the golden functions: special values, the edges of
  each bit trick and random inputs in CI; 10^8 random inputs per operation
  and the M1 golden model's exhaustive ranges (one bit-trick period for
  rsqrt and recip, every exp input with a normal result) in `slow`.
  Exhaustive (owner, 2026-10-22): `EXP`, `RECIP` and `RSQRT` on all 2³²
  inputs, split into chunks run in parallel on the CPU cores (a GPU cannot
  run Verilator); on demand, never in CI. Its wall time is measured on a
  chunk first, and the run is started only with the owner's OK (possibly
  overnight). Whole
  commands with M5's AXI memory model, which gains a write side that checks
  itself (every write answered once, the right bytes and strobes): random
  commands and random dependent sequences against `commands.run()`;
  summation-order inputs with large cancellations, lengths 1, 15, 16, 17,
  127, 128, 129 and a long row; `m` = 1, `t` < 16, `t` not a multiple of
  16, `t` = 1; `B_PER_ROW`/`B_ROW` with `NEG_A`/`NEG_B`; `cap` > `t` (stale
  data past a row stays unread and unwritten); random pauses on every AXI
  channel and long latencies (`KV_STORE` done only after every write
  response, D-044). A whole decode step's vector, `LOAD` and `KV_STORE`
  commands on the tiny config in CI. Real data (`slow`): SmolLM2's first
  and last layers and the final norm at positions 0, 63, 1023 and 8191,
  every command bit-exact with its cycles next to the perf model's. Hand
  mutations as in M5, one of them the final tree's order. The vector unit
  beside the engine on shared buffers is M7's.
- Latency in the perf model: a measured `vector_startup_cycles` (and a
  reduction tail if it differs) per vector command that waits for its input,
  almost all of them under D-045; `command_cycles` stays the controller's
  guess until M7. About 936 vector commands per SmolLM2 token × ~65 cycles
  ≈ 0.24 ms (estimate), so position 0 may land above 10%. If so, the report
  says so with the options (a short path for plain `ADD`/`MUL`/`FMA`,
  fused RMSNorm scalars, D-031); no tuning to pass. The vector unit takes
  the next command while the previous drains; the controller decides when
  that is safe (D-045).
- Area: Yosys cells for one slot and the whole unit, and an estimate in
  LUTs and DSPs from M3's Vivado numbers per unit; Vivado's numbers come
  with M8's build. No AWS in M6.

**Steps (one branch and PR each):** these answers; S and P at stride `cap`
and the buffer-port check in `commands.hazards()`; one slot's FP chain
(every operation) with its bulk tests and area; the vector unit with its
buffers (whole commands, sequences, the tiny step; two PRs, owner
2026-10-23: the buffers and the elementwise commands, then the reductions,
D-050; the fused softmax command waits, D-057); `LOAD` and `KV_STORE`
with the memory model's write side; real data, speed and
`vector_startup_cycles`, with the perf model's latencies (D-053); the
engine's placed Vivado build, then its pipelined step control only if
timing fails (D-059; it failed: the timing loop below, D-060); checkpoint
report.

**The timing loop (D-060, owner 2026-10-30; D-063):** one branch and PR per
iteration, a few placed builds, each with the owner's go.
1. ✅ (2026-10-31, D-061) Build D-051's pipelined step control in `rtl/engine.sv`: no global
   stall (the lanes take every step; the step is decided a few cycles
   ahead from what the FIFOs and the command's shape already tell), and
   the step, the x broadcast and the tile buffer's read addresses reach
   the lanes and banks through register stages, copied per group of lanes
   or banks. Same bits (every engine test passes unchanged); cycle counts
   re-measured, the perf model's `engine_startup_cycles` updated.
2. ✅ `platforms/f2/timing/timing.tcl` reports every family of failing
   paths, not only the worst 3: the failing paths grouped by endpoint,
   and the logic-level distribution.
   Done on `m6/step-pipeline`: the engine tests pass (226, and the 30 at
   128 lanes) with the same bits, one limit raised: `STARTUP_CYCLES`
   (a command's first result) 70 → 80, as it measures 73 at 128 lanes, 62
   before; `engine_startup_cycles` 64 → 72 (measured,
   reports/data/M5-engine.json); SmolLM2's commands within -4.5% to +1.7%
   of the perf model, the first one apart. A review before the first
   build (2026-10-31) found that Vivado would merge the per-group copies
   (timing.tcl now keeps them) and long paths left in the rotations, the
   per-bank addresses, the min of the FIFO counts, step's next unit and
   the drain; fixed before the build, which was stopped after 12 minutes. Predictions for its placed
   build, written before it: the worst slack better than -1 ns; the step
   to tile-address path gone; most of the 288,260 failing endpoints gone
   (they hang off that path's nets); what may remain: the drain's `go`
   (lane outputs to the drain and back), issue's handshake over the 32
   ports, the VALUES read data's cross-chip wires after the rotation,
   and step's own counter adders.
3. Placed build of `engine_path` (scripts/f2/time_loops.sh; c6i.4xlarge
   is enough: the first build peaked at 16.5 GB), with the owner's go and
   a time limit for each launch. Fix the paths it names; repeat.
   Build 1 (2026-10-31, commit 6dbfdae,
   reports/data/f2/2026-10-31-engine-placed.txt): worst slack -3.096 ns
   (-4.698 before), 177,270 of 1,110,481 endpoints failing (288,260 of
   986,478 before). Against the predictions: the step-to-tile-address
   path is gone; the slack is not better than -1 ns; 39% fewer failing
   endpoints, not most. The failing paths are now short wires, not deep
   logic: the worst is a bank's own read-address register to its LUT RAM,
   2 levels, 6.45 ns of a single net (fan-out 128, 104 rows away). Among
   the 177 worst paths the log kept, 0 to 3 levels each (one has 5), are the tile
   buffer's rotations (read rotation 71, shift copies 37 and their `_rep`
   ones, write side), the per-bank addresses (12), and the DMA FIFOs' own
   write pointers to their LUT RAM (`stream_fifo` `count_q`/`wr_q`,
   fan-out 674, up to -2.97 ns; code this step did not change). The
   middle SLR holds most of the LUT RAM (47,369 of 58,214 LUTs as memory)
   and is 76% full in LUTs. Cost of builds 0 and 1: about $2 (estimate,
   2.6 h of c6i.4xlarge).
   Next (owner, 2026-10-31): D-062 (KV cache in the lanes' order, no tile
   buffer, per-lane FIFOs), then build 2, which prints the full
   per-module utilization, a congestion report and every failing path
   family. What it names decides the next step: DSPs for the lanes'
   arithmetic (16 of 9,024 used), floorplanning, or neighbour chains for
   the x broadcast and the drain.
   D-062's engine side done on `m6/kv-lane-order`: SCORES and VALUES
   stream from the lanes' order like a MATVEC, the tile buffer and fill
   gone, each port's FIFO split per lane; every engine test passes with
   the same bits, unchanged (110, and 30 at 128 lanes). `KV_STORE`'s new
   layout is the next PR; build 2 runs on this one.
   D-062's store side done on `m6/kv-store-layout`: `KV_STORE` and a new
   `V_STORE` write the lanes' order with strobes (rtl/vec_unit.sv), checked
   bit for bit on the whole memory, SmolLM2's whole step included. See
   D-062's implementation note (2026-10-31).
   Build 2 (2026-10-31, commit 4fc7ca1, D-062's engine side,
   reports/data/f2/2026-10-31-engine-placed-build2.txt): only the console's
   tail survives, so the WNS line, the count of failing endpoints and
   the first rows of the path table are lost. The worst surviving path
   is -0.534 ns (-3.096 before). The 36 surviving rows (-0.534 to
   -0.409 ns) form three families, all logic, not wires:
   - The drain's look-ahead (14 rows, -0.534 to -0.464, 20 levels, 54%
     route): `drain_unit_q` → `unit_next` → `res_lanes` → the per-lane
     compare → `gen_wgroup[k].ok_next_q`, for 14 of the 16 write groups.
   - The lane's accumulator loop (16 rows, -0.446 to -0.410, 10 to 12
     levels, about 70% route): the slot's tag, `written_q` and `file_q`
     → the addend mux → `fp_add`'s first stage (`s1_q.lo`). The lane
     alone met 250 MHz (+0.17 ns, M4); in the full chip it misses.
   - Issue (2 rows, -0.409, 5 levels, 88% route): `iss_ports_q` → the
     AND over the 32 ports' `dma_cmd_ready` → the reset of `iss_taken_q`.
   One more row is a wire: the step's mask copy to a lane's tag
   (`pipe_q` → `tag_q` SRL, 0 levels, 98% route, -0.416). Paths better
   than -0.409 ns are not in the log. Size: 555,726 LUTs (was about
   641,000), middle SLR at 68% of LUTs (was 76%), 38,883 LUTs as memory,
   16 DSPs; no congestion window above level 5.
   Next (proposed, owner to confirm): register stages for the three
   families, then build 3. time_loops.sh now prints the WNS and the
   families again at the very end, so the console's tail holds them.
   Round 3 (owner, 2026-11-01), on `m6/timing-round3`: the drain's
   successors (the unit after, the queue head's first unit) registered
   with their lane counts two cycles ahead, their "results are there" for
   group 0 only; each lane unit's own copy of the read and write tags;
   issue's "every port took it" registered (2 cycles more per DMA
   command); no SRLs (`-shreg_min_size 16`). Same bits, and SmolLM2's
   engine cycles unchanged (reports/data/M5-engine.json). Build 3 runs
   the default flow and an `explore` flow (timing directives) side by
   side; the default flow is the gate, since the shell build (M8) runs
   it. Left for build 3 to confirm, from the review: the lane rows that
   start at `file_q` (-0.446) keep their logic, so the lane family may
   stay (then: read the file a cycle early, with a bypass of the sum);
   issue's gather and its reset broadcast still cross the chip, a cycle
   each (then: an AND per group of ports, as step's minimum); the
   drain's `go` (16 groups' ok to 128 lanes' ready) and the current
   unit's lane count to 128 compares are central broadcasts.
   Build 3 (2026-11-01, commit 9bd2faf, c6i.8xlarge, both runs side by
   side, reports/data/f2/2026-11-01-engine-placed-build3.txt; routed
   checkpoints in S3): the drain and issue families are gone in both.
   - Default flow: WNS -0.354 ns, 5,971 failing endpoints (1h26). The
     lane's accumulator loop is what is left: `rd_slot_q`, `written_q`,
     the tags and `file_q` → `fp_add`'s `s1_q.lo` (2,928 paths from
     `rd_slot_q` alone; worst 12 levels, 71% route). Then the step's
     `dlog` to the lanes' `x_q`, the multiplier's product to `p_q`, the
     step's unit to `step_active_q`, and the DMA buffers' write enables
     (-0.334).
   - Explore flow: WNS -0.079 ns, 586 failing endpoints (2h56). No
     family stands out: the x broadcast to the lanes (`gen_group.x_q` →
     lane `x_q`, 1 LUT, 97% route, about 2 ns per hop), `rd_slot_q` →
     `s1_q.lo` (280 paths), the DMA buffer's count, the scale unit's DSP
     product, the lane's FIFO write enable.
   - Size: 517,289 LUTs (was 555,726), middle SLR 66%, 37,088 LUTs as
     memory, no SRLs, 16 DSPs; no congestion window above level 5. Peak
     memory 16.3 GB per run.
   Correction to round 3: the shell build does not run Vivado's default
   flow. aws-fpga's `aws_build_dcp_from_cl.py` (branch f2, c34f88e)
   places with `SSI_SpreadLogic_high`, then runs `phys_opt_design` and
   `route_design -tns_cleanup` with `AggressiveExplore`, and takes each
   directive as an option (`--place`, `--phy_opt`, `--route`). So the
   explore run, not the default one, is the closer measure of M8.
   Next (proposed, owner to confirm): make the gate the shell's own
   directives; add a register stage to the x broadcast (the step's
   group → lanes) and read the lane's file a cycle early; build 4
   with the shell's directives only.
   Round 4 (owner, 2026-11-01): fix every known family at once, against
   a harness closer to the shell, then build 4. Harness, on
   `m6/round4-shell-harness`: a `shell` flow is now the gate. It places
   the design inside the Small Shell's CL region (`pblock_CL`, read at
   build time from the aws-fpga checkout: the file is under the Amazon
   Software License, so it is not copied into this repository), with the
   shell's directives (placement `SSI_SpreadLogic_high` or another
   given per run, `phys_opt_design` and `route_design -tns_cleanup` with
   `AggressiveExplore`) and no physical optimization after routing. The
   clock gets 0.065 ns of uncertainty on top of Vivado's 0.035 ns, for
   the shell MMCM's jitter (about 0.1 ns in all, an estimate), and a run
   passes at a worst slack of +0.1 ns or more, for the shell's clock
   tree and logic that an out-of-context build does not see (an
   estimate).
   RTL, on `m6/round4-rtl`, every family build 3 named, same bits (every
   test passes unchanged): the lane reads its accumulator file a cycle
   early into a register `fp_add` takes, the sum written that cycle
   bypassing it (the loop stays 4 cycles), and its product gets a second
   register (`MulRegs` 2); x reaches each port's lanes through a register
   per port holding the head they take (no mux or `dlog` at the lanes);
   issue's "every port took it" is an AND per group of 4 ports, then over
   the groups, with the same latency; issue's and step's next unit's
   ports come from registers (step's units under 3 beats, VALUES at t <=
   8, wait a cycle or two at their end); the DMA buffers' write enables
   come from registered full flags (`stream_fifo`); the scale's multiply
   gets a third register (`fp_product` Regs 3). SmolLM2's engine commands
   each take 2 cycles more (latency, reports/data/M5-engine.json);
   `engine_startup_cycles` 72 → 74. Not done: the drain's `go` and lane
   counts (not failing in build 3; a register there would cost SCORES a
   cycle per unit). If build 4 still shows issue's or step's unit paths:
   keep `rem` and `last_tile` in `unit_t` (from the review).
4. Stop when the worst slack is >= 0 at 250 MHz. If the builds stop
   closing named families (the owner decides, D-063), target 200 MHz (5 ns),
   with a decision-log entry and the perf model's clock changed (about 20%
   fewer tokens/s where the engine bounds them). Floorplanning (SLRs,
   the HBM side) waits for the full build inside the shell (M8, M9).

**Answers for `LOAD` and `KV_STORE` (owner, 2026-10-29):**
- Both run in the vector unit's own pipeline, in order with its other
  commands: `LOAD`'s beats take the place of a's read (BF16 converted up
  exactly), pass the chain unchanged and are written like any result;
  `KV_STORE` reads a, passes the chain, and its write stage rounds to BF16
  and sends the beats to AXI instead of a buffer. No write collides with
  another, so the buffers need no arbitration; the ext port stays the
  testbench's (and the engine's, M7).
- One AXI master of the vector unit's own (read and write channels, 256
  bits, one ID), with global HBM addresses (channel and offset; `KV_STORE`
  writes D-043's place, where the engine reads). In M8 the F2 wrapper
  merges it into one HBM port and the HBM switch carries it to the other
  channels (to verify); if the switch fails, `KV_STORE` moves to the engine
  ports' unused write channels.
- `LOAD BY_TOKEN` reads a row-ordered embedding: with tied embeddings the
  host stores a second copy for `LOAD` (about 57 MB for SmolLM2, 272 MB for
  Qwen2.5-0.5B), the classifier keeps the lanes' order.
- A `KV_STORE` is done after its last write response (D-044), and the unit
  keeps taking commands meanwhile; `done_o` stays one pulse per command in
  list order, so later commands' dones wait behind it.

**Answers for the real-data step (owner, 2026-10-30):**
- The real-data test runs SmolLM2's whole decode step (every vector,
  `LOAD` and `KV_STORE` command, the engine's stood in as in the tiny
  step), not only the first and last layers and the final norm: the tiny
  step's test reused, no starting state taken mid-step.
- The perf model counts `LOAD` and `KV_STORE`: a `LOAD` costs its groups
  and a `load_latency_cycles` (its read starts when the unit reaches it,
  and later commands wait behind it), a `KV_STORE` a
  `kv_store_latency_cycles` before its done (D-044; `SCORES` and, through
  the in-order dones, the next waiting command wait for it). Both measured
  in simulation at F2's documented read latency, again on F2 in M8.
- `command_cycles` becomes `issue_cycles`, the controller's cost to hand a
  command over, 4 a command (guess, M7 measures it); the stops are the
  latencies above, `vector_startup_cycles` and the reductions' trees.
- The test's memory holds only what the unit reads and writes (a map of
  its own: the caches at low offsets in every channel, the `LOAD` tensors
  in channel 0 above them); D-040's offsets stay M7's. The positions run
  one at a time (about 2 GiB of memory each at most, estimate).

**Progress:**
- [x] Answers above (PR #59).
- [x] S and P at the STRIDE register's stride, and the buffer-port check
  (PR #60). Owner, 2026-10-22: the stride is not `cap` (D-049). `run()`
  keeps every buffer at its full size, starting as NaN (stale data), and
  every command reads and writes only its span: a read of an element no
  command wrote gives NaN and fails the bit-exact tests. The KV cache's
  unwritten rows are NaN too. `Layout.stride` is the register's value
  (`cap` by default); the decode-step tests use strides above and below
  `cap`. `EngineCall` records the span an engine command reads and
  writes, the whole cache, `t` and the stride; `record_step` takes a
  `cap` and a stride, and `verif/engine.py`'s `real_attention` passes the
  recorded command through (a cap of whole tiles, the stride cap + 16).
  The engine takes `cmd_stride_i` (queued with each command) in place of
  `cap` for S and P; its tests draw the stride at random above and below
  `cap`, and a hand mutation (the stride from `cap`, as before) fails
  them. The engine tests compare only the rows a command writes
  (`written()`: scores past t stay stale, and the C++ driver rejects a
  write there). `commands.hazards()` finds a "read port" hazard: two
  commands of different units (the engine, the vector unit) reading one
  buffer at once; none in the ladder lists, two hand-made lists have them.
- [x] One slot's FP chain (D-048, PR #61): `rtl/fp32_chain.sv`, 7 stages of
  `fp32_fma` and integer steps; 50 cycles at `MulRegs` = 2. In Q3 the
  scalings of recip's and rsqrt's inputs were to be exponent edits; a
  subnormal input needs normalizing, so they stay multiplies, and rsqrt
  uses 7 stages, recip 6 (the chain stays 7). Tests
  (`verif/tests/test_fp32_chain.py`), bit for bit against the golden
  functions: specials (every triple for add, mul, fma), each function's
  edges (exp's ties and integers of t, clamp, subnormal results; the
  inputs recip and rsqrt scale) and random bits, operations interleaved;
  slow: 10^8 per operation and the M1 exhaustive ranges (87 s on 16
  cores). Every input of exp, recip and rsqrt (`CABOSSE_SWEEP=1`, on
  request): all 2^32 bit-exact for each, 21 min 26 s on 16 cores (exp
  457 s, recip 440 s, rsqrt 389 s; 2026-10-23, commit 05519c1). Hand
  mutations: 8 caught; one (exp's clamp at 151) gives the same bits (past
  ±150 the result is +inf or rounds to +0 either way). MAX is not in the
  chain: it is only a reduction (`MAX` per row), so it goes with the
  reductions (the vector-unit step). Area (`scripts/measure_m6.py area`,
  `reports/data/M6.json`): Yosys for UltraScale+, one `fp32_fma` 1,554
  LUTs, 409 registers, 2 DSPs; the chain 11,707 LUTs, 4,392 registers, 14
  DSPs. Scaled by M3's Vivado numbers for one FMA (estimate): 9,507 LUTs
  and 14 DSPs a slot, 152,116 LUTs, 91,921 registers and 224 DSPs for 16
  slots. Fewer Newton steps would not save area (the chain's length is
  exp's). Not done (simplify review): the delay lines carry every state
  field (about 1-2% of the LUTs could go, at some readability), and the
  register after each FMA (latency 50 -> 43) waits for Vivado's timing.
- [x] The vector unit's buffers and elementwise commands (step 4a):
  `rtl/vec_unit.sv`, every buffer as `Lanes` banks (`tile_buffer`, one read
  and one write of 16 elements a cycle, BF16 writes rounded), a walker over
  a command's groups (rows of `m`, rows of t at the stride, ROTATE_HALF's
  halves; packed rows as one), 16 `fp32_chain` slots, and an ext port for
  the testbench, LOAD and the engine later. The next command is taken while
  earlier ones drain (D-045). `commands.vector_command` is `run()`'s
  vector step, and `commands.overlap_ok` its buffer rules: distinct
  sources (one read port a buffer), and dst over a source only where each
  element is read once. Tests (`verif/tests/test_vec_unit.py`, 77, through
  `verif/bulk/vec_unit.cpp`): every buffer bit for bit after each command
  alone in every walk shape and row lengths around the lanes, softmax's
  rows of t at the stride, random sequences of 60 commands (12-16 of them
  waiting to read what a running command writes, the rest back to back,
  offered at random cycles in half), and back-to-back speed (a group a
  cycle). 10 hand mutations, all caught. Yosys synthesizes the unit at 2
  lanes (at 16 it passes 8 GB).
- [x] Reductions `SUM`, `SUMSQ`, `MAX` (D-050, step 4b): each slot keeps
  `Partials / Lanes` = 8 partials and accumulates with its chain's first
  FMA, tapped out of `fp32_chain` (`MAX` takes x through `add(x, -0)` and
  keeps `max_f32`, now in `fp_pkg`, from the canonical NaN). A row's first
  group starts its partials when its result lands, the cycle the tree takes
  the row before. `rtl/vec_tree.sv` runs the spec's tree on one `fp_add` (or
  `max_f32`) level by level, a level of P pairs in P + 3 cycles, and writes
  `dst[row]` once no elementwise result is on its way (so done_o stays in
  order for any parameters). A row's last group waits for the tree; after a
  reduction the next command waits for its last row. `TreeAdders` (D-050)
  is not a parameter yet: one adder. Tests (137 in `test_vec_unit.py`):
  each reduction on rows around the partials and their groups, the shared
  special values, the whole softmax at the stride, reductions in the random
  sequences, and the trees overlapping rows longer than a tree (each row
  after the first costs its groups alone). 11 hand mutations, all caught.
  Found on the way: a reduction's slots wrote with an empty mask over the
  tree's write.
- Fused softmax command (D-054): waits for a working core (D-057, Q-31).
- [x] `LOAD` and `KV_STORE` (answers above): in `rtl/vec_unit.sv`'s
  pipeline, with its AXI master: `port_dma` reads, the new
  `rtl/port_wdma.sv` writes a burst per KV head at D-043's place, through a
  queue of 64 beats with credits; dones in order through a queue of tags
  (one per command taken: KV_STORE or not, its heads). The addresses are
  registered the cycle after the take (by beats; a small second multiply
  for a head's region). `KV_STORE` needs m a power of two (one burst inside
  a 4 KB page) and `cap` a multiple of the channels, as the engine does.
  `commands.Layout` gives a tied `lm_head` its own copy, so `LOAD BY_TOKEN`
  reads `embed` by rows (`load_command` and `kv_store_command` are `run()`'s
  steps). The memory model gains a write port that checks itself
  (`verif/bulk/axi_mem.h`: every channel held while waiting, full strobes,
  WLAST on each burst's last beat, each burst answered once). Tests
  (`verif/tests/test_vec_mem.py`, 70): `LOAD` in every row mode from BF16
  and FP32 into FP32 and BF16 buffers, a row about 260 MiB in (Qwen2.5's
  last token), `KV_STORE` at positions in every
  channel with its done after its last response, later commands running
  while it waits, the queue filling, random sequences with the other
  commands and slow, pausing memories, error responses; a long `LOAD` of
  BF16 runs a group a cycle (312 cycles for 256 groups). 11 hand
  mutations, all caught (the queue's credits only once the test's rows
  were 16 beats and 4 heads a command), and 6 again after the /simplify
  changes to the done queue and the write's destinations. Found on the way: a BF16 NaN must
  become the canonical NaN on `LOAD` (`bf16_to_fp32`), as `up()` does;
  and (/code-review) `port_wdma`'s data waited for its address to be taken,
  which AXI forbids: a slave that takes an address only once its data is in
  (the memory model's `aw_after_w`, now in the slow tests) deadlocked it.
- [x] The tiny config's whole decode step (`verif/tests/test_vec_decode.py`,
  in CI). Owner, 2026-10-30: one run per token, kept small (the random
  sequences cover most of it; what is new is the real list's shapes and
  order). The driver gains engine steps: once every earlier command is
  done, it reads the engine's input and writes its result
  (`commands.run()`'s `EngineCall`) through the ext port. Checked bit for
  bit: each engine command's input, every buffer at the end (`run()` now
  takes the starting buffers: both start from the same NaN ones), and the
  whole memory.
  Memory map: channel 0 (256 KiB) holds `Layout`'s flat image, `LOAD`
  reads it at its flat addresses, and `KV_STORE` takes a cache's flat
  address as its offset in every channel (D-043; D-040's offsets come in
  M7). GQA (4 heads, 2 KV heads) at position 0 (cap 16, stride 16) and 5
  (cap 8, stride 32): 99 steps, 7,041 and 7,049 cycles (with the engine
  steps' ext reads and writes; the engine never overlaps the unit here). 3 hand mutations caught
  (an engine result, dropped `KV_STORE`s, a wrong token). `with_waits` no
  longer counts `NONE` as written, so a command after a `KV_STORE` no
  longer waits for it.
- [x] Real data, speed and the perf model's latencies (D-053; answers
  above). SmolLM2's whole step (`test_vec_decode.py`, slow, the shared
  part in `verif/decode.py`) at positions 0, 63, 1023 and 8191 of the long
  cache's text, F2's latency on both AXI channels: every engine command's
  input, every buffer and the memory bit for bit (108 s for the four, 3.1
  GB at most). The driver now waits where the controller stops
  (`commands.stops`, D-045; the tiny step's cycles are unchanged). The
  perf model's vector term is `perf.vector_cycles`, the unit in order: a
  group a cycle, and a start-up where the controller stops, a `LOAD`'s
  read, a `KV_STORE`'s answers, a reduction's tree; the commands after one
  the controller does not stop at overlap its latency. Measured
  (`scripts/measure_m6.py latency`, `reports/data/M6.json`), each one
  value at every position: `vector_startup_cycles` 53 (785 commands),
  `reduction_tail_cycles` 107, `reduction_row_cycles` 159 (a reduction's
  short rows queue on the one tree adder), `load_latency_cycles` 74,
  `kv_store_latency_cycles` 72; `issue_cycles` 4 (guess). Every command
  within 10% of the model (worst: the second `KV_STORE` of a layer, 13
  cycles against 12); the totals within 30 cycles (176,323 at position 0,
  790,333 at 8191). 3 hand mutations caught: a `KV_STORE` done before its
  answers and a reduction's rows waiting for the tree (by the cycles), the
  tree pairing i with i + P instead of neighbours (by the bits at 1023; at
  position 0 SmolLM2's 61 `SUMSQ`s give the same bits either way, and
  `test_vec_unit.py` catches it). **D-047's share at position 0 is not
  met** (perf model, in order, estimate; recorded, no longer a criterion: D-058): SmolLM2 36.8% (0.705 of 1.917
  ms), Qwen2.5-0.5B 16.6%; at 1023 38.6% and 19.5%, at the full window
  47.1% and 46.9%. At position 0 for SmolLM2, `SUM` and `MAX` (9 rows of
  one element, 159 cycles a row) take 86k of the unit's 176k cycles, the
  start-ups about 42k, the passes themselves about 27k. Q-30, Q-32.
- [x] The engine's placed build, as it is (D-059; 2026-10-30, r6i.4xlarge,
  2 h 18 min, peak 16.5 GB; reports/data/f2/2026-10-30-engine-placed.txt):
  timing fails, worst slack -4.698 ns at 250 MHz (an 8.68 ns path, 79% of
  it routing), 288,260 of 986,478 endpoints failing. The worst path is
  D-051's: the step decision (`step_unit_q`'s last-tile compare, the
  lanes' ready ANDed across the chip, the queue's step) into the tile
  buffer's read rotation, whose address fans out to 512 LUT RAMs (1.9 ns
  of wire), 16 logic levels. Use: 618,132 LUTs (47.4%), 378,367 registers,
  16 DSPs; by SLR 34.7% / 74.1% / 33.6% of LUTs. So the pipelined step
  control (D-051) is built: the timing loop (D-060).

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
**Notes (from M5):** the engine needs `rows` ≥ 1 and `cols` ≥ 1; a `cols` = 0
command hangs it, so the controller rejects one (or the list builder never
makes one). The model's `Layout` moves to channel offsets (D-040). The
controller sets `wait` on the commands `commands.engine_waits` marks and
pulses `go` once the command each depends on is done; a go never waits for
the engine to take a later command (D-042).
The controller stops at a non-engine command until its inputs are written
(D-045); the perf model's start-up (`engine_startup_cycles`) was measured
with weights read ahead at once, so compare the engine commands after such
a stop with it and report the loss.
Attention: `SCORES` writes head h's scores at `S[h·stride + i]` and
`VALUES` reads head h's probabilities at `P[h·stride + i]`, the STRIDE
register (D-049), which the controller passes to the engine
(`cmd_stride_i`) and the vector unit (`LEN_T` rows); never `cap`. The
engine needs `cap` a multiple of L, the stride a multiple of 16, a
power-of-two head dimension from 16 to 128 (`VALUES`: at most L), and
does not check them; `SCORES` and `VALUES` are given after the
`KV_STORE` they depend on (D-044).

### M8 — F2 bring-up (no model)
**What:** F2 wrapper with a trivial test design: registers, host DMA, and
device memory bandwidth. Runs beside M6 and M7 (D-052).
**Why:** keeps platform problems apart from design problems. Cloud time costs
money.
**Done when:** register access and DMA work, and HBM bandwidth per port and
in total is measured. **Cloud budget and instances are approved by the
owner before anything is created.**
Every Vivado run (timing harnesses, full builds) checks that the worst
slack at 250 MHz (4 ns) is ≥ 0, and its script fails otherwise; the slack
goes into the report next to the perf numbers (owner, 2026-10-08).
**From M5:** the engine's out-of-context Vivado run (128 lanes with DMA and
buffers: area and timing at 250 MHz) moved here (owner, 2026-10-09), and
back to the end of M6 as a placed build (D-051).
Its Yosys proxy (M5, attention) puts the deepest path in the scale's
`fp32_fma` and the engine's own control 4 cells shallower; if Vivado finds
the control path (unit end into the RAMs' next addresses) failing, register
the addresses a cycle earlier. The engine's four stage queues each hold the
whole command (about 178 bits); check in the utilization report that
synthesis drops the fields a stage never reads, or narrow them. Two
changes wait for that run: `port_dma`'s burst length registered a cycle
ahead (M5), and `fp_add`'s significand compare beside the alignment
shifter (M3; +142 Yosys cells), only if those paths limit.
The full build synthesizes with `-keep_equivalent_registers` and
`-shreg_min_size 16`, as platforms/f2/timing/timing.tcl does: the
engine's per-group copies of its control registers (D-051) and the lane
units' tag copies would otherwise merge back into one, and the short
copy pipelines become SRLs, whose output is slow. An option
if the lanes' room estimate ever goes stale: dot_lane exports its free
last-beat credits instead of `reserved_o`, so the engine stops modelling
the lane's ready rule (review of D-051, 2026-10-31).

### M9 — SmolLM2-135M-Instruct on F2, measured
**Done when:** bit-exact against the golden model. Tokens/s is measured over
≥ 1000 tokens and compared with the perf model and the bandwidth bound
(provisional floor: ≥ 50% of it). Clock and resource use are reported.
Compared with the Gemmini baseline (a different model) using normalized
figures: weight bytes/s and MACs/s, and with llama.cpp on the F2 host's
CPU at the same positions (D-056). The engine keeps a margin of 10-25% over
the memory (D-055).
**From M5:** with HBM at 450 MHz a port delivers about 1.8 beats per core
cycle, so lanes per port becomes 8 (256 lanes on 32 ports), with a clock
crossing on the memory side. `VALUES` runs at most 2 heads a pass (the
engine's `Heads`; the perf model follows it since M5, attention): at 256
lanes that leaves half the lanes idle on p.V, and docs/perf.md's 256-lane
row is limited by the engine at position 1023. Raise `Heads` to L / d
there.

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

New ones are added as a table:

| ID   | Question | Leaning / when |
|------|----------|----------------|
| Q-27 | Share K and V tiles between the query heads of a GQA group (q·K: one tile for the group's heads in turn; p·V: lanes that save and resume their partial sums, or a chunked summation order, a numerics change)? Today each query head reads K and V. | Only once the engine outruns the memory on attention. Perf model (docs/perf.md, "KV tiles shared", in order, the best case): sharing cuts SmolLM2's memory time at 8191 from 3.527 to 1.933 ms but gains under 1% (148 → 149 tokens/s), because attention's multiply-adds keep the engine as slow as the memory. |
| Q-28 | Use the ports a partial last pass leaves idle (rows not a multiple of `L`: SmolLM2's `q`, `o`, `down`, `k`, `v` at 128 lanes)? Either split the last pass's dot products across lanes and add the halves (a numerics change), or start the next command's first pass on the idle ports (controller and engine redesign). | After M7/M8 measure a whole token at F2's HBM rate; only if the idle ports show up there. |
| Q-29 | Bring the vector unit below 10% of token time at long positions? With D-054's fused pass (deferred, Q-31) and max in the drain, SmolLM2 at 8191 goes from 42.6% to 24.0% (in order, estimate). Left without a numerics change: 32 elements per cycle (D-030), softmax of head h beside the engine's scores of head h+1 (controller). With one: online softmax (D-022), which D-054 argues against. | After M7/M8 measure a whole token. |
| Q-30 | Cut the latency of dependent vector commands (D-053)? Options: residual adds and scalings on the engine's drain, as it already multiplies by `scale` (epilogue fusion); a short path through the chain for `ADD`, `MUL` and `FMA`; RMSNorm's three one-element commands on a small scalar unit; ROTATE_HALF as an addressing flag on `MUL` (today it goes through the 50-cycle chain), which saves RoPE one dependent command per rotation. All keep the bits. Measured (M6): 785 of SmolLM2's commands stop, 53 cycles each, about 42k cycles a token (0.17 ms). | After M9, on F2's measured numbers (D-058). |
| Q-31 | Add D-054's fused softmax pass (`exp(a - b)` per row and the row's sum, one command) and the max in the engine's drain? Same bits; softmax reads the scores twice instead of five times (perf model, in order, estimate: SmolLM2 609 → 618 tokens/s at position 0, 148 → 198 at 8191). The fused sum comes after the chain, and `SUMSQ` keeps `fma(x, x, p)`, so either a second accumulation point beside the first FMA or every reduction on an FMA after the chain (about 50 cycles more a reduction command, 32 DSPs). Deferred by D-057. | Once a whole token runs and is measured (M7/M8 or later). |
| Q-32 | Let a reduction's short rows stop queueing on the tree (D-050)? A row's last group waits for the tree to take the row before, so rows end at least 159 cycles apart (measured): `SUM` and `MAX` over 9 rows of t take about 9 × 159 cycles while t is below about 2,500, 86k of the vector unit's 176k cycles at position 0 for SmolLM2. Options: skip the tree's levels whose right half no element reached (those partials stay +0, or the canonical NaN for `MAX`, so the bits stay, to verify by test); more tree adders (`TreeAdders`). | After M9, on F2's measured numbers (D-058). |
