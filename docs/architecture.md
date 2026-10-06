# Cabosse architecture (v0, draft)

What the accelerator is made of, where its data lives, and why. M2 writes
it; M3 onward builds the RTL from it. The numerics (what each operation
computes, bit for bit) are in [numerics.md](numerics.md), the platform facts
in [f2.md](f2.md), and every performance number in [perf.md](perf.md), which
is generated from the perf model.

Status: draft. Sections marked with a
D-number are decided (PLAN.md decision log); **[proposed]** ones wait for the
owner. The tables of the controller, command and register sections are
generated from `model/commands.py` by `scripts/report_arch.py`.

## How a token runs

The host loads the image, the weights (into HBM, over PCIe) and the command
list for one decode step (on chip) once. Per token, it writes the token id
and the position into registers and starts the step; the controller runs
the whole command list with no host involvement until the logits are ready
(D-003). Weights, KV cache and command list stay on the card between
tokens. A new conversation resets the position; attention at position `p`
reads only cache entries `0…p`, so stale entries need no clearing.

## Blocks

| Block | What it does |
|---|---|
| Engine | matrix-vector products and attention's two dot products, in BF16 × BF16 → FP32 (numerics.md, section 3) |
| Vector unit | RMSNorm, softmax, RoPE, SwiGLU, residual adds, the BF16 roundings before each product (section 4) |
| Controller | reads the command list and starts each operation on the engine, the vector unit and the memory front end |
| Memory front end | the 32 HBM ports: streams weights into the engine, moves KV tiles between HBM and on-chip buffers |
| On-chip buffers | activations (the current vector, partial results), KV tiles, the command list |
| Host interface | AXI-Lite registers on the shell's OCL port (control); the shell's PCIS port to write HBM from the host (loading) |

The core exposes AXI-Lite and AXI only (D-004). Everything F2-specific (the
HBM IP, the shell ports, clocks) sits in `platforms/f2/`.

## Engine (D-027)

**Shape:** `L` = 128 lanes, each doing 4 multiply-adds per cycle, with `A` =
16 partial sums per lane: 512 multiply-adds per cycle.

- A lane computes one row at a time (numerics.md, section 3): each cycle
  the row's next 4 weights times the matching 4 elements of the input
  vector, which is broadcast to all lanes.
- Element `k` goes into partial sum `k mod 16`. Each of the 4 pipelined
  multiply-add units owns 4 of the sums, so its accumulate loop has 4
  cycles.
- At the end of a row the 16 sums are added in the fixed tree of section 3
  (15 adds), on one extra adder per lane, while the lane runs its next row.
- Rows go in passes of 128. A matrix whose row count is not a multiple of
  128 leaves lanes idle in its last pass.

**Why this shape.** 512 multiply-adds per cycle use exactly what the 32 HBM
ports deliver at 250 MHz: 32 × 32 B = 1024 B per cycle, 512 BF16 weights.
How the 512 are organised decides how many stay busy (perf.md, "Engine
use"):

- More lanes with fewer multiply-adds each (512 × 1) waste lanes on
  matrices with few rows: SmolLM2's 576-row and 192-row matrices fill 512
  lanes poorly.
- Fewer lanes with more multiply-adds each (64 × 8) need more partial sums
  (`A` = 32), so a longer final sum per row. Attention scores have rows only
  head-dimension long (64 for SmolLM2 and Qwen2.5-0.5B): 64 elements at 8
  per cycle take 8 cycles, but the final sum takes 31. At long positions,
  where attention dominates, the lanes wait.
- 128 × 4 sits between: a 64-long row takes 16 cycles, enough for its
  15-add final sum, and SmolLM2's 576- and 192-row matrices lose only part
  of their last pass. In general, with a 4-cycle loop, multiply-adds per
  lane should stay at or below √(head dimension ÷ 4): the final sum (4 × E
  − 1 adds) must fit in the next row's head dimension ÷ E cycles.

**Numerics.** Only `A` affects the result bits. `L` and the multiply-adds
per lane change the speed, not the bits, so the engine can grow (more
lanes) without a numerics change. `A` = 16 replaces the provisional `A` = 8
of numerics.md.

**Requirement for M3.** The multiply-add unit's accumulate loop must close
in 4 cycles at 250 MHz. If it needs more, `A` grows with it (8 cycles would
mean `A` = 32, with the losses of the 64 × 8 row above at long positions).

**Settled by this choice:**
- Q-22: keep `A` partial sums per row. Rotating rows through a lane works
  only with one multiply-add per lane.
- Q-25: answered above.

**Left for later:** concatenating matrices that share an input (q, k and v;
gate and up) into one, to fill the last pass; a small gain for SmolLM2.

## Vector unit (D-030)

**Shape:** 16 FP32 elements per cycle, with `S` = 128 partial sums for
reductions. Each of the 16 slots has the FP32 add, multiply, FMA and max of
numerics.md, section 2, and the rsqrt, recip and exp of section 5.

- Elementwise work (RoPE, SwiGLU, residual adds, the roundings to BF16)
  takes n ÷ 16 cycles.
- A sum (RMSNorm's sum of squares, softmax's denominator) puts element `k`
  into partial sum `k mod 128`. Each slot's adder owns 8 of the sums, so a
  4-cycle add loop has slack. The 128 sums then go through the fixed tree of
  section 3: 7 levels, about 30 cycles once per sum.
- `max` is exact, so its order is free.

**Why this width.** At short positions the vector unit is a few percent of
the memory time. At long ones softmax's passes over the scores grow with
the position, but so does attention on the engine, which stays the larger
of the two: at 16 wide the vector unit is about half the engine's time at
the full window of SmolLM2 and of Qwen2.5-0.5B (perf.md, "Predictions"). A
wider unit, or online softmax (D-022), is for when the engine stops being
the limit there.

**Numerics.** Only `S` affects the bits. 128 partial sums fit any width up
to 32 with a 4-cycle add loop, or 16 wide with an 8-cycle loop, so the unit
can widen without a numerics change. Online softmax changes the spec.
`S` = 128 replaces the provisional 8 of numerics.md.

**Commands:** fused per model block today; primitive commands are open
(Q-26).

## Memory (D-028)

| What | Where | Why |
|---|---|---|
| Weights | HBM, spread over all 32 channels | the only memory big and fast enough; all ports stream at once |
| KV cache | HBM, spread over all 32 channels | grows with the position; read every token |
| Command list | on chip, next to the controller | small (below, "One token as commands"); fetched with no HBM latency |
| Activations | on chip | small, used constantly; no HBM traffic (perf.md's assumption) |
| KV tiles | on chip | a block of K or V positions, staged for the engine (below) |
| DDR4 | unused in v0 | 27× slower than HBM (f2.md) |

**HBM ports.** The F2's HBM has 32 ports, each attached to its own memory
channel (512 MiB each), and a switch that lets any port reach any channel
(f2.md). v0 uses each port only on its own channel, so the switch is never
crossed and all 32 run in parallel (Q-10). Port `p` feeds lanes `4p` to
`4p+3`: one 32-byte beat holds the next 4 weights for each of its 4 lanes.

**Weight layout.** Each matrix is stored in the order the lanes read it. For
pass `j` (rows `128j` to `128j+127`), port `p` holds, for `k` = 0, 4, 8, …,
the beat `W[r][k…k+3]` for its 4 rows `r` = `128j+4p` … `128j+4p+3`. Each
port reads its share of a matrix as one sequential stream. The host writes
weights in this layout at load time (it is a reordering, not a conversion:
D-015).

**KV cache layout.** K and V are stored as written: one row per position,
`head_dim` BF16 values per KV head and layer, spread over the ports by
position. A token writes one row of K and one of V per KV head and layer.
For attention, the memory front end loads a tile of 128 positions (128 × 64
× 2 B = 16 KiB for SmolLM2) into an on-chip buffer, and the engine reads
it from there:
- q·K: one lane per position, 4 elements of that position's K per cycle.
- p·V: one lane per output dimension, 4 positions of that dimension per
  cycle: the buffer is read across positions, so V needs no transposed
  copy in HBM (Q-11).
- The tile is loaded once per KV head and used by all of its query heads
  (3 for SmolLM2), as the perf model assumes.
- A p·V pass covers 2 query heads (128 lanes = 2 × 64 dimensions), and the
  two can belong to different KV heads (SmolLM2's heads 2 and 3 use KV heads
  0 and 1), so the buffer holds the tiles of two KV heads at once.

The buffer must serve both access patterns at 512 values per cycle; its
banking is designed with the engine (M5).

## Clocks (D-028)

v0 runs everything on the shell's 250 MHz clock: the core, the HBM ports,
the shell interfaces. AWS fixes 250 MHz for the shell's interfaces; the HBM
ports are ours (we instantiate the HBM controller) and run up to 450 MHz,
so 250 MHz on them is a v0 choice that uses about 56% of the HBM's
bandwidth. In exchange, one clock means no clock-domain crossings in the
datapath of the first build. 250 MHz is a target, to verify in M3 (one lane) and in the first
full build. If it does not close, perf.md's "core at 125 MHz" scenario
shows what is lost; a slower core would run on its own clock from AWS's
recipes, with crossings to the shell.

The next step up (M9) is HBM at 450 MHz with 256 lanes (perf.md's last
scenario): about 1.8× the memory bandwidth, the same `A` and so the same
bits. It needs a second clock domain on the memory side.

## Controller (D-029)

A fixed-function sequencer, not a processor (Q-12): it reads the command
list in order and starts each command when the units it needs are free. A
decode step has no branches and no loops that depend on data: the list is
unrolled over the layers, and what changes from token to token (the token
id, the position) comes from registers, so the list is written once per
model. The position sets which RoPE row is read, which KV cache row is
written, and how many positions attention reads.

Commands run in order; each one waits for the buffers it reads. Overlap
comes from the memory front end streaming the next command's weights while
the current one finishes; the perf model's "overlapped" bound assumes it.

`model/commands.py` holds the command set, `build()` (the command list of
one decode step for any model config) and `run()`, a reference controller
built on the golden model's functions. A test runs the list token by token
and checks that logits and KV cache match the golden decode step bit for
bit: every operation of the step has a command, and no host work happens
mid-token.

## Commands (D-029; vector-unit commands: Q-26)

<!-- begin: opcodes -->
| opcode | command | does |
|---|---|---|
| 0 | `END` | token done: set STATUS done, raise the interrupt; the last command |
| 1 | `EMBED` | dst = up(table[token]), rows of m; table at addr |
| 2 | `RMSNORM` | dst = rmsnorm(a, gain at addr, eps = scalar), n elements |
| 3 | `MATVEC` | dst = W·a, W at addr: n rows × m columns |
| 4 | `ROPE` | dst = rope(a), n vectors of m; cos and sin at addr, row position |
| 5 | `KV_STORE` | cache[h][position] = a[h], n heads of m; cache at addr, cap rows per head |
| 6 | `SCORES` | dst[h][i] = dot(a[h], K[h // group][i]) · scalar, n heads of m, i < t |
| 7 | `SOFTMAX` | dst = softmax(a), n vectors of t |
| 8 | `VALUES` | dst[h] = Σᵢ a[h][i] · V[h // group][i], n heads of m, i < t |
| 9 | `SWIGLU` | dst = swiglu(a, b), n elements |
| 10 | `ADD` | dst = add(a, b), n elements |
| 11 | `OUTPUT` | a (n FP32) to host memory at LOGITS_HI:LOGITS_LO |
<!-- end: opcodes -->

Every command names its buffers (`dst`, `a`, `b`) and, where it reads or
writes HBM, an address. Results are FP32; writing to a BF16 buffer or to
the KV cache rounds them (round to nearest even). The BF16 rounding points
of D-011 are exactly the BF16 destinations, so no command rounds on its own
and every MATVEC input is already BF16.

**Encoding.** One fixed size for every command, unused fields zero:

<!-- begin: encoding -->
| byte | bytes | field | meaning | used by |
|---|---|---|---|---|
| 0 | 1 | `op` | opcode | all |
| 1 | 1 | `dst` | destination buffer | EMBED, RMSNORM, MATVEC, ROPE, SCORES, SOFTMAX, VALUES, SWIGLU, ADD |
| 2 | 1 | `a` | first source buffer | RMSNORM, MATVEC, ROPE, KV_STORE, SCORES, SOFTMAX, VALUES, SWIGLU, ADD, OUTPUT |
| 3 | 1 | `b` | second source buffer | SWIGLU, ADD |
| 4 | 4 | `n` | rows, elements, vectors or heads | RMSNORM, MATVEC, ROPE, KV_STORE, SCORES, SOFTMAX, VALUES, SWIGLU, ADD, OUTPUT |
| 8 | 4 | `m` | columns or vector length | EMBED, MATVEC, ROPE, KV_STORE, SCORES, VALUES |
| 12 | 4 | `kv_heads` | KV heads | SCORES, VALUES |
| 16 | 4 | `cap` | KV cache rows per head | KV_STORE, SCORES, VALUES |
| 20 | 8 | `addr` | HBM byte address | EMBED, RMSNORM, MATVEC, ROPE, KV_STORE, SCORES, VALUES |
| 28 | 4 | `scalar` | FP32 bits: eps or scale | RMSNORM, SCORES |

32 bytes per command, little-endian; unused fields are zero.
<!-- end: encoding -->

**On-chip buffers.** One per kind of value in a step. Their sizes are fixed
when the image is built, so they cap a model's dimensions: LOGITS the
vocabulary, S and P (scores and probabilities) heads × context, G, U and M
the intermediate size. They are sized for the largest model of the ladder at
its full context; a longer context or a larger model needs a rebuild, and
the host can always run a shorter context (a smaller `cap`) on the same
image. Qwen2.5-0.5B is read as a Llama config, so its q, k and v biases are
not counted (they need no buffer of their own).

<!-- begin: buffers -->
| buffer | format | elements | bytes | sized by |
|---|---|---|---|---|
| `H` | FP32 | 896 | 3,584 | Qwen2.5-0.5B-Instruct |
| `X` | BF16 | 896 | 1,792 | Qwen2.5-0.5B-Instruct |
| `Q` | FP32 | 896 | 3,584 | Qwen2.5-0.5B-Instruct |
| `K` | FP32 | 192 | 768 | SmolLM2-135M-Instruct |
| `V` | FP32 | 192 | 768 | SmolLM2-135M-Instruct |
| `QR` | BF16 | 896 | 1,792 | Qwen2.5-0.5B-Instruct |
| `KR` | FP32 | 192 | 768 | SmolLM2-135M-Instruct |
| `S` | FP32 | 458,752 | 1,835,008 | Qwen2.5-0.5B-Instruct |
| `P` | BF16 | 458,752 | 917,504 | Qwen2.5-0.5B-Instruct |
| `ATT` | BF16 | 896 | 1,792 | Qwen2.5-0.5B-Instruct |
| `T` | FP32 | 896 | 3,584 | Qwen2.5-0.5B-Instruct |
| `G` | FP32 | 4,864 | 19,456 | Qwen2.5-0.5B-Instruct |
| `U` | FP32 | 4,864 | 19,456 | Qwen2.5-0.5B-Instruct |
| `M` | BF16 | 4,864 | 9,728 | Qwen2.5-0.5B-Instruct |
| `LOGITS` | FP32 | 151,936 | 607,744 | Qwen2.5-0.5B-Instruct |

Total 3.27 MiB.
<!-- end: buffers -->

**Command lists of the ladder.** The list grows by one layer's commands per
layer, and fits the command buffer with a wide margin:

<!-- begin: ladder -->
| model | layers | context | commands | command bytes | HBM (MiB) |
|---|---|---|---|---|---|
| SmolLM2-135M-Instruct | 30 | 8192 | 575 | 18,400 | 440.7 |
| Qwen2.5-0.5B-Instruct | 24 | 32768 | 461 | 14,752 | 1,342.3 |

Command buffer: 64 KiB, 2,048 commands.
<!-- end: ladder -->

## Registers (D-029)

On the shell's OCL port (AXI-Lite, 32-bit). The host writes the command list
once, then per token: TOKEN, POSITION, CONTROL start; it waits for STATUS
done (or the interrupt) and finds the logits at the host address it set.

<!-- begin: registers -->
| offset | register | access | meaning |
|---|---|---|---|
| `0x0` | `ID` | R | 0x43424F53 ('CBOS') |
| `0x4` | `VERSION` | R | architecture version |
| `0x8` | `CONTROL` | W | bit 0: start a token; bit 1: reset the controller |
| `0xC` | `STATUS` | R | bit 0: busy; bit 1: done; bit 2: error |
| `0x10` | `TOKEN` | RW | token id of the next step |
| `0x14` | `POSITION` | RW | position of the next step |
| `0x18` | `COMMANDS` | RW | number of commands in the list |
| `0x1C` | `ERROR` | R | index of the command that failed, and why |
| `0x20` | `LOGITS_LO` | RW | host address for OUTPUT, low 32 bits |
| `0x24` | `LOGITS_HI` | RW | host address for OUTPUT, high 32 bits |
| `0x28` | `CYCLES_LO` | R | cycles of the last token, low 32 bits |
| `0x2C` | `CYCLES_HI` | R | cycles of the last token, high 32 bits |
| `0x10000` | `COMMAND_BUFFER` | W | the command list: 64 KiB, up to 2,048 commands |
<!-- end: registers -->

## One token as commands

Generated by `build()`:

<!-- begin: token -->
SmolLM2-135M-Instruct, cache of 8192 positions:

| # | command | dst | a | b | n | m | addr |
|---|---|---|---|---|---|---|---|
| 0 | EMBED | H |  |  |  | 576 | 0x0 |
| 1 | RMSNORM | X | H |  | 576 |  | 0x3A00000 |
| 2 | MATVEC | Q | X |  | 576 | 576 | 0x3A01000 |
| 3 | MATVEC | K | X |  | 192 | 576 | 0x3AA3000 |
| 4 | MATVEC | V | X |  | 192 | 576 | 0x3AD9000 |
| 5 | ROPE | KR | K |  | 3 | 64 | 0x3600000 |
| 6 | KV_STORE |  | KR |  | 3 | 64 | 0x40C2000 |
| 7 | KV_STORE |  | V |  | 3 | 64 | 0x43C2000 |
| 8 | ROPE | QR | Q |  | 9 | 64 | 0x3600000 |
| 9 | SCORES | S | QR |  | 9 | 64 | 0x40C2000 |
| 10 | SOFTMAX | P | S |  | 9 |  |  |
| 11 | VALUES | ATT | P |  | 9 | 64 | 0x43C2000 |
| 12 | MATVEC | T | ATT |  | 576 | 576 | 0x3B0F000 |
| 13 | ADD | H | H | T | 576 |  |  |
| 14 | RMSNORM | X | H |  | 576 |  | 0x3BB1000 |
| 15 | MATVEC | G | X |  | 1536 | 576 | 0x3BB2000 |
| 16 | MATVEC | U | X |  | 1536 | 576 | 0x3D62000 |
| 17 | SWIGLU | M | G | U | 1536 |  |  |
| 18 | MATVEC | T | M |  | 576 | 1536 | 0x3F12000 |
| 19 | ADD | H | H | T | 576 |  |  |
| … | layers 1 to 29: the same, at their own addresses |  |  |  |  |  |  |
| 571 | RMSNORM | X | H |  | 576 |  | 0x1B8BC000 |
| 572 | MATVEC | LOGITS | X |  | 49152 | 576 | 0x0 |
| 573 | OUTPUT |  | LOGITS |  | 49152 |  |  |
| 574 | END |  |  |  |  |  |  |

575 commands (19 per layer), 18,400 bytes. HBM in use: 440.7 MiB.
<!-- end: token -->
