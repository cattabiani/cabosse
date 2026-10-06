# Cabosse architecture (v0, draft)

What the accelerator is made of, where its data lives, and why. M2 writes
it; M3 onward builds the RTL from it. The numerics (what each operation
computes, bit for bit) are in [numerics.md](numerics.md), the platform facts
in [f2.md](f2.md), and every performance number in [perf.md](perf.md), which
is generated from the perf model.

Status: draft. This version covers the blocks, the engine, memory and
clocks. Still to come in M2: the register map, the command format, one token
written out as commands, the vector unit and the controller. Sections
marked with a D-number are decided (PLAN.md decision log).

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

## Memory (D-028)

| What | Where | Why |
|---|---|---|
| Weights | HBM, spread over all 32 channels | the only memory big and fast enough; all ports stream at once |
| KV cache | HBM, spread over all 32 channels | grows with the position; read every token |
| Command list | on chip, next to the controller | small (one command per operation: perf.md, "Work per token"); fetched with no HBM latency |
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
