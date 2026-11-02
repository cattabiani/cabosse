# platforms/f2/

AWS F2 wrapper: connects the core to the F2 shell (PCIe, host access, HBM)
and holds the Vivado build scripts. The Vivado builds run on an AWS instance
from the FPGA Developer AMI (D-032).

- `timing/`: out-of-context timing harnesses (M3 to M6).
- `bringup/`: M8's bring-up design, below.

## Bring-up design (M8)

One bitstream that measures the platform (PLAN.md, M8): HBM bandwidth,
latency and pauses per port and in total, the HBM switch, outstanding
writes, partial writes, host access through PCIS and OCL, and the clocks.
Every pattern, size and limit is a register, so one build answers all of
it.

```
host ── PCIe ── shell
                 ├─ OCL  (AXI-Lite, clk_main_a0) ── clock converter ──┐
                 ├─ PCIS (AXI4 512 b, clk_main_a0) ── SmartConnect ─┐ │
                 │     (to clk_hbm_axi, 512 → 256 b, AXI4 → AXI3)   │ │
                 ├─ SDA  (AXI-Lite) ── AWS_CLK_GEN ── clk_hbm_axi   │ │
                 └─ HBM APB monitors ───────────────────────┐       │ │
                                                            │       │ │
  clk_hbm_axi:  bringup_core ── registers ◄─────────────────┼───────┼─┘
                  └─ 32 × bringup_port (port_dma reader,    │       │
                     port_wdma writer, counters)            │       │
                       port 0 ◄── mux (HOST_PORT) ◄─────────┼───────┘
                       ports 0-31 (AXI3, 256 b) ── HBM IP (2 stacks, switch on)
  PCIM, DDR, interrupts: tied off. ILA on HBM port 0 (debug bridge).
```

The core clock is `clk_hbm_axi` (D-065): timed for 250 MHz, set by the
host at run time. AWS's pieces (AWS_CLK_GEN, the shell's ports, the HBM,
clock converter and SmartConnect configurations, the debug bridge) come
from the aws-fpga checkout on the build instance; nothing of theirs is
copied here (Amazon Software License).

### A run

The host sets the registers, writes `CTRL.start`, and polls `STATUS.busy`.
Each port in `PORT_EN` then, from its address `BASE + p × STRIDE`:

- writes `BEATS` beats in bursts of `WR_BURST` beats, each beat with
  strobes `WR_STRB`, at most `WR_LIMIT` bursts unanswered;
- reads `BEATS` beats in bursts of `RD_BURST` beats, at most `RD_LIMIT`
  bursts unanswered, and with `MODE.check` compares each beat with the
  pattern.

With both `MODE.read` and `MODE.write` the two run at once on the same
addresses (bandwidth only; leave `check` off). The pattern: the 32-bit word
at byte address `a` holds `(a >> 2) ^ SEED`. Bursts must not cross 4 KB:
`BASE` and `STRIDE` 4 KB-aligned, `WR_BURST` and `RD_BURST` powers of two
up to 16, `BEATS` a multiple of both. A read burst of 1 beat is issued at
half rate (port_dma takes a command every other cycle), and reads have at
most 128 beats in flight (port_dma's FIFO, as in the engine).

### Registers (OCL, BAR0)

32-bit, in the core clock's domain. Counters count core cycles and freeze
when the port is done; `start` clears them.

| offset | name | access | meaning |
|---|---|---|---|
| 0x000 | ID | RO | `0xCAB0B001` |
| 0x004 | BUILD | RO | the build's commit (first 8 hex digits) |
| 0x008 | SCRATCH | RW | for register access tests |
| 0x00C | STATUS | RO | bit 0 busy (any port), bit 1 HBM ready (both stacks) |
| 0x010 | CYCLES_LO | RO | free-running cycle counter; reading it latches CYCLES_HI |
| 0x014 | CYCLES_HI | RO | |
| 0x020 | CTRL | WO | bit 0 start |
| 0x024 | PORT_EN | RW | a bit per port |
| 0x028 | MODE | RW | bit 0 read, bit 1 write, bit 2 check |
| 0x02C | BASE_LO | RW | byte address, 4 KB-aligned |
| 0x030 | BASE_HI | RW | |
| 0x034 | STRIDE | RW | bytes between ports' addresses, 4 KB-aligned |
| 0x038 | BEATS | RW | beats per port, each way |
| 0x03C | RD_BURST | RW | 1, 2, 4, 8 or 16 |
| 0x040 | WR_BURST | RW | 1, 2, 4, 8 or 16 |
| 0x044 | RD_LIMIT | RW | read bursts unanswered, 1 to 255 |
| 0x048 | WR_LIMIT | RW | write bursts unanswered, 1 to 128 |
| 0x04C | WR_STRB | RW | strobes of every written beat (a bit per byte) |
| 0x050 | SEED | RW | |
| 0x054 | HOST_PORT | RW | bit 0: HBM port 0 serves PCIS (1, reset) or port 0's engine (0) |
| 0x100 + 0x40 p | port p's counters | RO | below |

| offset in a port | name | meaning |
|---|---|---|
| 0x00 | RD_BEATS | beats read |
| 0x04 | RD_CYCLES | start to the last beat |
| 0x08 | RD_FIRST | start to the first beat |
| 0x0C | RD_STALLS | cycles between the first and the last beat with no beat |
| 0x10 | RD_MAXGAP | the longest of those gaps |
| 0x14 | RD_BAD | beats unlike the pattern (with `check`) |
| 0x18 | RD_OCC_LO | the sum over cycles of read bursts unanswered (Little's law: mean latency = RD_OCC / bursts) |
| 0x1C | RD_OCC_HI | |
| 0x20 | WR_BEATS | beats written |
| 0x24 | WR_CYCLES | start to the last response |
| 0x28 | WR_FIRST | start to the first response |
| 0x2C | WR_MAXOUT | most write bursts unanswered at once |
| 0x30 | WR_OCC_LO | the sum over cycles of write bursts unanswered |
| 0x34 | WR_OCC_HI | |
| 0x38 | FLAGS | bit 0 read busy, bit 1 write busy, bit 2 read error response, bit 3 write error response |

Host-side measurements need no registers: PCIS throughput (host writes and
reads of card memory through BAR4), OCL latency (reads of SCRATCH), and
the core clock (CYCLES against the host's clock).
