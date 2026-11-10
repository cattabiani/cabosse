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

### Files

- `bringup_port.sv`, `bringup_host.sv`, `bringup_core.sv` (listed in
  `sources.f`): the vendor-free core, the host's port to HBM port 0
  included, tested in Verilator (verif/tests/test_bringup_core.py).
- `cabosse_bringup.sv`: the CL top (shell ports, crossings, HBM IP, ILA),
  built only in Vivado; verif/tests/test_bringup_core.py elaborates it with
  slang when `CABOSSE_AWS_FPGA` names an aws-fpga checkout
  (`scripts/test.sh -m on_request`).
- `../synth_cl.tcl` (both designs): its synthesis in AWS's shell flow;
  `../cl_shell.svh`: the shell body both CL tops include.

### Build

`scripts/f2/build_cl.sh` (DESIGN=bringup) is the user data of one build instance
(c6i.8xlarge, FPGA Developer AMI, the build instance profile): it makes a
CL directory from AWS's template, puts our design in it, runs
`aws_build_dcp_from_cl.py` (Small Shell, recipe H0) and our reports, and
copies the logs, reports and the AFI tarball to `S3` (under `dcp/`, where
the build role may write). A route with negative slack fails the run. The
AFI is then made from the tarball:

```
aws ec2 create-fpga-image --region eu-central-1 --name cabosse-bringup-<commit> \
  --input-storage-location Bucket=<bucket>,Key=dcp/<build>/<tag>.Developer_CL.tar \
  --logs-storage-location Bucket=<bucket>,Key=afi-logs/<build>/
```

### On F2

`scripts/f2/run_bringup.sh` is the user data of one f2.6xlarge (FPGA
Developer AMI, the build instance profile; `AGFI`, `S3` set at launch): it
builds the host tool (`host/bringup.cpp`, C++23 against AWS's SDK), loads
the AFI and runs `scripts/f2/bringup_plan.py`. The plan first sets the core
clock (the clock generator holds the design in reset until then), then
measures register access (OCL), host writes and reads through PCIS (checked
by port 0's engine), and at 250, 200 and 125 MHz: every port together and
alone, bursts, outstanding limits, the switch (own pseudo channel, the next,
the other stack, all on one), one-beat partial writes, read latency. Each
step is a JSON line with its provenance, copied to S3 as it comes.

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
up to 16, `BEATS` a multiple of both, both limits at least 1. A start that
breaks these rules, or comes during a run, is refused (`STATUS.refused`).
Port 0 runs only with `HOST_PORT` clear, and `HOST_PORT` keeps its value
during a run. The host uses PCIS only while `STATUS` says it owns port 0
and HBM is ready, with full-width beats (as AWS's example, HBM sees full
beats only), and before clearing `HOST_PORT` waits for its writes'
responses (a read through PCIS flushes posted writes); port 0 passes to the
engine once the host's side is idle (`STATUS` bit 3). A read burst of 1 beat is issued at
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
| 0x00C | STATUS | RO | bit 0 busy (any port), bit 1 HBM ready (both stacks), bit 2 refused (the last start), bit 3 the host owns HBM port 0 now, bits 15:8 start requests so far (mod 256: the host sees its start arrive, OCL's writes and reads not being ordered) |
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

## The core on F2 (M9)

The CL top holds `rtl/core.sv` in place of the bring-up's test engines and
keeps the bring-up's shell side: OCL and PCIS crossed into `clk_hbm_axi`,
the clock generator, the register slices, the ILA. The core's 32 engine
ports drive HBM ports 0-31, each on its own channel. The vector unit's AXI
port (`LOAD`, the KV stores, `OUTPUT`) shares HBM port 0 with engine port
0, and the HBM switch carries it to the other channels
(docs/architecture.md). PCIS reaches HBM port 0 through `HOST_PORT`, as in
the bring-up: the host owns port 0 between tokens, the core during one.

### Files

- `core/f2_port0.sv`, `core/f2_core.sv` (listed in `core/sources.f`, with
  the bring-up's `bringup_pipe.sv` and `bringup_host.sv`): the
  vendor-free wrapper. `f2_core` puts channel p into engine port p's
  addresses (the engine gives every port the same offset), and merges
  engine port 0 and the vector unit onto HBM port 0 (`f2_port0`: reads in
  turns, one ID, answers routed back in order). Register stages sit on
  each HBM channel the core uses, so no HBM port drives the core's logic
  directly, and on the vector unit's AXI port, one per SLR it crosses
  (M10). A read takes 4 cycles more than with none on engine ports 1-31,
  3 on engine port 0, 9 for the vector unit.
- `core/small_shell_cl_pnr_user.xdc`: the floorplan (one hard pblock per SLR),
  written by `scripts/f2/floorplan.py`, which says what goes
  where and why; `build_cl.sh` copies it into AWS's flow.
- `core/cabosse_core.sv`: the CL top, on `cl_shell.svh` as the bring-up's.
- Build: `scripts/f2/build_cl.sh` with `DESIGN=core` (a 9 h deadline:
  core build 1 took 6.6 h to route; the engine's own placed builds peaked
  at 14 GB). Core build 1 missed 250 MHz (WNS -5.466 ns,
  reports/data/f2/2026-11-04-core1.txt), so `scripts/f2/run_core.sh` runs
  it at `MHZ=100`.

### Registers (OCL, BAR0)

The core's registers (model/commands.py, `REGISTERS`) sit at their own
offsets, below 0x2_0000. The wrapper's sit above:

| Offset | Name | Access | Meaning |
|---|---|---|---|
| 0x2_0000 | WRAP_ID | RO | `0xCAB0C0E0` |
| 0x2_0004 | BUILD | RO | the build's commit (first 8 hex digits) |
| 0x2_0008 | HOST_PORT | RW | bit 0: HBM port 0 serves PCIS (1, reset) or the core (0) |
| 0x2_000C | WRAP_STATUS | RO | bit 1 HBM ready (both stacks), bit 3 the host owns HBM port 0 now |
| 0x2_0018 | PORTS | RO | the core's `Ports` |
| 0x2_001C | CHAN_W | RO | its `ChanW` |
| 0x2_0020 | LEN | RO | its `Len` |

HBM port 0 changes hands only between bursts: the host waits for its
PCIS writes to be done (a read after them) before clearing `HOST_PORT`,
and sets it only when the core is not running.

### Host side

`host/core_driver.cpp` (with `host/card_f2.cpp`; `make core_driver` on
the instance) serves the runtime's calls for `sw/f2_platform.py` in the
simulation driver's protocol (`sw/driver_protocol.h`). It hands HBM port 0
to the host before a memory request and to the core before a start.
PCIS writes are two 32-byte AVX2 stores a line; partial lines are byte
stores, carried as strobes. `sw/tests/test_f2_platform.py` runs it on a
fake card (`sw/tests/card_fake.cpp`), which fails a start while the host
holds port 0 and a memory access while the core holds it.

`host/core_clock.cpp` (`make core_clock`) sets clk_hbm_axi once after the
AFI's load (the clock generator holds the design in reset until then),
waits for HBM ready and prints the wrapper's ID, BUILD and parameters.

### On F2 (core)

`scripts/f2/run_core.sh` is the user data of one f2.6xlarge (`AGFI`, `S3`
set at launch): it builds the host tools, makes a venv with uv (Python
3.14 and `requirements.txt`), downloads SmolLM2 at the fixtures' revision,
loads the AFI and runs `scripts/f2/core_plan.py`: the clock, PCIS write and
read rates through the driver (5 repeats), the model's load at 2048
positions, the three 64-token fixtures and the long fixture, each token
checked bit for bit with its wall time and CYCLES. One JSON line a step,
copied to S3 as it comes. Then, with `CHAT=1` (the default), `scripts/f2/chat.py`
records a scripted chat (asciicast v2, transcript, JSON lines) to
`$S3/chat/`; `chat.py --golden` runs the same chat on the CPU. Then, with `BASELINE=1`
(the default), `scripts/f2/cpu_baseline.sh` times the host's CPU on the same
model: llama.cpp (a pinned release, GGUF BF16 converted by its own script,
`llama-bench` at positions 0, 46 and 1023) and `transformers`
(`scripts/measure_m7.py cpu`), to `$S3/baseline/`.
