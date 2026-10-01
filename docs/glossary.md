# Glossary

Hardware terms used in this project, in plain language. It is written for
people who know software and numerics but are new to hardware. Add a term
the first time a document uses it.

Grouped by topic; alphabetical within each group.

## Design and languages

- **ASIC (Application-Specific Integrated Circuit):** a chip manufactured for
  one design. Its logic is fixed once made, unlike an FPGA.
- **Clock / clock frequency (f_clk):** a periodic signal that steps all
  registers forward at once. At 250 MHz, the design does one step every 4 ns.
- **Clock domain / CDC (clock domain crossing):** logic driven by one clock
  is a clock domain. Passing signals between domains with different clocks
  needs special synchronizing circuits (CDC), or data gets corrupted.
- **Combinational logic:** logic whose output depends only on its current
  inputs (no memory), like a pure function.
- **Critical path:** the slowest combinational path between two registers. It
  sets the maximum clock frequency.
- **Fmax:** the highest clock frequency at which a design meets timing on a
  given device.
- **HDL (Hardware Description Language):** a language that describes circuits
  instead of programs. SystemVerilog and VHDL are HDLs.
- **Latency vs throughput:** latency is how many cycles one result takes from
  input to output. Throughput is how many results come out per cycle once
  the pipeline is full.
- **Netlist:** the design after synthesis, as a list of primitive cells (gates,
  flip-flops, LUTs) and the wires between them.
- **Pipeline:** splitting a computation into stages separated by registers, so
  a new input can start every cycle while earlier ones are still in flight.
  This raises throughput and adds latency.
- **Register / flip-flop (FF):** a 1-bit storage element that updates on a
  clock edge. Registers hold all state between clock cycles.
- **Reset:** a signal that puts all registers into a known initial state.
- **RTL (Register-Transfer Level):** the usual level at which hardware is
  written: registers, and the combinational logic that computes their next
  values each clock cycle. "The RTL" means the HDL source code of the design.
- **Synthesis:** compiling RTL into a netlist of primitives for a specific
  target (FPGA cells or ASIC standard cells).
- **SystemVerilog:** the HDL used in this project. It is an extension of
  Verilog.
- **Valid/ready handshake:** a common flow-control protocol. The sender
  raises `valid` when data is present, the receiver raises `ready` when it can
  accept it, and a transfer happens in any cycle where both are high.
  Backpressure means the receiver holding `ready` low.

## Verification

- **Bit-exact:** two implementations produce identical bit patterns, not just
  values within a tolerance.
- **cocotb:** a Python framework for writing testbenches. Python code drives
  and checks a design running in a simulator.
- **Gate-level simulation:** simulating the post-synthesis netlist instead of
  the RTL. It catches synthesis mismatches and is used at ASIC signoff.
- **Golden model:** a trusted reference implementation (here, Python/numpy)
  that hardware outputs are compared against.
- **Lint:** static checks on HDL code for likely bugs (width mismatches,
  unintended latches, unused signals).
- **Testbench:** code that feeds inputs into a design under simulation and
  checks its outputs.
- **Verilator:** an open-source simulator that compiles SystemVerilog into
  fast C++. It is a cycle-based simulator and supports only the synthesizable
  subset of the language, plus some extras.
- **Waveform (VCD/FST):** a file recording every signal's value over time
  during simulation, viewed with tools like GTKWave or Surfer for debugging.

## Arithmetic and numerics

- **BF16 (bfloat16):** a 16-bit float with 1 sign bit, 8 exponent bits (same
  range as FP32), and 7 stored mantissa bits (~3 significant decimal digits).
- **FP32:** IEEE 754 single precision. 1 sign bit, 8 exponent bits, 23 stored
  mantissa bits.
- **FTZ (flush to zero):** treating subnormal numbers as zero. It saves
  hardware but deviates from IEEE 754.
- **Kulisch accumulator:** a very wide fixed-point register that can sum
  floating-point products exactly, with no rounding. It costs much more area
  than an FP32 adder.
- **MAC (multiply-accumulate):** `acc += a * b`. The basic operation of a dot
  product.
- **RNE (round to nearest, ties to even):** the IEEE 754 default rounding
  mode. Round to the closest representable value. On an exact tie, pick the
  one with an even last bit.
- **Subnormal (denormal):** IEEE floats smaller than the smallest normal
  number. They have reduced precision and are expensive to support in
  hardware.
- **Summation order:** floating-point addition is not associative, so a sum's
  result depends on the order of additions. Bit-exact verification requires
  the golden model to use the hardware's order.

## Accelerator architecture

- **Command stream:** a list of commands in memory (e.g. "matvec from address
  A, rows R, cols C, to buffer B") that the accelerator's controller fetches
  and executes without the host.
- **Controller / sequencer:** the block that reads commands and starts the
  other units in order. It can be fixed-function logic or a small CPU.
- **CSR / register file / register map:** a small set of addressable registers
  that the host reads and writes to configure and control the device. The
  register map lists their addresses and meanings.
- **Doorbell:** a register the host writes to tell the device "new work is
  ready".
- **Dot-product lane:** one multiplier plus accumulator that computes one dot
  product, consuming one element pair per cycle. The matrix-vector engine has
  many lanes working on different rows in parallel.
- **Interrupt:** a signal from the device telling the host something
  happened (e.g. a token is done), so the host does not have to poll.
- **Matrix-vector engine:** the unit that computes `y = W·x` by streaming the
  matrix `W` from memory through many dot-product lanes.
- **On-chip buffer / scratchpad:** small, fast memory inside the chip that
  software or commands manage explicitly. Unlike a cache, it has no automatic
  replacement.
- **Systolic array:** a 2D grid of MAC units passing data to neighbours. It
  suits matrix-matrix work (e.g. Gemmini, TPU). It suits batch-1
  matrix-vector work poorly, because most of the grid sits idle.
- **Vector unit:** the unit that runs elementwise and reduction operations
  (RMSNorm, softmax, SiLU, RoPE, adds) on vectors.

## Memory and interconnect

- **AXI (AMBA AXI4):** ARM's standard on-chip bus protocol for memory-mapped
  transfers. It has separate channels for read/write addresses and data, and
  supports bursts. Most FPGA and ASIC IP uses it.
- **AXI-Lite:** a simplified AXI with no bursts, used for low-speed register
  access (control and status).
- **AXI master / slave (manager / subordinate):** the master starts
  transactions (e.g. a DMA engine reading memory). The slave responds (e.g. a
  memory controller or a register file).
- **Bandwidth:** bytes transferred per second. For batch-1 decode, tokens/s
  is roughly memory bandwidth divided by bytes read per token.
- **Burst:** one AXI transaction that moves several consecutive data beats
  for a single address. It is much more efficient than one address per word.
- **DDR (DDR4 SDRAM):** standard off-chip DRAM. It is large and cheap, with
  moderate bandwidth.
- **DMA (Direct Memory Access):** a hardware engine that copies data between
  memories (or between host and device) without a CPU moving each word.
- **HBM (High Bandwidth Memory):** DRAM stacked in the same package as the
  FPGA/chip, connected by a very wide interface. It has much higher bandwidth
  than DDR. It is split into many independent channels, each with its own
  port.
- **MMIO (memory-mapped I/O):** accessing device registers by reading and
  writing specific addresses, as if they were memory.
- **PCIe (PCI Express):** the high-speed serial link between a host CPU and
  plug-in devices (GPUs, FPGA cards). On F2, the host talks to the FPGA over
  PCIe.
- **SRAM:** fast static memory built into the chip. On-chip buffers are SRAM.

## Performance

- **Arithmetic intensity:** operations performed per byte moved from memory.
  Batch-1 matrix-vector work has about 1 MAC per 2 bytes of BF16 weight, which
  is very low. So it is memory bound.
- **Memory bound vs compute bound:** a workload is memory bound when it waits
  on data more than on arithmetic. Adding multipliers then does not help;
  more bandwidth does.
- **Roofline model:** a simple performance model. Achievable speed = min(peak
  compute, bandwidth × arithmetic intensity).
- **Utilization:** achieved throughput as a fraction of the theoretical peak
  (of bandwidth or of compute).

## FPGA

- **AFI (Amazon FPGA Image):** AWS's packaged form of an FPGA design that can
  be loaded onto an F2 instance.
- **AWS F2:** an AWS EC2 instance type with AMD Virtex UltraScale+ FPGAs that
  have HBM, attached to the host over PCIe.
- **Bitstream:** the final file that configures an FPGA with a design.
- **BRAM (Block RAM):** small dual-port SRAM blocks built into the FPGA fabric
  (36 Kb each on AMD UltraScale+).
- **DSP slice / DSP block:** a hard multiplier-adder block in the FPGA, much
  faster and smaller than building a multiplier from LUTs.
- **ECP5:** a Lattice FPGA family with good support in the fully open
  toolchain (Yosys + nextpnr). It is used on cheap boards.
- **FPGA (Field-Programmable Gate Array):** a chip made of configurable logic,
  memory, and DSP blocks that can be reprogrammed with any digital design.
- **HDK (Hardware Development Kit):** AWS's kit for F2 designs (shell
  interfaces, build scripts, examples).
- **LUT (Look-Up Table):** the basic FPGA logic cell. It implements any small
  boolean function of a few inputs (6 on AMD UltraScale+, 4 on ECP5).
- **nextpnr:** an open-source place-and-route tool for FPGAs, including ECP5.
- **Place and route (P&R):** deciding where each cell goes on the chip and
  how wires connect them. After synthesis, this is the step that determines
  real timing.
- **Shell (F2):** the fixed part of the F2 FPGA design, provided by AWS. It
  handles PCIe, DMA, and memory, and our design ("custom logic") plugs into
  its interfaces.
- **URAM (UltraRAM):** larger on-chip SRAM blocks (288 Kb each) on some AMD
  UltraScale+ FPGAs.
- **Vivado:** AMD's proprietary FPGA toolchain. It is required for F2 builds
  and runs on Linux and Windows.
- **Yosys:** an open-source synthesis tool. It is used for the ECP5 flow, the
  ASIC flow, and as a portability check.

## ASIC

- **DRC (Design Rule Check):** verifying that a layout follows the
  foundry's geometric rules (minimum widths, spacings). Violations make a
  chip unmanufacturable.
- **GDSII (GDS):** the file format of the final chip layout sent to the
  foundry.
- **LVS (Layout Versus Schematic):** verifying that the drawn layout is
  electrically the same circuit as the netlist.
- **MPW (Multi-Project Wafer) / shuttle:** many designs from different teams
  share one wafer run, splitting the cost. This is how small projects get
  real chips made.
- **OpenROAD / OpenLane / LibreLane:** open-source ASIC flows that go from RTL
  to GDS (synthesis, placement, clock tree, routing, signoff checks).
- **PDK (Process Design Kit):** the foundry's package describing a
  manufacturing process: rules, device models, standard cells. Open PDKs
  include SkyWater SKY130, GF180MCU, and IHP SG13G2.
- **Signoff:** the final set of checks (DRC, LVS, timing) a design must pass
  before tapeout.
- **Standard cell:** a pre-designed logic gate or flip-flop in a PDK's
  library. Synthesis maps the design onto these.
- **STA (Static Timing Analysis):** computing every path's delay without
  simulation, to check that the design meets its clock period across
  process/voltage/temperature corners.
- **Tapeout:** sending the final layout to the foundry for manufacturing.
- **Tiny Tapeout:** a program that puts many very small designs on one
  shared chip, cheaply.

## Timing

- **Slack:** clock period minus path delay. Negative slack means the path is
  too slow for the clock.
- **Timing closure:** the work of changing the design or constraints until
  every path meets the target clock (no negative slack).

## LLM inference

- **Decode:** generating one token at a time, each step reading all weights
  once. It is memory bound at batch size 1.
- **GQA (grouped-query attention):** several query heads share one key/value
  head, which shrinks the KV cache.
- **KV cache:** stored key and value vectors of all previous tokens for each
  layer, so attention does not recompute them.
- **Prefill:** processing the prompt. With many tokens at once it is
  matrix-matrix work and compute bound. Here it is initially done one token
  at a time.
- **RoPE (rotary position embedding):** encodes position by rotating pairs of
  query/key elements by position-dependent angles.
- **RMSNorm:** normalization that divides a vector by its root-mean-square,
  then scales it elementwise.
- **SwiGLU / SiLU:** the feed-forward activation in Llama-style models:
  `SiLU(W1·x) ⊙ (W3·x)`, where `SiLU(z) = z·sigmoid(z)`.
- **Teacher forcing:** when comparing two models, feeding both the same
  reference token sequence, so that one early disagreement does not make all
  later tokens diverge.

## Prior work

- **Gemmini:** an open-source systolic-array accelerator generator from UC
  Berkeley, written in Chisel. It is used in our hackathon baseline and not
  reused here.
- **Rocket:** an open-source RISC-V CPU core (Chisel), paired with Gemmini in
  the baseline.
