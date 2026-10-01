# platforms/

Thin wrappers that connect the accelerator core (`../rtl/`) to a specific
target. The core exposes one standard interface (AXI-Lite for control, AXI for
memory). Each platform adapts that interface to what the target provides.

- `f2/`: AWS F2 (AMD Virtex UltraScale+ with HBM, PCIe host, Vivado flow).
- `ecp5/`: Lattice ECP5 board, fully open flow (Yosys + nextpnr).
- `asic/`: open-PDK ASIC flow for a small slice of the design.

Rule: no accelerator logic lives here, only adapters, clocks, resets, pin
constraints, and build scripts.

License: Solderpad Hardware License v2.1 (`SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1`),
see [`LICENSE-HARDWARE`](../LICENSE-HARDWARE).
