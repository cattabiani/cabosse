# platforms/

Thin wrappers that connect the accelerator core (`../rtl/`) to a specific
target. The core exposes one standard interface (AXI-Lite for control, AXI for
memory). Each platform adapts that interface to what the target provides.

- `f2/`: AWS F2 (AMD Virtex UltraScale+ with HBM, PCIe host, Vivado flow).
- `asic/`: open-PDK ASIC flow for a small slice of the design.

Other FPGAs (such as Lattice ECP5) are out of scope (D-034); the core keeps no
vendor primitives so they can be added.

Rule: no accelerator logic lives here, only adapters, clocks, resets, pin
constraints, and build scripts.

License: Solderpad Hardware License v2.1 (`SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1`),
see [`LICENSE-HARDWARE`](../LICENSE-HARDWARE).
