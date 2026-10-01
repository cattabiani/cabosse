# rtl/

Synthesizable SystemVerilog for the accelerator core, and nothing else. Keeping
testbenches and platform code out of this folder means every synthesis tool
(Vivado, Yosys, OpenROAD) can use the same file list.

Planned contents (from M3 on): floating-point units, dot-product lanes, the
matrix-vector engine, the vector unit, DMA engines, on-chip buffers, the
controller, and the top level with one AXI-Lite control port and AXI memory
port(s).

Platform-specific code (clocking, vendor IP, shells) goes in `../platforms/`.

License: Solderpad Hardware License v2.1 (`SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1`),
see [`LICENSE-HARDWARE`](../LICENSE-HARDWARE).
