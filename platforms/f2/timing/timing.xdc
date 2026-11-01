# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
# clk_main_a0, 250 MHz (docs/f2.md)
create_clock -period 4.000 -name clk [get_ports clk_i]
# In the shell, clk_main_a0 comes out of an MMCM (aws-fpga, branch f2,
# aws_clock_properties.tcl, recipe A1: x 9.5, / 4.75), whose jitter
# an ideal out-of-context clock does not have. Vivado adds this to its own
# system jitter (0.035 ns, reports/data/f2/): about 0.1 ns in all, an
# estimate, not the shell's measured value.
set_clock_uncertainty -setup 0.065 [get_clocks clk]
