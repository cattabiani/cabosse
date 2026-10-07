# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# Out-of-context synthesis, placement and routing of acc_loop.sv on F2's part
# at clk_main_a0's 250 MHz (docs/f2.md). Reports go to the current directory.
# Usage: vivado -mode batch -source acc_loop.tcl -tclargs REPO PART NUM_PIPE_REGS

lassign $argv repo part n
set_param general.maxThreads 4

# rtl/sources.f, without its comments; paths are relative to rtl/.
set f [open $repo/rtl/sources.f]
foreach line [split [read $f] "\n"] {
  set line [string trim $line]
  if {$line eq "" || [string match "//*" $line]} continue
  if {[string match "+incdir+*" $line]} {
    lappend incdirs $repo/rtl/[string range $line 8 end]
  } else {
    read_verilog -sv $repo/rtl/$line
  }
}
close $f
read_verilog -sv $repo/platforms/f2/timing/acc_loop.sv
read_xdc -mode out_of_context $repo/platforms/f2/timing/acc_loop.xdc

synth_design -top acc_loop -part $part -mode out_of_context \
  -include_dirs $incdirs -generic NumPipeRegs=$n
opt_design
place_design
phys_opt_design
route_design

report_timing_summary -no_header -file timing_summary.rpt
report_timing -max_paths 3 -nworst 1 -path_type full -input_pins -file worst_paths.rpt
report_utilization -file utilization.rpt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "CABOSSE-RESULT loop_cycles=[expr {$n + 1}] wns_ns=$wns"
