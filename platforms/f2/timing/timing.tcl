# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# Out-of-context synthesis, placement and routing of a timing harness in
# this folder (acc_loop.sv, mac_loop.sv) on F2's part at clk_main_a0's
# 250 MHz (docs/f2.md). Reports go to the current directory.
# Usage: vivado -mode batch -source timing.tcl -tclargs REPO PART TOP PARAM VALUE LOOP RETIME
# PARAM=VALUE is the harness's loop-length parameter, LOOP the loop's cycles
# (for the report). RETIME 1 lets Vivado move registers across logic
# (synthesis and physical optimization).

lassign $argv repo part top param value loop retime
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
foreach harness [glob $repo/platforms/f2/timing/*.sv] {read_verilog -sv $harness}
read_xdc -mode out_of_context $repo/platforms/f2/timing/timing.xdc

synth_design -top $top -part $part -mode out_of_context \
  -include_dirs $incdirs -generic $param=$value \
  -global_retiming [expr {$retime ? "on" : "off"}]
opt_design
place_design
if {$retime} {phys_opt_design -retime} else {phys_opt_design}
route_design

report_timing_summary -no_header -file timing_summary.rpt
report_timing -max_paths 3 -nworst 1 -path_type full -input_pins -file worst_paths.rpt
report_utilization -file utilization.rpt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "CABOSSE-RESULT top=$top loop_cycles=$loop retime=$retime wns_ns=$wns"
