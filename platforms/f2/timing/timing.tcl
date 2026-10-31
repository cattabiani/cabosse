# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# Out-of-context synthesis, placement and routing of a timing harness in
# this folder (mac_loop.sv, fma_path.sv, lane_path.sv) on F2's part at
# clk_main_a0's 250 MHz (docs/f2.md). Reports go to the current directory.
# Usage: vivado -mode batch -source timing.tcl -tclargs REPO PART TOP PARAM VALUE LOOP RETIME
# PARAM=VALUE is the harness's length parameter, LOOP the loop's (or the
# path's) cycles, for the report. RETIME 1 lets Vivado move registers across logic
# (synthesis and physical optimization).

lassign $argv repo part top param value loop retime
set_param general.maxThreads 4

# rtl/sources.f, without its comments; paths are relative to rtl/.
set f [open $repo/rtl/sources.f]
foreach line [split [read $f] "\n"] {
  set line [string trim $line]
  if {$line eq "" || [string match "//*" $line]} continue
  read_verilog -sv $repo/rtl/$line
}
close $f
foreach harness [glob $repo/platforms/f2/timing/*.sv] {read_verilog -sv $harness}
read_xdc -mode out_of_context $repo/platforms/f2/timing/timing.xdc

# -keep_equivalent_registers: the engine copies its control registers per
# group of lanes and banks for the fan-out (D-051); merged, they would be
# one register driving the whole engine again.
synth_design -top $top -part $part -mode out_of_context \
  -generic $param=$value -keep_equivalent_registers \
  -global_retiming [expr {$retime ? "on" : "off"}]
opt_design
place_design
if {$retime} {phys_opt_design -retime} else {phys_opt_design}
route_design

report_timing_summary -no_header -file timing_summary.rpt
report_timing -max_paths 3 -nworst 1 -path_type full -input_pins -file worst_paths.rpt
# Logic levels per path, and the nets with the most loads (the failing
# paths' families: below).
report_design_analysis -logic_level_distribution -max_paths 10000 -file logic_levels.rpt
report_high_fanout_nets -timing -max_nets 50 -file fanout.rpt
report_utilization -file utilization.rpt
report_utilization -hierarchical -hierarchical_depth 4 -file utilization_hier.rpt
report_design_analysis -congestion -file congestion.rpt

# Every failing path's family, compactly: start and end cells with their
# indices dropped, the paths of each, and the worst slack.
set families [dict create]
foreach p [get_timing_paths -setup -max_paths 100000 -nworst 1 -slack_lesser_than 0] {
  set key "[regsub -all {\[[0-9]+\]} [get_property STARTPOINT_PIN $p] {[*]}] -> [regsub -all {\[[0-9]+\]} [get_property ENDPOINT_PIN $p] {[*]}]"
  set slack [get_property SLACK $p]
  if {[dict exists $families $key]} {
    lassign [dict get $families $key] n worst
    dict set families $key [list [expr {$n + 1}] [expr {min($worst, $slack)}]]
  } else {
    dict set families $key [list 1 $slack]
  }
}
set f [open families.rpt w]
foreach {key v} $families {puts $f "[lindex $v 1] [lindex $v 0] $key"}
close $f
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "CABOSSE-RESULT top=$top loop_cycles=$loop retime=$retime wns_ns=$wns"
# A design that misses 250 MHz must not pass as measured (PLAN.md, M8 notes).
if {$wns < 0} {
  puts "CABOSSE-TIMING-FAILED top=$top wns_ns=$wns"
  exit 1
}
