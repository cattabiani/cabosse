# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# Out-of-context synthesis, placement and routing of a timing harness in
# this folder (mac_loop.sv, fma_path.sv, lane_path.sv) on F2's part at
# clk_main_a0's 250 MHz (docs/f2.md). Reports go to the current directory.
# Usage: vivado -mode batch -source timing.tcl -tclargs REPO PART TOP PARAM VALUE LOOP RETIME [FLOW]
# PARAM=VALUE is the harness's length parameter, LOOP the loop's (or the
# path's) cycles, for the report. RETIME 1 lets Vivado move registers across logic
# (synthesis and physical optimization). FLOW "explore" runs placement,
# physical optimization and routing with their timing-driven directives,
# and a physical optimization after routing; "default" (or none) without.

lassign $argv repo part top param value loop retime flow
if {$flow eq ""} {set flow default}
set explore [expr {$flow eq "explore"}]
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
# one register driving the whole engine again. -shreg_min_size 16: no
# shift registers (SRLs) for the short pipelines, whose copies are there
# to cover distance (an SRL is one cell, with a slow output).
synth_design -top $top -part $part -mode out_of_context \
  -generic $param=$value -keep_equivalent_registers -shreg_min_size 16 \
  -global_retiming [expr {$retime ? "on" : "off"}]
opt_design
if {$explore} {
  place_design -directive ExtraTimingOpt
  phys_opt_design -directive AggressiveExplore
  route_design -directive AggressiveExplore
  phys_opt_design -directive AggressiveExplore
} else {
  place_design
  if {$retime} {phys_opt_design -retime} else {phys_opt_design}
  route_design
}
# The routed design, for reports without a rebuild (time_loops.sh keeps it).
write_checkpoint -force routed.dcp

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
# indices dropped, the paths of each, and the worst slack. The properties
# come as lists, a call each.
set paths [get_timing_paths -setup -max_paths 300000 -nworst 1 -slack_lesser_than 0]
set families [dict create]
if {[llength $paths]} {  # get_property fails on none: a build that meets timing
  foreach s [get_property STARTPOINT_PIN $paths] e [get_property ENDPOINT_PIN $paths] \
          slack [get_property SLACK $paths] {
    dict lappend families "[regsub -all {\[[0-9]+\]} $s {[*]}] -> [regsub -all {\[[0-9]+\]} $e {[*]}]" $slack
  }
}
set f [open families.rpt w]
dict for {key slacks} $families {
  puts $f "[tcl::mathfunc::min {*}$slacks] [llength $slacks] $key"
}
close $f
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "CABOSSE-RESULT top=$top loop_cycles=$loop retime=$retime flow=$flow wns_ns=$wns"
# A design that misses 250 MHz must not pass as measured (PLAN.md, M8 notes).
if {$wns < 0} {
  puts "CABOSSE-TIMING-FAILED top=$top wns_ns=$wns"
  exit 1
}
