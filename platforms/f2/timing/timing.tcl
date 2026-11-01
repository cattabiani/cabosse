# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# Out-of-context synthesis, placement and routing of a timing harness in
# this folder (mac_loop.sv, fma_path.sv, lane_path.sv) on F2's part at
# clk_main_a0's 250 MHz (docs/f2.md). Reports go to the current directory.
# Usage: vivado -mode batch -source timing.tcl -tclargs REPO PART TOP PARAM VALUE LOOP RETIME [FLOW [PLACE]]
# PARAM=VALUE is the harness's length parameter, LOOP the loop's (or the
# path's) cycles, for the report. RETIME 1 lets Vivado move registers across logic
# (synthesis and physical optimization; under explore and shell, synthesis only).
# FLOW:
# - "shell", the pass/fail gate: what the shell build (M8) runs on the CL
#   (aws-fpga, branch f2, hdk/common/shell_stable/build/scripts/
#   build_level_1_cl.tcl and aws_build_dcp_from_cl.py's defaults): the
#   design inside the Small Shell's CL region (pblock_CL, read from
#   small_shell_level_1_fp_cl.xdc in the aws-fpga checkout named by
#   CABOSSE_AWS_FPGA; Amazon Software License, so it is read there, not
#   copied here), placement with PLACE (default SSI_SpreadLogic_high),
#   phys_opt_design and route_design -tns_cleanup with AggressiveExplore,
#   no physical optimization after routing.
# - "explore": the timing-driven directives, the whole part, and a physical
#   optimization after routing while timing fails.
# - "default" (or none): Vivado's default directives, the whole part.
# A run passes with a worst slack of at least MARGIN_NS (below). Environment:
# CABOSSE_THREADS (default 4) for Vivado's threads, CABOSSE_DCP=1 to write
# the routed checkpoint, CABOSSE_AWS_FPGA for the shell flow.

lassign $argv repo part top param value loop retime flow place
if {$flow eq ""} {set flow default}
if {$flow ni {default explore shell}} {error "timing.tcl: unknown flow $flow"}
if {$place eq ""} {set place [expr {$flow eq "shell" ? "SSI_SpreadLogic_high" : "-"}]}
if {$flow ne "shell" && $place ne "-"} {error "timing.tcl: PLACE is for the shell flow"}
# The margin over the estimated clock uncertainty (timing.xdc) for what an
# out-of-context build does not see: the shell's clock tree across the
# SLRs and its logic around the CL (an estimate, PLAN.md M6).
set MARGIN_NS 0.1
set_param general.maxThreads [expr {[info exists env(CABOSSE_THREADS)] ? $env(CABOSSE_THREADS) : 4}]

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
if {$flow eq "shell"} {
  # The shell's floorplan, its one mapping (WRAPPER/CL) replaced by the
  # harness's top-level cells.
  if {![info exists env(CABOSSE_AWS_FPGA)]} {error "timing.tcl: shell flow needs CABOSSE_AWS_FPGA"}
  set f [open $env(CABOSSE_AWS_FPGA)/hdk/common/shell_stable/build/constraints/small_shell_level_1_fp_cl.xdc]
  set fp [read $f]
  close $f
  set n [regsub -all -line {^add_cells_to_pblock .*$} $fp {} fp]
  if {$n != 1} {error "timing.tcl: expected one add_cells_to_pblock in the shell's floorplan, found $n"}
  eval $fp
  add_cells_to_pblock pblock_CL [get_cells]
  place_design -directive $place -no_bufg_opt
  phys_opt_design -directive AggressiveExplore
  route_design -directive AggressiveExplore -tns_cleanup
} elseif {$flow eq "explore"} {
  place_design -directive ExtraTimingOpt
  phys_opt_design -directive Explore
  route_design -directive Explore
  if {[get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]] < 0} {
    phys_opt_design -directive Explore
  }
} else {
  place_design
  if {$retime} {phys_opt_design -retime} else {phys_opt_design}
  route_design
}
# The routed design, for reports without a rebuild (time_loops.sh keeps it).
if {[info exists env(CABOSSE_DCP)] && $env(CABOSSE_DCP) eq "1"} {write_checkpoint -force routed.dcp}

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
puts "CABOSSE-RESULT top=$top loop_cycles=$loop retime=$retime flow=$flow place=$place wns_ns=$wns"
# A design that misses 250 MHz with the margin must not pass as measured
# (PLAN.md, M8 notes).
if {$wns < $MARGIN_NS} {
  puts "CABOSSE-TIMING-FAILED top=$top wns_ns=$wns"
  exit 1
}
