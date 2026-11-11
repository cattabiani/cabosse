# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# The reports of a routed timing harness, into the current directory: run by
# timing.tcl after routing, or on a saved checkpoint with
#   vivado -mode batch -source report.tcl -tclargs routed.dcp

if {[llength $argv] == 1} {open_checkpoint [lindex $argv 0]}

# Paths under this slack are near-critical: they are what a better worst
# slack has to speed up (families.rpt, slack_histogram.rpt).
set NEAR_NS 0.2

report_timing_summary -no_header -file timing_summary.rpt
report_timing -max_paths 40 -nworst 1 -path_type full -input_pins -file worst_paths.rpt
# Per path: logic and net delay, skew, levels, SLR crossings, high fan-out (a table).
report_design_analysis -timing -setup -max_paths 300 -file design_analysis.rpt
# Start, end and slack of the worst 200 paths.
report_timing -max_paths 200 -nworst 1 -path_type summary -file worst_paths_summary.rpt
# Logic levels per path, and the nets with the most loads.
report_design_analysis -logic_level_distribution -max_paths 10000 -file logic_levels.rpt
report_high_fanout_nets -timing -max_nets 50 -file fanout.rpt
report_utilization -file utilization.rpt
report_utilization -hierarchical -hierarchical_depth 4 -file utilization_hier.rpt
report_design_analysis -congestion -file congestion.rpt

# Every near-critical path's family, compactly: start and end cells with
# their indices dropped, the paths of each, and the worst slack. The
# properties come as lists, a call each. Then the paths per 0.05 ns of slack.
set paths [get_timing_paths -setup -max_paths 300000 -nworst 1 -slack_lesser_than $NEAR_NS]
set families [dict create]
set bins [dict create]
if {[llength $paths]} {  # get_property fails on none
  foreach s [get_property STARTPOINT_PIN $paths] e [get_property ENDPOINT_PIN $paths] \
          slack [get_property SLACK $paths] {
    dict lappend families "[regsub -all {\[[0-9]+\]} $s {[*]}] -> [regsub -all {\[[0-9]+\]} $e {[*]}]" $slack
    dict incr bins [format %.2f [expr {floor($slack / 0.05) * 0.05}]]
  }
}
set f [open families.rpt w]
dict for {key slacks} $families {
  puts $f "[tcl::mathfunc::min {*}$slacks] [llength $slacks] $key"
}
close $f
set f [open slack_histogram.rpt w]
puts $f "# paths (one per endpoint) per slack bin, ns from the bin's lower edge, under $NEAR_NS ns"
foreach lo [lsort -real [dict keys $bins]] {
  puts $f "$lo [dict get $bins $lo]"
}
close $f
