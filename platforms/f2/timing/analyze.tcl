# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# Where a routed full core's area and slack go (M10): the vector unit's and
# the engine's utilization by sub-block, their cells per SLR, Vivado's QoR
# suggestions, path characteristics, complexity and control sets. Into the
# current directory, on a saved checkpoint:
#   vivado -mode batch -source analyze.tcl -tclargs routed.dcp
# Each report is caught on its own, so one failing does not lose the rest.

if {[llength $argv] == 1} {open_checkpoint [lindex $argv 0]}

set CORE WRAPPER/CL/u_core/u_core
set BLOCKS {u_vec u_engine u_controller}

proc try_report {name script} {
  set t [clock seconds]
  if {[catch {uplevel 1 $script} err]} {puts "CABOSSE-ANALYZE $name failed: $err"}
  puts "CABOSSE-ANALYZE $name [expr {[clock seconds] - $t}] s"
}

try_report utilization_core {
  report_utilization -hierarchical -hierarchical_depth 4 \
    -cells [get_cells $CORE] -file utilization_core.rpt
}
try_report utilization_slr {report_utilization -slr -file utilization_slr.rpt}

# Each block's primitives per SLR, by type (LUT, FF, URAM, ...).
try_report cells_per_slr {
  set f [open cells_per_slr.rpt w]
  foreach slr [get_slrs] {
    foreach b $BLOCKS {
      set cells [get_cells -quiet -of_objects $slr -filter "NAME =~ $CORE/$b/*"]
      set counts [dict create]
      foreach g [get_property PRIMITIVE_GROUP $cells] {dict incr counts $g}
      puts $f "$slr $b [llength $cells] [lsort -stride 2 -index 0 $counts]"
    }
  }
  close $f
}

try_report qor_suggestions {report_qor_suggestions -max_paths 100 -file qor_suggestions.rpt}
try_report design_timing {
  report_design_analysis -timing -max_paths 100 -setup -file design_timing.rpt
}
try_report complexity {
  report_design_analysis -complexity -hierarchical_depth 6 -file complexity.rpt
}
try_report control_sets {report_control_sets -hierarchical -hierarchical_depth 5 -file control_sets.rpt}
try_report fanout {report_high_fanout_nets -timing -max_nets 100 -file fanout.rpt}
