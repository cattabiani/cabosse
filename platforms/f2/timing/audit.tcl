# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# What Vivado kept of our placement hints, on a placed or routed checkpoint
# (into audit.rpt in the current directory):
#   vivado -mode batch -source audit.tcl -tclargs placed.dcp [floorplan.xdc]
# For each USER_SLR_ASSIGNMENT line of the floorplan: the cells it matches
# (hierarchical or leaf), whether the property is still on them, and the SLR
# each of their primitives was placed in. Then the RTL's synthesis
# attributes: whether the cells carry them and what they did. A hint that is
# not kept, or kept but not followed, does nothing (AGENTS.md).

set dcp [lindex $argv 0]
set xdc [expr {[llength $argv] > 1 ? [lindex $argv 1] \
  : "[file dirname [file normalize [info script]]]/../core/small_shell_cl_pnr_user.xdc"}]
if {$dcp ne ""} {open_checkpoint $dcp}
# Collections of thousands of cells are turned into lists below.
set_param tcl.collectionResultDisplayLimit 0
set out [open audit.rpt w]

# The SLR of every placed primitive, by name (three bulk queries; a query
# per cell or per site is far too slow on this design).
set CELL_SLR [dict create]
foreach slr [get_slrs] {
  set t [clock seconds]
  foreach c [get_property NAME [get_cells -quiet -of_objects $slr]] {dict set CELL_SLR $c $slr}
  puts "CABOSSE-AUDIT $slr [expr {[clock seconds] - $t}] s, [dict size $CELL_SLR] cells so far"
}

# Primitives per SLR, as "SLR0 n SLR1 n ..." sorted.
proc slr_counts {names} {
  global CELL_SLR
  set counts [dict create]
  foreach c $names {
    dict incr counts [expr {[dict exists $CELL_SLR $c] ? [dict get $CELL_SLR $c] : "unplaced"}]
  }
  return [lsort -stride 2 -index 0 $counts]
}

# The full names of the primitives under hierarchical cells.
proc prims_under {hier} {
  set names {}
  foreach h $hier {
    current_instance -quiet $h
    set prims [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE}]
    if {[llength $prims]} {lappend names {*}[get_property NAME $prims]}
    current_instance -quiet
  }
  return $names
}

# One floorplan line: its cells, what was kept, where they were placed.
proc audit_line {out n target expr} {
  global tot_in tot_all
  set cells [uplevel #0 $expr]
  set hier [filter -quiet $cells {!IS_PRIMITIVE}]
  set leaf [filter -quiet $cells {IS_PRIMITIVE}]
  set kept 0
  foreach v [get_property -quiet USER_SLR_ASSIGNMENT $hier] {if {$v eq $target} {incr kept}}
  set prims [prims_under $hier]
  set by [slr_counts $prims]
  set in [expr {[dict exists $by $target] ? [dict get $by $target] : 0}]
  incr tot_in $in
  incr tot_all [llength $prims]
  puts $out "$n $target [llength $cells] [llength $hier] [llength $leaf] $kept\
    $in/[llength $prims] {$by} | {[slr_counts [get_property -quiet NAME $leaf]]}"
}

puts $out "# line target matched hier leaf kept_on_hier placed_in_target/primitives by_slr (hier) | leaf by_slr"
set tot_in 0
set tot_all 0
set n 0
set fh [open $xdc r]
foreach line [split [read $fh] \n] {
  incr n
  if {![regexp {^set_property USER_SLR_ASSIGNMENT (\S+) \[(.*)\]\s*$} $line -> target expr]} {continue}
  if {[catch {audit_line $out $n $target $expr} err]} {puts $out "$n $target error: $err"}
  flush $out
}
close $fh
puts $out "# hierarchical assignments followed: $tot_in of $tot_all primitives in their target SLR"

# RTL attributes (rtl/fp_product.sv, rtl/dot_lane.sv).
set dt [get_cells -quiet -hierarchical -filter {NAME =~ *mul_fabric_q_reg*}]
set dt_kept [llength [filter -quiet $dt {DONT_TOUCH}]]
puts $out "dont_touch mul_fabric_q: [llength $dt] cells, $dt_kept with DONT_TOUCH"
set mf [get_cells -quiet -hierarchical -filter {NAME =~ *load_we_q_reg*}]
set mf_kept [llength [filter -quiet $mf {MAX_FANOUT != ""}]]
set mf_rep [llength [filter -quiet $mf {NAME =~ *_rep*}]]
set worst 0
foreach c $mf {
  set fo [get_property -quiet FLAT_PIN_COUNT [get_nets -quiet -of_objects [get_pins -quiet $c/Q]]]
  if {$fo ne "" && $fo > $worst} {set worst $fo}
}
puts $out "max_fanout load_we_q: [llength $mf] cells ($mf_rep replicas), $mf_kept with MAX_FANOUT,\
  largest net [expr {$worst - 1}] loads"
close $out
puts "CABOSSE-AUDIT done"
