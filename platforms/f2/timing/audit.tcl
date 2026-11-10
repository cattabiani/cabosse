# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# What Vivado kept of our placement hints, on a placed or routed checkpoint
# (into audit.rpt in the current directory):
#   vivado -mode batch -source audit.tcl -tclargs placed.dcp [floorplan.xdc]
# For each USER_SLR_ASSIGNMENT line of the floorplan: the cells it matches
# (hierarchical or leaf) and whether the property is still on them; then the
# RTL's synthesis attributes; then the SLR each assigned cell's primitives
# were placed in. A hint that is not kept, or kept but not followed, does
# nothing (AGENTS.md). Cheap parts first, each written as soon as known.

set dcp [lindex $argv 0]
set xdc [expr {[llength $argv] > 1 ? [lindex $argv 1] \
  : "[file dirname [file normalize [info script]]]/../core/small_shell_cl_pnr_user.xdc"}]
set T0 [clock seconds]
proc progress {msg} {puts "CABOSSE-AUDIT [expr {[clock seconds] - $::T0}] s: $msg"; flush stdout}
if {$dcp ne ""} {open_checkpoint $dcp}
progress "checkpoint open"
# Collections of thousands of cells are turned into lists below.
set_param tcl.collectionResultDisplayLimit 0
set out [open audit.rpt w]
proc emit {line} {puts $::out $line; flush $::out}

# 1. Each line: cells matched, hierarchical or leaf, property kept.
set LINES {}  ;# {line target hier_names leaf_names}
set n 0
set fh [open $xdc r]
set text [read $fh]
close $fh
emit "# line target matched hier leaf hier_with_property"
foreach line [split $text \n] {
  incr n
  if {![regexp {^set_property USER_SLR_ASSIGNMENT (\S+) \[(.*)\]\s*$} $line -> target expr]} {continue}
  if {[catch {
    set cells [uplevel #0 $expr]
    set hier [filter -quiet $cells {!IS_PRIMITIVE}]
    set leaf [filter -quiet $cells {IS_PRIMITIVE}]
    set kept 0
    foreach v [get_property -quiet USER_SLR_ASSIGNMENT $hier] {if {$v eq $target} {incr kept}}
    emit "$n $target [llength $cells] [llength $hier] [llength $leaf] $kept"
    lappend LINES [list $n $target [get_property -quiet NAME $hier] [get_property -quiet NAME $leaf]]
  } err]} {emit "$n $target error: $err"}
}
progress "lines read"

# 2. RTL attributes (rtl/fp_product.sv, rtl/dot_lane.sv).
set dt [get_cells -quiet -hierarchical -filter {NAME =~ *mul_fabric_q_reg*}]
emit "dont_touch mul_fabric_q: [llength $dt] cells, [llength [filter -quiet $dt {DONT_TOUCH}]] with DONT_TOUCH"
set mf [get_cells -quiet -hierarchical -filter {NAME =~ *load_we_q_reg*}]
set worst 0
foreach net [get_nets -quiet -of_objects [get_pins -quiet -of_objects $mf -filter {REF_PIN_NAME == Q}]] {
  set fo [get_property -quiet FLAT_PIN_COUNT $net]
  if {$fo ne "" && $fo > $worst} {set worst $fo}
}
emit "max_fanout load_we_q: [llength $mf] cells ([llength [filter -quiet $mf {NAME =~ *_rep*}]] replicas),\
  [llength [filter -quiet $mf {MAX_FANOUT != ""}]] with MAX_FANOUT, largest net [expr {$worst - 1}] loads"
progress "attributes"

# 3. Placement. Each assigned hierarchical cell and each assigned leaf, by
# name; then one pass over every placed cell of each SLR (three bulk
# queries: a query per cell is far too slow on this design), counting it
# for its nearest assigned ancestor, or for itself if it is an assigned leaf.
set OWNER [dict create]  ;# assigned cell name -> line
foreach l $LINES {
  lassign $l n target hier leaf
  foreach c [concat $hier $leaf] {dict set OWNER $c $n}
}
array set CNT {}  ;# (line,SLR) -> primitives
foreach slr [get_slrs] {
  foreach c [get_property NAME [get_cells -quiet -of_objects $slr -filter {IS_PRIMITIVE}]] {
    set p $c
    while {1} {
      if {[dict exists $OWNER $p]} {incr CNT([dict get $OWNER $p],$slr); break}
      set i [string last / $p]
      if {$i < 0} {break}
      set p [string range $p 0 [expr {$i - 1}]]
    }
  }
  progress "$slr counted"
}
emit "# line target placed_in_target/primitives by_slr"
set tot_in 0
set tot_all 0
foreach l $LINES {
  lassign $l n target
  set by [dict create]
  foreach slr [get_slrs] {if {[info exists CNT($n,$slr)]} {dict set by $slr $CNT($n,$slr)}}
  set all 0
  dict for {s k} $by {incr all $k}
  set in [expr {[dict exists $by $target] ? [dict get $by $target] : 0}]
  incr tot_in $in
  incr tot_all $all
  emit "$n $target $in/$all {[lsort -stride 2 -index 0 $by]}"
}
emit "# assignments followed: $tot_in of $tot_all primitives in their target SLR"
close $out
progress "done"
