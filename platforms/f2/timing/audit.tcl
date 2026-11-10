# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# What Vivado kept of our placement hints, on a placed or routed checkpoint
# (into audit.rpt in the current directory):
#   vivado -mode batch -source audit.tcl -tclargs placed.dcp [floorplan.xdc]
# For each USER_SLR_ASSIGNMENT line of the floorplan: the cells it matches
# (hierarchical or leaf) and whether the property is still on them; then the
# RTL's synthesis attributes. A hint that is not kept, or kept but not
# followed, does nothing (AGENTS.md). Each line is written as soon as known.

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

# Where the cells were placed is not checked here: three bulk queries for
# the cells of each SLR ran over 55 min on core build 4's checkpoint
# (r6i.2xlarge). For a pblock, report_utilization -pblocks says it.
close $out
progress "done"
