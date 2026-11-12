# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# What Vivado kept of our placement hints, on a placed or routed checkpoint
# (into audit.rpt in the current directory):
#   vivado -mode batch -source audit.tcl -tclargs placed.dcp [floorplan.xdc]
# For each add_cells_to_pblock line of the floorplan: the cells it matches
# (hierarchical or leaf); then the
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
emit "# line pblock matched hier leaf"
foreach line [split $text \n] {
  incr n
  if {![regexp {^add_cells_to_pblock (\S+) \[(.*)\]\s*$} $line -> target expr]} {continue}
  if {[catch {
    set cells [uplevel #0 $expr]
    set hier [filter -quiet $cells {!IS_PRIMITIVE}]
    set leaf [filter -quiet $cells {IS_PRIMITIVE}]
    emit "$n $target [llength $cells] [llength $hier] [llength $leaf]"
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
set worst 0
set reps 0
foreach net [get_nets -quiet -hierarchical -filter {NAME =~ *gen_stage*take*}] {
  if {[string match *_rep* $net]} {incr reps}
  set fo [get_property -quiet FLAT_PIN_COUNT $net]
  if {$fo ne "" && $fo > $worst} {set worst $fo}
}
emit "max_fanout bringup_pipe take: $reps replica nets, largest net [expr {$worst - 1}] loads"
foreach sig {unit_end step_load} {
  set nets [get_nets -quiet -hierarchical -filter "NAME =~ *u_engine/${sig}*"]
  set worst 0
  foreach net $nets {
    set fo [get_property -quiet FLAT_PIN_COUNT $net]
    if {$fo ne "" && $fo > $worst} {set worst $fo}
  }
  emit "max_fanout engine $sig: [llength $nets] nets, largest [expr {$worst - 1}] loads"
}
set dsp [get_cells -quiet -hierarchical -filter {REF_NAME =~ DSP48E2 && NAME =~ *u_engine/gen_lane*}]
emit "use_dsp fp_product (engine lanes): [llength $dsp] DSP48E2 cells (4 per lane: 512 at 32 ports)"
progress "attributes"

# Where the cells were placed is not checked here: three bulk queries for
# the cells of each SLR ran over 55 min on core build 4's checkpoint
# (r6i.2xlarge). For a pblock, report_utilization -pblocks says it.
close $out
progress "done"
