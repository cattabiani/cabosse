# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# Does the floorplan fit? On a linked or placed checkpoint of the core in the
# shell: read the floorplan (the build's own file, unless the checkpoint has
# it), then report each SLR
# pblock's cells and utilization against what its clock regions hold:
#   vivado -mode batch -source fitcheck.tcl -tclargs dcp.dcp [floorplan.xdc]
# Writes fit.rpt, pblocks.rpt and (with a CABOSSE-FIT line each) the log.
# Cells are only assigned, not placed, so nothing here depends on the placer.

set dcp [lindex $argv 0]
set xdc [expr {[llength $argv] > 1 ? [lindex $argv 1] \
  : "[file dirname [file normalize [info script]]]/../core/small_shell_cl_pnr_user.xdc"}]
open_checkpoint $dcp
set_param tcl.collectionResultDisplayLimit 0
# a checkpoint from the build has the floorplan already
if {[llength [get_pblocks -quiet pblock_slr0]] == 0} {read_xdc $xdc}
set out [open fit.rpt w]
set total [llength [get_cells -hierarchical -filter {IS_PRIMITIVE}]]
puts $out "leaf cells in the design: $total"
set assigned 0
foreach slr {0 1 2 hbm} {
  set name [expr {$slr eq "hbm" ? "pblock_hbm" : "pblock_slr$slr"}]
  set pb [get_pblocks $name]
  set n [llength [get_cells -hierarchical -filter "IS_PRIMITIVE && PBLOCK == $name"]]
  incr assigned $n
  set line "$name: parent [get_property PARENT $pb], IS_SOFT [get_property IS_SOFT $pb],\
    ranges [get_property GRID_RANGES $pb], leaf cells $n"
  puts $out $line
  puts "CABOSSE-FIT $line"
}
puts $out "leaf cells in a pblock: $assigned of $total"
puts "CABOSSE-FIT leaf cells in a pblock: $assigned of $total"
close $out
foreach slr {0 1 2 hbm} {
  set name [expr {$slr eq "hbm" ? "pblock_hbm" : "pblock_slr$slr"}]
  report_utilization -pblocks [get_pblocks $name] -file $name.rpt
  set fh [open $name.rpt]
  foreach l [split [read $fh] \n] {
    if {[regexp {^\| (CLB LUTs|CLB Registers|Block RAM Tile|URAM|DSPs)} $l]} {puts "CABOSSE-FIT slr$slr $l"}
  }
  close $fh
}
