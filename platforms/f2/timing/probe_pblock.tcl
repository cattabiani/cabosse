# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# Prints pblock_CL's ranges and each SLR's clock regions (to write the
# floorplan's pblocks as literals; an xdc cannot loop).
open_checkpoint [lindex $argv 0]
foreach slr [get_slrs] {puts "CABOSSE-FIT $slr: [get_clock_regions -of_objects $slr]"}
puts "CABOSSE-FIT pblock_CL GRID_RANGES: [get_property GRID_RANGES [get_pblocks pblock_CL]]"
puts "CABOSSE-FIT pblock_CL IS_SOFT: [get_property IS_SOFT [get_pblocks pblock_CL]]"
