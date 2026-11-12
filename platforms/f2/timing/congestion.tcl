# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# Why would a design not route? On a placed checkpoint of the core in the
# shell (post_place or post_phys_opt), save:
#   - congestion.rpt: report_design_analysis -congestion
#   - route_status.rpt: report_route_status (what is still unrouted)
#   - pblocks_dsp.rpt: report_utilization -pblocks of each of our pblocks
#     (LUTs, registers, DSPs, block RAM)
#   - partpins.rpt: where the shell's partition pins sit, for the CL's PCIS,
#     OCL and SDA interfaces (property HD.PARTPIN_LOCS, and the pins' own
#     locations if that property is empty on the locked static part)
#   vivado -mode batch -source congestion.tcl -tclargs dcp.dcp
# Each part runs on its own, so one that fails does not stop the others.

set dcp [lindex $argv 0]
open_checkpoint $dcp
set_param tcl.collectionResultDisplayLimit 0

proc part {name body} {
  if {[catch {uplevel 1 $body} err]} {puts "CABOSSE-CONGESTION $name failed: $err"}
}

part congestion {report_design_analysis -congestion -file congestion.rpt}
part route_status {report_route_status -file route_status.rpt}
part pblocks {
  foreach name [get_pblocks -quiet pblock_*] {
    report_utilization -pblocks $name -file $name.dsp.rpt
  }
}
part partpins {
  set out [open partpins.rpt w]
  foreach pat {*PCIS* *OCL* *SDA*} {
    foreach pin [get_pins -quiet -hierarchical -filter "NAME =~ $pat && IS_PARTITION_PIN"] {
      puts $out "[get_property NAME $pin] HD.PARTPIN_LOCS=[get_property HD.PARTPIN_LOCS $pin]\
        LOC=[get_property LOC [get_cells -quiet -of_objects $pin]]"
    }
  }
  close $out
}
