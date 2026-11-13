# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# Route a placed CL checkpoint (post_phys_opt) in a session of its own: the
# tail of AWS's build_level_1_cl.tcl (aws-fpga f2 branch), run from a
# checkpoint instead of after place in the same session, so a route (or a
# directive sweep) does not repeat synthesis and placement
# (scripts/f2/build_cl.sh, FROM_DCP; PLAN.md M10 step 7). The flow sets no
# parameters for place or route that the checkpoint would not carry.
#
# Output names are AWS's, so aws_build_dcp_from_cl.py --package-only makes the
# tarball from them: <cl>.<tag>.post_route.dcp, or .post_route.VIOLATED.dcp
# when any setup or hold slack is negative (AWS's check_timing_path), and
# <tag>.debug_probes.ltx. If route_design fails, or leaves nets unrouted or
# in conflict, the partly routed design goes to
# <cl>.<tag>.post_route.FAILED.dcp, with its route status, for later
# analysis, and Vivado exits with 1.
#
# vivado -mode batch -source route_cl.tcl \
#   -tclargs <placed.dcp> <checkpoints dir> <reports dir> <cl> <tag> <directive>

lassign $argv dcp checkpoints_dir reports_dir cl tag directive
set out $checkpoints_dir/$cl.$tag

open_checkpoint $dcp
set failed [catch {route_design -directive $directive -tns_cleanup} err]
report_route_status -file $reports_dir/$cl.$tag.post_route_status.rpt
# route_design can end without an error yet leave nets unrouted or
# overlapping (AWS's flow does not check): count them.
set bad [get_nets -quiet -hierarchical -top_net_of_hierarchical_group \
  -filter {ROUTE_STATUS == UNROUTED || ROUTE_STATUS == PARTIAL || ROUTE_STATUS == CONFLICTS}]
if {$failed || [llength $bad] > 0} {
  puts "CABOSSE-ROUTE failed: [llength $bad] nets not routed or in conflict; $err"
  foreach n [lrange $bad 0 9] { puts "CABOSSE-ROUTE net $n [get_property ROUTE_STATUS $n]" }
  write_checkpoint -force $out.post_route.FAILED.dcp
  exit 1
}

set setup [get_timing_paths -max_paths 1 -slack_lesser_than 0 -setup]
set hold [get_timing_paths -max_paths 1 -slack_lesser_than 0 -hold]
if {[llength $setup] == 0 && [llength $hold] == 0} {
  write_checkpoint -force $out.post_route.dcp
} else {
  write_checkpoint -force $out.post_route.VIOLATED.dcp
}
write_debug_probes -no_partial_ltxfile -force $checkpoints_dir/$tag.debug_probes.ltx
report_timing -delay_type max -path_type full_clock_expanded -max_paths 10 \
  -nworst 1 -input_pins -slice_pins -sort_by group -significant_digits 3 \
  -file $reports_dir/$cl.$tag.post_route_timing.rpt
catch {
  puts "CABOSSE-ROUTE WNS [get_property SLACK [get_timing_paths -max_paths 1 -setup]]\
    WHS [get_property SLACK [get_timing_paths -max_paths 1 -hold]]"
}
close_design
