# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
# Copyright 2026 The Cabosse Authors
#
# Synthesis of the bring-up CL (cabosse_bringup.sv) in AWS's shell flow:
# aws_build_dcp_from_cl.py's build_all.tcl sources it as
# $CL_DIR/build/scripts/synth_cabosse_bringup.tcl, with its variables set
# (CL, DEVICE_TYPE, HDK_SHELL_DIR, HDK_IP_SRC_DIR, HDK_BD_SRC_DIR,
# HDK_BD_GEN_DIR, src_post_enc_dir, constraints_dir). The shell's header and
# footer (the shell's sources, constraints and the post-synthesis checkpoint)
# and AWS's IP configurations come from the aws-fpga checkout. Our sources
# come from the repository at CABOSSE_REPO (environment), in the order of
# rtl/sources.f and platforms/f2/bringup/sources.f, then the CL top
# (scripts/f2/build_bringup.sh also puts the top in the CL's design folder,
# for AWS's scripts). CABOSSE_BUILD (environment): the commit, 8 hex digits,
# read back in BUILD.

source ${HDK_SHELL_DIR}/build/scripts/synth_cl_header.tcl

# The files a sources.f names, in order, without its // comments.
proc read_listed {listed} {
  set f [open $listed]
  foreach line [split [read $f] "\n"] {
    set line [string trim [regsub {//.*} $line ""]]
    if {$line ne ""} {read_verilog -sv [file dirname $listed]/$line}
  }
  close $f
}

print "Reading Cabosse sources"
set repo $::env(CABOSSE_REPO)
read_listed $repo/rtl/sources.f
read_listed $repo/platforms/f2/bringup/sources.f
read_verilog -sv $repo/platforms/f2/bringup/${CL}.sv

print "Reading AWS's IP: HBM, OCL clock converter, debug bridge, PCIS SmartConnect"
read_ip [list \
  ${HDK_IP_SRC_DIR}/cl_hbm/cl_hbm.xci \
  ${HDK_IP_SRC_DIR}/cl_axi_clock_converter_light/cl_axi_clock_converter_light.xci \
  ${HDK_IP_SRC_DIR}/cl_debug_bridge/cl_debug_bridge.xci \
]
add_files ${HDK_BD_SRC_DIR}/cl_axi_sc_1x1/cl_axi_sc_1x1.bd
read_verilog ${HDK_BD_GEN_DIR}/cl_axi_sc_1x1/hdl/cl_axi_sc_1x1_wrapper.v
# The clock generator's MMCMs and register crossbar, as AWS's examples read them.
set clk_ips $HDK_SHELL_DESIGN_DIR/../../ip/cl_ip/cl_ip.srcs/sources_1/ip
foreach ip {clk_mmcm_a clk_mmcm_b clk_mmcm_c clk_mmcm_hbm cl_clk_axil_xbar} {
  read_ip $clk_ips/$ip/$ip.xci
}

print "Generating the ILA on HBM port 0"
create_ip -name ila -vendor xilinx.com -library ip -module_name cabosse_ila
set_property -dict [list \
  CONFIG.C_NUM_OF_PROBES {3} \
  CONFIG.C_PROBE0_WIDTH {16} \
  CONFIG.C_PROBE1_WIDTH {34} \
  CONFIG.C_PROBE2_WIDTH {34} \
  CONFIG.C_DATA_DEPTH {1024} \
] [get_ips cabosse_ila]
generate_target all [get_ips cabosse_ila]
synth_ip [get_ips cabosse_ila]

print "Reading user constraints"
read_xdc [list ${constraints_dir}/cl_synth_user.xdc ${constraints_dir}/cl_timing_user.xdc]
set_property PROCESSING_ORDER LATE [get_files {cl_synth_user.xdc cl_timing_user.xdc}]
set build $::env(CABOSSE_BUILD)

print "Synthesizing ${CL}"
update_compile_order -fileset sources_1
# -keep_equivalent_registers and -shreg_min_size 16 as the engine's build
# needs them (PLAN.md, M8 notes; platforms/f2/timing/timing.tcl).
synth_design -mode out_of_context \
             -top ${CL} \
             -verilog_define XSDB_SLV_DIS \
             -verilog_define CABOSSE_BUILD=32'h${build} \
             -part ${DEVICE_TYPE} \
             -keep_equivalent_registers \
             -shreg_min_size 16

print "Connecting the ILA to the debug bridge"
connect_debug_cores -master [get_cells [get_debug_cores -filter {NAME=~*CL_DEBUG_BRIDGE*}]] \
                    -slaves [get_cells u_ila]

source ${HDK_SHELL_DIR}/build/scripts/synth_cl_footer.tcl
