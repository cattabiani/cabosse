// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The core's CL top for the F2 Small Shell (platforms/f2/README.md, "The core
// on F2"): f2_core.sv (rtl/core.sv and its HBM port 0) on clk_hbm_axi
// (D-065), in the shell body the bring-up design uses (../cl_shell.svh: OCL
// and PCIS crossed into clk, the HBM IP, the clock generator, the debug
// bridge), and an ILA on HBM port 0. Built only by Vivado, in the shell flow.
//
// The OCL port decodes the low LiteW bits of its address (the core's
// registers, then the wrapper's at 0x2_0000); the rest of BAR0 repeats them.
// PCIS (BAR4) reaches HBM through port 0 only while HOST_PORT is set, and
// only once STATUS shows HBM ready.

`ifndef CABOSSE_BUILD
`define CABOSSE_BUILD 32'h0
`endif

module cabosse_core #(
  parameter EN_DDR = 0,  // the shell flow's parameters (unused here)
  parameter EN_HBM = 1
) (
  `include "cl_ports.vh"
);

  localparam int unsigned Ports     = 32;
  localparam int unsigned AddrWidth = 34;
  localparam int unsigned DataWidth = 256;
  localparam int unsigned BeatBytes = DataWidth / 8;
  localparam int unsigned LiteW     = 18;

  `include "cl_shell.svh"

  // --- The core -------------------------------------------------------------------

  f2_core #(.Ports(Ports), .Build(`CABOSSE_BUILD), .LiteW(LiteW)) u_core (
    .clk_i       (clk),
    .rst_ni      (core_rst_nq),
    .hbm_ready_i (hbm_ready),
    .host_sel_o  (host_sel),
    .h_ar_valid_i(h_ar_valid),
    .h_ar_ready_o(h_ar_ready),
    .h_ar_addr_i (h_ar_addr[AddrWidth-1:0]),
    .h_ar_len_i  (8'(h_ar_len)),
    .h_ar_burst_i(h_ar_burst),
    .h_r_valid_o (h_r_valid),
    .h_r_ready_i (h_r_ready),
    .h_r_data_o  (h_r_data),
    .h_r_resp_o  (h_r_resp),
    .h_r_last_o  (h_r_last),
    .h_aw_valid_i(h_aw_valid),
    .h_aw_ready_o(h_aw_ready),
    .h_aw_addr_i (h_aw_addr[AddrWidth-1:0]),
    .h_aw_len_i  (8'(h_aw_len)),
    .h_aw_burst_i(h_aw_burst),
    .h_w_valid_i (h_w_valid),
    .h_w_ready_o (h_w_ready),
    .h_w_data_i  (h_w_data),
    .h_w_strb_i  (h_w_strb),
    .h_w_last_i  (h_w_last),
    .h_b_valid_o (h_b_valid),
    .h_b_ready_i (h_b_ready),
    .h_b_resp_o  (h_b_resp),
    .s_aw_valid_i(lite_aw_valid),
    .s_aw_ready_o(lite_aw_ready),
    .s_aw_addr_i (lite_aw_addr[LiteW-1:0]),
    .s_w_valid_i (lite_w_valid),
    .s_w_ready_o (lite_w_ready),
    .s_w_data_i  (lite_w_data),
    .s_b_valid_o (lite_b_valid),
    .s_b_ready_i (lite_b_ready),
    .s_b_resp_o  (lite_b_resp),
    .s_ar_valid_i(lite_ar_valid),
    .s_ar_ready_o(lite_ar_ready),
    .s_ar_addr_i (lite_ar_addr[LiteW-1:0]),
    .s_r_valid_o (lite_r_valid),
    .s_r_ready_i (lite_r_ready),
    .s_r_data_o  (lite_r_data),
    .s_r_resp_o  (lite_r_resp),
    .ar_valid_o  (ar_valid),
    .ar_ready_i  (ar_ready),
    .ar_addr_o   (ar_addr),
    .ar_len_o    (ar_len),
    .ar_size_o   (ar_size),
    .ar_burst_o  (ar_burst),
    .r_valid_i   (r_valid),
    .r_ready_o   (r_ready),
    .r_data_i    (r_data),
    .r_resp_i    (r_resp),
    .r_last_i    (r_last),
    .aw_valid_o  (aw_valid),
    .aw_ready_i  (aw_ready),
    .aw_addr_o   (aw_addr),
    .aw_len_o    (aw_len),
    .aw_size_o   (aw_size),
    .aw_burst_o  (aw_burst),
    .w_valid_o   (w_valid),
    .w_ready_i   (w_ready),
    .w_data_o    (w_data),
    .w_strb_o    (w_strb),
    .w_last_o    (w_last),
    .b_valid_i   (b_valid),
    .b_ready_o   (b_ready),
    .b_resp_i    (b_resp)
  );

  // --- Debug: the ILA on HBM port 0 --------------------------------------------

  // probe0: the handshakes, probe1/2: the read and write addresses
  // (synth_cl.tcl generates the ILA with these widths).
  cabosse_ila u_ila (
    .clk   (clk),
    .probe0({host_sel, hbm_ready, ar_valid[0], ar_ready[0], r_valid[0], r_ready[0],
             r_last[0], aw_valid[0], aw_ready[0], w_valid[0], w_ready[0], w_last[0],
             b_valid[0], b_ready[0], r_resp[0][1], b_resp[0][1]}),
    .probe1(ar_addr[0]),
    .probe2(aw_addr[0])
  );

endmodule
