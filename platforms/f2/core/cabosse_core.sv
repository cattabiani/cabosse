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
//
// PCIM (M11): f2_core's copies of the logits, crossed from clk into
// clk_main_a0 (the shell's) by an asynchronous FIFO per channel (AW, W; B
// back), beside the shell in SLR1 (the floorplan). Writes only.

`define CABOSSE_PCIM  // cl_shell.svh leaves PCIM's write channels to this top

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
  localparam int unsigned HostWidth = 64;
  localparam int unsigned PcimBytes = 64;  // PCIM's beat

  `include "cl_shell.svh"
`undef CABOSSE_PCIM

  // --- The core -------------------------------------------------------------------

  logic                     pm_aw_valid, pm_aw_ready, pm_w_valid, pm_w_ready, pm_b_valid;
  logic [HostWidth-1:0]     pm_aw_addr;
  logic [8*PcimBytes-1:0]   pm_w_data;
  logic [PcimBytes-1:0]     pm_w_strb;
  logic [1:0]               pm_b_resp;

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
    .pcim_aw_valid_o(pm_aw_valid),
    .pcim_aw_ready_i(pm_aw_ready),
    .pcim_aw_addr_o (pm_aw_addr),
    .pcim_w_valid_o (pm_w_valid),
    .pcim_w_ready_i (pm_w_ready),
    .pcim_w_data_o  (pm_w_data),
    .pcim_w_strb_o  (pm_w_strb),
    .pcim_b_valid_i (pm_b_valid),
    .pcim_b_resp_i  (pm_b_resp),
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

  // --- PCIM: f2_core's writes crossed into clk_main_a0 ------------------------------

  // Each FIFO first-word-fall-through (dout valid while !empty) in LUT RAM;
  // a side waits while its reset is busy. AW and W written on clk, B on
  // clk_main_a0, each FIFO reset on its write side.
  logic                 pm_rst_q;  // on clk
  logic                 aw_full, aw_empty, aw_wbusy, w_full, w_empty, w_wbusy;
  logic                 b_full, b_empty, b_wbusy;
  logic                 aw_rbusy, w_rbusy, b_rbusy;  // unread: empty stays set while busy
  always_ff @(posedge clk) pm_rst_q <= !rst_n;

  assign pm_aw_ready = !aw_full && !aw_wbusy;
  assign pm_w_ready  = !w_full && !w_wbusy;
  assign pm_b_valid  = !b_empty;  // f2_core takes every response

  xpm_fifo_async #(
    .CDC_SYNC_STAGES    (3),
    .FIFO_MEMORY_TYPE   ("distributed"),
    .FIFO_READ_LATENCY  (0),
    .FIFO_WRITE_DEPTH   (32),  // LUT RAM is 32 deep anyway; the thresholds well inside its range
    .PROG_EMPTY_THRESH  (10),
    .PROG_FULL_THRESH   (10),
    .READ_DATA_WIDTH    (HostWidth),
    .READ_MODE          ("fwft"),
    .USE_ADV_FEATURES   ("0000"),
    .WRITE_DATA_WIDTH   (HostWidth)
  ) u_pcim_aw (
    .rst          (pm_rst_q),
    .wr_clk       (clk),
    .wr_en        (pm_aw_valid && pm_aw_ready),
    .din          (pm_aw_addr),
    .full         (aw_full),
    .wr_rst_busy  (aw_wbusy),
    .rd_clk       (clk_main_a0),
    .rd_en        (cl_sh_pcim_awvalid && sh_cl_pcim_awready),
    .dout         (cl_sh_pcim_awaddr),
    .empty        (aw_empty),
    .rd_rst_busy  (aw_rbusy),
    .sleep        (1'b0),
    .injectsbiterr(1'b0),
    .injectdbiterr(1'b0),
    .almost_empty (),
    .almost_full  (),
    .data_valid   (),
    .dbiterr      (),
    .overflow     (),
    .prog_empty   (),
    .prog_full    (),
    .rd_data_count(),
    .sbiterr      (),
    .underflow    (),
    .wr_ack       (),
    .wr_data_count()
  );

  xpm_fifo_async #(
    .CDC_SYNC_STAGES    (3),
    .FIFO_MEMORY_TYPE   ("distributed"),
    .FIFO_READ_LATENCY  (0),
    .FIFO_WRITE_DEPTH   (32),  // LUT RAM is 32 deep anyway; the thresholds well inside its range
    .PROG_EMPTY_THRESH  (10),
    .PROG_FULL_THRESH   (10),
    .READ_DATA_WIDTH    (9 * PcimBytes),
    .READ_MODE          ("fwft"),
    .USE_ADV_FEATURES   ("0000"),
    .WRITE_DATA_WIDTH   (9 * PcimBytes)
  ) u_pcim_w (
    .rst          (pm_rst_q),
    .wr_clk       (clk),
    .wr_en        (pm_w_valid && pm_w_ready),
    .din          ({pm_w_data, pm_w_strb}),
    .full         (w_full),
    .wr_rst_busy  (w_wbusy),
    .rd_clk       (clk_main_a0),
    .rd_en        (cl_sh_pcim_wvalid && sh_cl_pcim_wready),
    .dout         ({cl_sh_pcim_wdata, cl_sh_pcim_wstrb}),
    .empty        (w_empty),
    .rd_rst_busy  (w_rbusy),
    .sleep        (1'b0),
    .injectsbiterr(1'b0),
    .injectdbiterr(1'b0),
    .almost_empty (),
    .almost_full  (),
    .data_valid   (),
    .dbiterr      (),
    .overflow     (),
    .prog_empty   (),
    .prog_full    (),
    .rd_data_count(),
    .sbiterr      (),
    .underflow    (),
    .wr_ack       (),
    .wr_data_count()
  );

  xpm_fifo_async #(
    .CDC_SYNC_STAGES    (3),
    .FIFO_MEMORY_TYPE   ("distributed"),
    .FIFO_READ_LATENCY  (0),
    .FIFO_WRITE_DEPTH   (32),  // LUT RAM is 32 deep anyway; the thresholds well inside its range
    .PROG_EMPTY_THRESH  (10),
    .PROG_FULL_THRESH   (10),
    .READ_DATA_WIDTH    (2),
    .READ_MODE          ("fwft"),
    .USE_ADV_FEATURES   ("0000"),
    .WRITE_DATA_WIDTH   (2)
  ) u_pcim_b (
    .rst          (!cross_rst_n),
    .wr_clk       (clk_main_a0),
    .wr_en        (sh_cl_pcim_bvalid && cl_sh_pcim_bready),
    .din          (sh_cl_pcim_bresp),
    .full         (b_full),
    .wr_rst_busy  (b_wbusy),
    .rd_clk       (clk),
    .rd_en        (!b_empty),
    .dout         (pm_b_resp),
    .empty        (b_empty),
    .rd_rst_busy  (b_rbusy),
    .sleep        (1'b0),
    .injectsbiterr(1'b0),
    .injectdbiterr(1'b0),
    .almost_empty (),
    .almost_full  (),
    .data_valid   (),
    .dbiterr      (),
    .overflow     (),
    .prog_empty   (),
    .prog_full    (),
    .rd_data_count(),
    .sbiterr      (),
    .underflow    (),
    .wr_ack       (),
    .wr_data_count()
  );

  // PCIM's other write signals: one ID, single full-width INCR beats
  assign cl_sh_pcim_awid    = '0;
  assign cl_sh_pcim_awlen   = '0;
  assign cl_sh_pcim_awsize  = 3'($clog2(PcimBytes));
  assign cl_sh_pcim_awburst = 2'b01;
  assign cl_sh_pcim_awcache = '0;
  assign cl_sh_pcim_awlock  = '0;
  assign cl_sh_pcim_awprot  = '0;
  assign cl_sh_pcim_awqos   = '0;
  assign cl_sh_pcim_awuser  = '0;
  assign cl_sh_pcim_awvalid = !aw_empty;
  assign cl_sh_pcim_wid     = '0;
  assign cl_sh_pcim_wlast   = 1'b1;
  assign cl_sh_pcim_wuser   = '0;
  assign cl_sh_pcim_wvalid  = !w_empty;
  assign cl_sh_pcim_bready  = !b_full && !b_wbusy;

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
