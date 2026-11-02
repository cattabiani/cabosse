// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The bring-up design's CL top for the F2 Small Shell (platforms/f2/README.md):
// bringup_core.sv on clk_hbm_axi (D-065), the shell's OCL and PCIS crossed
// into it, the HBM IP's 32 ports, the clock generator on SDA, an ILA on HBM
// port 0. PCIM, DDR and interrupts are tied off. Built only by Vivado, in
// the shell flow: it uses the shell's ports (cl_ports.vh), AWS's clock
// generator and AMD's IP, which the build takes from the aws-fpga checkout
// (cl_hbm, cl_axi_clock_converter_light, cl_axi_sc_1x1, cl_debug_bridge) or
// generates (cabosse_ila), none of them in this repository.
//
// The OCL port decodes the low 12 bits of its address; the rest of BAR0
// repeats the registers. PCIS (BAR4) reaches HBM through port 0 only while
// HOST_PORT is set, and only once STATUS.hbm_ready: the host keeps to both
// (with HOST_PORT clear its accesses wait, unanswered). The port switches
// once its host side is idle.

`ifndef CABOSSE_BUILD
`define CABOSSE_BUILD 32'h0
`endif

module cabosse_bringup #(
  parameter EN_DDR = 0,  // the shell flow's parameters (unused here)
  parameter EN_HBM = 1
) (
  `include "cl_ports.vh"
);

  localparam int unsigned Ports     = 32;
  localparam int unsigned AddrWidth = 34;
  localparam int unsigned DataWidth = 256;
  localparam int unsigned BeatBytes = DataWidth / 8;

  // --- Ties ---------------------------------------------------------------------

  // The PCI IDs of AWS's template CL.
  localparam logic [31:0] ClId0 = 32'hF010_1D0F;
  localparam logic [31:0] ClId1 = 32'h1D51_FEDC;

  always_comb begin
    cl_sh_status0       = '0;
    cl_sh_status1       = '0;
    cl_sh_status2       = '0;
    cl_sh_id0           = ClId0;
    cl_sh_id1           = ClId1;
    cl_sh_status_vled   = '0;
    cl_sh_dma_wr_full   = '0;
    cl_sh_dma_rd_full   = '0;
    cl_sh_apppf_irq_req = '0;
    cl_sh_ddr_stat_ack  = 1'b1;
    cl_sh_ddr_stat_rdata = '0;
    cl_sh_ddr_stat_int  = '0;
    PCIE_EP_TXP         = '0;
    PCIE_EP_TXN         = '0;
    PCIE_RP_PERSTN      = '0;
    PCIE_RP_TXP         = '0;
    PCIE_RP_TXN         = '0;
    // PCIM: unused (PLAN.md, M8; Q-33)
    cl_sh_pcim_awid     = '0;
    cl_sh_pcim_awaddr   = '0;
    cl_sh_pcim_awlen    = '0;
    cl_sh_pcim_awsize   = '0;
    cl_sh_pcim_awburst  = '0;
    cl_sh_pcim_awcache  = '0;
    cl_sh_pcim_awlock   = '0;
    cl_sh_pcim_awprot   = '0;
    cl_sh_pcim_awqos    = '0;
    cl_sh_pcim_awuser   = '0;
    cl_sh_pcim_awvalid  = '0;
    cl_sh_pcim_wid      = '0;
    cl_sh_pcim_wdata    = '0;
    cl_sh_pcim_wstrb    = '0;
    cl_sh_pcim_wlast    = '0;
    cl_sh_pcim_wuser    = '0;
    cl_sh_pcim_wvalid   = '0;
    cl_sh_pcim_bready   = '0;
    cl_sh_pcim_arid     = '0;
    cl_sh_pcim_araddr   = '0;
    cl_sh_pcim_arlen    = '0;
    cl_sh_pcim_arsize   = '0;
    cl_sh_pcim_arburst  = '0;
    cl_sh_pcim_arcache  = '0;
    cl_sh_pcim_arlock   = '0;
    cl_sh_pcim_arprot   = '0;
    cl_sh_pcim_arqos    = '0;
    cl_sh_pcim_aruser   = '0;
    cl_sh_pcim_arvalid  = '0;
    cl_sh_pcim_rready   = '0;
    cl_sh_dma_pcis_ruser = '0;
  end

  // --- Resets and clocks --------------------------------------------------------

  logic rst_main_sync_n;  // rst_main_n on clk_main_a0
  xpm_cdc_async_rst #(.DEST_SYNC_FF(4), .RST_ACTIVE_HIGH(0)) u_rst_main (
    .src_arst (rst_main_n),
    .dest_clk (clk_main_a0),
    .dest_arst(rst_main_sync_n)
  );

  // Function-level reset: done a cycle after it is asked for.
  logic flr_q;
  always_ff @(posedge clk_main_a0) begin
    if (!rst_main_sync_n) begin
      flr_q          <= 1'b0;
      cl_sh_flr_done <= 1'b0;
    end else begin
      flr_q          <= sh_cl_flr_assert;
      cl_sh_flr_done <= flr_q && !cl_sh_flr_done;
    end
  end

  // The clock generator, on SDA (PF1 BAR4; the SDK's fpga_clkgen). Its
  // instance name is AWS's rule (hdk/docs/AWS_CLK_GEN_spec.md).
  logic clk, clk_rst_n;  // clk_hbm_axi: the core's clock (D-065), its reset from the generator
  logic clk_ref, clk_ref_rst_n;  // clk_hbm_ref, 100 MHz: the HBM's reference and APB clock
  logic gen_rst_main_n;  // the generator's reset on clk_main_a0, with clk's: the crossings' sides together
  aws_clk_gen #(
    .CLK_GRP_A_EN(1),
    .CLK_GRP_B_EN(1),
    .CLK_GRP_C_EN(1),
    .CLK_HBM_EN  (1)
  ) AWS_CLK_GEN (
    .i_clk_main_a0      (clk_main_a0),
    .i_rst_main_n       (rst_main_sync_n),
    .i_clk_hbm_ref      (clk_hbm_ref),
    .s_axil_ctrl_awaddr (sda_cl_awaddr),
    .s_axil_ctrl_awvalid(sda_cl_awvalid),
    .s_axil_ctrl_awready(cl_sda_awready),
    .s_axil_ctrl_wdata  (sda_cl_wdata),
    .s_axil_ctrl_wstrb  (sda_cl_wstrb),
    .s_axil_ctrl_wvalid (sda_cl_wvalid),
    .s_axil_ctrl_wready (cl_sda_wready),
    .s_axil_ctrl_bresp  (cl_sda_bresp),
    .s_axil_ctrl_bvalid (cl_sda_bvalid),
    .s_axil_ctrl_bready (sda_cl_bready),
    .s_axil_ctrl_araddr (sda_cl_araddr),
    .s_axil_ctrl_arvalid(sda_cl_arvalid),
    .s_axil_ctrl_arready(cl_sda_arready),
    .s_axil_ctrl_rdata  (cl_sda_rdata),
    .s_axil_ctrl_rresp  (cl_sda_rresp),
    .s_axil_ctrl_rvalid (cl_sda_rvalid),
    .s_axil_ctrl_rready (sda_cl_rready),
    .o_clk_hbm_ref      (clk_ref),
    .o_clk_main_a0      (),
    .o_clk_extra_a1     (),
    .o_clk_extra_a2     (),
    .o_clk_extra_a3     (),
    .o_clk_extra_b0     (),
    .o_clk_extra_b1     (),
    .o_clk_extra_c0     (),
    .o_clk_extra_c1     (),
    .o_clk_hbm_axi      (clk),
    .o_cl_rst_hbm_axi_n (clk_rst_n),
    .o_cl_rst_hbm_ref_n (clk_ref_rst_n),
    .o_cl_rst_c1_n      (),
    .o_cl_rst_c0_n      (),
    .o_cl_rst_b1_n      (),
    .o_cl_rst_b0_n      (),
    .o_cl_rst_a3_n      (),
    .o_cl_rst_a2_n      (),
    .o_cl_rst_a1_n      (),
    .o_cl_rst_main_n    (gen_rst_main_n)
  );

  // The generator's resets come synchronized to their clocks.
  logic rst_n, ref_rst_n, cross_rst_n;  // on clk, clk_ref, clk_main_a0
  assign rst_n       = clk_rst_n;
  assign ref_rst_n   = clk_ref_rst_n;
  assign cross_rst_n = gen_rst_main_n;

  // --- OCL: the registers, crossed into clk ---------------------------------------

  logic        lite_aw_valid, lite_aw_ready, lite_w_valid, lite_w_ready, lite_b_valid, lite_b_ready;
  logic        lite_ar_valid, lite_ar_ready, lite_r_valid, lite_r_ready;
  logic [31:0] lite_aw_addr, lite_w_data, lite_ar_addr, lite_r_data;
  logic [1:0]  lite_b_resp, lite_r_resp;

  cl_axi_clock_converter_light u_ocl_cdc (
    .s_axi_aclk   (clk_main_a0),
    .s_axi_aresetn(cross_rst_n),
    .s_axi_awaddr (ocl_cl_awaddr),
    .s_axi_awprot (3'b000),
    .s_axi_awvalid(ocl_cl_awvalid),
    .s_axi_awready(cl_ocl_awready),
    .s_axi_wdata  (ocl_cl_wdata),
    .s_axi_wstrb  (ocl_cl_wstrb),
    .s_axi_wvalid (ocl_cl_wvalid),
    .s_axi_wready (cl_ocl_wready),
    .s_axi_bresp  (cl_ocl_bresp),
    .s_axi_bvalid (cl_ocl_bvalid),
    .s_axi_bready (ocl_cl_bready),
    .s_axi_araddr (ocl_cl_araddr),
    .s_axi_arprot (3'b000),
    .s_axi_arvalid(ocl_cl_arvalid),
    .s_axi_arready(cl_ocl_arready),
    .s_axi_rdata  (cl_ocl_rdata),
    .s_axi_rresp  (cl_ocl_rresp),
    .s_axi_rvalid (cl_ocl_rvalid),
    .s_axi_rready (ocl_cl_rready),
    .m_axi_aclk   (clk),
    .m_axi_aresetn(rst_n),
    .m_axi_awaddr (lite_aw_addr),
    .m_axi_awprot (),
    .m_axi_awvalid(lite_aw_valid),
    .m_axi_awready(lite_aw_ready),
    .m_axi_wdata  (lite_w_data),
    .m_axi_wstrb  (),
    .m_axi_wvalid (lite_w_valid),
    .m_axi_wready (lite_w_ready),
    .m_axi_bresp  (lite_b_resp),
    .m_axi_bvalid (lite_b_valid),
    .m_axi_bready (lite_b_ready),
    .m_axi_araddr (lite_ar_addr),
    .m_axi_arprot (),
    .m_axi_arvalid(lite_ar_valid),
    .m_axi_arready(lite_ar_ready),
    .m_axi_rdata  (lite_r_data),
    .m_axi_rresp  (lite_r_resp),
    .m_axi_rvalid (lite_r_valid),
    .m_axi_rready (lite_r_ready)
  );

  // --- The core -------------------------------------------------------------------

  logic [Ports-1:0]                ar_valid, ar_ready, r_valid, r_ready, r_last;
  logic [Ports-1:0]                aw_valid, aw_ready, w_valid, w_ready, w_last, b_valid, b_ready;
  logic [Ports-1:0][AddrWidth-1:0] ar_addr, aw_addr;
  logic [Ports-1:0][7:0]           ar_len, aw_len;
  logic [Ports-1:0][2:0]           ar_size, aw_size;
  logic [Ports-1:0][1:0]           ar_burst, aw_burst, r_resp, b_resp;
  logic [Ports-1:0][DataWidth-1:0] r_data, w_data;
  logic [Ports-1:0][BeatBytes-1:0] w_strb;
  logic                            host_sel, hbm_ready;
  // The host's side of the SmartConnect (PCIS, below)
  logic                 h_ar_valid, h_ar_ready, h_r_valid, h_r_ready, h_r_last;
  logic                 h_aw_valid, h_aw_ready, h_w_valid, h_w_ready, h_w_last, h_b_valid, h_b_ready;
  logic [63:0]          h_ar_addr, h_aw_addr;
  logic [3:0]           h_ar_len, h_aw_len;
  logic [2:0]           h_ar_size, h_aw_size;
  logic [1:0]           h_ar_burst, h_aw_burst, h_r_resp, h_b_resp;
  logic [DataWidth-1:0] h_r_data, h_w_data;
  logic [BeatBytes-1:0] h_w_strb;
  logic [1:0]                      apb_complete, apb_complete_sync;

  xpm_cdc_array_single #(.DEST_SYNC_FF(4), .SRC_INPUT_REG(0), .WIDTH(2)) u_hbm_ready (
    .src_clk (clk_ref),
    .src_in  (apb_complete),
    .dest_clk(clk),
    .dest_out(apb_complete_sync)
  );
  assign hbm_ready = &apb_complete_sync;

  bringup_core #(.Ports(Ports), .Build(`CABOSSE_BUILD)) u_core (
    .clk_i       (clk),
    .rst_ni      (rst_n),
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
    .s_aw_addr_i (lite_aw_addr[11:0]),
    .s_w_valid_i (lite_w_valid),
    .s_w_ready_o (lite_w_ready),
    .s_w_data_i  (lite_w_data),
    .s_b_valid_o (lite_b_valid),
    .s_b_ready_i (lite_b_ready),
    .s_b_resp_o  (lite_b_resp),
    .s_ar_valid_i(lite_ar_valid),
    .s_ar_ready_o(lite_ar_ready),
    .s_ar_addr_i (lite_ar_addr[11:0]),
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

  // --- PCIS: the host's access to HBM, through port 0 ----------------------------

  // SmartConnect: clk_main_a0 to clk, 512 to 256 bits, AXI4 to AXI3 (bursts
  // of at most 16). The 64-bit address keeps HBM's 34 bits.

  cl_axi_sc_1x1_wrapper u_pcis (
    .aclk_250    (clk_main_a0),
    .aresetn_250 (cross_rst_n),
    .aclk_450    (clk),  // the BD's name for the HBM side's clock
    .AXI4_araddr (sh_cl_dma_pcis_araddr),
    .AXI4_arburst(sh_cl_dma_pcis_arburst),
    .AXI4_arcache(4'b0011),
    .AXI4_arid   (sh_cl_dma_pcis_arid),
    .AXI4_arlen  (sh_cl_dma_pcis_arlen),
    .AXI4_arlock (1'b0),
    .AXI4_arprot (3'b000),
    .AXI4_arqos  (4'b0000),
    .AXI4_arready(cl_sh_dma_pcis_arready),
    .AXI4_arsize (sh_cl_dma_pcis_arsize),
    .AXI4_arvalid(sh_cl_dma_pcis_arvalid),
    .AXI4_awaddr (sh_cl_dma_pcis_awaddr),
    .AXI4_awburst(sh_cl_dma_pcis_awburst),
    .AXI4_awcache(4'b0011),
    .AXI4_awid   (sh_cl_dma_pcis_awid),
    .AXI4_awlen  (sh_cl_dma_pcis_awlen),
    .AXI4_awlock (1'b0),
    .AXI4_awprot (3'b000),
    .AXI4_awqos  (4'b0000),
    .AXI4_awready(cl_sh_dma_pcis_awready),
    .AXI4_awsize (sh_cl_dma_pcis_awsize),
    .AXI4_awvalid(sh_cl_dma_pcis_awvalid),
    .AXI4_bid    (cl_sh_dma_pcis_bid),
    .AXI4_bready (sh_cl_dma_pcis_bready),
    .AXI4_bresp  (cl_sh_dma_pcis_bresp),
    .AXI4_bvalid (cl_sh_dma_pcis_bvalid),
    .AXI4_rdata  (cl_sh_dma_pcis_rdata),
    .AXI4_rid    (cl_sh_dma_pcis_rid),
    .AXI4_rlast  (cl_sh_dma_pcis_rlast),
    .AXI4_rready (sh_cl_dma_pcis_rready),
    .AXI4_rresp  (cl_sh_dma_pcis_rresp),
    .AXI4_rvalid (cl_sh_dma_pcis_rvalid),
    .AXI4_wdata  (sh_cl_dma_pcis_wdata),
    .AXI4_wlast  (sh_cl_dma_pcis_wlast),
    .AXI4_wready (cl_sh_dma_pcis_wready),
    .AXI4_wstrb  (sh_cl_dma_pcis_wstrb),
    .AXI4_wvalid (sh_cl_dma_pcis_wvalid),
    .AXI3_araddr (h_ar_addr),
    .AXI3_arburst(h_ar_burst),
    .AXI3_arcache(),
    .AXI3_arlen  (h_ar_len),
    .AXI3_arlock (),
    .AXI3_arprot (),
    .AXI3_arqos  (),
    .AXI3_arready(h_ar_ready),
    .AXI3_arsize (h_ar_size),
    .AXI3_arvalid(h_ar_valid),
    .AXI3_awaddr (h_aw_addr),
    .AXI3_awburst(h_aw_burst),
    .AXI3_awcache(),
    .AXI3_awlen  (h_aw_len),
    .AXI3_awlock (),
    .AXI3_awprot (),
    .AXI3_awqos  (),
    .AXI3_awready(h_aw_ready),
    .AXI3_awsize (h_aw_size),
    .AXI3_awvalid(h_aw_valid),
    .AXI3_bready (h_b_ready),
    .AXI3_bresp  (h_b_resp),
    .AXI3_bvalid (h_b_valid),
    .AXI3_rdata  (h_r_data),
    .AXI3_rlast  (h_r_last),
    .AXI3_rready (h_r_ready),
    .AXI3_rresp  (h_r_resp),
    .AXI3_rvalid (h_r_valid),
    .AXI3_wdata  (h_w_data),
    .AXI3_wlast  (h_w_last),
    .AXI3_wready (h_w_ready),
    .AXI3_wstrb  (h_w_strb),
    .AXI3_wvalid (h_w_valid)
  );

  // --- HBM ------------------------------------------------------------------------

  // HBM sees full beats only, as in AWS's example (ar_size, aw_size and the
  // host's unread; its narrow accesses keep their strobes), and 34-bit
  // addresses (the host's upper bits unread).
  localparam logic [2:0] FullBeat = 3'($clog2(BeatBytes));

  // One HBM port's pins, P its two-digit name and I its index.
`define CABOSSE_HBM_PORT(P, I) \
    .AXI_``P``_ACLK        (clk), \
    .AXI_``P``_ARESET_N    (rst_n), \
    .AXI_``P``_ARADDR      (ar_addr[I]), \
    .AXI_``P``_ARBURST     (ar_burst[I]), \
    .AXI_``P``_ARID        (6'd0), \
    .AXI_``P``_ARLEN       (ar_len[I][3:0]), \
    .AXI_``P``_ARSIZE      (FullBeat), \
    .AXI_``P``_ARVALID     (ar_valid[I]), \
    .AXI_``P``_ARREADY     (ar_ready[I]), \
    .AXI_``P``_RDATA       (r_data[I]), \
    .AXI_``P``_RDATA_PARITY(), \
    .AXI_``P``_RID         (), \
    .AXI_``P``_RLAST       (r_last[I]), \
    .AXI_``P``_RRESP       (r_resp[I]), \
    .AXI_``P``_RVALID      (r_valid[I]), \
    .AXI_``P``_RREADY      (r_ready[I]), \
    .AXI_``P``_AWADDR      (aw_addr[I]), \
    .AXI_``P``_AWBURST     (aw_burst[I]), \
    .AXI_``P``_AWID        (6'd0), \
    .AXI_``P``_AWLEN       (aw_len[I][3:0]), \
    .AXI_``P``_AWSIZE      (FullBeat), \
    .AXI_``P``_AWVALID     (aw_valid[I]), \
    .AXI_``P``_AWREADY     (aw_ready[I]), \
    .AXI_``P``_WDATA       (w_data[I]), \
    .AXI_``P``_WDATA_PARITY(32'd0), \
    .AXI_``P``_WLAST       (w_last[I]), \
    .AXI_``P``_WSTRB       (w_strb[I]), \
    .AXI_``P``_WVALID      (w_valid[I]), \
    .AXI_``P``_WREADY      (w_ready[I]), \
    .AXI_``P``_BID         (), \
    .AXI_``P``_BRESP       (b_resp[I]), \
    .AXI_``P``_BVALID      (b_valid[I]), \
    .AXI_``P``_BREADY      (b_ready[I]),

  // The APB monitors go to the shell (AWS refuses an image without them);
  // AWS's own APB ports stay idle.
  cl_hbm u_hbm (
    .HBM_REF_CLK_0           (clk_ref),
    .HBM_REF_CLK_1           (clk_ref),
    .APB_0_PCLK              (clk_ref),
    .APB_0_PRESET_N          (ref_rst_n),
    .APB_0_PADDR             (22'd0),
    .APB_0_PENABLE           (1'b0),
    .APB_0_PSEL              (1'b0),
    .APB_0_PWDATA            (32'd0),
    .APB_0_PWRITE            (1'b0),
    .APB_0_PRDATA            (),
    .APB_0_PREADY            (),
    .APB_0_PSLVERR           (),
    .APB_1_PCLK              (clk_ref),
    .APB_1_PRESET_N          (ref_rst_n),
    .APB_1_PADDR             (22'd0),
    .APB_1_PENABLE           (1'b0),
    .APB_1_PSEL              (1'b0),
    .APB_1_PWDATA            (32'd0),
    .APB_1_PWRITE            (1'b0),
    .APB_1_PRDATA            (),
    .APB_1_PREADY            (),
    .APB_1_PSLVERR           (),
    .MON_APB_0_PRESET_N      (hbm_apb_preset_n_0),
    .APB_0_PWDATA_MON        (hbm_apb_pwdata_0),
    .APB_0_PADDR_MON         (hbm_apb_paddr_0),
    .APB_0_PENABLE_MON       (hbm_apb_penable_0),
    .APB_0_PSEL_MON          (hbm_apb_psel_0),
    .APB_0_PWRITE_MON        (hbm_apb_pwrite_0),
    .APB_0_PRDATA_MON        (hbm_apb_prdata_0),
    .APB_0_PREADY_MON        (hbm_apb_pready_0),
    .APB_0_PSLVERR_MON       (hbm_apb_pslverr_0),
    .MON_APB_1_PRESET_N      (hbm_apb_preset_n_1),
    .APB_1_PWDATA_MON        (hbm_apb_pwdata_1),
    .APB_1_PADDR_MON         (hbm_apb_paddr_1),
    .APB_1_PENABLE_MON       (hbm_apb_penable_1),
    .APB_1_PSEL_MON          (hbm_apb_psel_1),
    .APB_1_PWRITE_MON        (hbm_apb_pwrite_1),
    .APB_1_PRDATA_MON        (hbm_apb_prdata_1),
    .APB_1_PREADY_MON        (hbm_apb_pready_1),
    .APB_1_PSLVERR_MON       (hbm_apb_pslverr_1),
    `CABOSSE_HBM_PORT(00, 0)
    `CABOSSE_HBM_PORT(01, 1)
    `CABOSSE_HBM_PORT(02, 2)
    `CABOSSE_HBM_PORT(03, 3)
    `CABOSSE_HBM_PORT(04, 4)
    `CABOSSE_HBM_PORT(05, 5)
    `CABOSSE_HBM_PORT(06, 6)
    `CABOSSE_HBM_PORT(07, 7)
    `CABOSSE_HBM_PORT(08, 8)
    `CABOSSE_HBM_PORT(09, 9)
    `CABOSSE_HBM_PORT(10, 10)
    `CABOSSE_HBM_PORT(11, 11)
    `CABOSSE_HBM_PORT(12, 12)
    `CABOSSE_HBM_PORT(13, 13)
    `CABOSSE_HBM_PORT(14, 14)
    `CABOSSE_HBM_PORT(15, 15)
    `CABOSSE_HBM_PORT(16, 16)
    `CABOSSE_HBM_PORT(17, 17)
    `CABOSSE_HBM_PORT(18, 18)
    `CABOSSE_HBM_PORT(19, 19)
    `CABOSSE_HBM_PORT(20, 20)
    `CABOSSE_HBM_PORT(21, 21)
    `CABOSSE_HBM_PORT(22, 22)
    `CABOSSE_HBM_PORT(23, 23)
    `CABOSSE_HBM_PORT(24, 24)
    `CABOSSE_HBM_PORT(25, 25)
    `CABOSSE_HBM_PORT(26, 26)
    `CABOSSE_HBM_PORT(27, 27)
    `CABOSSE_HBM_PORT(28, 28)
    `CABOSSE_HBM_PORT(29, 29)
    `CABOSSE_HBM_PORT(30, 30)
    `CABOSSE_HBM_PORT(31, 31)
    .apb_complete_0          (apb_complete[0]),
    .apb_complete_1          (apb_complete[1]),
    .DRAM_0_STAT_CATTRIP     (),
    .DRAM_0_STAT_TEMP        (),
    .DRAM_1_STAT_CATTRIP     (),
    .DRAM_1_STAT_TEMP        ()
  );
`undef CABOSSE_HBM_PORT

  // The monitors' protection and strobes: no pins on the HBM IP for them.
  assign hbm_apb_pprot_0 = '0;
  assign hbm_apb_pstrb_0 = '0;
  assign hbm_apb_pprot_1 = '0;
  assign hbm_apb_pstrb_1 = '0;

  // --- DDR: none (the shell's stub drives its pins) ---------------------------------

  sh_ddr #(.DDR_PRESENT(0)) SH_DDR (
    .clk                  (clk_main_a0),
    .rst_n                (rst_main_sync_n),
    .stat_clk             (clk_main_a0),
    .stat_rst_n           (rst_main_sync_n),
    .CLK_DIMM_DP          (CLK_DIMM_DP),
    .CLK_DIMM_DN          (CLK_DIMM_DN),
    .M_ACT_N              (M_ACT_N),
    .M_MA                 (M_MA),
    .M_BA                 (M_BA),
    .M_BG                 (M_BG),
    .M_CKE                (M_CKE),
    .M_ODT                (M_ODT),
    .M_CS_N               (M_CS_N),
    .M_CLK_DN             (M_CLK_DN),
    .M_CLK_DP             (M_CLK_DP),
    .M_PAR                (M_PAR),
    .M_DQ                 (M_DQ),
    .M_ECC                (M_ECC),
    .M_DQS_DP             (M_DQS_DP),
    .M_DQS_DN             (M_DQS_DN),
    .cl_RST_DIMM_N        (RST_DIMM_N),
    .cl_sh_ddr_axi_awid   ('0),
    .cl_sh_ddr_axi_awaddr ('0),
    .cl_sh_ddr_axi_awlen  ('0),
    .cl_sh_ddr_axi_awsize (3'd6),
    .cl_sh_ddr_axi_awvalid(1'b0),
    .cl_sh_ddr_axi_awburst(2'b01),
    .cl_sh_ddr_axi_awuser (1'b0),
    .cl_sh_ddr_axi_awready(),
    .cl_sh_ddr_axi_wdata  ('0),
    .cl_sh_ddr_axi_wstrb  ('0),
    .cl_sh_ddr_axi_wlast  (1'b0),
    .cl_sh_ddr_axi_wvalid (1'b0),
    .cl_sh_ddr_axi_wready (),
    .cl_sh_ddr_axi_bid    (),
    .cl_sh_ddr_axi_bresp  (),
    .cl_sh_ddr_axi_bvalid (),
    .cl_sh_ddr_axi_bready (1'b0),
    .cl_sh_ddr_axi_arid   ('0),
    .cl_sh_ddr_axi_araddr ('0),
    .cl_sh_ddr_axi_arlen  ('0),
    .cl_sh_ddr_axi_arsize (3'd6),
    .cl_sh_ddr_axi_arvalid(1'b0),
    .cl_sh_ddr_axi_arburst(2'b01),
    .cl_sh_ddr_axi_aruser (1'b0),
    .cl_sh_ddr_axi_arready(),
    .cl_sh_ddr_axi_rid    (),
    .cl_sh_ddr_axi_rdata  (),
    .cl_sh_ddr_axi_rresp  (),
    .cl_sh_ddr_axi_rlast  (),
    .cl_sh_ddr_axi_rvalid (),
    .cl_sh_ddr_axi_rready (1'b0),
    .sh_ddr_stat_bus_addr (8'd0),
    .sh_ddr_stat_bus_wdata(32'd0),
    .sh_ddr_stat_bus_wr   (1'b0),
    .sh_ddr_stat_bus_rd   (1'b0),
    .sh_ddr_stat_bus_ack  (),
    .sh_ddr_stat_bus_rdata(),
    .ddr_sh_stat_int      (),
    .sh_cl_ddr_is_ready   ()
  );

  // --- Debug: an ILA on HBM port 0, through the shell's virtual JTAG -------------

  cl_debug_bridge CL_DEBUG_BRIDGE (
    .clk               (clk_main_a0),
    .S_BSCAN_drck      (drck),
    .S_BSCAN_shift     (shift),
    .S_BSCAN_tdi       (tdi),
    .S_BSCAN_update    (update),
    .S_BSCAN_sel       (sel),
    .S_BSCAN_tdo       (tdo),
    .S_BSCAN_tms       (tms),
    .S_BSCAN_tck       (tck),
    .S_BSCAN_runtest   (runtest),
    .S_BSCAN_reset     (reset),
    .S_BSCAN_capture   (capture),
    .S_BSCAN_bscanid_en(bscanid_en)
  );

  // probe0: the handshakes, probe1/2: the read and write addresses
  // (synth_cabosse_bringup.tcl generates the ILA with these widths).
  cabosse_ila u_ila (
    .clk   (clk),
    .probe0({host_sel, hbm_ready, ar_valid[0], ar_ready[0], r_valid[0], r_ready[0],
             r_last[0], aw_valid[0], aw_ready[0], w_valid[0], w_ready[0], w_last[0],
             b_valid[0], b_ready[0], r_resp[0][1], b_resp[0][1]}),
    .probe1(ar_addr[0]),
    .probe2(aw_addr[0])
  );

endmodule
