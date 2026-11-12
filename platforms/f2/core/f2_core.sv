// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// rtl/core.sv on F2, vendor-free (platforms/f2/README.md, "The core on F2"):
// the CL top (cabosse_core.sv) adds the shell, the clock crossings and the
// HBM. Its ports are bringup_core.sv's, but for the registers' address
// (LiteW bits).
//
// HBM: engine port p reads its own channel, so its address gets the channel
// p in the bits above ChanW (the engine gives every port the same offset;
// the HBM switch decodes the channel from those bits). Ports 1 to 31 only
// read. Port 0 is engine port 0's reads and the vector unit's AXI port
// (global addresses already) merged (f2_port0.sv), shared with the host
// through bringup_host.sv, which HOST_PORT drives.
//
// AXI-Lite: one access at a time, a write's address and data in the same
// cycle (as the OCL clock converter gives them), the strobes ignored. Below
// WrapBase the core's registers, passed on; from WrapBase the wrapper's
// (README.md): its ID, BUILD, HOST_PORT, STATUS and the build's parameters.

module f2_core #(
  parameter  int unsigned Ports     = 32,
  parameter  int unsigned ChanW     = 29,     // 512 MiB channels
  parameter  int unsigned Len       = 49152,  // the buffers' elements (SmolLM2's LOGITS, M9)
  parameter  logic [31:0] Build     = 32'h0,  // the commit's first 8 hex digits
  parameter  int unsigned LiteW     = 18,
  // The engine's regions (rtl/engine.sv), one per SLR: groups of 4 ports
  // before Region1 in SLR0, before Region2 in SLR1, the rest in SLR2.
  // scripts/f2/floorplan.py's REGION1, REGION2 must match (its defaults do).
  parameter  int unsigned Region1   = Ports / 8,
  parameter  int unsigned Region2   = Ports / 4,
  localparam int unsigned AddrWidth = 34,
  localparam int unsigned DataWidth = 256,
  localparam int unsigned BeatBytes = DataWidth / 8,
  localparam int unsigned AxilW     = 17      // the core's
) (
  input  logic                                 clk_i,
  input  logic                                 rst_ni,
  input  logic                                 hbm_ready_i,  // both stacks initialized, synchronized to clk_i
  output logic                                 host_sel_o,   // HBM port 0 serves the host now
  // AXI-Lite (registers)
  input  logic                                 s_aw_valid_i,
  output logic                                 s_aw_ready_o,
  input  logic [LiteW-1:0]                     s_aw_addr_i,
  input  logic                                 s_w_valid_i,
  output logic                                 s_w_ready_o,
  input  logic [31:0]                          s_w_data_i,
  output logic                                 s_b_valid_o,
  input  logic                                 s_b_ready_i,
  output logic [1:0]                           s_b_resp_o,
  input  logic                                 s_ar_valid_i,
  output logic                                 s_ar_ready_o,
  input  logic [LiteW-1:0]                     s_ar_addr_i,
  output logic                                 s_r_valid_o,
  input  logic                                 s_r_ready_i,
  output logic [31:0]                          s_r_data_o,
  output logic [1:0]                           s_r_resp_o,
  // The host's AXI (PCIS, through the CL top), to HBM port 0
  input  logic                                 h_ar_valid_i,
  output logic                                 h_ar_ready_o,
  input  logic [AddrWidth-1:0]                 h_ar_addr_i,
  input  logic [7:0]                           h_ar_len_i,
  input  logic [1:0]                           h_ar_burst_i,
  output logic                                 h_r_valid_o,
  input  logic                                 h_r_ready_i,
  output logic [DataWidth-1:0]                 h_r_data_o,
  output logic [1:0]                           h_r_resp_o,
  output logic                                 h_r_last_o,
  input  logic                                 h_aw_valid_i,
  output logic                                 h_aw_ready_o,
  input  logic [AddrWidth-1:0]                 h_aw_addr_i,
  input  logic [7:0]                           h_aw_len_i,
  input  logic [1:0]                           h_aw_burst_i,
  input  logic                                 h_w_valid_i,
  output logic                                 h_w_ready_o,
  input  logic [DataWidth-1:0]                 h_w_data_i,
  input  logic [BeatBytes-1:0]                 h_w_strb_i,
  input  logic                                 h_w_last_i,
  output logic                                 h_b_valid_o,
  input  logic                                 h_b_ready_i,
  output logic [1:0]                           h_b_resp_o,
  // AXI, a port each (sizes: always the full beat)
  output logic [Ports-1:0]                     ar_valid_o,
  input  logic [Ports-1:0]                     ar_ready_i,
  output logic [Ports-1:0][AddrWidth-1:0]      ar_addr_o,
  output logic [Ports-1:0][7:0]                ar_len_o,
  output logic [Ports-1:0][2:0]                ar_size_o,
  output logic [Ports-1:0][1:0]                ar_burst_o,
  input  logic [Ports-1:0]                     r_valid_i,
  output logic [Ports-1:0]                     r_ready_o,
  input  logic [Ports-1:0][DataWidth-1:0]      r_data_i,
  input  logic [Ports-1:0][1:0]                r_resp_i,
  input  logic [Ports-1:0]                     r_last_i,
  output logic [Ports-1:0]                     aw_valid_o,
  // verilator lint_off UNUSEDSIGNAL
  // Writes go to port 0 only: the other ports' write answers are unread.
  input  logic [Ports-1:0]                     aw_ready_i,
  output logic [Ports-1:0][AddrWidth-1:0]      aw_addr_o,
  output logic [Ports-1:0][7:0]                aw_len_o,
  output logic [Ports-1:0][2:0]                aw_size_o,
  output logic [Ports-1:0][1:0]                aw_burst_o,
  output logic [Ports-1:0]                     w_valid_o,
  input  logic [Ports-1:0]                     w_ready_i,
  output logic [Ports-1:0][DataWidth-1:0]      w_data_o,
  output logic [Ports-1:0][BeatBytes-1:0]      w_strb_o,
  output logic [Ports-1:0]                     w_last_o,
  input  logic [Ports-1:0]                     b_valid_i,
  output logic [Ports-1:0]                     b_ready_o,
  input  logic [Ports-1:0][1:0]                b_resp_i
  // verilator lint_on UNUSEDSIGNAL
);

  localparam logic [31:0]      Id       = 32'hCAB0_C0E0;
  localparam logic [LiteW-1:0] WrapBase = LiteW'(1) << AxilW;
  // The wrapper's registers, by offset from WrapBase (README.md)
  localparam logic [7:0] RegId = 8'h00, RegBuild = 8'h04, RegHostPort = 8'h08,
                         RegStatus = 8'h0C, RegPorts = 8'h18, RegChanW = 8'h1C, RegLen = 8'h20;
  localparam logic [2:0] FullBeat = 3'($clog2(BeatBytes));
  localparam int unsigned AxW = AddrWidth + 8 + 2;
  localparam int unsigned WW  = DataWidth + BeatBytes + 1;
  localparam int unsigned RW  = DataWidth + 2 + 1;

  if (Ports < 1 || Ports > 32 || ChanW + $clog2(Ports) > AddrWidth) begin : gen_bad_ports
    $error("f2_core: Ports must be 1 to 32, their channels within the address");
  end

  // --- AXI-Lite: the core's registers or the wrapper's ---------------------------

  // The core's side
  logic             c_aw_valid, c_aw_ready, c_w_valid, c_w_ready, c_b_valid, c_b_ready;
  logic             c_ar_valid, c_ar_ready, c_r_valid, c_r_ready;
  logic [31:0]      c_r_data;
  logic [1:0]       c_b_resp, c_r_resp;
  // One write and one read at a time: what each waits for
  logic             wr_busy_q, wr_core_q, aw_out_q, w_out_q, wb_q;
  logic             rd_busy_q, rd_core_q, ar_out_q, rv_q;
  logic [AxilW-1:0] wr_addr_q, rd_addr_q;
  logic [31:0]      wr_data_q, rdata_q;
  logic             host_port_q, wrap_wr, wrap_rd;
  logic             lite_rst_nq;  // this logic's copy of the reset (M10)
  logic [7:0]       wr_reg, rd_reg;

  // The shell's AXI-Lite requests through a stage each first: they come
  // from another SLR (M10). The stages run on rst_ni, so nothing comes out
  // of them before lite_rst_nq is high.
  logic             l_aw_valid, l_aw_ready, l_w_valid, l_w_ready, l_ar_valid, l_ar_ready;
  logic [LiteW-1:0] l_aw_addr, l_ar_addr;
  logic [31:0]      l_w_data;
  // verilator lint_off PINCONNECTEMPTY
  bringup_pipe #(.Width(LiteW)) u_lite_aw (
    .clk_i, .rst_ni, .in_valid_i(s_aw_valid_i), .in_ready_o(s_aw_ready_o),
    .in_data_i(s_aw_addr_i), .busy_o(), .out_valid_o(l_aw_valid), .out_ready_i(l_aw_ready),
    .out_data_o(l_aw_addr)
  );
  bringup_pipe #(.Width(32)) u_lite_w (
    .clk_i, .rst_ni, .in_valid_i(s_w_valid_i), .in_ready_o(s_w_ready_o),
    .in_data_i(s_w_data_i), .busy_o(), .out_valid_o(l_w_valid), .out_ready_i(l_w_ready),
    .out_data_o(l_w_data)
  );
  bringup_pipe #(.Width(LiteW)) u_lite_ar (
    .clk_i, .rst_ni, .in_valid_i(s_ar_valid_i), .in_ready_o(s_ar_ready_o),
    .in_data_i(s_ar_addr_i), .busy_o(), .out_valid_o(l_ar_valid), .out_ready_i(l_ar_ready),
    .out_data_o(l_ar_addr)
  );
  // verilator lint_on PINCONNECTEMPTY

  assign l_aw_ready   = !wr_busy_q && l_aw_valid && l_w_valid;
  assign l_w_ready    = l_aw_ready;
  assign wrap_wr      = l_aw_ready && l_aw_addr >= WrapBase;
  assign wr_reg       = l_aw_addr[7:0];
  assign c_aw_valid   = aw_out_q;
  assign c_w_valid    = w_out_q;
  assign s_b_valid_o  = wr_core_q ? c_b_valid : wb_q;
  assign s_b_resp_o   = wr_core_q ? c_b_resp : 2'b00;
  assign c_b_ready    = wr_core_q && s_b_ready_i;

  assign l_ar_ready   = !rd_busy_q && l_ar_valid;
  assign wrap_rd      = l_ar_ready && l_ar_addr >= WrapBase;
  assign rd_reg       = l_ar_addr[7:0];
  assign c_ar_valid   = ar_out_q;
  assign s_r_valid_o  = rd_core_q ? c_r_valid : rv_q;
  assign s_r_data_o   = rd_core_q ? c_r_data : rdata_q;
  assign s_r_resp_o   = rd_core_q ? c_r_resp : 2'b00;
  assign c_r_ready    = rd_core_q && s_r_ready_i;

  always_ff @(posedge clk_i) lite_rst_nq <= rst_ni;
  always_ff @(posedge clk_i) begin
    if (!lite_rst_nq) begin
      {wr_busy_q, wr_core_q, aw_out_q, w_out_q, wb_q} <= '0;
      {rd_busy_q, rd_core_q, ar_out_q, rv_q}          <= '0;
      host_port_q <= 1'b1;
    end else begin
      // Writes
      if (l_aw_ready) begin
        wr_busy_q <= 1'b1;
        wr_core_q <= !wrap_wr;
        aw_out_q  <= !wrap_wr;
        w_out_q   <= !wrap_wr;
        wb_q      <= wrap_wr;
        wr_addr_q <= l_aw_addr[AxilW-1:0];
        wr_data_q <= l_w_data;
        if (wrap_wr && wr_reg == RegHostPort) host_port_q <= l_w_data[0];
      end else begin
        if (c_aw_ready) aw_out_q <= 1'b0;
        if (c_w_ready) w_out_q <= 1'b0;
        if (s_b_valid_o && s_b_ready_i) {wr_busy_q, wr_core_q, wb_q} <= '0;
      end
      // Reads
      if (l_ar_ready) begin
        rd_busy_q <= 1'b1;
        rd_core_q <= !wrap_rd;
        ar_out_q  <= !wrap_rd;
        rv_q      <= wrap_rd;
        rd_addr_q <= l_ar_addr[AxilW-1:0];
        unique case (rd_reg)
          RegId:       rdata_q <= Id;
          RegBuild:    rdata_q <= Build;
          RegHostPort: rdata_q <= {31'd0, host_port_q};
          RegStatus:   rdata_q <= {28'd0, host_sel_o, 1'b0, hbm_ready_i, 1'b0};
          RegPorts:    rdata_q <= 32'(Ports);
          RegChanW:    rdata_q <= 32'(ChanW);
          RegLen:      rdata_q <= 32'(Len);
          default:     rdata_q <= '0;
        endcase
      end else begin
        if (c_ar_ready) ar_out_q <= 1'b0;
        if (s_r_valid_o && s_r_ready_i) {rd_busy_q, rd_core_q, rv_q} <= '0;
      end
    end
  end

  // --- The HBM's side: a register slice on every channel -------------------------

  // The HBM's AXI ports sit at the die's edge, far from the core's logic
  // (core build 1: their r_valid and w_ready into the DMAs' counters, -5.1
  // ns), and the engine's ports in up to three SLRs (the floorplan,
  // small_shell_cl_pnr_user.xdc). Each port's AR and R go through 2 + its
  // SLR stages: one beside the HBM (the bottom clock-region row), one
  // beside the port's DMA, and between them one per SLR crossed, which no
  // pblock holds (M10 step 6). AR: bringup_pipe stages (all registered,
  // valid and data forward, ready back). R on ports 1 and up: plain
  // registers, no ready: the engine's DMAs take every beat (port_dma.sv's
  // credits: a burst is asked only with room for it), so r_ready stays
  // high. Port 0 (shared with the vector unit and the host) keeps a
  // bringup_pipe stage for R, and two for W (the HBM's w_ready into a
  // register beside it first).
  localparam int GroupPorts = 4;  // rtl/engine.sv's GP: its regions are of groups of 4 ports
  function automatic int unsigned port_slr(int p);  // rtl/engine.sv's region_of(p / GP)
    return p / GroupPorts < int'(Region1) ? 0 : p / GroupPorts < int'(Region2) ? 1 : 2;
  endfunction

  logic [Ports-1:0]                hb_ar_valid, hb_ar_ready, hb_r_valid, hb_r_ready, hb_r_last;
  logic [Ports-1:0][AddrWidth-1:0] hb_ar_addr;
  logic [Ports-1:0][7:0]           hb_ar_len;
  logic [Ports-1:0][1:0]           hb_ar_burst, hb_r_resp;
  logic [Ports-1:0][DataWidth-1:0] hb_r_data;
  // Port 0's writes (the vector unit's and the host's)
  logic                            hb_aw_valid, hb_aw_ready, hb_w_valid, hb_w_ready, hb_w_last;
  logic                            hb_b_valid, hb_b_ready;
  logic [AddrWidth-1:0]            hb_aw_addr;
  logic [7:0]                      hb_aw_len;
  logic [1:0]                      hb_aw_burst, hb_b_resp;
  logic [DataWidth-1:0]            hb_w_data;
  logic [BeatBytes-1:0]            hb_w_strb;

  // verilator lint_off PINCONNECTEMPTY

  for (genvar p = 0; p < Ports; p++) begin : gen_hbm
    localparam int unsigned S = 2 + port_slr(p);  // stages: gen_stage[S - 1] beside the HBM
    bringup_pipe #(.Width(AxW), .Stages(S)) u_ar (
      .clk_i,
      .rst_ni,
      .in_valid_i (hb_ar_valid[p]),
      .in_ready_o (hb_ar_ready[p]),
      .in_data_i  ({hb_ar_addr[p], hb_ar_len[p], hb_ar_burst[p]}),
      .busy_o     (),
      .out_valid_o(ar_valid_o[p]),
      .out_ready_i(ar_ready_i[p]),
      .out_data_o ({ar_addr_o[p], ar_len_o[p], ar_burst_o[p]})
    );
    if (p == 0) begin : gen_r_pipe
      bringup_pipe #(.Width(RW)) u_r (
        .clk_i,
        .rst_ni,
        .in_valid_i (r_valid_i[p]),
        .in_ready_o (r_ready_o[p]),
        .in_data_i  ({r_data_i[p], r_resp_i[p], r_last_i[p]}),
        .busy_o     (),
        .out_valid_o(hb_r_valid[p]),
        .out_ready_i(hb_r_ready[p]),
        .out_data_o ({hb_r_data[p], hb_r_resp[p], hb_r_last[p]})
      );
    end else begin : gen_r_regs
      logic [S:0]         valid;  // gen_stage[0] beside the HBM, gen_stage[S - 1] beside the DMA
      logic [S:0][RW-1:0] data;
      assign valid[0] = r_valid_i[p];
      assign data[0]  = {r_data_i[p], r_resp_i[p], r_last_i[p]};
      for (genvar s = 0; s < S; s++) begin : gen_stage
        logic          valid_q;
        logic [RW-1:0] data_q;
        always_ff @(posedge clk_i) begin
          valid_q <= rst_ni && valid[s];
          data_q  <= data[s];
        end
        assign valid[s+1] = valid_q;
        assign data[s+1]  = data_q;
      end
      assign r_ready_o[p] = 1'b1;
      assign hb_r_valid[p] = valid[S];
      assign {hb_r_data[p], hb_r_resp[p], hb_r_last[p]} = data[S];
`ifndef SYNTHESIS
      always_ff @(posedge clk_i) begin
        if (rst_ni && hb_r_valid[p] && !hb_r_ready[p]) $error("f2_core: port %0d refused a read beat", p);
      end
`endif
    end
  end

  bringup_pipe #(.Width(AxW)) u_aw (
    .clk_i,
    .rst_ni,
    .in_valid_i (hb_aw_valid),
    .in_ready_o (hb_aw_ready),
    .in_data_i  ({hb_aw_addr, hb_aw_len, hb_aw_burst}),
    .busy_o     (),
    .out_valid_o(aw_valid_o[0]),
    .out_ready_i(aw_ready_i[0]),
    .out_data_o ({aw_addr_o[0], aw_len_o[0], aw_burst_o[0]})
  );
  bringup_pipe #(.Width(WW), .Stages(2)) u_w (
    .clk_i,
    .rst_ni,
    .in_valid_i (hb_w_valid),
    .in_ready_o (hb_w_ready),
    .in_data_i  ({hb_w_data, hb_w_strb, hb_w_last}),
    .busy_o     (),
    .out_valid_o(w_valid_o[0]),
    .out_ready_i(w_ready_i[0]),
    .out_data_o ({w_data_o[0], w_strb_o[0], w_last_o[0]})
  );
  bringup_pipe #(.Width(2)) u_b (
    .clk_i,
    .rst_ni,
    .in_valid_i (b_valid_i[0]),
    .in_ready_o (b_ready_o[0]),
    .in_data_i  (b_resp_i[0]),
    .busy_o     (),
    .out_valid_o(hb_b_valid),
    .out_ready_i(hb_b_ready),
    .out_data_o (hb_b_resp)
  );
  // verilator lint_on PINCONNECTEMPTY

  // --- The core ------------------------------------------------------------------

  logic [Ports-1:0]                e_ar_valid, e_ar_ready, e_r_valid, e_r_ready;
  logic [Ports-1:0][AddrWidth-1:0] e_ar_addr;
  logic [Ports-1:0][7:0]           e_ar_len;
  logic [Ports-1:0][1:0]           e_ar_burst;
  logic                            v_ar_valid, v_ar_ready, v_r_valid, v_r_ready;
  logic                            v_aw_valid, v_aw_ready, v_w_valid, v_w_ready, v_w_last;
  logic                            v_b_valid, v_b_ready;
  logic [AddrWidth-1:0]            v_ar_addr, v_aw_addr;
  logic [7:0]                      v_ar_len, v_aw_len;
  logic [1:0]                      v_ar_burst, v_aw_burst;
  logic [DataWidth-1:0]            v_w_data;
  logic [BeatBytes-1:0]            v_w_strb;
  logic [DataWidth-1:0]            port0_r_data, v_r_data;
  logic [1:0]                      port0_r_resp, port0_b_resp, v_r_resp, v_b_resp;
  // verilator lint_off UNUSEDSIGNAL
  logic [Ports-1:0][2:0]           e_ar_size;  // always the full beat
  logic [2:0]                      v_ar_size, v_aw_size;
  logic                            irq;
  logic [15:0][31:0]               host_r_data;  // the simulation's (M7)
  logic [Ports-1:0]                r_last;       // the engine counts its beats
  // verilator lint_on UNUSEDSIGNAL

  core #(
    .Ports  (Ports),
    .ChanW  (ChanW),
    .Len    (Len),
    .Region1(Region1),
    .Region2(Region2)
  ) u_core (
    .clk_i,
    .rst_ni,
    .s_aw_valid_i  (c_aw_valid),
    .s_aw_ready_o  (c_aw_ready),
    .s_aw_addr_i   (wr_addr_q),
    .s_w_valid_i   (c_w_valid),
    .s_w_ready_o   (c_w_ready),
    .s_w_data_i    (wr_data_q),
    .s_b_valid_o   (c_b_valid),
    .s_b_ready_i   (c_b_ready),
    .s_b_resp_o    (c_b_resp),
    .s_ar_valid_i  (c_ar_valid),
    .s_ar_ready_o  (c_ar_ready),
    .s_ar_addr_i   (rd_addr_q),
    .s_r_valid_o   (c_r_valid),
    .s_r_ready_i   (c_r_ready),
    .s_r_data_o    (c_r_data),
    .s_r_resp_o    (c_r_resp),
    .irq_o         (irq),
    .e_ar_valid_o  (e_ar_valid),
    .e_ar_ready_i  (e_ar_ready),
    .e_ar_addr_o   (e_ar_addr),
    .e_ar_len_o    (e_ar_len),
    .e_ar_size_o   (e_ar_size),
    .e_ar_burst_o  (e_ar_burst),
    .e_r_valid_i   (e_r_valid),
    .e_r_ready_o   (e_r_ready),
    .e_r_data_i    ({hb_r_data[Ports-1:1], port0_r_data}),
    .e_r_resp_i    ({hb_r_resp[Ports-1:1], port0_r_resp}),
    .v_ar_valid_o  (v_ar_valid),
    .v_ar_ready_i  (v_ar_ready),
    .v_ar_addr_o   (v_ar_addr),
    .v_ar_len_o    (v_ar_len),
    .v_ar_size_o   (v_ar_size),
    .v_ar_burst_o  (v_ar_burst),
    .v_r_valid_i   (v_r_valid),
    .v_r_ready_o   (v_r_ready),
    .v_r_data_i    (v_r_data),
    .v_r_resp_i    (v_r_resp),
    .v_aw_valid_o  (v_aw_valid),
    .v_aw_ready_i  (v_aw_ready),
    .v_aw_addr_o   (v_aw_addr),
    .v_aw_len_o    (v_aw_len),
    .v_aw_size_o   (v_aw_size),
    .v_aw_burst_o  (v_aw_burst),
    .v_w_valid_o   (v_w_valid),
    .v_w_ready_i   (v_w_ready),
    .v_w_data_o    (v_w_data),
    .v_w_strb_o    (v_w_strb),
    .v_w_last_o    (v_w_last),
    .v_b_valid_i   (v_b_valid),
    .v_b_ready_o   (v_b_ready),
    .v_b_resp_i    (v_b_resp),
    .host_r_valid_i(1'b0),
    .host_r_buf_i  ('0),
    .host_r_addr_i ('0),
    .host_r_data_o (host_r_data)
  );

  // --- Ports 1 .. Ports - 1: the engine's reads, on their channels ---------------

  for (genvar p = 1; p < Ports; p++) begin : gen_port
    assign hb_ar_valid[p]  = e_ar_valid[p];
    assign e_ar_ready[p]   = hb_ar_ready[p];
    assign hb_ar_addr[p]   = AddrWidth'(p) << ChanW | AddrWidth'(e_ar_addr[p][ChanW-1:0]);
    assign hb_ar_len[p]    = e_ar_len[p];
    assign ar_size_o[p]    = FullBeat;
    assign hb_ar_burst[p]  = e_ar_burst[p];
    assign e_r_valid[p]    = hb_r_valid[p];
    assign hb_r_ready[p]   = e_r_ready[p];
    assign aw_valid_o[p]   = 1'b0;
    assign aw_addr_o[p]    = '0;
    assign aw_len_o[p]     = '0;
    assign aw_size_o[p]    = FullBeat;
    assign aw_burst_o[p]   = 2'b01;
    assign w_valid_o[p]    = 1'b0;
    assign w_data_o[p]     = '0;
    assign w_strb_o[p]     = '0;
    assign w_last_o[p]     = 1'b0;
    assign b_ready_o[p]    = 1'b1;
  end
  assign r_last = hb_r_last;

  // --- Port 0: engine port 0 and the vector unit, shared with the host ------------

  logic           p0_ar_valid, p0_ar_ready, p0_r_valid, p0_r_ready;
  logic           p0_aw_valid, p0_aw_ready, p0_w_valid, p0_w_ready, p0_b_valid, p0_b_ready;
  logic [AxW-1:0] p0_ar, p0_aw;
  logic [WW-1:0]  p0_w;
  logic [AxW-1:0] m_ar, m_aw;
  logic [WW-1:0]  m_w;
  logic [RW-1:0]  h_r;

  // The vector unit (SLR2 in the floorplan) to port 0 (SLR0): three
  // register stages each way, one per SLR (stage 0 the vector unit's). Its
  // reads and write responses as plain registers: it takes every beat
  // (port_dma.sv's credits, port_wdma.sv's b_ready).
  logic           pv_ar_valid, pv_ar_ready, pv_aw_valid, pv_aw_ready, pv_w_valid, pv_w_ready;
  logic           pv_r_valid, pv_b_valid;
  logic [AxW-1:0] pv_ar, pv_aw;
  logic [WW-1:0]  pv_w;

  // verilator lint_off PINCONNECTEMPTY
  bringup_pipe #(.Width(AxW), .Stages(3)) u_v_ar (
    .clk_i,
    .rst_ni,
    .in_valid_i (v_ar_valid),
    .in_ready_o (v_ar_ready),
    .in_data_i  ({v_ar_addr, v_ar_len, v_ar_burst}),
    .busy_o     (),
    .out_valid_o(pv_ar_valid),
    .out_ready_i(pv_ar_ready),
    .out_data_o (pv_ar)
  );
  bringup_pipe #(.Width(AxW), .Stages(3)) u_v_aw (
    .clk_i,
    .rst_ni,
    .in_valid_i (v_aw_valid),
    .in_ready_o (v_aw_ready),
    .in_data_i  ({v_aw_addr, v_aw_len, v_aw_burst}),
    .busy_o     (),
    .out_valid_o(pv_aw_valid),
    .out_ready_i(pv_aw_ready),
    .out_data_o (pv_aw)
  );
  bringup_pipe #(.Width(WW), .Stages(3)) u_v_w (
    .clk_i,
    .rst_ni,
    .in_valid_i (v_w_valid),
    .in_ready_o (v_w_ready),
    .in_data_i  ({v_w_data, v_w_strb, v_w_last}),
    .busy_o     (),
    .out_valid_o(pv_w_valid),
    .out_ready_i(pv_w_ready),
    .out_data_o (pv_w)
  );
  // verilator lint_on PINCONNECTEMPTY

  // Back: stage 0 beside port 0, stage 2 the vector unit's.
  logic [2:0]                            v_r_valid_q, v_b_valid_q;
  logic [2:0][DataWidth+1:0]             v_r_q;
  logic [2:0][1:0]                       v_b_q;
  always_ff @(posedge clk_i) begin
    v_r_valid_q <= rst_ni ? {v_r_valid_q[1:0], pv_r_valid} : '0;
    v_b_valid_q <= rst_ni ? {v_b_valid_q[1:0], pv_b_valid} : '0;
    v_r_q       <= {v_r_q[1:0], {port0_r_data, port0_r_resp}};
    v_b_q       <= {v_b_q[1:0], port0_b_resp};
  end
  assign v_r_valid              = v_r_valid_q[2];
  assign {v_r_data, v_r_resp}   = v_r_q[2];
  assign v_b_valid              = v_b_valid_q[2];
  assign v_b_resp               = v_b_q[2];
`ifndef SYNTHESIS
  always_ff @(posedge clk_i) begin
    if (rst_ni && v_r_valid && !v_r_ready) $error("f2_core: the vector unit refused a read beat");
    if (rst_ni && v_b_valid && !v_b_ready) $error("f2_core: the vector unit refused a write response");
  end
`endif

  f2_port0 #(
    .AddrWidth(AddrWidth),
    .DataWidth(DataWidth)
  ) u_port0 (
    .clk_i,
    .rst_ni,
    .e_ar_valid_i(e_ar_valid[0]),
    .e_ar_ready_o(e_ar_ready[0]),
    .e_ar_i      ({AddrWidth'(e_ar_addr[0][ChanW-1:0]), e_ar_len[0], e_ar_burst[0]}),
    .e_r_valid_o (e_r_valid[0]),
    .e_r_ready_i (e_r_ready[0]),
    .v_ar_valid_i(pv_ar_valid),
    .v_ar_ready_o(pv_ar_ready),
    .v_ar_i      (pv_ar),
    .v_r_valid_o (pv_r_valid),
    .v_r_ready_i (1'b1),
    .v_aw_valid_i(pv_aw_valid),
    .v_aw_ready_o(pv_aw_ready),
    .v_aw_i      (pv_aw),
    .v_w_valid_i (pv_w_valid),
    .v_w_ready_o (pv_w_ready),
    .v_w_i       (pv_w),
    .v_b_valid_o (pv_b_valid),
    .v_b_ready_i (1'b1),
    .m_ar_valid_o(p0_ar_valid),
    .m_ar_ready_i(p0_ar_ready),
    .m_ar_o      (p0_ar),
    .m_r_valid_i (p0_r_valid),
    .m_r_ready_o (p0_r_ready),
    .m_r_last_i  (hb_r_last[0]),
    .m_aw_valid_o(p0_aw_valid),
    .m_aw_ready_i(p0_aw_ready),
    .m_aw_o      (p0_aw),
    .m_w_valid_o (p0_w_valid),
    .m_w_ready_i (p0_w_ready),
    .m_w_o       (p0_w),
    .m_b_valid_i (p0_b_valid),
    .m_b_ready_o (p0_b_ready)
  );
  assign port0_r_data = hb_r_data[0];
  assign port0_r_resp = hb_r_resp[0];
  assign port0_b_resp = hb_b_resp;

  bringup_host #(
    .AddrWidth(AddrWidth),
    .DataWidth(DataWidth)
  ) u_host (
    .clk_i,
    .rst_ni,
    .sel_i       (host_port_q),
    .sel_o       (host_sel_o),
    .h_ar_valid_i(h_ar_valid_i),
    .h_ar_ready_o(h_ar_ready_o),
    .h_ar_i      ({h_ar_addr_i, h_ar_len_i, h_ar_burst_i}),
    .h_r_valid_o (h_r_valid_o),
    .h_r_ready_i (h_r_ready_i),
    .h_r_o       (h_r),
    .h_aw_valid_i(h_aw_valid_i),
    .h_aw_ready_o(h_aw_ready_o),
    .h_aw_i      ({h_aw_addr_i, h_aw_len_i, h_aw_burst_i}),
    .h_w_valid_i (h_w_valid_i),
    .h_w_ready_o (h_w_ready_o),
    .h_w_i       ({h_w_data_i, h_w_strb_i, h_w_last_i}),
    .h_b_valid_o (h_b_valid_o),
    .h_b_ready_i (h_b_ready_i),
    .h_b_resp_o  (h_b_resp_o),
    .e_ar_valid_i(p0_ar_valid),
    .e_ar_ready_o(p0_ar_ready),
    .e_ar_i      (p0_ar),
    .e_r_valid_o (p0_r_valid),
    .e_r_ready_i (p0_r_ready),
    .e_aw_valid_i(p0_aw_valid),
    .e_aw_ready_o(p0_aw_ready),
    .e_aw_i      (p0_aw),
    .e_w_valid_i (p0_w_valid),
    .e_w_ready_o (p0_w_ready),
    .e_w_i       (p0_w),
    .e_b_valid_o (p0_b_valid),
    .e_b_ready_i (p0_b_ready),
    .m_ar_valid_o(hb_ar_valid[0]),
    .m_ar_ready_i(hb_ar_ready[0]),
    .m_ar_o      (m_ar),
    .m_r_valid_i (hb_r_valid[0]),
    .m_r_ready_o (hb_r_ready[0]),
    .m_r_i       ({hb_r_data[0], hb_r_resp[0], hb_r_last[0]}),
    .m_aw_valid_o(hb_aw_valid),
    .m_aw_ready_i(hb_aw_ready),
    .m_aw_o      (m_aw),
    .m_w_valid_o (hb_w_valid),
    .m_w_ready_i (hb_w_ready),
    .m_w_o       (m_w),
    .m_b_valid_i (hb_b_valid),
    .m_b_ready_o (hb_b_ready),
    .m_b_resp_i  (hb_b_resp)
  );
  assign {hb_ar_addr[0], hb_ar_len[0], hb_ar_burst[0]} = m_ar;
  assign {hb_aw_addr, hb_aw_len, hb_aw_burst} = m_aw;
  assign {hb_w_data, hb_w_strb, hb_w_last}    = m_w;
  assign ar_size_o[0] = FullBeat;
  assign aw_size_o[0] = FullBeat;
  assign {h_r_data_o, h_r_resp_o, h_r_last_o} = h_r;

endmodule
