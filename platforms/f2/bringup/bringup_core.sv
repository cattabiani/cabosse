// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The bring-up design's core (platforms/f2/README.md): the registers on an
// AXI-Lite port and Ports test engines (bringup_port.sv), one per HBM port,
// all on one clock. Vendor-free; the CL top (cabosse_bringup.sv) adds the
// shell, the clock crossings and the HBM.
//
// AXI-Lite: one access at a time; a write takes its address and data in the
// same cycle and ignores the strobes (whole words). A read answers 3 cycles
// after its address: the ports' counters are a register stage in each port
// and one here.
//
// A start is refused (STATUS.refused) while busy, and when the settings
// break README.md's rules, which would hang a port or end a run with bursts
// in flight: a burst not 1, 2, 4, 8 or 16 beats, a limit of 0, BEATS not a
// multiple of both bursts or past CountWidth, BASE or STRIDE not 4 KB-aligned.
// Port 0 does not start while it serves the host, and HOST_PORT does not
// change during a run.

module bringup_core #(
  parameter  int unsigned Ports      = 32,
  parameter  logic [31:0] Build      = 32'h0,  // the commit's first 8 hex digits
  localparam int unsigned AddrWidth  = 34,
  localparam int unsigned DataWidth  = 256,
  localparam int unsigned BeatBytes  = DataWidth / 8,
  localparam int unsigned CountWidth = 24,
  localparam int unsigned LenW       = 5,  // a burst's beats, 1 .. 16 (bringup_port's)
  localparam int unsigned LimitW     = 8   // bursts unanswered (bringup_port's)
) (
  input  logic                                 clk_i,
  input  logic                                 rst_ni,
  input  logic                                 hbm_ready_i,  // both stacks initialized, synchronized to clk_i
  output logic                                 host_port_o,  // HBM port 0 serves the host
  // AXI-Lite (registers)
  input  logic                                 s_aw_valid_i,
  output logic                                 s_aw_ready_o,
  input  logic [11:0]                          s_aw_addr_i,
  input  logic                                 s_w_valid_i,
  output logic                                 s_w_ready_o,
  input  logic [31:0]                          s_w_data_i,
  output logic                                 s_b_valid_o,
  input  logic                                 s_b_ready_i,
  output logic [1:0]                           s_b_resp_o,
  input  logic                                 s_ar_valid_i,
  output logic                                 s_ar_ready_o,
  input  logic [11:0]                          s_ar_addr_i,
  output logic                                 s_r_valid_o,
  input  logic                                 s_r_ready_i,
  output logic [31:0]                          s_r_data_o,
  output logic [1:0]                           s_r_resp_o,
  // AXI, a port each
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
);

  localparam logic [31:0] Id = 32'hCAB0_B001;
  // Register offsets (README.md)
  localparam logic [11:0] RegId = 12'h000, RegBuild = 12'h004, RegScratch = 12'h008,
                          RegStatus = 12'h00C, RegCyclesLo = 12'h010, RegCyclesHi = 12'h014,
                          RegCtrl = 12'h020, RegPortEn = 12'h024, RegMode = 12'h028,
                          RegBaseLo = 12'h02C, RegBaseHi = 12'h030, RegStride = 12'h034,
                          RegBeats = 12'h038, RegRdBurst = 12'h03C, RegWrBurst = 12'h040,
                          RegRdLimit = 12'h044, RegWrLimit = 12'h048, RegWrStrb = 12'h04C,
                          RegSeed = 12'h050, RegHostPort = 12'h054;
  localparam logic [11:0] RegPorts = 12'h100;  // port p's counters at RegPorts + 0x40 p

  if (Ports < 1 || Ports > 32) begin : gen_bad_ports
    $error("bringup_core: Ports must be 1 to 32");
  end

  // --- Registers ----------------------------------------------------------------

  logic                  start_q;
  logic [31:0]           port_en_q, base_lo_q, base_hi_q, stride_q, beats_q, seed_q, strb_q;
  logic [2:0]            mode_q;  // check, write, read
  logic [LenW-1:0]       rd_burst_q, wr_burst_q;
  logic [LimitW-1:0]     rd_limit_q, wr_limit_q;
  logic [31:0]           scratch_q;
  logic                  host_port_q;
  logic [63:0]           cycles_q;
  logic [31:0]           cycles_hi_q;  // CYCLES_HI as of the last CYCLES_LO read
  logic                  busy_q, hbm_ready_q, refused_q;
  logic [Ports-1:0]      busy;
  logic                  start_wr, settings_ok;
  logic [1:0]            rd_stage_q;  // reads: 0 idle, then 1, 2, 3 (answer held until taken)
  logic [11:0]           rd_addr_q;   // the read's address, its low bits a port's counter

  // A burst of 1, 2, 4, 8 or 16 beats: a power of two up to 16.
  function automatic logic burst_ok(logic [LenW-1:0] beats);
    return beats != '0 && (beats & (beats - 1'b1)) == '0 && beats <= LenW'(16);
  endfunction

  // README.md's rules for a run (the module's header).
  assign settings_ok = burst_ok(rd_burst_q) && burst_ok(wr_burst_q)
                    && rd_limit_q != '0 && wr_limit_q != '0
                    && beats_q[31:CountWidth] == '0
                    && (beats_q[LenW-1:0] & (rd_burst_q - 1'b1)) == '0
                    && (beats_q[LenW-1:0] & (wr_burst_q - 1'b1)) == '0
                    && base_lo_q[11:0] == '0 && stride_q[11:0] == '0;

  logic                  wr_fire, b_pend_q;
  assign s_aw_ready_o = s_aw_valid_i && s_w_valid_i && !b_pend_q;
  assign s_w_ready_o  = s_aw_ready_o;
  assign wr_fire      = s_aw_ready_o;
  assign s_b_valid_o  = b_pend_q;
  assign s_b_resp_o   = 2'b00;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      b_pend_q    <= 1'b0;
      start_q     <= 1'b0;
      port_en_q   <= '0;
      mode_q      <= '0;
      base_lo_q   <= '0;
      base_hi_q   <= '0;
      stride_q    <= '0;
      beats_q     <= '0;
      rd_burst_q  <= LenW'(16);
      wr_burst_q  <= LenW'(16);
      rd_limit_q  <= LimitW'(8);
      wr_limit_q  <= LimitW'(8);
      strb_q      <= '1;
      seed_q      <= '0;
      scratch_q   <= '0;
      host_port_q <= 1'b1;
      refused_q   <= 1'b0;
    end else begin
      if (wr_fire) b_pend_q <= 1'b1;
      else if (s_b_ready_i) b_pend_q <= 1'b0;
      start_q <= start_wr && !busy_q && settings_ok;
      if (start_wr) refused_q <= busy_q || !settings_ok;
      if (wr_fire) begin
        unique case (s_aw_addr_i)
          RegScratch:  scratch_q   <= s_w_data_i;
          RegPortEn:   port_en_q   <= s_w_data_i;
          RegMode:     mode_q      <= s_w_data_i[2:0];
          RegBaseLo:   base_lo_q   <= s_w_data_i;
          RegBaseHi:   base_hi_q   <= s_w_data_i;
          RegStride:   stride_q    <= s_w_data_i;
          RegBeats:    beats_q     <= s_w_data_i;
          RegRdBurst:  rd_burst_q  <= s_w_data_i[LenW-1:0];
          RegWrBurst:  wr_burst_q  <= s_w_data_i[LenW-1:0];
          RegRdLimit:  rd_limit_q  <= s_w_data_i[LimitW-1:0];
          RegWrLimit:  wr_limit_q  <= s_w_data_i[LimitW-1:0];
          RegWrStrb:   strb_q      <= s_w_data_i;
          RegSeed:     seed_q      <= s_w_data_i;
          RegHostPort: if (!busy_q) host_port_q <= s_w_data_i[0];
          default: ;
        endcase
      end
    end
  end

  assign start_wr    = wr_fire && s_aw_addr_i == RegCtrl && s_w_data_i[0];
  assign host_port_o = host_port_q;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) cycles_q <= '0;
    else cycles_q <= cycles_q + 1'b1;
    if (!rst_ni) cycles_hi_q <= '0;
    else if (rd_stage_q == 2'd2 && rd_addr_q == RegCyclesLo) cycles_hi_q <= cycles_q[63:32];
    busy_q      <= |busy || start_q;  // a port is busy from its start, a cycle after start_q
    hbm_ready_q <= hbm_ready_i;
  end

  // --- Ports ----------------------------------------------------------------------

  logic [AddrWidth-1:0]        base;
  logic [Ports-1:0][31:0]      stat;
  assign base = {base_hi_q[AddrWidth-33:0], base_lo_q};

  for (genvar p = 0; p < Ports; p++) begin : gen_port
    // The port's address, two register stages (its offset, then the sum),
    // settled long before a start; its start, one.
    logic [AddrWidth-1:0] offset_q, addr_q;
    logic                 start_q2;
    always_ff @(posedge clk_i) begin
      offset_q <= AddrWidth'(p) * AddrWidth'(stride_q);
      addr_q   <= base + offset_q;
      start_q2 <= start_q && port_en_q[p] && !(p == 0 && host_port_q);
    end

    bringup_port #(
      .AddrWidth (AddrWidth),
      .DataWidth (DataWidth),
      .CountWidth(CountWidth)
    ) u_port (
      .clk_i,
      .rst_ni,
      .start_i   (start_q2),
      .rd_en_i   (mode_q[0]),
      .wr_en_i   (mode_q[1]),
      .check_i   (mode_q[2]),
      .addr_i    (addr_q),
      .beats_i   (beats_q[CountWidth-1:0]),
      .rd_burst_i(rd_burst_q),
      .wr_burst_i(wr_burst_q),
      .rd_limit_i(rd_limit_q),
      .wr_limit_i(wr_limit_q),
      .wr_strb_i (strb_q),
      .seed_i    (seed_q),
      .busy_o    (busy[p]),
      .stat_sel_i(rd_addr_q[5:2]),
      .stat_o    (stat[p]),
      .ar_valid_o(ar_valid_o[p]),
      .ar_ready_i(ar_ready_i[p]),
      .ar_addr_o (ar_addr_o[p]),
      .ar_len_o  (ar_len_o[p]),
      .ar_size_o (ar_size_o[p]),
      .ar_burst_o(ar_burst_o[p]),
      .r_valid_i (r_valid_i[p]),
      .r_ready_o (r_ready_o[p]),
      .r_data_i  (r_data_i[p]),
      .r_resp_i  (r_resp_i[p]),
      .r_last_i  (r_last_i[p]),
      .aw_valid_o(aw_valid_o[p]),
      .aw_ready_i(aw_ready_i[p]),
      .aw_addr_o (aw_addr_o[p]),
      .aw_len_o  (aw_len_o[p]),
      .aw_size_o (aw_size_o[p]),
      .aw_burst_o(aw_burst_o[p]),
      .w_valid_o (w_valid_o[p]),
      .w_ready_i (w_ready_i[p]),
      .w_data_o  (w_data_o[p]),
      .w_strb_o  (w_strb_o[p]),
      .w_last_o  (w_last_o[p]),
      .b_valid_i (b_valid_i[p]),
      .b_ready_o (b_ready_o[p]),
      .b_resp_i  (b_resp_i[p])
    );
  end

  // --- Reads: address taken (stage 1), the port's counter (2), the answer (3) --

  logic        rd_is_port_q;  // the address is a port's counter
  logic [4:0]  rd_port_q;
  logic [31:0] r_data_q;
  logic [5:0]  ar_port;      // the port the address would be, 0 .. 59

  assign s_ar_ready_o = rd_stage_q == 2'd0;
  assign s_r_valid_o  = rd_stage_q == 2'd3;
  assign s_r_data_o   = r_data_q;
  assign s_r_resp_o   = 2'b00;
  assign ar_port      = s_ar_addr_i[11:6] - 6'(RegPorts >> 6);

  always_ff @(posedge clk_i) begin
    if (!rst_ni) rd_stage_q <= '0;
    else if (s_ar_valid_i && s_ar_ready_o) rd_stage_q <= 2'd1;
    else if (rd_stage_q == 2'd1 || rd_stage_q == 2'd2) rd_stage_q <= rd_stage_q + 1'b1;
    else if (s_r_valid_o && s_r_ready_i) rd_stage_q <= 2'd0;
  end

  always_ff @(posedge clk_i) begin
    if (s_ar_valid_i && s_ar_ready_o) begin
      rd_addr_q    <= s_ar_addr_i;
      rd_is_port_q <= s_ar_addr_i >= RegPorts && 32'(ar_port) < 32'(Ports);
      rd_port_q    <= 5'(ar_port);
    end
    if (rd_stage_q == 2'd2) begin
      if (rd_is_port_q) begin
        r_data_q <= stat[rd_port_q];
      end else begin
        unique case (rd_addr_q)
          RegId:       r_data_q <= Id;
          RegBuild:    r_data_q <= Build;
          RegScratch:  r_data_q <= scratch_q;
          RegStatus:   r_data_q <= {29'd0, refused_q, hbm_ready_q, busy_q};
          RegCyclesLo: r_data_q <= cycles_q[31:0];
          RegCyclesHi: r_data_q <= cycles_hi_q;
          RegPortEn:   r_data_q <= port_en_q;
          RegMode:     r_data_q <= {29'd0, mode_q};
          RegBaseLo:   r_data_q <= base_lo_q;
          RegBaseHi:   r_data_q <= base_hi_q;
          RegStride:   r_data_q <= stride_q;
          RegBeats:    r_data_q <= beats_q;
          RegRdBurst:  r_data_q <= 32'(rd_burst_q);
          RegWrBurst:  r_data_q <= 32'(wr_burst_q);
          RegRdLimit:  r_data_q <= 32'(rd_limit_q);
          RegWrLimit:  r_data_q <= 32'(wr_limit_q);
          RegWrStrb:   r_data_q <= strb_q;
          RegSeed:     r_data_q <= seed_q;
          RegHostPort: r_data_q <= {31'd0, host_port_q};
          default: r_data_q <= 32'd0;
        endcase
      end
    end
  end

endmodule
