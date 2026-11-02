// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// One HBM port's test engines for the bring-up design (platforms/f2/README.md,
// "A run"): from start_i, the writer sends beats_i beats from addr_i in bursts
// of wr_burst_i beats, the data the pattern, and the reader (rtl/port_dma.sv)
// reads beats_i beats in bursts of rd_burst_i beats, each compared with the
// pattern; each keeps at most its limit of bursts unanswered. The port takes
// its settings at start_i, which comes only while it is idle (busy_o low).
// The counters count cycles from start_i and stop when their side is done;
// start_i clears them.
//
// The pattern: the 32-bit word at byte address a holds (a >> 2) ^ seed.
//
// AXI: the subset rtl/port_dma.sv and rtl/port_wdma.sv use, plus RLAST (to
// count read bursts answered). The burst length is AXI4's 8 bits; the
// wrapper keeps AXI3's low 4.

module bringup_port #(
  parameter  int unsigned AddrWidth     = 34,
  parameter  int unsigned DataWidth     = 256,
  parameter  int unsigned CountWidth    = 24,   // beats per run
  parameter  int unsigned Depth         = 128,  // port_dma's FIFO, beats (as the engine's)
  parameter  int unsigned WrOutstanding = 128,  // port_wdma's limit (as the vector unit's)
  localparam int unsigned BeatBytes     = DataWidth / 8,
  localparam int unsigned Words         = DataWidth / 32,
  localparam int unsigned LenW          = 5     // a burst's beats, 1 .. 16
) (
  input  logic                  clk_i,
  input  logic                  rst_ni,
  // Settings, taken at start_i
  input  logic                  start_i,
  input  logic                  rd_en_i,
  input  logic                  wr_en_i,
  input  logic                  check_i,
  input  logic [AddrWidth-1:0]  addr_i,      // 4 KB-aligned
  input  logic [CountWidth-1:0] beats_i,     // a multiple of both bursts
  input  logic [LenW-1:0]       rd_burst_i,  // 1, 2, 4, 8 or 16
  input  logic [LenW-1:0]       wr_burst_i,
  input  logic [7:0]            rd_limit_i,  // bursts unanswered, at least 1
  input  logic [7:0]            wr_limit_i,
  input  logic [BeatBytes-1:0]  wr_strb_i,
  input  logic [31:0]           seed_i,
  output logic                  busy_o,      // from start_i on, until both sides are done
  // Counters: stat_o is counter stat_sel_i (README.md's port table, by word), a cycle later
  input  logic [3:0]            stat_sel_i,
  output logic [31:0]           stat_o,
  // AXI read
  output logic                  ar_valid_o,
  input  logic                  ar_ready_i,
  output logic [AddrWidth-1:0]  ar_addr_o,
  output logic [7:0]            ar_len_o,
  output logic [2:0]            ar_size_o,
  output logic [1:0]            ar_burst_o,
  input  logic                  r_valid_i,
  output logic                  r_ready_o,
  input  logic [DataWidth-1:0]  r_data_i,
  input  logic [1:0]            r_resp_i,
  input  logic                  r_last_i,
  // AXI write
  output logic                  aw_valid_o,
  input  logic                  aw_ready_i,
  output logic [AddrWidth-1:0]  aw_addr_o,
  output logic [7:0]            aw_len_o,
  output logic [2:0]            aw_size_o,
  output logic [1:0]            aw_burst_o,
  output logic                  w_valid_o,
  input  logic                  w_ready_i,
  output logic [DataWidth-1:0]  w_data_o,
  output logic [BeatBytes-1:0]  w_strb_o,
  output logic                  w_last_o,
  input  logic                  b_valid_i,
  output logic                  b_ready_o,
  input  logic [1:0]            b_resp_i
);

  localparam int unsigned OffW = $clog2(BeatBytes);

  // The pattern's beat at a beat-aligned address: its words' indices are the
  // address's word index with the low bits the word's place.
  function automatic logic [DataWidth-1:0] pattern(logic [AddrWidth-1:0] addr, logic [31:0] seed);
    logic [DataWidth-1:0] beat;
    for (int w = 0; w < Words; w++) begin
      beat[w*32 +: 32] = 32'((addr >> 2) | AddrWidth'(w)) ^ seed;
    end
    return beat;
  endfunction

  // The run's settings, held here: the shared registers then reach the ports
  // only at start, and nothing broadcast sits on an AXI path.
  logic                  check_q;
  logic [CountWidth-1:0] beats_q;
  logic [LenW-1:0]       rd_burst_q, wr_burst_q;
  logic [7:0]            rd_limit_q, wr_limit_q;
  logic [BeatBytes-1:0]  strb_q;
  logic [31:0]           seed_q;

  always_ff @(posedge clk_i) begin
    if (start_i) begin
      check_q    <= check_i;
      beats_q    <= beats_i;
      rd_burst_q <= rd_burst_i;
      wr_burst_q <= wr_burst_i;
      rd_limit_q <= rd_limit_i;
      wr_limit_q <= wr_limit_i;
      strb_q     <= wr_strb_i;
      seed_q     <= seed_i;
    end
  end

  // Cycles since start, counting the first cycle after it as 1: a counter
  // stores the count of the cycle its event happens in.
  logic [31:0] cyc_q;

  always_ff @(posedge clk_i) begin
    if (!rst_ni || start_i) cyc_q <= 32'd1;
    else cyc_q <= cyc_q + 1'b1;
  end

  // --- Reader -------------------------------------------------------------------

  logic                  rd_busy_q, rd_err_q;
  logic [CountWidth-1:0] rd_left_q;  // beats not yet commanded
  logic [AddrWidth-1:0]  rd_addr_q;
  logic [7:0]            rd_out_q;   // bursts requested, not yet answered
  logic [31:0]           rd_beats_q, rd_cycles_q, rd_first_q, rd_stalls_q, rd_gap_q, rd_maxgap_q;
  logic [31:0]           rd_bad_q;
  logic [63:0]           rd_occ_q;
  logic                  rd_cmd_valid, rd_cmd_ready, rd_room, dma_ar_valid, ar_fire, r_fire;
  logic                  out_valid;
  logic [DataWidth-1:0]  out_data;
  logic [CountWidth-1:0] chk_beats_q;  // beats out of port_dma
  logic [AddrWidth-1:0]  chk_addr_q;
  logic                  chk_valid_q;
  logic [DataWidth-1:0]  chk_data_q, chk_want_q;

  assign rd_cmd_valid = rd_busy_q && rd_left_q != '0;
  assign rd_room      = rd_out_q < rd_limit_q;
  assign ar_valid_o   = dma_ar_valid && rd_room;
  assign ar_fire      = ar_valid_o && ar_ready_i;
  assign r_fire       = r_valid_i && r_ready_o;

  // verilator lint_off UNUSEDSIGNAL
  logic                       unused_dma_err;  // the run's own flag instead, cleared by start
  logic [$clog2(Depth+1)-1:0] unused_count;
  // verilator lint_on UNUSEDSIGNAL
  port_dma #(
    .AddrWidth (AddrWidth),
    .DataWidth (DataWidth),
    .CountWidth(CountWidth),
    .Depth     (Depth)
  ) u_rd (
    .clk_i,
    .rst_ni,
    .cmd_valid_i(rd_cmd_valid),
    .cmd_ready_o(rd_cmd_ready),
    .cmd_addr_i (rd_addr_q),
    .cmd_beats_i(CountWidth'(rd_burst_q)),
    .out_valid_o(out_valid),
    .out_ready_i(1'b1),
    .out_data_o (out_data),
    .out_count_o(unused_count),
    .ar_valid_o (dma_ar_valid),
    .ar_ready_i (ar_ready_i && rd_room),
    .ar_addr_o,
    .ar_len_o,
    .ar_size_o,
    .ar_burst_o,
    .r_valid_i,
    .r_ready_o,
    .r_data_i,
    .r_resp_i,
    .err_o      (unused_dma_err)
  );

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      rd_busy_q <= 1'b0;
      rd_out_q  <= '0;
    end else if (start_i) begin
      rd_busy_q <= rd_en_i && beats_i != '0;
      rd_out_q  <= '0;
    end else begin
      rd_out_q <= rd_out_q + 8'(ar_fire) - 8'(r_fire && r_last_i);
      if (out_valid && chk_beats_q + 1'b1 == beats_q) rd_busy_q <= 1'b0;
    end
  end

  always_ff @(posedge clk_i) begin
    if (start_i) begin
      rd_left_q   <= beats_i;
      rd_addr_q   <= addr_i;
      chk_addr_q  <= addr_i;
      chk_beats_q <= '0;
    end else begin
      if (rd_cmd_valid && rd_cmd_ready) begin
        rd_left_q <= rd_left_q - CountWidth'(rd_burst_q);
        rd_addr_q <= rd_addr_q + (AddrWidth'(rd_burst_q) << OffW);
      end
      if (out_valid) begin
        chk_addr_q  <= chk_addr_q + AddrWidth'(BeatBytes);
        chk_beats_q <= chk_beats_q + 1'b1;
      end
    end
  end

  // The check, a register stage before the compare.
  always_ff @(posedge clk_i) begin
    if (!rst_ni) chk_valid_q <= 1'b0;
    else chk_valid_q <= out_valid && check_q;
    chk_data_q <= out_data;
    chk_want_q <= pattern(chk_addr_q, seed_q);
  end

  // Counters, from the AXI side: r_fire is a beat arriving.
  always_ff @(posedge clk_i) begin
    if (start_i) begin
      rd_err_q    <= 1'b0;
      rd_beats_q  <= '0;
      rd_cycles_q <= '0;
      rd_first_q  <= '0;
      rd_stalls_q <= '0;
      rd_gap_q    <= '0;
      rd_maxgap_q <= '0;
      rd_bad_q    <= '0;
      rd_occ_q    <= '0;
    end else begin
      if (chk_valid_q && chk_data_q != chk_want_q) rd_bad_q <= rd_bad_q + 1'b1;
      if (rd_busy_q) rd_occ_q <= rd_occ_q + 64'(rd_out_q);
      if (r_fire) begin
        rd_beats_q  <= rd_beats_q + 1'b1;
        rd_cycles_q <= cyc_q;
        rd_gap_q    <= '0;
        if (r_resp_i != 2'b00) rd_err_q <= 1'b1;
        if (rd_beats_q == '0) rd_first_q <= cyc_q;
      end else if (rd_beats_q != '0 && rd_beats_q != 32'(beats_q)) begin
        rd_stalls_q <= rd_stalls_q + 1'b1;
        rd_gap_q    <= rd_gap_q + 1'b1;
        if (rd_gap_q + 1'b1 > rd_maxgap_q) rd_maxgap_q <= rd_gap_q + 1'b1;
      end
    end
  end

  // --- Writer -------------------------------------------------------------------

  logic                  wr_busy_q, wr_err_q;
  logic [CountWidth-1:0] wr_left_q, wr_dleft_q, wr_done_q;  // beats not commanded, not sent; answered
  logic [AddrWidth-1:0]  wr_addr_q, wr_daddr_q;
  logic [7:0]            wr_out_q;
  logic [31:0]           wr_beats_q, wr_cycles_q, wr_first_q, wr_maxout_q;
  logic [63:0]           wr_occ_q;
  logic                  wr_cmd_valid, wr_cmd_ready, wr_room, wdma_aw_valid, aw_fire;
  logic                  in_valid, in_ready, done;

  assign wr_cmd_valid = wr_busy_q && wr_left_q != '0;
  assign in_valid     = wr_busy_q && wr_dleft_q != '0;
  assign wr_room      = wr_out_q < wr_limit_q;
  assign aw_valid_o   = wdma_aw_valid && wr_room;
  assign aw_fire      = aw_valid_o && aw_ready_i;

  // verilator lint_off UNUSEDSIGNAL
  logic unused_wdma_err, unused_tag;
  // verilator lint_on UNUSEDSIGNAL
  port_wdma #(
    .AddrWidth  (AddrWidth),
    .DataWidth  (DataWidth),
    .Outstanding(WrOutstanding)
  ) u_wr (
    .clk_i,
    .rst_ni,
    .cmd_valid_i(wr_cmd_valid),
    .cmd_ready_o(wr_cmd_ready),
    .cmd_addr_i (wr_addr_q),
    .cmd_beats_i(wr_burst_q),
    .cmd_tag_i  (1'b0),
    .in_valid_i (in_valid),
    .in_ready_o (in_ready),
    .in_data_i  (pattern(wr_daddr_q, seed_q)),
    .in_strb_i  (strb_q),
    .done_o     (done),
    .done_tag_o (unused_tag),
    .aw_valid_o (wdma_aw_valid),
    .aw_ready_i (aw_ready_i && wr_room),
    .aw_addr_o,
    .aw_len_o,
    .aw_size_o,
    .aw_burst_o,
    .w_valid_o,
    .w_ready_i,
    .w_data_o,
    .w_strb_o,
    .w_last_o,
    .b_valid_i,
    .b_ready_o,
    .b_resp_i,
    .err_o      (unused_wdma_err)
  );

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      wr_busy_q <= 1'b0;
      wr_out_q  <= '0;
    end else if (start_i) begin
      wr_busy_q <= wr_en_i && beats_i != '0;
      wr_out_q  <= '0;
    end else begin
      wr_out_q <= wr_out_q + 8'(aw_fire) - 8'(done);
      if (done && wr_done_q + CountWidth'(wr_burst_q) == beats_q) wr_busy_q <= 1'b0;
    end
  end

  always_ff @(posedge clk_i) begin
    if (start_i) begin
      wr_left_q  <= beats_i;
      wr_dleft_q <= beats_i;
      wr_addr_q  <= addr_i;
      wr_daddr_q <= addr_i;
    end else begin
      if (wr_cmd_valid && wr_cmd_ready) begin
        wr_left_q <= wr_left_q - CountWidth'(wr_burst_q);
        wr_addr_q <= wr_addr_q + (AddrWidth'(wr_burst_q) << OffW);
      end
      if (in_valid && in_ready) begin
        wr_dleft_q <= wr_dleft_q - 1'b1;
        wr_daddr_q <= wr_daddr_q + AddrWidth'(BeatBytes);
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (start_i) begin
      wr_err_q    <= 1'b0;
      wr_done_q   <= '0;
      wr_beats_q  <= '0;
      wr_cycles_q <= '0;
      wr_first_q  <= '0;
      wr_maxout_q <= '0;
      wr_occ_q    <= '0;
    end else begin
      if (w_valid_o && w_ready_i) wr_beats_q <= wr_beats_q + 1'b1;
      if (wr_busy_q) wr_occ_q <= wr_occ_q + 64'(wr_out_q);
      if (32'(wr_out_q) > wr_maxout_q) wr_maxout_q <= 32'(wr_out_q);
      if (done) begin
        wr_done_q   <= wr_done_q + CountWidth'(wr_burst_q);
        wr_cycles_q <= cyc_q;
        if (b_resp_i != 2'b00) wr_err_q <= 1'b1;
        if (wr_done_q == '0) wr_first_q <= cyc_q;
      end
    end
  end

  // --- Counters out -------------------------------------------------------------

  assign busy_o = start_i || rd_busy_q || wr_busy_q;

  always_ff @(posedge clk_i) begin
    unique case (stat_sel_i)
      4'd0:    stat_o <= rd_beats_q;
      4'd1:    stat_o <= rd_cycles_q;
      4'd2:    stat_o <= rd_first_q;
      4'd3:    stat_o <= rd_stalls_q;
      4'd4:    stat_o <= rd_maxgap_q;
      4'd5:    stat_o <= rd_bad_q;
      4'd6:    stat_o <= rd_occ_q[31:0];
      4'd7:    stat_o <= rd_occ_q[63:32];
      4'd8:    stat_o <= wr_beats_q;
      4'd9:    stat_o <= wr_cycles_q;
      4'd10:   stat_o <= wr_first_q;
      4'd11:   stat_o <= wr_maxout_q;
      4'd12:   stat_o <= wr_occ_q[31:0];
      4'd13:   stat_o <= wr_occ_q[63:32];
      4'd14:   stat_o <= {28'd0, wr_err_q, rd_err_q, wr_busy_q, rd_busy_q};
      default: stat_o <= '0;
    endcase
  end

endmodule
