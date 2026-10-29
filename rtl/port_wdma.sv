// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// One AXI port's writer (PLAN.md, M6: KV_STORE): a command is one burst, a
// beat-aligned address and 1 to MaxBurst beats inside one 4 KB page (the
// caller's to keep); its beats come from the data stream in order, full beats
// (every strobe set). A burst's data goes out once its address is taken, and
// done_o pulses for each write response, in order, err_o (sticky) for one
// that is not OKAY. At most Outstanding bursts are sent and not yet answered.
//
// AXI: write channels only, in the subset AXI3 and AXI4 share: incrementing
// bursts of the full data width, one ID (responses in order; no ID ports).

module port_wdma #(
  parameter  int unsigned AddrWidth   = 34,
  parameter  int unsigned DataWidth   = 256,
  parameter  int unsigned MaxBurst    = 16,
  parameter  int unsigned Outstanding = 8,
  localparam int unsigned BeatBytes   = DataWidth / 8,
  localparam int unsigned LenW        = $clog2(MaxBurst + 1)
) (
  input  logic                  clk_i,
  input  logic                  rst_ni,
  // Command: one burst
  input  logic                  cmd_valid_i,
  output logic                  cmd_ready_o,
  input  logic [AddrWidth-1:0]  cmd_addr_i,   // beat-aligned
  input  logic [LenW-1:0]       cmd_beats_i,  // 1 .. MaxBurst
  // Beats in
  input  logic                  in_valid_i,
  output logic                  in_ready_o,
  input  logic [DataWidth-1:0]  in_data_i,
  output logic                  done_o,       // a burst's write response
  // AXI write address channel
  output logic                  aw_valid_o,
  input  logic                  aw_ready_i,
  output logic [AddrWidth-1:0]  aw_addr_o,
  output logic [7:0]            aw_len_o,     // beats - 1
  output logic [2:0]            aw_size_o,    // log2(bytes per beat)
  output logic [1:0]            aw_burst_o,   // INCR
  // AXI write data channel
  output logic                  w_valid_o,
  input  logic                  w_ready_i,
  output logic [DataWidth-1:0]  w_data_o,
  output logic [BeatBytes-1:0]  w_strb_o,
  output logic                  w_last_o,
  // AXI write response channel
  input  logic                  b_valid_i,
  output logic                  b_ready_o,
  input  logic [1:0]            b_resp_i,
  output logic                  err_o         // sticky: a response was not OKAY
);

  localparam int unsigned OW = $clog2(Outstanding + 1);
  localparam logic [1:0]  Incr = 2'b01;

  if (MaxBurst < 1 || MaxBurst > 16) begin : gen_bad_burst
    $error("port_wdma: MaxBurst must be 1 to 16 (AXI3)");
  end
  if (DataWidth < 8 || (DataWidth & (DataWidth - 1)) != 0 || BeatBytes > 128) begin : gen_bad_w
    $error("port_wdma: DataWidth must be a power of two, 8 to 1024 bits");
  end

  // The address channel holds one burst; once taken, its length waits in
  // lens for its data. sent_q counts bursts taken and not yet answered.
  logic [AddrWidth-1:0] addr_q;
  logic [LenW-1:0]      beats_q, lens_out, beat_q;
  logic                 aw_q, aw_fire, lens_valid, w_fire;
  logic [OW-1:0]        sent_q;

  assign cmd_ready_o = !aw_q;
  assign aw_valid_o  = aw_q && sent_q != OW'(Outstanding);
  assign aw_fire     = aw_valid_o && aw_ready_i;
  assign aw_addr_o   = addr_q;
  assign aw_len_o    = 8'(beats_q - 1'b1);
  assign aw_size_o   = 3'($clog2(BeatBytes));
  assign aw_burst_o  = Incr;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      aw_q <= 1'b0;
    end else if (cmd_valid_i && cmd_ready_o) begin
      aw_q    <= 1'b1;
      addr_q  <= cmd_addr_i;
      beats_q <= cmd_beats_i;
    end else if (aw_fire) begin
      aw_q <= 1'b0;
    end
  end

  // Every burst taken has a place: sent_q < Outstanding when it is.
  // verilator lint_off UNUSEDSIGNAL
  logic unused_lens_ready;
  // verilator lint_on UNUSEDSIGNAL
  stream_fifo #(.Width(LenW), .Depth(Outstanding)) u_lens (
    .clk_i,
    .rst_ni,
    .in_valid_i (aw_fire),
    .in_ready_o (unused_lens_ready),
    .in_data_i  (beats_q),
    .out_valid_o(lens_valid),
    .out_ready_i(w_fire && w_last_o),
    .out_data_o (lens_out)
  );

  assign w_valid_o  = lens_valid && in_valid_i;
  assign in_ready_o = lens_valid && w_ready_i;
  assign w_fire     = w_valid_o && w_ready_i;
  assign w_data_o   = in_data_i;
  assign w_strb_o   = '1;
  assign w_last_o   = beat_q + 1'b1 == lens_out;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) beat_q <= '0;
    else if (w_fire) beat_q <= w_last_o ? '0 : beat_q + 1'b1;
  end

  assign b_ready_o = 1'b1;
  assign done_o    = b_valid_i;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      sent_q <= '0;
      err_o  <= 1'b0;
    end else begin
      sent_q <= sent_q + OW'(aw_fire) - OW'(b_valid_i);
      if (b_valid_i && b_resp_i != 2'b00) err_o <= 1'b1;
    end
  end

endmodule
