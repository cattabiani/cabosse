// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// One AXI port's writer (PLAN.md, M6: KV_STORE): a command is one burst, a
// beat-aligned address and 1 to MaxBurst beats inside one 4 KB page (the
// caller's to keep), and a tag; its beats come from the data stream in
// order, each with its strobes. A burst's data goes out from its command's
// take, not waiting for its address to be taken (AXI: WVALID must not wait
// for AWREADY; a slave may wait for data before taking an address), and
// done_o pulses for each write response, in order, with done_tag_o the
// burst's tag; err_o (sticky) for one that is not OKAY. At most Outstanding
// bursts are sent and not yet answered, and at most Outstanding taken and
// not yet written.
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
  input  logic                  cmd_tag_i,    // returned with its response
  // Beats in
  input  logic                  in_valid_i,
  output logic                  in_ready_o,
  input  logic [DataWidth-1:0]  in_data_i,
  input  logic [BeatBytes-1:0]  in_strb_i,
  output logic                  done_o,       // a burst's write response
  output logic                  done_tag_o,   // its tag
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

  // The address channel holds two bursts, so it takes one a cycle; a
  // burst's length waits in lens for its data from the command's take, its
  // tag in tags for its response. sent_q counts bursts sent and not yet
  // answered.
  logic [AddrWidth-1:0] aw_addr;
  logic [LenW-1:0]      aw_beats, lens_out, beat_q;
  logic                 aw_ready, aw_valid, aw_fire, lens_valid, lens_ready, tags_ready;
  logic                 cmd_fire, w_fire;
  logic [OW-1:0]        sent_q;

  assign cmd_ready_o = aw_ready && lens_ready && tags_ready;
  assign cmd_fire    = cmd_valid_i && cmd_ready_o;
  assign aw_valid_o  = aw_valid && sent_q != OW'(Outstanding);
  assign aw_fire     = aw_valid_o && aw_ready_i;
  assign aw_addr_o   = aw_addr;
  assign aw_len_o    = 8'(aw_beats - 1'b1);
  assign aw_size_o   = 3'($clog2(BeatBytes));
  assign aw_burst_o  = Incr;

  stream_fifo #(.Width(AddrWidth + LenW), .Depth(2)) u_aw (
    .clk_i,
    .rst_ni,
    .in_valid_i (cmd_fire),
    .in_ready_o (aw_ready),
    .in_data_i  ({cmd_addr_i, cmd_beats_i}),
    .out_valid_o(aw_valid),
    .out_ready_i(aw_fire),
    .out_data_o ({aw_addr, aw_beats}),
    // verilator lint_off PINCONNECTEMPTY
    .count_o    ()
    // verilator lint_on PINCONNECTEMPTY
  );

  stream_fifo #(.Width(LenW), .Depth(Outstanding)) u_lens (
    .clk_i,
    .rst_ni,
    .in_valid_i (cmd_fire),
    .in_ready_o (lens_ready),
    .in_data_i  (cmd_beats_i),
    .out_valid_o(lens_valid),
    .out_ready_i(w_fire && w_last_o),
    .out_data_o (lens_out),
    // verilator lint_off PINCONNECTEMPTY
    .count_o    ()
    // verilator lint_on PINCONNECTEMPTY
  );

  // Taken and not yet answered: Outstanding sent, two in the address channel.
  // verilator lint_off UNUSEDSIGNAL
  logic unused_tag_valid;
  // verilator lint_on UNUSEDSIGNAL
  stream_fifo #(.Width(1), .Depth(Outstanding + 2)) u_tags (
    .clk_i,
    .rst_ni,
    .in_valid_i (cmd_fire),
    .in_ready_o (tags_ready),
    .in_data_i  (cmd_tag_i),
    .out_valid_o(unused_tag_valid),  // a response has its burst's
    .out_ready_i(b_valid_i),
    .out_data_o (done_tag_o),
    // verilator lint_off PINCONNECTEMPTY
    .count_o    ()
    // verilator lint_on PINCONNECTEMPTY
  );

  assign w_valid_o  = lens_valid && in_valid_i;
  assign in_ready_o = lens_valid && w_ready_i;
  assign w_fire     = w_valid_o && w_ready_i;
  assign w_data_o   = in_data_i;
  assign w_strb_o   = in_strb_i;
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
