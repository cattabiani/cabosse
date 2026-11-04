// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// HBM port 0's core side (platforms/f2/README.md, "The core on F2"): engine
// port 0's reads and the vector unit's reads and writes merged into one AXI
// port, for bringup_host.sv's engine side. Reads: the two address streams
// take turns (round robin; a request waiting for the port keeps its turn, as
// AXI asks), all with one ID, so the memory answers in their order; a queue
// of who sent each burst sends its beats back to the sender, a burst ending
// at RLAST. With Bursts bursts unanswered, new reads wait. Writes: the vector
// unit's alone.
//
// Packed channels as bringup_host.sv's: {addr, len, burst}, {data, strb,
// last}. Full beats and INCR bursts only (the core's).

module f2_port0 #(
  parameter  int unsigned AddrWidth = 34,
  parameter  int unsigned DataWidth = 256,
  parameter  int unsigned Bursts    = 32,   // reads unanswered at most
  localparam int unsigned BeatBytes = DataWidth / 8,
  localparam int unsigned AxW       = AddrWidth + 8 + 2,
  localparam int unsigned WW        = DataWidth + BeatBytes + 1
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,
  // Engine port 0: reads
  input  logic                 e_ar_valid_i,
  output logic                 e_ar_ready_o,
  input  logic [AxW-1:0]       e_ar_i,
  output logic                 e_r_valid_o,
  input  logic                 e_r_ready_i,
  // The vector unit: reads and writes
  input  logic                 v_ar_valid_i,
  output logic                 v_ar_ready_o,
  input  logic [AxW-1:0]       v_ar_i,
  output logic                 v_r_valid_o,
  input  logic                 v_r_ready_i,
  input  logic                 v_aw_valid_i,
  output logic                 v_aw_ready_o,
  input  logic [AxW-1:0]       v_aw_i,
  input  logic                 v_w_valid_i,
  output logic                 v_w_ready_o,
  input  logic [WW-1:0]        v_w_i,
  output logic                 v_b_valid_o,
  input  logic                 v_b_ready_i,
  // The port (R data and B's response go to both senders as they are: only
  // RLAST is read here)
  output logic                 m_ar_valid_o,
  input  logic                 m_ar_ready_i,
  output logic [AxW-1:0]       m_ar_o,
  input  logic                 m_r_valid_i,
  output logic                 m_r_ready_o,
  input  logic                 m_r_last_i,
  output logic                 m_aw_valid_o,
  input  logic                 m_aw_ready_i,
  output logic [AxW-1:0]       m_aw_o,
  output logic                 m_w_valid_o,
  input  logic                 m_w_ready_i,
  output logic [WW-1:0]        m_w_o,
  input  logic                 m_b_valid_i,
  output logic                 m_b_ready_o
);

  // --- Reads: whose turn, and who sent each burst ------------------------------

  logic pick_v, prefer_v_q, lock_q, lock_v_q, room, ar_fire;
  logic sender, sender_valid, r_last;  // the oldest burst's sender (1: the vector unit)
  // verilator lint_off UNUSEDSIGNAL
  logic order_ready;  // room keeps a place: unread
  // verilator lint_on UNUSEDSIGNAL

  // A waiting request keeps the port; else the preferred one, else the other.
  always_comb begin
    if (lock_q) pick_v = lock_v_q;
    else if (e_ar_valid_i && v_ar_valid_i) pick_v = prefer_v_q;
    else pick_v = v_ar_valid_i;
  end

  assign m_ar_valid_o = room && (pick_v ? v_ar_valid_i : e_ar_valid_i);
  assign m_ar_o       = pick_v ? v_ar_i : e_ar_i;
  assign ar_fire      = m_ar_valid_o && m_ar_ready_i;
  assign e_ar_ready_o = room && !pick_v && m_ar_ready_i;
  assign v_ar_ready_o = room && pick_v && m_ar_ready_i;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      prefer_v_q <= 1'b0;
      lock_q     <= 1'b0;
      lock_v_q   <= 1'b0;
    end else begin
      lock_q   <= m_ar_valid_o && !m_ar_ready_i;
      lock_v_q <= pick_v;
      if (ar_fire) prefer_v_q <= !pick_v;
    end
  end

  logic [$clog2(Bursts + 1)-1:0] count;
  stream_fifo #(.Width(1), .Depth(Bursts)) u_order (
    .clk_i,
    .rst_ni,
    .in_valid_i (ar_fire),
    .in_ready_o (order_ready),
    .in_data_i  (pick_v),
    .out_valid_o(sender_valid),
    .out_ready_i(m_r_valid_i && m_r_ready_o && r_last),
    .out_data_o (sender),
    .count_o    (count)
  );
  assign room = count < ($clog2(Bursts + 1))'(Bursts);

  assign r_last      = m_r_last_i;
  assign e_r_valid_o = m_r_valid_i && sender_valid && !sender;
  assign v_r_valid_o = m_r_valid_i && sender_valid && sender;
  assign m_r_ready_o = sender_valid && (sender ? v_r_ready_i : e_r_ready_i);

  // --- Writes: the vector unit's ---------------------------------------------------

  assign m_aw_valid_o = v_aw_valid_i;
  assign m_aw_o       = v_aw_i;
  assign v_aw_ready_o = m_aw_ready_i;
  assign m_w_valid_o  = v_w_valid_i;
  assign m_w_o        = v_w_i;
  assign v_w_ready_o  = m_w_ready_i;
  assign v_b_valid_o  = m_b_valid_i;
  assign m_b_ready_o  = v_b_ready_i;

endmodule
