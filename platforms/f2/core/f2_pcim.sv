// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The logits to the host (PLAN.md M11 step 1, Q-34): on the vector unit's
// way to HBM port 0, a copy of each write beat inside a window (OUTPUT's
// logits) for PCIM, at the host address host_delta_i + its HBM address.
// Every write still goes to port 0, unchanged; the core does not see the
// copies (f2_core.sv sends them on and counts their answers).
//
// The window: a beat is copied if its address minus win_base_i is below
// win_bytes_i (0: none), each beat of a burst on its own. Its settings change only while the core is idle;
// they are registered here first (a copy beside the tap, a plain register
// path from the wrapper's registers).
//
// AW and W are forks: a request is taken once both its ways have it, each
// way taking it on its own (a flag each), so neither waits for the other's
// ready. AW's second way is a queue of the bursts' offsets in the window and
// host addresses, W's the copies. A W beat leaves only once its
// burst's decision is in the queue: the queue takes AW without waiting for
// port 0's AWREADY, so a W beat never waits for AWREADY (AXI: a slave may
// wait for data before taking an address). Bursts of any length; each beat
// copied at its own address.

module f2_pcim #(
  parameter  int unsigned AddrWidth = 34,
  parameter  int unsigned DataWidth = 256,
  parameter  int unsigned HostWidth = 64,
  parameter  int unsigned Depth     = 8,    // bursts decided and not yet written
  localparam int unsigned BeatBytes = DataWidth / 8,
  localparam int unsigned AxW       = AddrWidth + 8 + 2,
  localparam int unsigned WW        = DataWidth + BeatBytes + 1,
  localparam int unsigned CW        = HostWidth + DataWidth + BeatBytes
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,
  // The window (static while the core runs)
  input  logic [AddrWidth-1:0] win_base_i,
  input  logic [AddrWidth-1:0] win_bytes_i,
  input  logic [HostWidth-1:0] host_delta_i,  // host address - HBM address, mod 2^HostWidth
  // The vector unit's writes in, packed as f2_port0.sv's: {addr, len, burst}, {data, strb, last}
  input  logic                 s_aw_valid_i,
  output logic                 s_aw_ready_o,
  input  logic [AxW-1:0]       s_aw_i,
  input  logic                 s_w_valid_i,
  output logic                 s_w_ready_o,
  input  logic [WW-1:0]        s_w_i,
  // ... and out to port 0, unchanged
  output logic                 m_aw_valid_o,
  input  logic                 m_aw_ready_i,
  output logic [AxW-1:0]       m_aw_o,
  output logic                 m_w_valid_o,
  input  logic                 m_w_ready_i,
  output logic [WW-1:0]        m_w_o,
  // The copies: {host address, data, strb}, a beat each
  output logic                 c_valid_o,
  input  logic                 c_ready_i,
  output logic [CW-1:0]        c_o
);

  localparam int unsigned QW = AddrWidth + HostWidth;  // {its offset in the window, its host address}
  localparam int unsigned BB = $clog2(BeatBytes);

  if (BeatBytes < 2 || 2 ** $clog2(BeatBytes) != BeatBytes) begin : gen_bad_width
    $error("f2_pcim: DataWidth must be a power-of-two number of bytes");
  end

  logic [AddrWidth-1:0] base_q, bytes_q;
  logic [HostWidth-1:0] delta_q;
  always_ff @(posedge clk_i) begin
    base_q  <= win_base_i;
    bytes_q <= win_bytes_i;
    delta_q <= host_delta_i;
  end

  // --- AW: to port 0 and to the queue -------------------------------------------

  logic [AddrWidth-1:0] aw_addr, aw_off;
  logic                 aw_m_done_q, aw_q_done_q;  // the request taken on that way
  logic                 q_in_valid, q_in_ready, q_out_valid, q_out_ready;
  logic [QW-1:0]        q_in, q_out;

  assign aw_addr      = s_aw_i[AxW-1 -: AddrWidth];
  assign aw_off       = aw_addr - base_q;  // below the base: wraps to above any window
  assign m_aw_valid_o = s_aw_valid_i && !aw_m_done_q;
  assign m_aw_o       = s_aw_i;
  assign q_in_valid   = s_aw_valid_i && !aw_q_done_q;
  assign q_in         = {aw_off, delta_q + HostWidth'(aw_addr)};
  assign s_aw_ready_o = (aw_m_done_q || m_aw_ready_i) && (aw_q_done_q || q_in_ready);

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      {aw_m_done_q, aw_q_done_q} <= '0;
    end else if (s_aw_valid_i && s_aw_ready_o) begin
      {aw_m_done_q, aw_q_done_q} <= '0;
    end else begin
      if (m_aw_valid_o && m_aw_ready_i) aw_m_done_q <= 1'b1;
      if (q_in_valid && q_in_ready) aw_q_done_q <= 1'b1;
    end
  end

  // verilator lint_off PINCONNECTEMPTY
  stream_fifo #(.Width(QW), .Depth(Depth)) u_queue (
    .clk_i,
    .rst_ni,
    .in_valid_i (q_in_valid),
    .in_ready_o (q_in_ready),
    .in_data_i  (q_in),
    .out_valid_o(q_out_valid),
    .out_ready_i(q_out_ready),
    .out_data_o (q_out),
    .count_o    ()
  );
  // verilator lint_on PINCONNECTEMPTY

  // --- W: to port 0 and, inside the window, to the copies -----------------------

  logic                 copy, w_last, w_fire;
  logic [AddrWidth-1:0] off;  // the burst's
  logic [HostWidth-1:0] host;
  logic                 w_m_done_q, w_c_done_q;
  logic [7:0]           beat_q;  // the beat's index in its burst

  assign {off, host}  = q_out;
  assign copy         = off + AddrWidth'({beat_q, BB'(0)}) < bytes_q;
  assign w_last       = s_w_i[0];
  assign m_w_valid_o  = s_w_valid_i && q_out_valid && !w_m_done_q;
  assign m_w_o        = s_w_i;
  assign c_valid_o    = s_w_valid_i && q_out_valid && copy && !w_c_done_q;
  assign c_o          = {host + HostWidth'({beat_q, BB'(0)}), s_w_i[WW-1:1]};
  assign s_w_ready_o  = q_out_valid && (w_m_done_q || m_w_ready_i) && (!copy || w_c_done_q || c_ready_i);
  assign w_fire       = s_w_valid_i && s_w_ready_o;
  assign q_out_ready  = w_fire && w_last;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      {w_m_done_q, w_c_done_q} <= '0;
      beat_q                   <= '0;
    end else if (w_fire) begin
      {w_m_done_q, w_c_done_q} <= '0;
      beat_q                   <= w_last ? '0 : beat_q + 1'b1;
    end else begin
      if (m_w_valid_o && m_w_ready_i) w_m_done_q <= 1'b1;
      if (c_valid_o && c_ready_i) w_c_done_q <= 1'b1;
    end
  end

endmodule
