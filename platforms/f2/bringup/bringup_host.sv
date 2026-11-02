// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// HBM port 0 shared between the host (PCIS, through the CL top's
// SmartConnect) and port 0's test engine (platforms/f2/README.md,
// HOST_PORT). sel_i asks for the owner; sel_o is the owner: it turns to the
// host at once (the core asks only between runs, the engine idle) and back
// to the engine once the host's side is idle: no burst unanswered, its
// address or its data sent (W may go ahead of AW), and none offered. The host's channels each pass a 2-entry FIFO, a register stage
// on its long way from the shell. With the engine owning the port, the
// host's requests wait in their FIFOs.
//
// AXI: the subset of bringup_port.sv (full beats, INCR, one ID), plus the
// host's write strobes and RLAST.

module bringup_host #(
  parameter  int unsigned AddrWidth = 34,
  parameter  int unsigned DataWidth = 256,
  localparam int unsigned BeatBytes = DataWidth / 8,
  localparam int unsigned AxW       = AddrWidth + 8 + 2,           // address, length, burst
  localparam int unsigned WW        = DataWidth + BeatBytes + 1,   // data, strobes, last
  localparam int unsigned RW        = DataWidth + 2 + 1            // data, response, last
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,
  input  logic                 sel_i,  // 1: the host should own the port
  output logic                 sel_o,  // 1: the host owns it
  // The host's side
  input  logic                 h_ar_valid_i,
  output logic                 h_ar_ready_o,
  input  logic [AxW-1:0]       h_ar_i,      // {addr, len, burst}
  output logic                 h_r_valid_o,
  input  logic                 h_r_ready_i,
  output logic [RW-1:0]        h_r_o,       // {data, resp, last}
  input  logic                 h_aw_valid_i,
  output logic                 h_aw_ready_o,
  input  logic [AxW-1:0]       h_aw_i,
  input  logic                 h_w_valid_i,
  output logic                 h_w_ready_o,
  input  logic [WW-1:0]        h_w_i,       // {data, strb, last}
  output logic                 h_b_valid_o,
  input  logic                 h_b_ready_i,
  output logic [1:0]           h_b_resp_o,
  // The engine's side
  input  logic                 e_ar_valid_i,
  output logic                 e_ar_ready_o,
  input  logic [AxW-1:0]       e_ar_i,
  output logic                 e_r_valid_o,
  input  logic                 e_r_ready_i,
  input  logic                 e_aw_valid_i,
  output logic                 e_aw_ready_o,
  input  logic [AxW-1:0]       e_aw_i,
  input  logic                 e_w_valid_i,
  output logic                 e_w_ready_o,
  input  logic [WW-1:0]        e_w_i,
  output logic                 e_b_valid_o,
  input  logic                 e_b_ready_i,
  // The memory's side (R and B data go to both sides)
  output logic                 m_ar_valid_o,
  input  logic                 m_ar_ready_i,
  output logic [AxW-1:0]       m_ar_o,
  input  logic                 m_r_valid_i,
  output logic                 m_r_ready_o,
  input  logic [RW-1:0]        m_r_i,
  output logic                 m_aw_valid_o,
  input  logic                 m_aw_ready_i,
  output logic [AxW-1:0]       m_aw_o,
  output logic                 m_w_valid_o,
  input  logic                 m_w_ready_i,
  output logic [WW-1:0]        m_w_o,
  input  logic                 m_b_valid_i,
  output logic                 m_b_ready_o,
  input  logic [1:0]           m_b_resp_i
);

  logic           sel_q;
  logic           ar_valid, aw_valid, w_valid, r_ready, b_ready;  // the host's FIFOs, memory side
  logic [AxW-1:0] ar, aw;
  logic [WW-1:0]  w;
  // The host's bursts unanswered: reads by address, writes by address and by
  // last data beat. The SmartConnect lets at most 64 out, so 8 bits hold them.
  logic [7:0]     reads_q, writes_q, wlasts_q;

  assign sel_o = sel_q;

  // verilator lint_off PINCONNECTEMPTY
  // The FIFOs' counts go unread.
  stream_fifo #(.Width(AxW), .Depth(2)) u_ar (
    .clk_i, .rst_ni,
    .in_valid_i (h_ar_valid_i), .in_ready_o (h_ar_ready_o), .in_data_i (h_ar_i),
    .out_valid_o(ar_valid), .out_ready_i(sel_q && m_ar_ready_i), .out_data_o(ar), .count_o()
  );
  stream_fifo #(.Width(AxW), .Depth(2)) u_aw (
    .clk_i, .rst_ni,
    .in_valid_i (h_aw_valid_i), .in_ready_o (h_aw_ready_o), .in_data_i (h_aw_i),
    .out_valid_o(aw_valid), .out_ready_i(sel_q && m_aw_ready_i), .out_data_o(aw), .count_o()
  );
  stream_fifo #(.Width(WW), .Depth(2)) u_w (
    .clk_i, .rst_ni,
    .in_valid_i (h_w_valid_i), .in_ready_o (h_w_ready_o), .in_data_i (h_w_i),
    .out_valid_o(w_valid), .out_ready_i(sel_q && m_w_ready_i), .out_data_o(w), .count_o()
  );
  stream_fifo #(.Width(RW), .Depth(2)) u_r (
    .clk_i, .rst_ni,
    .in_valid_i (sel_q && m_r_valid_i), .in_ready_o (r_ready), .in_data_i (m_r_i),
    .out_valid_o(h_r_valid_o), .out_ready_i(h_r_ready_i), .out_data_o(h_r_o), .count_o()
  );
  stream_fifo #(.Width(2), .Depth(2)) u_b (
    .clk_i, .rst_ni,
    .in_valid_i (sel_q && m_b_valid_i), .in_ready_o (b_ready), .in_data_i (m_b_resp_i),
    .out_valid_o(h_b_valid_o), .out_ready_i(h_b_ready_i), .out_data_o(h_b_resp_o), .count_o()
  );
  // verilator lint_on PINCONNECTEMPTY

  assign m_ar_valid_o = sel_q ? ar_valid : e_ar_valid_i;
  assign m_ar_o       = sel_q ? ar : e_ar_i;
  assign m_aw_valid_o = sel_q ? aw_valid : e_aw_valid_i;
  assign m_aw_o       = sel_q ? aw : e_aw_i;
  assign m_w_valid_o  = sel_q ? w_valid : e_w_valid_i;
  assign m_w_o        = sel_q ? w : e_w_i;
  assign m_r_ready_o  = sel_q ? r_ready : e_r_ready_i;
  assign m_b_ready_o  = sel_q ? b_ready : e_b_ready_i;
  assign e_ar_ready_o = !sel_q && m_ar_ready_i;
  assign e_aw_ready_o = !sel_q && m_aw_ready_i;
  assign e_w_ready_o  = !sel_q && m_w_ready_i;
  assign e_r_valid_o  = !sel_q && m_r_valid_i;
  assign e_b_valid_o  = !sel_q && m_b_valid_i;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      sel_q    <= 1'b1;
      reads_q  <= '0;
      writes_q <= '0;
      wlasts_q <= '0;
    end else begin
      reads_q  <= reads_q + 8'(sel_q && ar_valid && m_ar_ready_i)
                - 8'(sel_q && m_r_valid_i && r_ready && m_r_i[0]);
      writes_q <= writes_q + 8'(sel_q && aw_valid && m_aw_ready_i) - 8'(sel_q && m_b_valid_i && b_ready);
      wlasts_q <= wlasts_q + 8'(sel_q && w_valid && m_w_ready_i && w[0])
                - 8'(sel_q && m_b_valid_i && b_ready);
      if (sel_i) sel_q <= 1'b1;
      else if (reads_q == '0 && writes_q == '0 && wlasts_q == '0 && !ar_valid && !aw_valid && !w_valid) begin
        sel_q <= 1'b0;
      end
    end
  end

endmodule
