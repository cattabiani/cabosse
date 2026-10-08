// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Timing harness for the lane (rtl/dot_lane.sv, M4): every input and output
// registered, so every path is register to register, the accumulate loop
// with its accumulator file and the final sum's contexts included. Not part
// of the design.

module lane_path #(
  parameter  int unsigned MulRegs = 1,
  localparam int unsigned E       = 4
) (
  input  logic               clk_i,
  input  logic               rst_ni,
  input  logic               in_valid_i,
  output logic               in_ready_o,
  input  logic [E-1:0][15:0] w_i,
  input  logic [E-1:0][15:0] x_i,
  input  logic [E-1:0]       mask_i,
  input  logic               last_i,
  output logic               out_valid_o,
  input  logic               out_ready_i,
  output logic [31:0]        out_o
);

  logic               rst_nq, in_valid_q, last_q, out_ready_q;
  logic [E-1:0][15:0] w_q, x_q;
  logic [E-1:0]       mask_q;
  always_ff @(posedge clk_i) begin
    rst_nq      <= rst_ni;
    in_valid_q  <= in_valid_i;
    w_q         <= w_i;
    x_q         <= x_i;
    mask_q      <= mask_i;
    last_q      <= last_i;
    out_ready_q <= out_ready_i;
  end

  logic        in_ready, out_valid;
  logic [31:0] out;
  dot_lane #(
    .E      (E),
    .MulRegs(MulRegs)
  ) u_lane (
    .clk_i,
    .rst_ni     (rst_nq),
    .in_valid_i (in_valid_q),
    .in_ready_o (in_ready),
    .w_i        (w_q),
    .x_i        (x_q),
    .mask_i     (mask_q),
    .last_i     (last_q),
    .out_valid_o(out_valid),
    .out_ready_i(out_ready_q),
    .out_o      (out)
  );

  always_ff @(posedge clk_i) begin
    in_ready_o  <= in_ready;
    out_valid_o <= out_valid;
    out_o       <= out;
  end

endmodule
