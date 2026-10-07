// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Timing harness for the lane's accumulate loop (docs/architecture.md,
// "Requirement for M3"): acc = fma(up(w), up(x), acc), with the FMA's
// NumPipeRegs registers plus the accumulator register in the loop, so the
// loop has NumPipeRegs + 1 cycles. Inputs and output are registered, so every
// path is register to register. Not part of the design: M4 builds the lane.

module acc_loop #(
  parameter int unsigned NumPipeRegs = 3
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  logic [15:0] w_i,
  input  logic [15:0] x_i,
  output logic [31:0] acc_o
);

  logic [15:0] w_q, x_q;
  logic        valid_q;
  always_ff @(posedge clk_i) begin
    w_q     <= w_i;
    x_q     <= x_i;
    valid_q <= valid_i;
  end

  logic [31:0] w_f32, x_f32, sum;
  logic        sum_valid;
  bf16_to_fp32 u_up_w (.a_i(w_q), .y_o(w_f32));
  bf16_to_fp32 u_up_x (.a_i(x_q), .y_o(x_f32));

  logic [31:0] acc_q;
  fp32_fma #(
    .NumPipeRegs(NumPipeRegs)
  ) u_fma (
    .clk_i,
    .rst_ni,
    .valid_i(valid_q),
    .op_i   (2'd0),  // fma
    .a_i    (w_f32),
    .b_i    (x_f32),
    .c_i    (acc_q),
    .valid_o(sum_valid),
    .y_o    (sum)
  );

  always_ff @(posedge clk_i) begin
    if (!rst_ni) acc_q <= '0;
    else if (sum_valid) acc_q <= sum;
  end

  assign acc_o = acc_q;

endmodule
