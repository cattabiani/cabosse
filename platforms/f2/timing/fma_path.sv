// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Timing harness for the vector unit's FP32 fma (D-039): registered inputs,
// fp32_fma with MulRegs registers in its product (MulRegs + 4 cycles), and
// the result registered, so every path is register to register. No loop: the
// product's registers only add latency. Not part of the design: M6 builds the
// vector unit.

module fma_path #(
  parameter int unsigned MulRegs = 2
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  logic [1:0]  op_i,
  input  logic [31:0] a_i,
  input  logic [31:0] b_i,
  input  logic [31:0] c_i,
  output logic        valid_o,
  output logic [31:0] y_o
);

  logic        valid_q;
  logic [1:0]  op_q;
  logic [31:0] a_q, b_q, c_q;
  always_ff @(posedge clk_i) begin
    valid_q <= valid_i;
    op_q    <= op_i;
    a_q     <= a_i;
    b_q     <= b_i;
    c_q     <= c_i;
  end

  logic        valid;
  logic [31:0] y;
  fp32_fma #(
    .MulRegs(MulRegs)
  ) u_fma (
    .clk_i,
    .rst_ni,
    .valid_i(valid_q),
    .op_i   (op_q),
    .a_i    (a_q),
    .b_i    (b_q),
    .c_i    (c_q),
    .valid_o(valid),
    .y_o    (y)
  );

  always_ff @(posedge clk_i) begin
    y_o     <= y;
    valid_o <= valid;
  end

endmodule
