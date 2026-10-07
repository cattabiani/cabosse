// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Timing harness for the vector unit's FP32 fma (D-039): InRegs registers on
// the inputs, then fp32_fma (4 cycles), and the result registered. No loop:
// with retiming, Vivado may move the extra input registers into the fma's
// stages, which shows whether one more stage would close 250 MHz. Every path
// is register to register. Not part of the design: M6 builds the vector unit.

module fma_path #(
  parameter int unsigned InRegs = 1
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

  typedef struct packed {
    logic        valid;
    logic [1:0]  op;
    logic [31:0] a;
    logic [31:0] b;
    logic [31:0] c;
  } in_t;

  in_t in_q[InRegs+1];
  assign in_q[0] = '{valid: valid_i, op: op_i, a: a_i, b: b_i, c: c_i};
  for (genvar i = 0; i < InRegs; i++) begin : gen_in
    always_ff @(posedge clk_i) in_q[i+1] <= in_q[i];
  end

  logic        valid;
  logic [31:0] y;
  fp32_fma u_fma (
    .clk_i,
    .rst_ni,
    .valid_i(in_q[InRegs].valid),
    .op_i   (in_q[InRegs].op),
    .a_i    (in_q[InRegs].a),
    .b_i    (in_q[InRegs].b),
    .c_i    (in_q[InRegs].c),
    .valid_o(valid),
    .y_o    (y)
  );

  always_ff @(posedge clk_i) begin
    y_o     <= y;
    valid_o <= valid;
  end

endmodule
