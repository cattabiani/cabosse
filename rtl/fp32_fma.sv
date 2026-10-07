// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// FP32 fused multiply-add, add and multiply, round to nearest even, one
// rounding per operation (docs/numerics.md, section 2; D-039). Golden model:
// golden.arith.fma, add, mul.
//   OpAdd: y = a + b      = fma(a, 1, b)
//   OpMul: y = a * b      = fma(a, b, -0)  (x + -0 = x for every x, -0 too)
//   any other op_i (fma): y = a * b + c
// fp_product with M = 24, a register, then fp_add with W = 51: a result leaves 4
// cycles after its operands enter; valid_o marks it.

module fp32_fma
  import fp_pkg::*;
(
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

  localparam int unsigned W     = 51;  // fp_product's, for M = 24
  localparam logic [1:0]  OpAdd = 2'd1;
  localparam logic [1:0]  OpMul = 2'd2;
  localparam logic [31:0] One   = 32'h3F80_0000;
  localparam logic [31:0] NegZ  = 32'h8000_0000;

  logic [31:0] fa, fb, fc;  // the fma's operands: fa * fb + fc
  always_comb begin
    {fa, fb, fc} = {a_i, b_i, c_i};
    case (op_i)
      OpAdd:   {fa, fb, fc} = {a_i, One, b_i};
      OpMul:   {fa, fb, fc} = {a_i, b_i, NegZ};
      default: ;
    endcase
  end

  logic [W+11:0] p_d, p_q;
  fp_product #(
    .M(24)
  ) u_mul (
    .a_i(fa),
    .b_i(fb),
    .p_o(p_d)
  );

  // The addend in the product's layout: FP32's significand at the top.
  operand_t      fc_op;
  logic [W+11:0] fc_q;
  logic          valid_q;
  assign fc_op = decode_f32(fc);
  always_ff @(posedge clk_i) begin
    p_q     <= p_d;
    fc_q    <= {fc_op.sign, fc_op.exp, fc_op.sig, 24'h0, fc_op.is_inf, fc_op.is_nan};
    valid_q <= rst_ni && valid_i;
  end

  fp_add #(
    .W(W)
  ) u_add (
    .clk_i,
    .rst_ni,
    .valid_i(valid_q),
    .a_i    (p_q),
    .b_i    (fc_q),
    .valid_o,
    .y_o
  );

endmodule
