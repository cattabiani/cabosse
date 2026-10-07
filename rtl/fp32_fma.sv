// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// FP32 fused multiply-add, add and multiply, round to nearest even, one
// rounding per operation (docs/numerics.md, section 2; D-039). Golden model:
// golden.arith.fma, add, mul.
//   OpAdd: y = a + b      = fma(a, 1, b)
//   OpMul: y = a * b      = fma(a, b, -0)  (x + -0 = x for every x, -0 too)
//   any other op_i (fma): y = a * b + c
// fp32_product, a register, then fp_add with W = 51: a result leaves 4
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

  localparam int unsigned W     = 51;
  localparam logic [1:0]  OpAdd = 2'd1;
  localparam logic [1:0]  OpMul = 2'd2;
  localparam logic [31:0] One   = 32'h3F80_0000;
  localparam logic [31:0] NegZ  = 32'h8000_0000;

  logic [31:0] x, y, z;  // y = x * y' + z, the operands of the fma
  always_comb begin
    {x, y, z} = {a_i, b_i, c_i};
    case (op_i)
      OpAdd:   {x, y, z} = {a_i, One, b_i};
      OpMul:   {x, y, z} = {a_i, b_i, NegZ};
      default: ;
    endcase
  end

  logic [W+11:0] p_d, p_q;
  fp32_product u_mul (
    .a_i(x),
    .b_i(y),
    .p_o(p_d)
  );

  // The addend in the product's layout: FP32's significand at the top.
  operand_t      z_d;
  logic [W+11:0] z_q;
  logic          valid_q;
  assign z_d = decode_f32(z);
  always_ff @(posedge clk_i) begin
    p_q     <= p_d;
    z_q     <= {z_d.sign, z_d.exp, z_d.sig, 24'h0, z_d.is_inf, z_d.is_nan};
    valid_q <= rst_ni && valid_i;
  end

  fp_add #(
    .W(W)
  ) u_add (
    .clk_i,
    .rst_ni,
    .valid_i(valid_q),
    .a_i    (p_q),
    .b_i    (z_q),
    .valid_o,
    .y_o
  );

endmodule
