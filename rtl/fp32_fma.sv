// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// FP32 fused multiply-add, add and multiply, round to nearest even, one
// rounding per operation (docs/numerics.md, section 2; D-039). Golden model:
// golden.arith.fma, add, mul.
//   Fp32Add: y = a + b    = fma(a, 1, b)       (fp_pkg's op codes)
//   Fp32Mul: y = a * b    = fma(a, b, -0)  (x + -0 = x for every x, -0 too)
//   any other op_i (fma): y = a * b + c
// fp_mul_add with M = 24 and MulRegs registers in the product (D-039:
// 250 MHz on F2 needs them): a result leaves MulRegs + 4 cycles after its
// operands enter; valid_o marks it.

module fp32_fma
  import fp_pkg::*;
#(
  parameter int unsigned MulRegs = 2,
  parameter bit          Ftz     = 1'b0  // flush-to-zero variant (fp_mul_add)
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

  localparam logic [31:0] NegZ = 32'h8000_0000;

  logic [31:0] fa, fb, fc;  // the fma's operands: fa * fb + fc
  always_comb begin
    {fa, fb, fc} = {a_i, b_i, c_i};
    case (op_i)
      Fp32Add: {fa, fb, fc} = {a_i, FP32One, b_i};
      Fp32Mul: {fa, fb, fc} = {a_i, b_i, NegZ};
      default: ;
    endcase
  end

  fp_mul_add #(
    .M      (24),
    .MulRegs(MulRegs),
    .Ftz    (Ftz)
  ) u_fma (
    .clk_i,
    .rst_ni,
    .valid_i,
    .a_i(fa),
    .b_i(fb),
    .c_i(fc),
    .valid_o,
    .y_o
  );

endmodule
