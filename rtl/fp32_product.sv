// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// FP32 x FP32, exact, for fp32_fma (D-039): the 48-bit product of the
// significands, normalized, as an fp_add operand with W = 51 (fp_pkg's
// layout). A product below exponent 1 is shifted to it with a sticky bit, as
// in bf16_mul. Subnormal inputs are kept; Inf * 0 is NaN. Combinational.

module fp32_product
  import fp_pkg::*;
#(
  localparam int unsigned W = 51  // the 48-bit product, guard, round, sticky
) (
  input  logic [31:0]  a_i,
  input  logic [31:0]  b_i,
  output logic [W+11:0] p_o
);

  operand_t a, b;
  assign a = decode_f32(a_i);
  assign b = decode_f32(b_i);

  // The 24-bit significands (hidden bit included) and their exact product.
  logic [23:0] ma, mb;
  logic [47:0] prod;
  assign ma   = a.sig[SigW-1-:24];
  assign mb   = b.sig[SigW-1-:24];
  assign prod = ma * mb;

  logic [5:0] lz;
  logic       prod_zero;
  leading_zeros #(
    .Width(48)
  ) u_lz (
    .in_i   (prod),
    .cnt_o  (lz),
    .empty_o(prod_zero)
  );

  // a = ma * 2^(a.exp - 150), so the product is prod * 2^(a.exp + b.exp - 300)
  // = (prod << (3 + lz)) * 2^(exp - 177) with exp = a.exp + b.exp - 126 - lz,
  // from -171 (both smallest subnormals) to 382.
  logic signed [10:0] exp;
  logic [W-1:0]       sig, sig_below;
  logic [8:0]         below;  // 1 - exp: how far below exponent 1
  assign exp   = 11'(a.exp) + 11'(b.exp) - 11'sd126 - 11'(lz);
  assign sig   = {prod << lz, 3'b000};
  assign below = 9'(11'sd1 - exp);

  sticky_shift #(
    .Width(W)
  ) u_below (
    .in_i   (sig),
    .shift_i(below),
    .out_o  (sig_below)
  );

  logic         nan, inf;
  logic [8:0]   p_exp;
  assign nan   = a.is_nan || b.is_nan || (a.is_inf && mb == 0) || (ma == 0 && b.is_inf);
  assign inf   = (a.is_inf || b.is_inf) && !nan;
  assign p_exp = (exp >= 11'sd1 && !prod_zero) ? exp[8:0] : 9'd1;  // a zero at 1
  assign p_o   = {a.sign ^ b.sign, p_exp, (exp >= 11'sd1) ? sig : sig_below, inf, nan};

endmodule
