// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// a * b, exact, as an fp_add operand (fp_pkg's layout), for FP32 values whose
// significands (hidden bit included) fit in their top M bits: M = 8 for BF16
// values widened to FP32 (the lanes, D-038), M = 24 for any FP32 value
// (fp32_fma, D-039). The 2M-bit product of the significands keeps all its
// bits, normalized to the top of a W-bit significand, with an exponent wider
// than FP32's. A product below FP32's smallest exponent is shifted to
// exponent 1 with a sticky bit, which lets fp_add order its operands by
// exponent and still round once, exactly as an fma. Subnormal inputs are
// kept; Inf * 0 is NaN. Combinational.

module fp_product
  import fp_pkg::*;
#(
  parameter  int unsigned M   = 24,
  localparam int unsigned PW  = 2 * M,                // the product's bits
  localparam int unsigned W   = (PW > 24 ? PW : 24) + 3,  // and guard, round, sticky
  localparam int unsigned LzW = $clog2(PW)
) (
  input  logic [31:0]   a_i,
  input  logic [31:0]   b_i,
  output logic [W+11:0] p_o
);

  operand_t a, b;
  assign a = decode_f32(a_i);
  assign b = decode_f32(b_i);

  logic [M-1:0]  ma, mb;
  logic [PW-1:0] prod;
  assign ma   = a.sig[SigW-1-:M];
  assign mb   = b.sig[SigW-1-:M];
  assign prod = ma * mb;

  logic [LzW-1:0] lz;
  logic           prod_zero;
  leading_zeros #(
    .Width(PW)
  ) u_lz (
    .in_i   (prod),
    .cnt_o  (lz),
    .empty_o(prod_zero)
  );

  // a = ma * 2^(a.exp - 126 - M), so the product is
  // prod * 2^(a.exp + b.exp - 252 - 2M) = sig * 2^(exp - 126 - W) with
  // sig = prod << lz, padded to W bits, and exp = a.exp + b.exp - 126 - lz:
  // from -124 - (2M - 1) (both smallest subnormals) to 382.
  logic signed [10:0] exp;
  logic [W-1:0]       sig, sig_below;
  logic [8:0]         below;  // 1 - exp: how far below exponent 1
  assign exp   = 11'(a.exp) + 11'(b.exp) - 11'sd126 - 11'(lz);
  assign sig   = {prod << lz, (W - PW)'(0)};
  assign below = 9'(11'sd1 - exp);

  sticky_shift #(
    .Width(W)
  ) u_below (
    .in_i   (sig),
    .shift_i(below),
    .out_o  (sig_below)
  );

  logic       nan, inf;
  logic [8:0] p_exp;
  assign nan   = a.is_nan || b.is_nan || (a.is_inf && mb == 0) || (ma == 0 && b.is_inf);
  assign inf   = (a.is_inf || b.is_inf) && !nan;
  assign p_exp = (exp >= 11'sd1 && !prod_zero) ? exp[8:0] : 9'd1;  // a zero at 1, as FP32's
  assign p_o   = {a.sign ^ b.sign, p_exp, (exp >= 11'sd1) ? sig : sig_below, inf, nan};

endmodule
