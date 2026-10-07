// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// BF16 x BF16 for the lanes' multiply-add (D-038): up(w) * up(x) needs no
// rounding, so the product keeps all 16 bits of its significand and an
// exponent wider than FP32's. A product below FP32's smallest exponent is
// shifted to exponent 1 with a sticky bit (bf16_mac_pkg::operand_t), which
// lets mac_add order its operands by exponent and still round once, exactly
// as fma(up(w), up(x), acc). Subnormal inputs are kept; Inf * 0 is NaN.
// Combinational: the lane registers the product, outside the accumulate loop.

module bf16_mul
  import bf16_mac_pkg::*;
(
  input  logic [15:0] w_i,
  input  logic [15:0] x_i,
  output operand_t    p_o
);

  operand_t w, x;
  assign w = decode_f32({w_i, 16'h0});  // up(w)
  assign x = decode_f32({x_i, 16'h0});

  // The 8-bit significands (hidden bit included) and their exact product.
  logic [7:0]  mw, mx;
  logic [15:0] prod;
  assign mw   = w.sig[W-1-:8];
  assign mx   = x.sig[W-1-:8];
  assign prod = mw * mx;

  // Normalize: the leading one to bit 15 (then bit W-1 of sig).
  logic [3:0] lz;
  logic       prod_zero;
  lzc #(
    .WIDTH(16),
    .MODE (1'b1)
  ) u_lzc (
    .in_i   (prod),
    .cnt_o  (lz),
    .empty_o(prod_zero)
  );

  // up(w) = mw * 2^(w.exp - 134), so the product is
  // prod * 2^(w.exp + x.exp - 268) = (prod << (11 + lz)) * 2^(exp - 153) with
  // exp = w.exp + x.exp - 126 - lz, from -139 (both smallest subnormals) to 382.
  logic signed [10:0] exp;
  logic [W-1:0]       sig;
  logic [8:0]         below;  // 1 - exp: how far below exponent 1 (up to 140)
  assign exp   = 11'(w.exp) + 11'(x.exp) - 11'sd126 - 11'(lz);
  assign sig   = {prod << lz, 11'h000};
  assign below = 9'(11'sd1 - exp);

  always_comb begin
    p_o.sign   = w.sign ^ x.sign;
    p_o.exp    = (exp >= 11'sd1 && !prod_zero) ? exp[8:0] : 9'd1;  // a zero at 1, as FP32's
    p_o.sig    = (exp >= 11'sd1) ? sig : shift_right_sticky(sig, below);
    p_o.is_nan = w.is_nan || x.is_nan || (w.is_inf && mx == 0) || (mw == 0 && x.is_inf);
    p_o.is_inf = (w.is_inf || x.is_inf) && !p_o.is_nan;
  end

endmodule
