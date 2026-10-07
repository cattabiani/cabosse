// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// BF16 x BF16 for the lanes' multiply-add (D-038): up(w) * up(x) needs no
// rounding, so the product keeps all 16 bits of its significand and an
// exponent wider than FP32's. A product below FP32's smallest exponent is
// shifted to exponent 1 with a sticky bit (bf16_mac_pkg::product_t), which
// lets mac_add order its operands by exponent and still round once, exactly
// as fma(up(w), up(x), acc). Subnormal inputs are kept; Inf * 0 is NaN.
// Combinational: the lane registers the product, outside the accumulate loop.

module bf16_mul (
  input  logic [15:0]            w_i,
  input  logic [15:0]            x_i,
  output bf16_mac_pkg::product_t p_o
);

  localparam logic [7:0]  ExpMax = 8'hFF;
  localparam int unsigned W      = 27;  // significand, guard, round, sticky

  logic [7:0] ew, ex;
  logic [6:0] fw, fx;
  assign {ew, fw} = w_i[14:0];
  assign {ex, fx} = x_i[14:0];

  logic w_zero, x_zero, w_inf, x_inf, w_nan, x_nan;
  assign w_zero = (ew == 0) && (fw == 0);
  assign x_zero = (ex == 0) && (fx == 0);
  assign w_inf  = (ew == ExpMax) && (fw == 0);
  assign x_inf  = (ex == ExpMax) && (fx == 0);
  assign w_nan  = (ew == ExpMax) && (fw != 0);
  assign x_nan  = (ex == ExpMax) && (fx != 0);

  // Significands with the hidden bit (0 for a subnormal, whose exponent is 1).
  logic [7:0]  mw, mx;
  logic [15:0] prod;
  assign mw   = {ew != 0, fw};
  assign mx   = {ex != 0, fx};
  assign prod = mw * mx;

  // Normalize: the leading one to bit 15 (then bit 26 of sig).
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

  // up(w) = mw * 2^(ew' - 134) with ew' = max(ew, 1), so the product is
  // prod * 2^(ew' + ex' - 268) = (prod << (11 + lz)) * 2^(exp - 153) with
  // exp = ew' + ex' - 126 - lz.
  logic signed [10:0] ew_eff, ex_eff, exp;
  logic [W-1:0]       sig;
  assign ew_eff = (ew == 0) ? 11'sd1 : 11'(signed'({1'b0, ew}));
  assign ex_eff = (ex == 0) ? 11'sd1 : 11'(signed'({1'b0, ex}));
  assign exp    = ew_eff + ex_eff - 11'sd126 - 11'(signed'({1'b0, lz}));
  assign sig    = {prod << lz, 11'h000};

  // Below exponent 1: shift right to it, the bits shifted out into sticky.
  logic [10:0]    below;
  logic [4:0]     shift;
  logic [2*W-1:0] shifted;
  assign below   = 11'(11'sd1 - exp);
  assign shift   = (exp >= 11'sd1) ? 5'd0 : (below > 11'(W)) ? 5'(W + 1) : below[4:0];
  assign shifted = {sig, W'(0)} >> shift;

  always_comb begin
    p_o.sign   = w_i[15] ^ x_i[15];
    p_o.exp    = (exp >= 11'sd1 && !prod_zero) ? exp : 11'sd1;  // a zero at 1, as FP32's
    p_o.sig    = {shifted[2*W-1:W+1], shifted[W] | (|shifted[W-1:0])};
    p_o.is_nan = w_nan || x_nan || (w_inf && x_zero) || (w_zero && x_inf);
    p_o.is_inf = (w_inf || x_inf) && !p_o.is_nan;
  end

endmodule
