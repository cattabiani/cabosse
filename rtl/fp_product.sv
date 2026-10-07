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
// kept; Inf * 0 is NaN.
//
// Three steps: multiply | normalize | shift below exponent 1. Regs registers
// split them (0: combinational; 1: after the multiply, which lets Vivado use
// the DSPs' own register; 2: after normalizing too). A result leaves Regs
// cycles after its operands enter.

module fp_product
  import fp_pkg::*;
#(
  parameter  int unsigned M    = 24,
  parameter  int unsigned Regs = 0,
  localparam int unsigned PW   = 2 * M,  // the product's bits
  localparam int unsigned W    = prod_sig_w(M),
  localparam int unsigned LzW  = $clog2(PW)
) (
  input  logic          clk_i,
  input  logic [31:0]   a_i,
  input  logic [31:0]   b_i,
  output logic [W+11:0] p_o
);

  // --- Multiply ---------------------------------------------------------------

  operand_t a, b;
  assign a = decode_f32(a_i);
  assign b = decode_f32(b_i);

  typedef struct packed {
    logic          sign;
    logic [9:0]    exp_sum;  // a.exp + b.exp
    logic [PW-1:0] prod;     // of the significands
    logic          nan;
    logic          inf;
  } mul_t;
  mul_t mul_d, mul;

  logic [M-1:0] ma, mb;
  assign ma = a.sig[SigW-1-:M];
  assign mb = b.sig[SigW-1-:M];
  always_comb begin
    mul_d.sign    = a.sign ^ b.sign;
    mul_d.exp_sum = 10'(a.exp) + 10'(b.exp);
    mul_d.prod    = ma * mb;
    mul_d.nan     = a.is_nan || b.is_nan || (a.is_inf && mb == 0) || (ma == 0 && b.is_inf);
    mul_d.inf     = (a.is_inf || b.is_inf) && !mul_d.nan;
  end

  // --- Normalize ----------------------------------------------------------------

  logic [LzW-1:0] lz;
  logic           prod_zero;
  leading_zeros #(
    .Width(PW)
  ) u_lz (
    .in_i   (mul.prod),
    .cnt_o  (lz),
    .empty_o(prod_zero)
  );

  // a = ma * 2^(a.exp - 126 - M), so the product is
  // prod * 2^(a.exp + b.exp - 252 - 2M) = sig * 2^(exp - 126 - W) with
  // sig = prod << lz, padded to W bits, and exp = a.exp + b.exp - 126 - lz:
  // from -124 - (2M - 1) (both smallest subnormals) to 382.
  typedef struct packed {
    logic               sign;
    logic signed [10:0] exp;
    logic [W-1:0]       sig;
    logic               zero;
    logic               nan;
    logic               inf;
  } norm_t;
  norm_t norm_d, norm;
  always_comb begin
    norm_d.sign = mul.sign;
    norm_d.exp  = 11'(mul.exp_sum) - 11'sd126 - 11'(lz);
    norm_d.sig  = {mul.prod << lz, (W - PW)'(0)};
    norm_d.zero = prod_zero;
    norm_d.nan  = mul.nan;
    norm_d.inf  = mul.inf;
  end

  // --- Shift below exponent 1, pack ----------------------------------------------

  logic [W-1:0] sig_below;
  logic [8:0]   below;  // 1 - exp: how far below exponent 1
  assign below = 9'(11'sd1 - norm.exp);
  sticky_shift #(
    .Width(W)
  ) u_below (
    .in_i   (norm.sig),
    .shift_i(below),
    .out_o  (sig_below)
  );

  logic [8:0] p_exp;
  assign p_exp = (norm.exp >= 11'sd1 && !norm.zero) ? norm.exp[8:0] : 9'd1;  // a zero at 1, as FP32's
  assign p_o   = {norm.sign, p_exp, (norm.exp >= 11'sd1) ? norm.sig : sig_below, norm.inf, norm.nan};

  // --- Registers ------------------------------------------------------------------

  if (Regs > 2) begin : gen_bad_regs
    $error("fp_product: Regs must be 0, 1 or 2");
  end

  if (Regs >= 1) begin : gen_mul_q
    always_ff @(posedge clk_i) mul <= mul_d;
  end else begin : gen_mul_comb
    logic unused_clk;  // no register, no clock
    assign unused_clk = clk_i;
    assign mul = mul_d;
  end

  if (Regs >= 2) begin : gen_norm_q
    always_ff @(posedge clk_i) norm <= norm_d;
  end else begin : gen_norm_comb
    assign norm = norm_d;
  end

endmodule
