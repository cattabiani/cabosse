// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// a * b, exact, as an fp_add operand (fp_pkg's layout), for a and b in a
// format of M + 8 bits: a sign, FP32's 8-bit exponent and M - 1 fraction
// bits. M = 8 is BF16 (the lanes, D-038), M = 24 is FP32 (fp32_fma, D-039).
// The 2M-bit product of the significands keeps all its
// bits, normalized to the top of a W-bit significand, with an exponent wider
// than FP32's. A product below FP32's smallest exponent is shifted to
// exponent 1 with a sticky bit, which lets fp_add order its operands by
// exponent and still round once, exactly as an fma. Subnormal inputs are
// kept; Inf * 0 is NaN. Ftz = 1 flushes subnormal inputs to zero instead
// (fp_pkg::flush_f32); the product is still exact.
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
  parameter  bit          Ftz  = 1'b0,
  localparam int unsigned PW   = 2 * M,  // the product's bits
  localparam int unsigned W    = prod_sig_w(M),
  localparam int unsigned LzW  = $clog2(PW)
) (
  input  logic          clk_i,
  input  logic [M+7:0]  a_i,
  input  logic [M+7:0]  b_i,
  output logic [W+11:0] p_o
);

  // --- Multiply ---------------------------------------------------------------

  operand_t a, b;
  // Widened to FP32, exactly.
  assign a = decode_f32(flush_f32(32'(a_i) << (F32SigW - M), Ftz));
  assign b = decode_f32(flush_f32(32'(b_i) << (F32SigW - M), Ftz));

  typedef struct packed {
    logic          sign;
    logic [9:0]    exp_sum;  // a.exp + b.exp
    logic [PW-1:0] prod;     // of the significands
    logic          nan;
    logic          inf;
  } mul_t;
  mul_t mul_d, mul_q;  // mul_q is mul_d when Regs = 0

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
  if (Ftz) begin : gen_lz_ftz
    // Both significands are normal or zero, so the product is zero or in
    // [1, 4): its leading one is in the top two bits.
    assign lz        = LzW'(!mul_q.prod[PW-1]);
    assign prod_zero = !mul_q.prod[PW-1] && !mul_q.prod[PW-2];
  end else begin : gen_lz
    leading_zeros #(
      .Width(PW)
    ) u_lz (
      .in_i   (mul_q.prod),
      .cnt_o  (lz),
      .empty_o(prod_zero)
    );
  end

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
  norm_t norm_d, norm_q;  // norm_q is norm_d when Regs < 2
  always_comb begin
    norm_d.sign = mul_q.sign;
    norm_d.exp  = 11'(mul_q.exp_sum) - 11'sd126 - 11'(lz);
    norm_d.sig  = {mul_q.prod << lz, (W - PW)'(0)};
    norm_d.zero = prod_zero;
    norm_d.nan  = mul_q.nan;
    norm_d.inf  = mul_q.inf;
  end

  // --- Shift below exponent 1, pack ----------------------------------------------

  logic [W-1:0] sig_below;
  logic [8:0]   below;  // 1 - exp: how far below exponent 1
  assign below = 9'(11'sd1 - norm_q.exp);
  sticky_shift #(
    .Width(W)
  ) u_below (
    .in_i   (norm_q.sig),
    .shift_i(below),
    .out_o  (sig_below)
  );

  logic [8:0] p_exp;  // a zero at exponent 1, as FP32's
  assign p_exp = (norm_q.exp >= 11'sd1 && !norm_q.zero) ? norm_q.exp[8:0] : 9'd1;
  assign p_o   = {
    norm_q.sign, p_exp, (norm_q.exp >= 11'sd1) ? norm_q.sig : sig_below, norm_q.inf, norm_q.nan
  };

  // --- Registers ------------------------------------------------------------------

  if (Regs > 2) begin : gen_bad_regs
    $error("fp_product: Regs must be 0, 1 or 2");
  end
  if (M > F32SigW) begin : gen_bad_m
    $error("fp_product: M must be at most 24, FP32's significand");
  end

  if (Regs >= 1) begin : gen_mul_q
    always_ff @(posedge clk_i) mul_q <= mul_d;
  end else begin : gen_mul_comb
    logic unused_clk;  // no register, no clock
    assign unused_clk = clk_i;
    assign mul_q = mul_d;
  end

  if (Regs >= 2) begin : gen_norm_q
    always_ff @(posedge clk_i) norm_q <= norm_d;
  end else begin : gen_norm_comb
    assign norm_q = norm_d;
  end

endmodule
