// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// y = a + b with one rounding to FP32, round to nearest even, for an operand
// a in fp_pkg's layout with a W-bit significand (fp_product's) and an FP32
// value b (D-038, D-039). Subnormals are kept, Inf - Inf and any NaN give the
// canonical NaN, an exact zero sum is -0 only if both operands are -0.
// Ftz = 1 flushes a subnormal b and a subnormal result to zero
// (fp_pkg::flush_f32). fp_product flushes its own inputs, but a product
// below exponent 1 still reaches a exactly, so the fma rounds once.
// W = 27 (fp_pkg::SigW) adds an exact BF16 product; fp32_fma uses W = 51 for
// its exact FP32 products. The bits below the rounding point are guard (bit
// W-25) and sticky (the rest, ORed); the operands' own sticky bits stay below
// them.
//
// Three register stages: align | add, count leading zeros | normalize,
// round. A result leaves 3 cycles after its operands enter; valid_o marks it.

module fp_add
  import fp_pkg::*;
#(
  parameter  int unsigned W   = SigW,
  parameter  bit          Ftz = 1'b0,
  localparam int unsigned OpW = W + 12,  // fp_pkg::operand_t's layout
  localparam int unsigned LzW = $clog2(W)
) (
  input  logic           clk_i,
  input  logic           rst_ni,
  input  logic           valid_i,
  input  logic [OpW-1:0] a_i,
  input  logic [31:0]    b_i,
  output logic           valid_o,
  output logic [31:0]    y_o
);

  typedef struct packed {
    logic         sign;
    logic [8:0]   exp;
    logic [W-1:0] sig;
    logic         is_inf;
    logic         is_nan;
  } op_t;

  // --- Stage 1: specials, order by magnitude, align -------------------------

  // b in a's layout: FP32's significand at the top of W bits.
  op_t      a, b;
  operand_t b_f32;
  assign a     = a_i;
  assign b_f32 = decode_f32(flush_f32(b_i, Ftz));
  assign b     = {b_f32.sign, b_f32.exp, W'(b_f32.sig) << (W - SigW), b_f32.is_inf, b_f32.is_nan};

  // The shift needs only the exponents (equal exponents shift by 0); the
  // larger magnitude goes first, so the difference is never negative. Each
  // subtraction's borrow gives the exponents' order.
  logic         a_exp_hi, b_exp_hi, a_hi;
  logic [8:0]   a_minus_b, b_minus_a, diff;
  logic         hi_sign;
  logic [W-1:0] hi_sig, lo_sig, lo_aligned;
  assign {b_exp_hi, a_minus_b} = {1'b0, a.exp} - {1'b0, b.exp};
  assign {a_exp_hi, b_minus_a} = {1'b0, b.exp} - {1'b0, a.exp};
  assign diff = a_exp_hi ? a_minus_b : b_minus_a;
  assign a_hi = a_exp_hi || (!b_exp_hi && (a.sig > b.sig));
  assign {hi_sign, hi_sig} = a_hi ? {a.sign, a.sig} : {b.sign, b.sig};
  assign lo_sig = a_hi ? b.sig : a.sig;

  sticky_shift #(
    .Width(W)
  ) u_align (
    .in_i   (lo_sig),
    .shift_i(diff),
    .out_o  (lo_aligned)
  );

  logic nan_s1, inf_s1;
  assign nan_s1 = a.is_nan || b.is_nan || (a.is_inf && b.is_inf && (a.sign != b.sign));
  assign inf_s1 = (a.is_inf || b.is_inf) && !nan_s1;

  typedef struct packed {
    logic         valid;
    logic         sub;   // effective subtraction
    logic         sign;  // of the result (of the infinity, for inf)
    logic [8:0]   exp;
    logic [W-1:0] hi;
    logic [W-1:0] lo;
    logic         nan;
    logic         inf;
  } s1_t;
  s1_t s1_d, s1_q;
  always_comb begin
    s1_d.valid = valid_i;
    s1_d.sub   = a.sign != b.sign;
    s1_d.sign  = !inf_s1 ? hi_sign : a.is_inf ? a.sign : b.sign;
    s1_d.exp   = a_hi ? a.exp : b.exp;
    s1_d.hi    = hi_sig;
    s1_d.lo    = lo_aligned;
    s1_d.nan   = nan_s1;
    s1_d.inf   = inf_s1;
  end

  // --- Stage 2: add, count leading zeros -------------------------------------

  logic [W:0] sum;
  assign sum = s1_q.sub ? {1'b0, s1_q.hi} - {1'b0, s1_q.lo} : {1'b0, s1_q.hi} + {1'b0, s1_q.lo};

  logic [LzW-1:0] lz;
  logic           sum_low_zero;
  leading_zeros #(
    .Width(W)
  ) u_lz (
    .in_i   (sum[W-1:0]),
    .cnt_o  (lz),
    .empty_o(sum_low_zero)
  );

  typedef struct packed {
    logic           valid;
    logic           sign;
    logic [8:0]     exp;
    logic [W:0]     sum;
    logic [LzW-1:0] lz;
    logic           nan;
    logic           inf;
  } s2_t;
  s2_t s2_d, s2_q;
  always_comb begin
    s2_d.valid = s1_q.valid;
    // An exact zero from a subtraction is +0; from an addition both were -0
    // or both +0, and the sign stays.
    s2_d.sign  = s1_q.sign && !(s1_q.sub && sum_low_zero && !sum[W] && !s1_q.inf);
    s2_d.exp   = s1_q.exp;
    s2_d.sum   = sum;
    s2_d.lz    = lz;
    s2_d.nan   = s1_q.nan;
    s2_d.inf   = s1_q.inf;
  end

  // --- Stage 3: normalize, round to nearest even, pack -----------------------

  // A carry shifts right by one; otherwise shift left past the leading zeros,
  // but not below exponent 1 (a subnormal result keeps its leading zeros).
  logic [W-1:0]   norm;
  logic [8:0]     norm_exp;
  logic [LzW-1:0] left;
  always_comb begin
    left = '0;
    if (s2_q.sum[W]) begin
      norm     = {s2_q.sum[W:2], s2_q.sum[1] | s2_q.sum[0]};
      norm_exp = s2_q.exp + 9'd1;
    end else begin
      left     = (9'(s2_q.lz) < s2_q.exp) ? s2_q.lz : LzW'(s2_q.exp - 9'd1);
      norm     = s2_q.sum[W-1:0] << left;
      norm_exp = s2_q.exp - 9'(left);
    end
  end

  // Round at FP32's 24 bits; an all-ones significand that rounds up carries
  // into the exponent. From exponent 254 that gives exponent 255 and a zero
  // fraction, which is Inf's pattern, so overflow needs no test of the carry.
  logic               guard, sticky, round_up, overflow;
  logic [F32SigW:0]   rounded;
  logic [F32SigW-1:0] sig;
  logic [7:0]         exp;  // when there is no overflow
  always_comb begin
    guard    = norm[W-F32SigW-1];
    sticky   = |norm[W-F32SigW-2:0];
    round_up = guard && (sticky || norm[W-F32SigW]);
    rounded  = {1'b0, norm[W-1-:F32SigW]} + (F32SigW + 1)'(round_up);
    sig      = rounded[F32SigW] ? rounded[F32SigW:1] : rounded[F32SigW-1:0];
    exp      = norm_exp[7:0] + 8'(rounded[F32SigW]);
    overflow = norm_exp >= 9'd255;
  end

  logic [31:0] y_d;
  always_comb begin
    if (s2_q.nan) y_d = CanonicalNaN;
    else if (s2_q.inf || overflow) y_d = {s2_q.sign, 8'hFF, 23'h0};
    else y_d = {s2_q.sign, sig[F32SigW-1] ? exp : 8'h00, sig[F32SigW-2:0]};  // a zero: sig = 0
    y_d = flush_f32(y_d, Ftz);  // after rounding, as the golden model
  end

  // --- Registers --------------------------------------------------------------

  always_ff @(posedge clk_i) begin
    s1_q    <= s1_d;
    s2_q    <= s2_d;
    y_o     <= y_d;
    valid_o <= s2_q.valid;
    if (!rst_ni) begin
      s1_q.valid <= 1'b0;
      s2_q.valid <= 1'b0;
      valid_o    <= 1'b0;
    end
  end

endmodule
