// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The accumulate step of the lanes' multiply-add (D-038): y = p + acc with
// one rounding to FP32, round to nearest even, where p is bf16_mul's
// product. Together they compute mac(w, x, acc) = fma(up(w), up(x), acc)
// (docs/numerics.md, section 2; golden model: golden.arith.mac). Subnormals
// are kept, Inf - Inf and any NaN give the canonical NaN, an exact zero sum
// is -0 only if both operands are -0.
//
// Three register stages: align | add, count leading zeros | normalize,
// round. A result leaves 3 cycles after its operands enter; valid_o marks it.
// With the lane's accumulator register the loop has 4 cycles (D-027).

module mac_add
  import bf16_mac_pkg::*;
(
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  operand_t    p_i,
  input  logic [31:0] acc_i,
  output logic        valid_o,
  output logic [31:0] y_o
);

  localparam logic [31:0] QNaN = 32'h7FC0_0000;

  // --- Stage 1: specials, order by magnitude, align -------------------------

  operand_t a;
  assign a = decode_f32(acc_i);

  // The shift needs only the exponents (equal exponents shift by 0); the
  // larger magnitude goes first, so the difference is never negative.
  logic         p_exp_hi, p_hi;
  logic [8:0]   diff;
  logic         hi_sign, lo_sign;
  logic [W-1:0] hi_sig, lo_sig;
  assign p_exp_hi = p_i.exp > a.exp;
  assign diff     = p_exp_hi ? p_i.exp - a.exp : a.exp - p_i.exp;
  assign p_hi     = p_exp_hi || ((p_i.exp == a.exp) && (p_i.sig > a.sig));
  assign {hi_sign, hi_sig} = p_hi ? {p_i.sign, p_i.sig} : {a.sign, a.sig};
  assign {lo_sign, lo_sig} = p_hi ? {a.sign, a.sig} : {p_i.sign, p_i.sig};

  logic nan_s1, inf_s1;
  assign nan_s1 = p_i.is_nan || a.is_nan || (p_i.is_inf && a.is_inf && (p_i.sign != a.sign));
  assign inf_s1 = (p_i.is_inf || a.is_inf) && !nan_s1;

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
    s1_d.sub   = hi_sign != lo_sign;
    s1_d.sign  = !inf_s1 ? hi_sign : p_i.is_inf ? p_i.sign : a.sign;
    s1_d.exp   = p_hi ? p_i.exp : a.exp;
    s1_d.hi    = hi_sig;
    s1_d.lo    = shift_right_sticky(lo_sig, diff);
    s1_d.nan   = nan_s1;
    s1_d.inf   = inf_s1;
  end

  // --- Stage 2: add, count leading zeros -------------------------------------

  logic [W:0] sum;
  assign sum = s1_q.sub ? {1'b0, s1_q.hi} - {1'b0, s1_q.lo} : {1'b0, s1_q.hi} + {1'b0, s1_q.lo};

  logic [4:0] lz;
  logic       sum_low_zero;
  lzc #(
    .WIDTH(W),
    .MODE (1'b1)
  ) u_lzc (
    .in_i   (sum[W-1:0]),
    .cnt_o  (lz),
    .empty_o(sum_low_zero)
  );

  typedef struct packed {
    logic       valid;
    logic       sign;
    logic [8:0] exp;
    logic [W:0] sum;
    logic [4:0] lz;
    logic       nan;
    logic       inf;
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
  logic [W-1:0] norm;
  logic [8:0]   norm_exp;
  logic [4:0]   left;
  always_comb begin
    left = '0;
    if (s2_q.sum[W]) begin
      norm     = {s2_q.sum[W:2], s2_q.sum[1] | s2_q.sum[0]};
      norm_exp = s2_q.exp + 9'd1;
    end else begin
      left     = (s2_q.exp > 9'(W) || s2_q.lz < s2_q.exp[4:0]) ? s2_q.lz : 5'(s2_q.exp - 9'd1);
      norm     = s2_q.sum[W-1:0] << left;
      norm_exp = s2_q.exp - 9'(left);
    end
  end

  // Round; an all-ones significand that rounds up carries into the exponent,
  // so the overflow test looks at the exponent before the carry.
  logic        round_up, overflow;
  logic [24:0] rounded;
  logic [23:0] sig;
  logic [7:0]  exp;  // when there is no overflow
  always_comb begin
    round_up = norm[2] && (norm[1] || norm[0] || norm[3]);
    rounded  = {1'b0, norm[W-1:3]} + 25'(round_up);
    sig      = rounded[24] ? rounded[24:1] : rounded[23:0];
    exp      = norm_exp[7:0] + 8'(rounded[24]);
    overflow = rounded[24] ? (norm_exp >= 9'd254) : (norm_exp >= 9'd255);
  end

  logic [31:0] y_d;
  always_comb begin
    if (s2_q.nan) y_d = QNaN;
    else if (s2_q.inf || overflow) y_d = {s2_q.sign, 8'hFF, 23'h0};
    else y_d = {s2_q.sign, sig[23] ? exp : 8'h00, sig[22:0]};  // a zero: sig = 0
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
