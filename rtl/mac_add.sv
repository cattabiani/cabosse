// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The accumulate step of the lanes' multiply-add (D-038): y = p + acc with
// one rounding to FP32, round to nearest even, where p is bf16_mul's
// product. Together they compute mac(w, x, acc) = fma(up(w), up(x),
// acc) (docs/numerics.md, section 2; golden model: golden.arith.mac).
// Subnormals are kept, Inf - Inf and any NaN give the canonical NaN, an exact
// zero sum is -0 only if both operands are -0.
//
// Three register stages: align | add, count leading zeros | normalize,
// round. A result leaves 3 cycles after its operands enter; valid_o marks it.
// With the lane's accumulator register the loop has 4 cycles (D-027).

module mac_add (
  input  logic                   clk_i,
  input  logic                   rst_ni,
  input  logic                   valid_i,
  input  bf16_mac_pkg::product_t p_i,
  input  logic [31:0]            acc_i,
  output logic                   valid_o,
  output logic [31:0]            y_o
);

  localparam logic [7:0]  ExpMax = 8'hFF;
  localparam logic [31:0] QNaN   = 32'h7FC0_0000;
  localparam int unsigned W      = 27;  // significand, guard, round, sticky

  // An operand: (-1)^sign * sig * 2^(exp - 153), with exp >= 1 and sig[W-1]
  // set whenever exp > 1, so (exp, sig) orders magnitudes (product_t).
  typedef struct packed {
    logic               sign;
    logic signed [10:0] exp;
    logic [W-1:0]       sig;
  } operand_t;

  // --- Stage 1: specials, order by magnitude, align -------------------------

  logic a_nan, a_inf;
  assign a_nan = (acc_i[30:23] == ExpMax) && (acc_i[22:0] != 0);
  assign a_inf = (acc_i[30:23] == ExpMax) && (acc_i[22:0] == 0);

  operand_t a, p, hi, lo;
  always_comb begin
    a.sign = acc_i[31];
    a.exp  = (acc_i[30:23] == 0) ? 11'sd1 : 11'(signed'({1'b0, acc_i[30:23]}));
    a.sig  = {acc_i[30:23] != 0, acc_i[22:0], 3'b000};
    p.sign = p_i.sign;
    p.exp  = p_i.exp;
    p.sig  = p_i.sig;
  end

  // The larger magnitude goes first, so the difference is never negative.
  logic p_hi;
  assign p_hi = (p.exp > a.exp) || ((p.exp == a.exp) && (p.sig > a.sig));
  assign hi   = p_hi ? p : a;
  assign lo   = p_hi ? a : p;

  // Shift the smaller right by the exponent difference; bits shifted out
  // collect into the sticky bit (bit 0).
  logic [11:0]      diff;
  logic [4:0]       shift;
  logic [2*W-1:0]   shifted;
  logic [W-1:0]     hi_s1, lo_s1;
  assign diff     = 12'(hi.exp - lo.exp);  // both in 1 .. 382
  assign shift    = (diff > 12'(W)) ? 5'(W + 1) : diff[4:0];
  assign shifted  = {lo.sig, W'(0)} >> shift;
  assign hi_s1    = hi.sig;
  assign lo_s1    = {shifted[2*W-1:W+1], shifted[W] | (|shifted[W-1:0])};

  // Results that bypass the arithmetic.
  logic nan_s1, inf_s1;
  assign nan_s1 = p_i.is_nan || a_nan || (p_i.is_inf && a_inf && (p_i.sign != acc_i[31]));
  assign inf_s1 = (p_i.is_inf || a_inf) && !nan_s1;

  typedef struct packed {
    logic               valid;
    logic               sub;        // effective subtraction
    logic               sign;       // of the result, unless it is zero
    logic               zero_sign;  // of an exact zero: -0 only for -0 + -0
    logic signed [10:0] exp;
    logic [W-1:0]       hi;
    logic [W-1:0]       lo;
    logic               nan;
    logic               inf;
    logic               inf_sign;
  } s1_t;
  s1_t s1_d, s1_q;
  always_comb begin
    s1_d.valid     = valid_i;
    s1_d.sub       = hi.sign != lo.sign;
    s1_d.sign      = hi.sign;
    s1_d.zero_sign = hi.sign && lo.sign;
    s1_d.exp       = hi.exp;
    s1_d.hi        = hi_s1;
    s1_d.lo        = lo_s1;
    s1_d.nan       = nan_s1;
    s1_d.inf       = inf_s1;
    s1_d.inf_sign  = p_i.is_inf ? p_i.sign : acc_i[31];
  end

  // --- Stage 2: add, count leading zeros -------------------------------------

  logic [W:0] sum;
  assign sum = s1_q.sub ? {1'b0, s1_q.hi} - {1'b0, s1_q.lo}
                        : {1'b0, s1_q.hi} + {1'b0, s1_q.lo};

  logic [4:0] lz;
  logic       sum_zero;
  lzc #(
    .WIDTH(W),
    .MODE (1'b1)
  ) u_lzc (
    .in_i   (sum[W-1:0]),
    .cnt_o  (lz),
    .empty_o(sum_zero)
  );

  typedef struct packed {
    logic               valid;
    logic               sign;
    logic               zero_sign;
    logic signed [10:0] exp;
    logic [W:0]         sum;
    logic [4:0]         lz;
    logic               zero;
    logic               nan;
    logic               inf;
    logic               inf_sign;
  } s2_t;
  s2_t s2_d, s2_q;
  always_comb begin
    s2_d.valid     = s1_q.valid;
    s2_d.sign      = s1_q.sign;
    s2_d.zero_sign = s1_q.zero_sign;
    s2_d.exp       = s1_q.exp;
    s2_d.sum       = sum;
    s2_d.lz        = lz;
    s2_d.zero      = sum_zero && !sum[W];
    s2_d.nan       = s1_q.nan;
    s2_d.inf       = s1_q.inf;
    s2_d.inf_sign  = s1_q.inf_sign;
  end

  // --- Stage 3: normalize, round to nearest even, pack -----------------------

  // A carry shifts right by one; otherwise shift left past the leading zeros,
  // but not below exponent 1 (a subnormal result keeps its leading zeros).
  logic [W-1:0]       norm;
  logic signed [10:0] norm_exp;
  logic [4:0]         left;
  always_comb begin
    left = '0;
    if (s2_q.sum[W]) begin
      norm     = {s2_q.sum[W:2], s2_q.sum[1] | s2_q.sum[0]};
      norm_exp = s2_q.exp + 11'sd1;
    end else begin
      left     = (11'(signed'({1'b0, s2_q.lz})) < s2_q.exp) ? s2_q.lz : 5'(s2_q.exp - 11'sd1);
      norm     = s2_q.sum[W-1:0] << left;
      norm_exp = s2_q.exp - 11'(signed'({1'b0, left}));
    end
  end

  logic               round_up;
  logic [24:0]        rounded;
  logic [23:0]        sig;
  logic signed [10:0] exp;
  always_comb begin
    round_up = norm[2] && (norm[1] || norm[0] || norm[3]);
    rounded  = {1'b0, norm[W-1:3]} + 25'(round_up);
    if (rounded[24]) begin  // 1.11..1 rounded up to 10.0
      sig = rounded[24:1];
      exp = norm_exp + 11'sd1;
    end else begin
      sig = rounded[23:0];
      exp = norm_exp;
    end
  end

  logic [31:0] y_d;
  always_comb begin
    if (s2_q.nan) y_d = QNaN;
    else if (s2_q.inf) y_d = {s2_q.inf_sign, ExpMax, 23'h0};
    else if (s2_q.zero) y_d = {s2_q.zero_sign, 31'h0};
    else if (exp >= 11'sd255) y_d = {s2_q.sign, ExpMax, 23'h0};  // overflow
    else y_d = {s2_q.sign, sig[23] ? exp[7:0] : 8'h00, sig[22:0]};
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
