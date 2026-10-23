// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// One vector-unit slot's arithmetic (D-048; docs/numerics.md, sections 2 and
// 5): a chain of 7 stages, each one fp32_fma and the integer steps around
// it, that every operation flows through. add, mul and fma use the first
// stage and pass the others; exp and rsqrt use all seven, recip six; pass
// (the vector unit's ROTATE_HALF) gives a unchanged, bit for bit. Golden
// model: golden.arith.add, mul, fma and golden.funcs.exp, recip, rsqrt, bit
// for bit.
//
//   stage  exp                  recip              rsqrt
//   1      t = x * log2(e)      xs = |x| * k       xs = x * k
//   2      f = t + -i           e = -xs * y + 1    a = h * y
//   3      p = c4 * f + c3      y = y * e + y      b = -a * y + 1.5
//   4      p = p * f + c2       e = -xs * y + 1    y = y * b
//   5      p = p * f + c1       y = y * e + y      a = h * y
//   6      p = p * f + c0       y = y * k          b = -a * y + 1.5
//   7      y = 2^i p (* 2^-64)  -                  y = y * b
//
// k scales an input the first guess cannot take (recip: 2^24 below 2^-126,
// 2^-24 from 2^125; rsqrt: 2^24 below 2^-125). Between the stages, integer
// steps and no FP unit: exp's clamp of t to +-150, i = round(t) and -i in
// FP32, and 2^i p by an add to p's exponent field; the first guesses of
// recip and rsqrt; h = xs / 2 by a decrement of the exponent (exact: xs >=
// 2^-125, so h is normal), and rsqrt's final 2^12 by an increment (exact: y
// is normal). Special inputs (NaN, zeros, infinities, rsqrt of a negative)
// are classified at the entry and set the result at the exit.
//
// Each stage is its fp32_fma (MulRegs + 4 cycles) and a register: a result
// leaves 1 + 7 * (MulRegs + 5) cycles after its operands enter (50 at the
// default); valid_o marks it. No stall (AGENTS.md: a fixed-latency
// arithmetic pipeline).

module fp32_chain
  import fp_pkg::*;
#(
  parameter int unsigned MulRegs = 2
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  chain_op_e   op_i,
  input  logic [31:0] a_i,
  input  logic [31:0] b_i,
  input  logic [31:0] c_i,
  output logic        valid_o,
  output logic [31:0] y_o
);

  localparam int unsigned Stages = ChainStages;
  localparam int unsigned FmaLatency = mul_add_latency(MulRegs);

  // FP32 bits (docs/numerics.md, section 5; golden.funcs).
  localparam logic [31:0] OneHalf = 32'h3FC0_0000, PosInf = 32'h7F80_0000;
  localparam logic [31:0] Log2E = 32'h3FB8_AA3B;
  localparam logic [31:0] ExpClamp = 32'h4316_0000;  // 150
  localparam logic [31:0] Exp2C[5] = '{
    32'h3F80_0000, 32'h3F31_7061, 32'h3E75_FD26, 32'h3D65_0E71, 32'h3C1E_5FB0
  };  // c0 ... c4
  localparam logic [31:0] RRsqrt = 32'h5F37_59DF, RRecip = 32'h7EF3_11C3;
  localparam logic [31:0] Two24 = 32'h4B80_0000, TwoM24 = 32'h3380_0000;
  localparam logic [31:0] TwoM64 = 32'h1F80_0000;

  // The result an input sets by itself, with state_t.sign.
  typedef enum logic [1:0] {
    Computed,
    SpecialNan,
    SpecialInf,
    SpecialZero
  } special_e;

  typedef struct packed {
    chain_op_e         op;
    logic [31:0]       u, v, w;  // the values of the table: add, mul, fma's a, b, c, then v
    logic signed [8:0] i;        // exp: round(t), |i| <= 150
    logic [1:0]        k;        // a scale (scale()): recip's, rsqrt's input; exp's 2^-64
    logic              sign;     // recip: x's sign; the special result's sign
    special_e          special;
  } state_t;

  typedef struct packed {
    logic [1:0]  op;  // fp32_fma's
    logic [31:0] a, b, c;
  } fma_in_t;

  function automatic logic [31:0] neg(logic [31:0] x);  // an exact sign flip
    return {~x[31], x[30:0]};
  endfunction

  function automatic logic [31:0] scale(logic [1:0] k);
    case (k)
      2'd1: return Two24;
      2'd2: return TwoM24;
      2'd3: return TwoM64;
      default: return Fp32One;
    endcase
  endfunction

  // The input's state: its scale and the special result it sets.
  function automatic state_t entry(chain_op_e op, logic [31:0] a, b, c);
    state_t     s;
    logic [7:0] e;
    logic       nan, zero;
    s = '{op: op, u: a, v: op == ChainPass ? a : b, w: c, sign: a[31], special: Computed,
          default: '0};
    e = a[30:23];
    nan = is_nan_f32(a[30:0]);
    zero = a[30:0] == '0;
    case (op)
      ChainExp: if (nan) s.special = SpecialNan;
      ChainRecip: begin
        s.u = {1'b0, a[30:0]};
        s.k = e == 8'd0 ? 2'd1 : e >= 8'd252 ? 2'd2 : 2'd0;
        if (nan) s.special = SpecialNan;
        else if (zero) s.special = SpecialInf;
        else if (e == 8'hFF) s.special = SpecialZero;
      end
      ChainRsqrt: begin
        s.k = {1'b0, !a[31] && e <= 8'd1 && !zero};  // 0 < x < 2^-125
        if (nan || (a[31] && !zero)) s.special = SpecialNan;
        else if (zero) s.special = SpecialInf;  // rsqrt(-0) = -inf
        else if (a == PosInf) s.special = SpecialZero;
      end
      default: ;
    endcase
    return s;
  endfunction

  // exp: t clamped to [-150, 150] (a NaN too, by its bits: the entry set
  // its result).
  function automatic logic [31:0] exp_clamp(logic [31:0] t);
    return t[30:0] > ExpClamp[30:0] ? {t[31], ExpClamp[30:0]} : t;
  endfunction

  // t (|t| <= 150) rounded to the nearest integer, ties to even.
  function automatic logic signed [8:0] round_even(logic [31:0] t);
    logic [23:0]       m;
    logic [7:0]        q;
    logic              guard, sticky;
    int unsigned       sh;
    logic signed [8:0] r;
    if (t[30:23] < 8'd126) return '0;  // |t| < 0.5
    m = {1'b1, t[22:0]};
    sh = 150 - 32'(t[30:23]);  // 24 (|t| in [0.5, 1)) down to 16 (|t| in [128, 150])
    q = 8'(m >> sh);
    guard = m[sh-1];
    sticky = (m & ((24'd1 << (sh - 1)) - 24'd1)) != '0;
    if (guard && (sticky || q[0])) q = q + 8'd1;
    r = {1'b0, q};
    return t[31] ? -r : r;
  endfunction

  // -i in FP32 (exact; +0 for i = 0).
  function automatic logic [31:0] neg_int_f32(logic signed [8:0] i);
    logic [7:0]  n;
    logic [31:0] f;
    n = i[8] ? 8'(-i) : i[7:0];
    f = '0;
    for (int unsigned b = 0; b < 8; b++) begin
      if (n[b]) f = {!i[8], 8'(127 + b), 23'(n) << (23 - b)};  // the highest set bit wins
    end
    return f;
  endfunction

  // exp after stage 6: 2^i p by an add to p's exponent field (p is in
  // [0.7, 1.5], so the field stays in range); below the normal range,
  // 2^(i+64) p now and 2^-64 at stage 7 (one rounding).
  function automatic state_t exp_scale(state_t s, logic [31:0] p);
    logic signed [10:0] e;
    e = $signed({3'b0, p[30:23]}) + 11'(s.i);
    if (e >= 11'sd255) begin
      s.v = PosInf;
    end else begin
      if (e < 11'sd1) begin
        e   = e + 11'sd64;
        s.k = 2'd3;
      end
      s.v = {p[31], e[7:0], p[22:0]};
    end
    return s;
  endfunction

  // The helpers below take the whole state and read the fields they need.
  // verilator lint_off UNUSEDSIGNAL

  // Stage `stage`'s FMA: a column of the table. A pass leaves it idle, zeros.
  function automatic fma_in_t operands(int unsigned stage, state_t s);
    fma_in_t f;
    f = '0;
    case (s.op)
      ChainAdd: if (stage == 1) f = '{Fp32Add, s.u, s.v, '0};
      ChainMul: if (stage == 1) f = '{Fp32Mul, s.u, s.v, '0};
      ChainFma: if (stage == 1) f = '{Fp32Fma, s.u, s.v, s.w};
      ChainExp:
      case (stage)
        1: f = '{Fp32Mul, s.u, Log2E, '0};
        2: f = '{Fp32Add, s.u, neg_int_f32(s.i), '0};
        3: f = '{Fp32Fma, Exp2C[4], s.u, Exp2C[3]};  // p = c4 * f + c3, then p * f + c
        4: f = '{Fp32Fma, s.v, s.u, Exp2C[2]};
        5: f = '{Fp32Fma, s.v, s.u, Exp2C[1]};
        6: f = '{Fp32Fma, s.v, s.u, Exp2C[0]};
        7: f = '{Fp32Mul, s.v, scale(s.k), '0};
        default: ;
      endcase
      ChainRecip:
      case (stage)
        1: f = '{Fp32Mul, s.u, scale(s.k), '0};
        2, 4: f = '{Fp32Fma, neg(s.u), s.v, Fp32One};
        3, 5: f = '{Fp32Fma, s.v, s.w, s.v};
        6: f = '{Fp32Mul, s.v, scale(s.k), '0};
        default: ;
      endcase
      ChainRsqrt:
      case (stage)
        1: f = '{Fp32Mul, s.u, scale(s.k), '0};
        2, 5: f = '{Fp32Mul, s.u, s.v, '0};
        3, 6: f = '{Fp32Fma, neg(s.w), s.v, OneHalf};
        4, 7: f = '{Fp32Mul, s.v, s.w, '0};
        default: ;
      endcase
      default: ;
    endcase
    return f;
  endfunction

  // The state after stage `stage`, whose FMA gave r. A pass keeps it.
  function automatic state_t result(int unsigned stage, state_t s, logic [31:0] r);
    case (s.op)
      ChainAdd, ChainMul, ChainFma: if (stage == 1) s.v = r;
      ChainExp:
      case (stage)
        1: begin
          s.u = exp_clamp(r);  // t
          s.i = round_even(s.u);
        end
        2: s.u = r;  // f
        6: s = exp_scale(s, r);
        default: s.v = r;  // p
      endcase
      ChainRecip:
      case (stage)
        1: begin
          s.u = r;  // xs
          s.v = RRecip - r;
        end
        2, 4: s.w = r;  // e
        3, 5, 6: s.v = r;  // y
        default: ;
      endcase
      ChainRsqrt:
      case (stage)
        1: begin
          s.u = {r[31], r[30:23] - 8'd1, r[22:0]};  // h = xs / 2
          s.v = RRsqrt - {1'b0, r[31:1]};
        end
        4, 7: s.v = r;  // y
        default: s.w = r;  // a, b
      endcase
      default: ;
    endcase
    return s;
  endfunction

  // The result: recip's sign, rsqrt's 2^12, or the special result.
  function automatic logic [31:0] exit_value(state_t s);
    logic [31:0] y;
    y = s.v;
    if (s.op == ChainRecip) y = {s.sign, s.v[30:0]};
    if (s.op == ChainRsqrt && s.k[0]) y = {s.v[31], s.v[30:23] + 8'd12, s.v[22:0]};
    case (s.special)
      SpecialNan:  y = CanonicalNaN;
      SpecialInf:  y = {s.sign, PosInf[30:0]};
      SpecialZero: y = {s.sign, 31'd0};
      default: ;
    endcase
    return y;
  endfunction

  // verilator lint_on UNUSEDSIGNAL

  // st_q[s]: stage s + 1's input; st_q[Stages]: the result. valid_q
  // alongside: each stage's from its FMA's valid_o.
  state_t st_q[Stages+1];
  logic   valid_q[Stages+1];
  always_ff @(posedge clk_i) begin
    st_q[0]    <= entry(op_i, a_i, b_i, c_i);
    valid_q[0] <= rst_ni && valid_i;
  end

  for (genvar s = 0; s < Stages; s++) begin : gen_stage
    fma_in_t     f;
    logic [31:0] r;
    logic        r_valid;
    state_t      d_q[FmaLatency];  // the state, delayed to meet r
    assign f = operands(s + 1, st_q[s]);
    fp32_fma #(
      .MulRegs(MulRegs)
    ) u_fma (
      .clk_i,
      .rst_ni,
      .valid_i(valid_q[s]),
      .op_i   (f.op),
      .a_i    (f.a),
      .b_i    (f.b),
      .c_i    (f.c),
      .valid_o(r_valid),
      .y_o    (r)
    );
    always_ff @(posedge clk_i) begin
      d_q[0] <= st_q[s];
      for (int unsigned j = 1; j < FmaLatency; j++) d_q[j] <= d_q[j-1];
      st_q[s+1]    <= result(s + 1, d_q[FmaLatency-1], r);
      valid_q[s+1] <= rst_ni && r_valid;
    end
  end

  assign valid_o = valid_q[Stages];
  assign y_o     = exit_value(st_q[Stages]);

endmodule
