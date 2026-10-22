// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// One vector-unit slot's arithmetic (D-048; docs/numerics.md, sections 2 and
// 5): a chain of 7 stages, each one fp32_fma and the integer steps around
// it, that every operation flows through. add, mul and fma use the first
// stage and pass the others; exp and rsqrt use all seven, recip six. Golden
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

  localparam int unsigned Stages = 7;
  localparam int unsigned FmaLatency = MulRegs + 4;
  localparam logic [1:0] Fp32Fma = 2'd0;  // fp32_fma's op_i: any but add and mul

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
    logic              valid;
    chain_op_e         op;
    logic [31:0]       u, v, w;  // the values of the table: add, mul, fma's a, b, c, then v
    logic signed [8:0] i;        // exp: round(t), |i| <= 150
    logic [1:0]        k;        // the input's scale: 1: 2^24, 2: 2^-24, else 1
    logic              lift;     // exp: below the normal range, times 2^-64 at stage 7
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
    return k == 2'd1 ? Two24 : k == 2'd2 ? TwoM24 : Fp32One;
  endfunction

  // The input's state: its scale and the special result it sets.
  function automatic state_t entry(logic valid, chain_op_e op, logic [31:0] a, b, c);
    state_t     s;
    logic [7:0] e;
    logic       nan, zero;
    s = '{valid: valid, op: op, u: a, v: b, w: c, sign: a[31], special: Computed, default: '0};
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

  // exp: t clamped to [-150, 150], a NaN to 0.
  function automatic logic [31:0] exp_clamp(logic [31:0] t);
    if (is_nan_f32(t[30:0])) return '0;
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

  // exp after stage 6: 2^i p, by an add to p's exponent field when it stays
  // normal; below, 2^(i+64) p now and 2^-64 at stage 7 (one rounding).
  function automatic state_t exp_scale(state_t s, logic [31:0] p);
    logic signed [10:0] e;
    e = $signed({3'b0, p[30:23]}) + 11'(s.i);
    if (e >= 11'sd255) begin
      s.v = PosInf;
    end else if (e >= 11'sd1) begin
      s.v = p + (32'(s.i) << 23);
    end else begin
      s.v    = p + (32'(s.i + 9'sd64) << 23);
      s.lift = 1'b1;
    end
    return s;
  endfunction

  // The helpers below take the whole state and read the fields they need.
  // verilator lint_off UNUSEDSIGNAL
  // Stage `stage`'s FMA (the table): a pass leaves it idle, zeros.
  function automatic fma_in_t operands(int unsigned stage, state_t s);
    fma_in_t f;
    f = '0;
    case (stage)
      1:
      case (s.op)
        ChainAdd: f = '{Fp32Add, s.u, s.v, '0};
        ChainMul: f = '{Fp32Mul, s.u, s.v, '0};
        ChainFma: f = '{Fp32Fma, s.u, s.v, s.w};
        ChainExp: f = '{Fp32Mul, s.u, Log2E, '0};
        ChainRecip, ChainRsqrt: f = '{Fp32Mul, s.u, scale(s.k), '0};
        default: ;
      endcase
      2:
      case (s.op)
        ChainExp: f = '{Fp32Add, s.u, neg_int_f32(s.i), '0};
        ChainRecip: f = '{Fp32Fma, neg(s.u), s.v, Fp32One};
        ChainRsqrt: f = '{Fp32Mul, s.u, s.v, '0};
        default: ;
      endcase
      3:
      case (s.op)
        ChainExp: f = '{Fp32Fma, Exp2C[4], s.u, Exp2C[3]};
        ChainRecip: f = '{Fp32Fma, s.v, s.w, s.v};
        ChainRsqrt: f = '{Fp32Fma, neg(s.w), s.v, OneHalf};
        default: ;
      endcase
      4:
      case (s.op)
        ChainExp: f = '{Fp32Fma, s.v, s.u, Exp2C[2]};
        ChainRecip: f = '{Fp32Fma, neg(s.u), s.v, Fp32One};
        ChainRsqrt: f = '{Fp32Mul, s.v, s.w, '0};
        default: ;
      endcase
      5:
      case (s.op)
        ChainExp: f = '{Fp32Fma, s.v, s.u, Exp2C[1]};
        ChainRecip: f = '{Fp32Fma, s.v, s.w, s.v};
        ChainRsqrt: f = '{Fp32Mul, s.u, s.v, '0};
        default: ;
      endcase
      6:
      case (s.op)
        ChainExp: f = '{Fp32Fma, s.v, s.u, Exp2C[0]};
        ChainRecip: f = '{Fp32Mul, s.v, scale(s.k), '0};
        ChainRsqrt: f = '{Fp32Fma, neg(s.w), s.v, OneHalf};
        default: ;
      endcase
      default:
      case (s.op)
        ChainExp: f = '{Fp32Mul, s.v, s.lift ? TwoM64 : Fp32One, '0};
        ChainRsqrt: f = '{Fp32Mul, s.v, s.w, '0};
        default: ;
      endcase
    endcase
    return f;
  endfunction

  // The state after stage `stage`, whose FMA gave r (the table).
  function automatic state_t result(int unsigned stage, state_t s, logic [31:0] r);
    case (stage)
      1:
      case (s.op)
        ChainExp: begin
          s.u = exp_clamp(r);
          s.i = round_even(s.u);
        end
        ChainRecip: begin
          s.u = r;
          s.v = RRecip - r;
        end
        ChainRsqrt: begin
          s.u = {r[31], r[30:23] - 8'd1, r[22:0]};  // h = xs / 2
          s.v = RRsqrt - {1'b0, r[31:1]};
        end
        default: s.v = r;  // add, mul, fma: done
      endcase
      2:
      case (s.op)
        ChainExp: s.u = r;
        ChainRecip, ChainRsqrt: s.w = r;
        default: ;
      endcase
      3, 5:
      case (s.op)
        ChainExp, ChainRecip: s.v = r;
        ChainRsqrt: s.w = r;
        default: ;
      endcase
      4:
      case (s.op)
        ChainExp, ChainRsqrt: s.v = r;
        ChainRecip: s.w = r;
        default: ;
      endcase
      6:
      case (s.op)
        ChainExp: s = exp_scale(s, r);
        ChainRecip: s.v = r;
        ChainRsqrt: s.w = r;
        default: ;
      endcase
      default:
      case (s.op)
        ChainExp, ChainRsqrt: s.v = r;
        default: ;
      endcase
    endcase
    // A pass (add, mul and fma after stage 1, recip at 7) keeps the state.
    return s;
  endfunction

  // The result: recip's sign, rsqrt's 2^12, or the special result.
  function automatic logic [31:0] exit_value(state_t s);
    logic [31:0] y;
    y = s.v;
    if (s.op == ChainRecip) y = {s.sign, s.v[30:0]};
    if (s.op == ChainRsqrt && s.k[0]) y = s.v + (32'd12 << 23);
    case (s.special)
      SpecialNan:  y = CanonicalNaN;
      SpecialInf:  y = {s.sign, PosInf[30:0]};
      SpecialZero: y = {s.sign, 31'd0};
      default: ;
    endcase
    return y;
  endfunction

  // verilator lint_on UNUSEDSIGNAL

  function automatic state_t gate(state_t s, logic rst_n);  // valid only out of reset
    s.valid = s.valid && rst_n;
    return s;
  endfunction

  state_t st_q[Stages+1];  // st_q[s]: stage s + 1's input; st_q[Stages]: the result
  always_ff @(posedge clk_i) st_q[0] <= entry(rst_ni && valid_i, op_i, a_i, b_i, c_i);

  for (genvar s = 0; s < Stages; s++) begin : gen_stage
    fma_in_t     f;
    logic [31:0] r;
    state_t      d_q[FmaLatency];  // the state, delayed to meet r
    assign f = operands(s + 1, st_q[s]);
    fp32_fma #(
      .MulRegs(MulRegs)
    ) u_fma (
      .clk_i,
      .rst_ni,
      .valid_i(st_q[s].valid),
      .op_i   (f.op),
      .a_i    (f.a),
      .b_i    (f.b),
      .c_i    (f.c),
      // verilator lint_off PINCONNECTEMPTY
      .valid_o(),
      // verilator lint_on PINCONNECTEMPTY
      .y_o    (r)
    );
    always_ff @(posedge clk_i) begin
      d_q[0] <= gate(st_q[s], rst_ni);
      for (int unsigned j = 1; j < FmaLatency; j++) d_q[j] <= gate(d_q[j-1], rst_ni);
      st_q[s+1] <= gate(result(s + 1, d_q[FmaLatency-1], r), rst_ni);
    end
  end

  assign valid_o = st_q[Stages].valid;
  assign y_o     = exit_value(st_q[Stages]);

endmodule
