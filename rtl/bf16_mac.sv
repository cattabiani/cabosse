// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The lanes' multiply-add (D-038), whole: y = mac(w, x, acc) = fma(up(w),
// up(x), acc), one rounding (docs/numerics.md, section 2; golden model:
// golden.arith.mac). fp_product with M = 8, a register, then fp_add: a result leaves 4
// cycles after its operands enter; valid_o marks it. The lane uses the parts
// directly, with the accumulator fed back into fp_add.

module bf16_mac
  import fp_pkg::*;
(
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  logic [15:0] w_i,
  input  logic [15:0] x_i,
  input  logic [31:0] acc_i,
  output logic        valid_o,
  output logic [31:0] y_o
);

  operand_t    p_d, p_q;
  logic [31:0] acc_q;
  logic        valid_q;

  fp_product #(
    .M(8)
  ) u_mul (
    .clk_i,
    .a_i({w_i, 16'h0}),  // up(w)
    .b_i({x_i, 16'h0}),
    .p_o(p_d)
  );

  always_ff @(posedge clk_i) begin
    p_q     <= p_d;
    acc_q   <= acc_i;
    valid_q <= rst_ni && valid_i;
  end

  fp_add u_add (
    .clk_i,
    .rst_ni,
    .valid_i(valid_q),
    .a_i    (p_q),
    .b_i    (decode_f32(acc_q)),
    .valid_o,
    .y_o
  );

endmodule
