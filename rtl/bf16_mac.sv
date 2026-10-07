// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The lanes' multiply-add (D-038), whole: y = mac(w, x, acc) = fma(up(w),
// up(x), acc), one rounding (docs/numerics.md, section 2; golden model:
// golden.arith.mac). bf16_mul, a register, then mac_add: a result
// leaves 4 cycles after its operands enter; valid_o marks it. The lane uses
// the two parts directly, with the accumulator fed back into mac_add.

module bf16_mac (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  logic [15:0] w_i,
  input  logic [15:0] x_i,
  input  logic [31:0] acc_i,
  output logic        valid_o,
  output logic [31:0] y_o
);

  bf16_mac_pkg::operand_t p_d, p_q;
  logic [31:0]            acc_q;
  logic                   valid_q;

  bf16_mul u_mul (
    .w_i,
    .x_i,
    .p_o(p_d)
  );

  always_ff @(posedge clk_i) begin
    p_q     <= p_d;
    acc_q   <= acc_i;
    valid_q <= rst_ni && valid_i;
  end

  mac_add u_add (
    .clk_i,
    .rst_ni,
    .valid_i(valid_q),
    .p_i    (p_q),
    .acc_i  (acc_q),
    .valid_o,
    .y_o
  );

endmodule
