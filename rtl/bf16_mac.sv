// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The lanes' multiply-add (D-038), whole: y = mac(w, x, acc) = fma(up(w),
// up(x), acc), one rounding (docs/numerics.md, section 2; golden model:
// golden.arith.mac). fp_product with M = 8 and MulRegs registers inside
// (250 MHz on F2 needs one), a register, then fp_add: a result leaves
// MulRegs + 4 cycles after its operands enter; valid_o marks it. The lane uses
// the parts directly, with the accumulator fed back into fp_add; the
// product's registers are outside that loop.

module bf16_mac
  import fp_pkg::*;
#(
  parameter int unsigned MulRegs = 1
) (
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
  logic [31:0] acc_q[MulRegs+2];
  logic        valid_q[MulRegs+2];

  fp_product #(
    .M   (8),
    .Regs(MulRegs)
  ) u_mul (
    .clk_i,
    .a_i({w_i, 16'h0}),  // up(w)
    .b_i({x_i, 16'h0}),
    .p_o(p_d)
  );

  // The accumulator and valid, delayed to meet the product.
  assign acc_q[0]   = acc_i;
  assign valid_q[0] = valid_i;
  for (genvar i = 0; i <= MulRegs; i++) begin : gen_delay
    always_ff @(posedge clk_i) begin
      acc_q[i+1]   <= acc_q[i];
      valid_q[i+1] <= rst_ni && valid_q[i];
    end
  end
  always_ff @(posedge clk_i) p_q <= p_d;

  fp_add u_add (
    .clk_i,
    .rst_ni,
    .valid_i(valid_q[MulRegs+1]),
    .a_i    (p_q),
    .b_i    (decode_f32(acc_q[MulRegs+1])),
    .valid_o,
    .y_o
  );

endmodule
