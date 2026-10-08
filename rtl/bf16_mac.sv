// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The lanes' multiply-add (D-038), whole: y = mac(w, x, acc) = fma(up(w),
// up(x), acc), one rounding (docs/numerics.md, section 2; golden model:
// golden.arith.mac). fp_mul_add with M = 8 and MulRegs registers in the
// product (250 MHz on F2 needs one): a result leaves MulRegs + 4 cycles
// after its operands enter; valid_o marks it. The lane uses the parts
// directly, with the accumulator fed back into fp_add; the product's
// registers are outside that loop.

module bf16_mac #(
  parameter int unsigned MulRegs = 1,
  parameter bit          Ftz     = 1'b0  // flush-to-zero variant (fp_mul_add)
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

  fp_mul_add #(
    .M      (8),
    .MulRegs(MulRegs),
    .Ftz    (Ftz)
  ) u_mac (
    .clk_i,
    .rst_ni,
    .valid_i,
    .a_i(w_i),
    .b_i(x_i),
    .c_i(acc_i),
    .valid_o,
    .y_o
  );

endmodule
