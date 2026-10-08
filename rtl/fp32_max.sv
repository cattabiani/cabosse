// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// FP32 max(a, b), IEEE 754-2019 maximumNumber (D-037, D-039): a NaN operand
// gives the other, two NaNs give the canonical NaN, max(-0, +0) = +0.
// Exact: the result is one of the operands, bit for bit. Golden model:
// golden.arith.maximum. One register: a result leaves 1 cycle after its
// operands enter; valid_o marks it.

module fp32_max
  import fp_pkg::*;
(
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  logic [31:0] a_i,
  input  logic [31:0] b_i,
  output logic        valid_o,
  output logic [31:0] y_o
);

  logic a_nan, b_nan, mag_gt, mag_eq, a_bigger;
  assign a_nan  = is_nan_f32(a_i[30:0]);
  assign b_nan  = is_nan_f32(b_i[30:0]);
  assign mag_gt = a_i[30:0] > b_i[30:0];
  assign mag_eq = a_i[30:0] == b_i[30:0];

  // Signed order of the bit patterns: positives by magnitude above negatives
  // in reverse; +0 above -0. One magnitude compare serves both signs.
  always_comb begin
    unique case ({a_i[31], b_i[31]})
      2'b00:   a_bigger = mag_gt;
      2'b11:   a_bigger = !mag_gt && !mag_eq;
      2'b01:   a_bigger = 1'b1;
      default: a_bigger = 1'b0;
    endcase
  end

  logic [31:0] y_d;
  always_comb begin
    if (a_nan && b_nan) y_d = CanonicalNaN;
    else if (a_nan) y_d = b_i;
    else if (b_nan) y_d = a_i;
    else y_d = a_bigger ? a_i : b_i;
  end

  always_ff @(posedge clk_i) begin
    y_o     <= y_d;
    valid_o <= rst_ni && valid_i;
  end

endmodule
