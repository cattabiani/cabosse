// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// FP32 max(a, b), IEEE 754-2019 maximumNumber (D-037, D-039): fp_pkg::max_f32,
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

  logic [31:0] y_d;
  assign y_d = max_f32(a_i, b_i);

  always_ff @(posedge clk_i) begin
    y_o     <= y_d;
    valid_o <= rst_ni && valid_i;
  end

endmodule
