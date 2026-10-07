// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Timing harness for our own BF16 multiply-add's accumulate loop (D-038):
// acc = fp_add(fp_product(up(w), up(x)), acc). fp_add has 3 registers and AccRegs
// more close the loop, so it has 3 + AccRegs cycles (4 for D-027). The
// product is registered outside the loop, and so are the inputs and the
// output, so every path is register to register. Not part of the design: M4
// builds the lane.

module mac_loop
  import fp_pkg::*;
#(
  parameter int unsigned AccRegs = 1
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  logic [15:0] w_i,
  input  logic [15:0] x_i,
  output logic [31:0] acc_o
);

  logic [15:0] w_q, x_q;
  logic        valid_q, p_valid_q;
  always_ff @(posedge clk_i) begin
    w_q     <= w_i;
    x_q     <= x_i;
    valid_q <= valid_i;
  end

  operand_t p_d, p_q;
  fp_product #(
    .M(8)
  ) u_mul (
    .a_i({w_q, 16'h0}),
    .b_i({x_q, 16'h0}),
    .p_o(p_d)
  );
  always_ff @(posedge clk_i) begin
    p_q       <= p_d;
    p_valid_q <= valid_q;
  end

  logic [31:0] acc, sum;
  logic        unused_sum_valid;  // the loop runs every cycle
  fp_add u_add (
    .clk_i,
    .rst_ni,
    .valid_i(p_valid_q),
    .a_i    (p_q),
    .b_i    (decode_f32(acc)),
    .valid_o(unused_sum_valid),
    .y_o    (sum)
  );

  // AccRegs registers from the sum back to fp_add's accumulator input.
  logic [31:0] acc_q[AccRegs+1];
  assign acc_q[0] = sum;
  for (genvar i = 0; i < AccRegs; i++) begin : gen_acc
    always_ff @(posedge clk_i) begin
      if (!rst_ni) acc_q[i+1] <= '0;
      else acc_q[i+1] <= acc_q[i];
    end
  end
  assign acc   = acc_q[AccRegs];
  assign acc_o = acc;

endmodule
