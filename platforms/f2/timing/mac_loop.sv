// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Timing harness for our own BF16 multiply-add's accumulate loop (D-038):
// acc = fp_add(fp_product(up(w), up(x)), acc). fp_add has 3 registers and AccRegs
// more close the loop, so it has 3 + AccRegs cycles (4 for D-027). The
// product has MulRegs registers inside (bf16_mac's default: 1) and one after,
// all outside the loop; the inputs and the output are registered too, so
// every path is register to register. Not part of the design: M4 builds the
// lane.

module mac_loop
  import fp_pkg::*;
#(
  parameter  int unsigned AccRegs = 1,
  parameter  int unsigned MulRegs = 1,
  localparam int unsigned W       = prod_sig_w(8)
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  logic [15:0] w_i,
  input  logic [15:0] x_i,
  output logic [31:0] acc_o
);

  logic [15:0] w_q, x_q;
  always_ff @(posedge clk_i) begin
    w_q <= w_i;
    x_q <= x_i;
  end

  logic [W+11:0] p_d, p_q;
  fp_product #(
    .M   (8),
    .Regs(MulRegs)
  ) u_mul (
    .clk_i,
    .a_i(w_q),
    .b_i(x_q),
    .p_o(p_d)
  );
  always_ff @(posedge clk_i) p_q <= p_d;

  // valid, delayed like the product: the input register, MulRegs, p_q.
  logic valid_q[MulRegs+2];
  always_ff @(posedge clk_i) valid_q[0] <= valid_i;
  for (genvar i = 0; i <= MulRegs; i++) begin : gen_valid
    always_ff @(posedge clk_i) valid_q[i+1] <= valid_q[i];
  end

  logic [31:0] acc, sum;
  logic        unused_sum_valid;  // the loop runs every cycle
  fp_add u_add (
    .clk_i,
    .rst_ni,
    .valid_i(valid_q[MulRegs+1]),
    .a_i    (p_q),
    .b_i    (acc),
    .valid_o(unused_sum_valid),
    .y_o    (sum)
  );

  // AccRegs registers from the sum back to fp_add's accumulator input.
  // acc_pipe[0] is the sum; acc_pipe[1..AccRegs] are registers.
  logic [31:0] acc_pipe[AccRegs+1];
  assign acc_pipe[0] = sum;
  for (genvar i = 0; i < AccRegs; i++) begin : gen_acc
    always_ff @(posedge clk_i) begin
      if (!rst_ni) acc_pipe[i+1] <= '0;
      else acc_pipe[i+1] <= acc_pipe[i];
    end
  end
  assign acc   = acc_pipe[AccRegs];
  assign acc_o = acc;

endmodule
