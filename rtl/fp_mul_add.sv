// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// y = a * b + c with one rounding to FP32 (D-038, D-039), for a and b in
// fp_product's M + 8-bit format (BF16 for M = 8, FP32 for M = 24) and an
// FP32 value c: fp_product with MulRegs registers inside (0 to 2), a
// register, then fp_add. A result leaves MulRegs + 4 cycles after its operands enter;
// valid_o marks it. bf16_mac (M = 8) and fp32_fma (M = 24) are this unit.

module fp_mul_add
  import fp_pkg::*;
#(
  parameter  int unsigned M       = 24,
  parameter  int unsigned MulRegs = 2,
  localparam int unsigned W       = prod_sig_w(M)
) (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         valid_i,
  input  logic [M+7:0] a_i,
  input  logic [M+7:0] b_i,
  input  logic [31:0]  c_i,
  output logic         valid_o,
  output logic [31:0]  y_o
);

  logic [W+11:0] p_d, p_q;
  fp_product #(
    .M   (M),
    .Regs(MulRegs)
  ) u_mul (
    .clk_i,
    .a_i,
    .b_i,
    .p_o(p_d)
  );
  always_ff @(posedge clk_i) p_q <= p_d;

  // The addend and valid, delayed MulRegs + 1 cycles to meet the product.
  logic [31:0] c_q[MulRegs+1];
  logic        valid_q[MulRegs+1];
  always_ff @(posedge clk_i) begin
    c_q[0]     <= c_i;
    valid_q[0] <= rst_ni && valid_i;
  end
  for (genvar i = 0; i < MulRegs; i++) begin : gen_delay
    always_ff @(posedge clk_i) begin
      c_q[i+1]     <= c_q[i];
      valid_q[i+1] <= rst_ni && valid_q[i];
    end
  end

  fp_add #(
    .W(W)
  ) u_add (
    .clk_i,
    .rst_ni,
    .valid_i(valid_q[MulRegs]),
    .a_i    (p_q),
    .b_i    (c_q[MulRegs]),
    .valid_o,
    .y_o
  );

endmodule
