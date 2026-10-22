// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors
//
// A stand-in for rtl/fp32_fma.sv with its ports and one register stage, no
// FP logic: rtl.logic_depth(..., stubs=...) measures a block's own logic
// without the FMA, whose depth is measured alone (it met 250 MHz in M3).

module fp32_fma
  import fp_pkg::*;
#(
  parameter int unsigned MulRegs = 2,
  parameter bit          Ftz     = 1'b0
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  logic [1:0]  op_i,
  input  logic [31:0] a_i,
  input  logic [31:0] b_i,
  input  logic [31:0] c_i,
  output logic        valid_o,
  output logic [31:0] y_o
);
  always_ff @(posedge clk_i) begin
    valid_o <= valid_i;
    y_o     <= a_i ^ b_i ^ c_i ^ {30'b0, op_i};
  end
endmodule
