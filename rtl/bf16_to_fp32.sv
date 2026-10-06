// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// BF16 -> FP32, exact: the BF16 bits become the high half and the low 16 bits
// are zero. A NaN becomes the canonical NaN (docs/numerics.md, section 2).
// Combinational. Golden model: golden.arith.up.

`default_nettype none

module bf16_to_fp32 (
  input  logic [15:0] a_i,
  output logic [31:0] y_o
);

  localparam logic [31:0] CanonicalNaN = 32'h7FC0_0000;

  logic is_nan;
  assign is_nan = (a_i[14:7] == 8'hFF) && (a_i[6:0] != 7'h00);
  assign y_o = is_nan ? CanonicalNaN : {a_i, 16'h0000};

endmodule

`default_nettype wire
