// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// BF16 -> FP32, exact: the BF16 bits become the high half and the low 16 bits
// are zero. A NaN becomes the canonical NaN (docs/numerics.md, section 2).
// Combinational. Golden model: golden.arith.up.

module bf16_to_fp32
  import fp_pkg::*;
(
  input  logic [15:0] a_i,
  output logic [31:0] y_o
);

  assign y_o = is_nan_f32({a_i[14:0], 16'h0000}) ? CanonicalNaN : {a_i, 16'h0000};

endmodule
