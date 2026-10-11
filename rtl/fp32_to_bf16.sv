// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// FP32 -> BF16, round to nearest even (docs/numerics.md, section 2): keep the
// high half, plus one if the low half is above half an ulp, or exactly half
// and the kept part is odd (golden: add 0x7FFF and the kept part's last bit,
// keep the high half; the carry is this round-up). Overflow rounds
// to infinity; a NaN becomes the canonical NaN. Subnormals are kept (no
// flush-to-zero variant). Combinational. Golden model: golden.arith.bf16.

module fp32_to_bf16
  import fp_pkg::*;
(
  input  logic [31:0] a_i,
  output logic [15:0] y_o
);

  logic round_up;
  assign round_up = a_i[15] && (a_i[16] || |a_i[14:0]);
  assign y_o      = is_nan_f32(a_i[30:0]) ? CanonicalNaN[31:16] : a_i[31:16] + 16'(round_up);

endmodule
