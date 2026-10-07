// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// in_i >> shift_i, with every bit shifted out ORed into bit 0 (the sticky
// bit of a significand). Combinational.

module sticky_shift #(
  parameter int unsigned Width = 27
) (
  input  logic [Width-1:0] in_i,
  input  logic [8:0]       shift_i,
  output logic [Width-1:0] out_o
);

  localparam int unsigned ShiftBits = $clog2(Width);

  logic [Width-1:0] lost_mask;  // the bits shifted out
  always_comb begin
    lost_mask = '0;
    if (shift_i >= 9'(Width)) begin
      out_o = Width'(|in_i);
    end else begin
      lost_mask = ~({Width{1'b1}} << shift_i[ShiftBits-1:0]);
      out_o     = (in_i >> shift_i[ShiftBits-1:0]) | Width'(|(in_i & lost_mask));
    end
  end

endmodule
