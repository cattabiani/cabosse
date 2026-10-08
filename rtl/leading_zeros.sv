// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Count of the leading zeros of in_i, in log2 steps: at each step, if the
// top half of what is left is zero, count it and shift it out. cnt_o is
// meaningful only when empty_o (in_i == 0) is low. Combinational.

module leading_zeros #(
  parameter  int unsigned Width    = 27,
  localparam int unsigned CntWidth = $clog2(Width)
) (
  input  logic [Width-1:0]    in_i,
  output logic [CntWidth-1:0] cnt_o,
  output logic                empty_o
);

  localparam int unsigned Padded = 2 ** CntWidth;  // Width rounded up to a power of 2

  // The input left-justified; ones below it stop the count at the input's end.
  // The array looks to Verilator like one signal feeding itself (UNOPTFLAT) when
  // a parent inlines several counters (dot_lane); each step reads only the
  // step before, so there is no loop.
  /* verilator lint_off UNOPTFLAT */
  logic [Padded-1:0] v[CntWidth+1];
  /* verilator lint_on UNOPTFLAT */
  if (Padded > Width) begin : gen_pad
    assign v[0] = {in_i, {(Padded - Width) {1'b1}}};
  end else begin : gen_no_pad
    assign v[0] = in_i;
  end

  for (genvar i = 0; i < CntWidth; i++) begin : gen_step
    localparam int unsigned Half = Padded >> (i + 1);
    assign cnt_o[CntWidth-1-i] = ~|v[i][Padded-1-:Half];
    assign v[i+1] = cnt_o[CntWidth-1-i] ? v[i] << Half : v[i];
  end

  assign empty_o = ~|in_i;

endmodule
