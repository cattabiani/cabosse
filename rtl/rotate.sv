// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Rotation of N entries of W bits: out_o[i] = in_i[(i + shift_i) mod N], for
// shift_i < N. log2(N) stages, stage b rotating by 2^b mod N when bit b of
// shift_i is set: rotations add up, so any N works, not only powers of two.

module rotate #(
  parameter  int unsigned N  = 8,
  parameter  int unsigned W  = 1,
  localparam int unsigned SW = N > 1 ? $clog2(N) : 1
) (
  input  logic [N-1:0][W-1:0] in_i,
  input  logic [SW-1:0]       shift_i,
  output logic [N-1:0][W-1:0] out_o
);

  logic [N-1:0][W-1:0] stage[SW+1];

  assign stage[0] = in_i;
  for (genvar b = 0; b < SW; b++) begin : gen_stage
    localparam int unsigned Amount = (1 << b) % N;
    for (genvar i = 0; i < N; i++) begin : gen_entry
      assign stage[b+1][i] = shift_i[b] ? stage[b][(i + Amount) % N] : stage[b][i];
    end
  end
  assign out_o = stage[SW];

endmodule
