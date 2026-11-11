// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors
//
// A stand-in for rtl/dot_lane.sv with its ports and its boundary logic as
// the RTL has it (in_ready_o from last_i and reserved_q; the output FIFO's
// valid, read mux and pop), no FP logic: rtl.logic_depth(..., stubs=...)
// measures a block's own logic without the lanes, whose depth is measured
// alone (test_lane_logic_depth). In the flattened engine, the mapper may
// build a lane's arithmetic deeper than it builds the lane alone, and that
// would hide the engine's own paths.

module dot_lane
  import fp_pkg::*;
#(
  parameter  int unsigned E       = 4,
  parameter  int unsigned A       = 16,
  parameter  int unsigned MulRegs = 1,
  parameter  bit          Ftz     = 1'b0,
  parameter  int unsigned Banks   = 2,
  localparam int unsigned S       = A / E,
  localparam int unsigned BW      = $clog2(Banks)
) (
  input  logic                clk_i,
  input  logic                rst_ni,
  input  logic                in_valid_i,
  output logic                in_ready_o,
  input  logic [E-1:0][15:0]  w_i,
  input  logic [E-1:0][15:0]  x_i,
  input  logic [E-1:0]        mask_i,
  input  logic                last_i,
  output logic                out_valid_o,
  input  logic                out_ready_i,
  output logic [31:0]         out_o,
  output logic [BW:0]         reserved_o
);
  localparam int unsigned OutDepth = 2;  // as dot_lane's
  localparam int unsigned OW       = $clog2(OutDepth);

  logic [BW:0]   reserved_q;
  logic          fire, pop, root_q;
  logic [31:0]   root_sum_q;
  logic [31:0]   fifo_q[OutDepth];
  logic [OW-1:0] rd_ptr_q, wr_ptr_q;
  logic [OW:0]   count_q;

  assign in_ready_o  = !last_i || (reserved_q < (BW + 1)'(Banks));
  assign reserved_o  = reserved_q;
  assign fire        = in_valid_i && in_ready_o;
  assign out_valid_o = count_q != 0;
  assign out_o       = fifo_q[rd_ptr_q];
  assign pop         = out_valid_o && out_ready_i;

  // A row's end becomes a result a cycle later (the lane: many cycles), its
  // value every input bit folded in.
  logic [31:0] fold;
  always_comb begin
    fold = 32'(mask_i) ^ 32'(Ftz) ^ 32'(MulRegs) ^ 32'(S);
    for (int unsigned e = 0; e < E; e++) fold = fold ^ {w_i[e], x_i[e]};
  end
  always_ff @(posedge clk_i) begin
    root_sum_q <= fold;
    if (root_q) fifo_q[wr_ptr_q] <= root_sum_q;
  end
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      reserved_q <= '0;
      root_q     <= 1'b0;
      rd_ptr_q   <= '0;
      wr_ptr_q   <= '0;
      count_q    <= '0;
    end else begin
      root_q     <= fire && last_i;
      reserved_q <= reserved_q + (BW + 1)'(fire && last_i) - (BW + 1)'(root_q);
      if (root_q) wr_ptr_q <= wr_ptr_q + 1'b1;
      if (pop) rd_ptr_q <= rd_ptr_q + 1'b1;
      count_q <= count_q + (OW + 1)'(root_q) - (OW + 1)'(pop);
    end
  end
endmodule
