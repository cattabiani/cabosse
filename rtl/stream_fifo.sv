// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// A first-in first-out queue of Depth entries of Width bits, with
// valid/ready on both sides. Push and pop may happen in the same cycle; a
// full queue takes no push, even if it pops in that cycle. The output reads
// the storage directly (no output register), so data pushed in one cycle can
// leave in the next. count_o is how many entries it holds.

module stream_fifo #(
  parameter  int unsigned Width = 32,
  parameter  int unsigned Depth = 4,
  localparam int unsigned PW    = Depth > 1 ? $clog2(Depth) : 1,  // a slot's index
  localparam int unsigned CW    = $clog2(Depth + 1)                // a count 0 .. Depth
) (
  input  logic             clk_i,
  input  logic             rst_ni,
  input  logic             in_valid_i,
  output logic             in_ready_o,
  input  logic [Width-1:0] in_data_i,
  output logic             out_valid_o,
  input  logic             out_ready_i,
  output logic [Width-1:0] out_data_o,
  output logic [CW-1:0]    count_o
);

  if (Depth == 0) begin : gen_bad_depth
    $error("stream_fifo: Depth must be at least 1");
  end

  logic [Width-1:0] mem_q[Depth];
  logic [PW-1:0]    rd_q, wr_q;
  logic [CW-1:0]    count_q;
  // count_q == Depth, == 0, a register each (the memory's write enable, the
  // output's valid), each next value from push and pop and a compare of
  // count_q with a constant.
  logic             full_q, empty_q;
  logic             push, pop;

  assign in_ready_o  = !full_q;
  assign out_valid_o = !empty_q;
  assign out_data_o  = mem_q[rd_q];
  assign count_o     = count_q;
  assign push        = in_valid_i && in_ready_o;
  assign pop         = out_valid_o && out_ready_i;

  function automatic logic [PW-1:0] next_slot(logic [PW-1:0] slot);
    return slot == PW'(Depth - 1) ? '0 : slot + 1'b1;
  endfunction

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      rd_q    <= '0;
      wr_q    <= '0;
      count_q <= '0;
      full_q  <= 1'b0;
      empty_q <= 1'b1;
    end else begin
      if (push) wr_q <= next_slot(wr_q);
      if (pop) rd_q <= next_slot(rd_q);
      count_q <= count_q + CW'(push) - CW'(pop);
      if (push != pop) begin
        full_q  <= push && count_q == CW'(Depth - 1);
        empty_q <= pop && count_q == CW'(1);
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (push) mem_q[wr_q] <= in_data_i;
  end

endmodule
