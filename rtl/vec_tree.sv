// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The vector unit's tree unit (D-050): a reduction row's Width partial sums
// (or maxima) to one value, in docs/numerics.md section 3's fixed tree:
// neighbours paired level by level (golden: dot.tree_sum). Sums go through
// one fp_add, as the lanes' tree (dot_lane), maxima through fp_pkg::max_f32
// (exact, so any order gives the golden reduce_max's bits). start_i takes a
// row's partials; each of the log2(Width) levels issues a pair a cycle and
// waits for its last result; done_o then gives the row's value, held in y_o
// until the next start_i. A level of P pairs takes P + AddLatency cycles, so
// a row about Width + 3 * log2(Width). busy_o from start_i to done_o;
// start_i only when idle.

module vec_tree
  import fp_pkg::*;
#(
  parameter  int unsigned Width = 128,
  localparam int unsigned IW    = $clog2(Width)
) (
  input  logic                   clk_i,
  input  logic                   rst_ni,
  input  logic                   start_i,
  input  logic                   max_i,    // maxima, else sums
  input  logic [Width-1:0][31:0] parts_i,
  output logic                   busy_o,
  output logic                   done_o,
  output logic [31:0]            y_o
);

  localparam int unsigned Lat = AddLatency;

  if (Width < 2 || Width != 2 ** IW) begin : gen_width_check
    $error("Width must be a power of two, at least 2");
  end

  // v_q: the level's values, each pair's result written over the first of
  // the pairs still to come (index i < 2i, so no value is overwritten before
  // it is read). pairs_q: the level's pairs - 1; next_q: the next to issue.
  logic [Width-1:0][31:0] v_q;
  logic                   busy_q, max_q;
  logic [IW-1:0]          pairs_q, next_q;

  // A result on its way back: its index, and a maximum's value (a sum's
  // comes from the adder).
  typedef struct packed {
    logic          valid;
    logic [IW-1:0] idx;
    logic [31:0]   max;
  } back_t;
  back_t back_q[Lat];

  // A level ends once every pair is issued and its last result is written
  // (this cycle at the latest); the next one reads the values from the cycle
  // after. Past the level's pairs the index wraps, so no read is out of
  // range (an unused read).
  logic        all_issued, issue, in_flight, level_end, sum_valid;
  logic [31:0] a, b, sum;
  always_comb begin
    in_flight = 1'b0;
    for (int unsigned j = 0; j + 1 < Lat; j++) in_flight |= back_q[j].valid;
  end
  assign all_issued = {1'b0, next_q} == {1'b0, pairs_q} + 1'b1;
  assign issue      = busy_q && !all_issued;
  assign level_end  = busy_q && all_issued && !in_flight;
  assign a          = v_q[IW'(2*next_q)];
  assign b          = v_q[IW'(2*next_q+1)];

  fp_add u_add (
    .clk_i,
    .rst_ni,
    .valid_i(issue && !max_q),
    .a_i    (decode_f32(a)),
    .b_i    (b),
    .valid_o(sum_valid),
    .y_o    (sum)
  );

  // verilator lint_off UNUSEDSIGNAL
  logic unused_sum_valid;  // back_q's valid marks the adder's result too
  assign unused_sum_valid = sum_valid;
  // verilator lint_on UNUSEDSIGNAL

  always_ff @(posedge clk_i) begin
    back_q[0] <= '{valid: rst_ni && issue, idx: next_q, max: max_f32(a, b)};
    for (int unsigned j = 1; j < Lat; j++) back_q[j] <= back_q[j-1];
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      busy_q <= 1'b0;
    end else if (start_i) begin
      busy_q  <= 1'b1;
      max_q   <= max_i;
      v_q     <= parts_i;
      pairs_q <= IW'(Width / 2 - 1);
      next_q  <= '0;
    end else if (busy_q) begin
      if (issue) next_q <= next_q + 1'b1;
      if (back_q[Lat-1].valid) v_q[back_q[Lat-1].idx] <= max_q ? back_q[Lat-1].max : sum;
      if (level_end) begin
        if (pairs_q == '0) begin
          busy_q <= 1'b0;
        end else begin
          pairs_q <= pairs_q >> 1;
          next_q  <= '0;
        end
      end
    end
  end

  assign busy_o = busy_q;
  assign done_o = level_end && pairs_q == '0;
  assign y_o    = v_q[0];

endmodule
