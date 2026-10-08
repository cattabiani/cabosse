// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// One lane of the engine (D-027): FP32 dot products of BF16 rows, E pairs
// per cycle, in the order of docs/numerics.md, section 3 (golden model:
// golden.dot.dot). Element k of a row goes to partial sum k mod A; at the
// row's end the A partial sums are added in the fixed pairwise tree.
//
// Input: one beat per cycle with valid/ready, carrying E pairs (w, x),
// elements 4c..4c+3 of the row for E = 4. mask_i marks which elements are
// there; a missing element leaves its partial sum unchanged, as the golden
// model's `valid`. last_i ends the row. Output: one FP32 result per row, in
// row order, valid/ready. Ready may drop only on a beat with last_i set: a
// row's end needs a level-1 buffer for its final sum (below).
//
// Units: unit e takes element e of each beat, so the S = A / E partial sums
// e, e + E, e + 2E, ... are its own. Group g of a row (its g-th beat) adds
// to sum slot g mod S, kept in the unit's accumulator file: read when the
// product enters fp_add, written back when the sum leaves, so the loop is
// fp_add's 3 registers plus the file, LoopCycles cycles. Group g + S reads
// that slot at least S cycles later, so S >= LoopCycles makes any pattern of
// bubbles safe. A row's first S groups add to +0 instead of the file, which
// still holds the previous row's sums; so does a later group whose slot no
// element of this row has reached yet (missing elements): the written flags
// tell, as by then they hold this row's slots only.
//
// Final sum: LoopCycles cycles after the row's last group enters fp_add, its
// sums (+0 for a slot no element of the row reached) are copied into a free
// level-1 buffer, Banks of them. One more fp_add adds them in the tree of
// section 3 like a conveyor belt, one add per cycle: level 1 takes pairs
// (2m, 2m + 1) from the buffer in order; level l > 1 takes the front pair of
// its queue, which the results of level l - 1 enter in the order they leave
// the adder. Every row brings an even number of items to each level, so a
// pair never mixes rows, and its two items are the two the tree adds. Each
// cycle the deepest level with a pair, and room for its result in the next
// queue (the output FIFO for the last level), goes; any order gives the same
// bits, the deepest first keeps the queues short. Results leave in row order.
// A tree is A - 1 adds, so the lane keeps E pairs per cycle on rows of
// (A - 1) * E elements or more; shorter rows make ready drop at their end.

module dot_lane
  import fp_pkg::*;
#(
  parameter  int unsigned E       = 4,     // pairs per cycle (multiply-add units)
  parameter  int unsigned A       = 16,    // partial sums per row (D-027)
  parameter  int unsigned MulRegs = 1,     // registers inside each product (bf16_mac)
  parameter  bit          Ftz     = 1'b0,  // flush-to-zero variant (fp_mul_add)
  localparam int unsigned S       = A / E  // partial sums per unit
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
  output logic [31:0]         out_o
);

  localparam int unsigned W          = prod_sig_w(8);
  localparam int unsigned LoopCycles = 4;  // fp_add's 3 registers, then the file
  localparam int unsigned SW         = $clog2(S);
  localparam int unsigned Levels     = $clog2(A);  // of the tree; level i here is i + 1 there
  localparam int unsigned LW         = $clog2(Levels);
  localparam int unsigned PW         = $clog2(A / 2);  // a level-1 pair's index
  localparam int unsigned Banks      = 2;  // level-1 buffers
  localparam int unsigned BW         = $clog2(Banks);
  localparam int unsigned OutDepth   = 2;
  localparam int unsigned OW         = $clog2(OutDepth);
  // Stages of a beat's tag: 0 is the input register; it enters fp_add at
  // Rd, leaves it at Wr, and its row's sums are copied at Snap.
  localparam int unsigned Rd         = MulRegs + 1;
  localparam int unsigned Wr         = Rd + 3;
  localparam int unsigned Snap       = Wr + 1;

  if (A < 4 || (A & (A - 1)) != 0) begin : gen_bad_a
    $error("dot_lane: A must be a power of two, at least 4");
  end
  if (E == 0 || A % E != 0 || S < LoopCycles) begin : gen_bad_e
    $error("dot_lane: E must divide A with A / E >= 4, the accumulate loop's cycles");
  end

  // --- Input: handshake, group counter, input register ------------------------

  logic [BW:0] reserved_q;  // rows past their last beat whose level-1 pairs are not all taken
  logic        fire, bank_done, root_issue;
  assign in_ready_o = !last_i || (reserved_q < (BW + 1)'(Banks));
  assign fire       = in_valid_i && in_ready_o;

  // The beat's place in its row: slot = group mod S, first = group < S.
  logic [SW-1:0] slot_q;
  logic          first_q;
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      slot_q  <= '0;
      first_q <= 1'b1;
    end else if (fire) begin
      slot_q  <= last_i ? '0 : slot_q + 1'b1;
      first_q <= last_i || (first_q && slot_q != SW'(S - 1));
    end
  end

  typedef struct packed {
    logic          valid;
    logic          last;
    logic          first;
    logic [SW-1:0] slot;
    logic [E-1:0]  mask;
  } tag_t;

  tag_t              tag_q[Snap+1];
  logic [E-1:0][15:0] w_q, x_q;
  always_ff @(posedge clk_i) begin
    tag_q[0] <= '{valid: fire, last: last_i, first: first_q, slot: slot_q, mask: mask_i};
    w_q      <= w_i;
    x_q      <= x_i;
    if (!rst_ni) tag_q[0].valid <= 1'b0;
  end
  for (genvar i = 0; i < Snap; i++) begin : gen_tag
    always_ff @(posedge clk_i) begin
      tag_q[i+1] <= tag_q[i];
      if (!rst_ni) tag_q[i+1].valid <= 1'b0;
    end
  end

  logic snap;  // copy the row's sums into a context this cycle
  assign snap = tag_q[Snap].valid && tag_q[Snap].last;

  // --- Units: product, add, accumulator file ---------------------------------

  logic [31:0] acc[E][S];  // the files, for the copy into a context
  logic [S-1:0] written[E];  // which slots the current row has reached

  for (genvar e = 0; e < E; e++) begin : gen_unit
    logic [W+11:0] p_d, p_q;
    fp_product #(
      .M   (8),
      .Regs(MulRegs),
      .Ftz (Ftz)
    ) u_mul (
      .clk_i,
      .a_i(w_q[e]),
      .b_i(x_q[e]),
      .p_o(p_d)
    );
    always_ff @(posedge clk_i) p_q <= p_d;

    logic [31:0] file_q[S];
    logic [S-1:0] written_q;
    logic [31:0] addend, sum;
    logic        sum_valid;
    assign addend = (tag_q[Rd].first || !written_q[tag_q[Rd].slot]) ? 32'h0
                    : file_q[tag_q[Rd].slot];

    fp_add #(
      .W  (W),
      .Ftz(Ftz)
    ) u_add (
      .clk_i,
      .rst_ni,
      .valid_i(tag_q[Rd].valid && tag_q[Rd].mask[e]),
      .a_i    (p_q),
      .b_i    (addend),
      .valid_o(sum_valid),
      .y_o    (sum)
    );

    always_ff @(posedge clk_i) begin
      if (sum_valid) file_q[tag_q[Wr].slot] <= sum;
    end
    // The copy clears the flags in the cycle the next row's first sum may be
    // written; that write sets its flag (the later assignment wins).
    always_ff @(posedge clk_i) begin
      if (!rst_ni) written_q <= '0;
      else begin
        if (snap) written_q <= '0;
        if (sum_valid) written_q[tag_q[Wr].slot] <= 1'b1;
      end
    end

    assign acc[e]     = file_q;
    assign written[e] = written_q;
  end

  // --- Final sum: level-1 buffers, queues and the tree's adder ---------------

  // Per tree level i (0 is level 1): whether a pair is ready, its two items,
  // and whether its result has room.
  logic [Levels-1:0] has_pair, room;
  logic [31:0]       front_a[Levels], front_b[Levels];
  logic              out_room;

  logic          pick;
  logic [LW-1:0] pick_lvl;
  always_comb begin
    pick     = 1'b0;
    pick_lvl = '0;
    for (int i = Levels - 1; i >= 0; i--) begin
      if (!pick && has_pair[i] && room[i]) begin
        pick     = 1'b1;
        pick_lvl = LW'(i);
      end
    end
  end
  assign root_issue = pick && pick_lvl == LW'(Levels - 1);

  // Level 1: the buffers, read pair by pair.
  logic [31:0]      bank_q[Banks][A];
  logic [Banks-1:0] full_q;
  logic [BW-1:0]    load_ptr_q, read_ptr_q;
  logic [PW-1:0]    pair_q;
  assign has_pair[0] = full_q[read_ptr_q];
  assign front_a[0]  = bank_q[read_ptr_q][{pair_q, 1'b0}];
  assign front_b[0]  = bank_q[read_ptr_q][{pair_q, 1'b1}];
  assign bank_done   = pick && pick_lvl == '0 && pair_q == PW'(A / 2 - 1);

  always_ff @(posedge clk_i) begin
    if (snap) begin
      for (int unsigned s = 0; s < S; s++) begin
        for (int unsigned e = 0; e < E; e++) begin
          bank_q[load_ptr_q][s*E+e] <= written[e][s] ? acc[e][s] : 32'h0;
        end
      end
    end
  end
  // A copy goes to a free buffer (ready keeps the rows in flight to Banks).
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      full_q     <= '0;
      load_ptr_q <= '0;
      read_ptr_q <= '0;
      pair_q     <= '0;
      reserved_q <= '0;
    end else begin
      if (snap) begin
        full_q[load_ptr_q] <= 1'b1;
        load_ptr_q         <= load_ptr_q + 1'b1;
      end
      if (pick && pick_lvl == '0) pair_q <= pair_q + 1'b1;
      if (bank_done) begin
        full_q[read_ptr_q] <= 1'b0;
        read_ptr_q         <= read_ptr_q + 1'b1;
      end
      reserved_q <= reserved_q + (BW + 1)'(fire && last_i) - (BW + 1)'(bank_done);
    end
  end

  // The adder, after an issue register.
  logic          iss_valid_q;
  logic [LW-1:0] iss_lvl_q;
  logic [31:0]   op_a_q, op_b_q;
  always_ff @(posedge clk_i) begin
    iss_valid_q <= rst_ni && pick;
    iss_lvl_q   <= pick_lvl;
    op_a_q      <= front_a[pick_lvl];
    op_b_q      <= front_b[pick_lvl];
  end

  logic [LW-1:0] tree_lvl_q[3];  // iss_lvl_q through fp_add's registers
  logic          tree_valid;
  logic [31:0]   tree_sum;
  always_ff @(posedge clk_i) begin
    tree_lvl_q[0] <= iss_lvl_q;
    tree_lvl_q[1] <= tree_lvl_q[0];
    tree_lvl_q[2] <= tree_lvl_q[1];
  end

  fp_add #(
    .W  (W),
    .Ftz(Ftz)
  ) u_tree (
    .clk_i,
    .rst_ni,
    .valid_i(iss_valid_q),
    .a_i    (decode_f32(op_a_q)),
    .b_i    (op_b_q),
    .valid_o(tree_valid),
    .y_o    (tree_sum)
  );

  logic root_result;  // the row's result
  assign root_result = tree_valid && tree_lvl_q[2] == LW'(Levels - 1);
  assign room[Levels-1] = out_room;

  // Levels 2 and up: a queue each, one row's items deep (A >> i), counting
  // the results still in the adder against its room.
  for (genvar i = 1; i < Levels; i++) begin : gen_queue
    localparam int unsigned D  = A >> i;
    localparam int unsigned QW = $clog2(D);
    logic [31:0] q_q[D];
    logic [QW-1:0] rd_q, wr_q;
    logic [QW:0]   count_q, flying_q;  // items held, results on their way
    logic          push, pop2, issue_below;
    assign push        = tree_valid && tree_lvl_q[2] == LW'(i - 1);
    assign pop2        = pick && pick_lvl == LW'(i);
    assign issue_below = pick && pick_lvl == LW'(i - 1);
    assign has_pair[i] = count_q >= 2;
    assign front_a[i]  = q_q[rd_q];
    assign front_b[i]  = q_q[rd_q+1'b1];
    assign room[i-1]   = count_q + flying_q < (QW + 1)'(D);

    always_ff @(posedge clk_i) begin
      if (push) q_q[wr_q] <= tree_sum;
    end
    always_ff @(posedge clk_i) begin
      if (!rst_ni) begin
        rd_q     <= '0;
        wr_q     <= '0;
        count_q  <= '0;
        flying_q <= '0;
      end else begin
        if (push) wr_q <= wr_q + 1'b1;
        if (pop2) rd_q <= rd_q + QW'(2);
        count_q  <= count_q + (QW + 1)'(push) - (pop2 ? (QW + 1)'(2) : '0);
        flying_q <= flying_q + (QW + 1)'(issue_below) - (QW + 1)'(push);
      end
    end
  end

  // --- Output FIFO --------------------------------------------------------------

  logic [31:0] fifo_q[OutDepth];
  logic [OW-1:0] rd_ptr_q, wr_ptr_q;
  logic [OW:0]   count_q, roots_q;  // results held, roots in the adder
  logic          pop;
  assign out_valid_o = count_q != 0;
  assign out_o       = fifo_q[rd_ptr_q];
  assign pop         = out_valid_o && out_ready_i;
  assign out_room    = count_q + roots_q < (OW + 1)'(OutDepth);

  always_ff @(posedge clk_i) begin
    if (root_result) fifo_q[wr_ptr_q] <= tree_sum;
  end
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      rd_ptr_q <= '0;
      wr_ptr_q <= '0;
      count_q  <= '0;
      roots_q  <= '0;
    end else begin
      if (root_result) wr_ptr_q <= OW'((wr_ptr_q + 1) % OutDepth);
      if (pop) rd_ptr_q <= OW'((rd_ptr_q + 1) % OutDepth);
      count_q <= count_q + (OW + 1)'(root_result) - (OW + 1)'(pop);
      roots_q <= roots_q + (OW + 1)'(root_issue) - (OW + 1)'(root_result);
    end
  end

endmodule
