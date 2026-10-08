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
// row's end needs a context for its final sum (below).
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
// context, Ctx of them (3: a final sum takes about 30 cycles from the
// row's last beat, two rows' worth at 16 beats a row). One more fp_add sums each context in the tree of
// section 3, one add per cycle, oldest context first; a node issues once
// its two children are done. An add writes its result over its left operand.
// The root, which frees the context, issues only for the oldest context and
// when the output FIFO has room for it, so results leave in row order (oldest
// first already gives that order; the condition makes it certain).
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
  localparam int unsigned Levels     = $clog2(A);
  localparam int unsigned Nodes      = A - 1;  // adds of the tree; the root is the last
  localparam int unsigned NW         = $clog2(Nodes);
  localparam int unsigned AW         = $clog2(A);
  localparam int unsigned Ctx        = 3;  // rows whose final sum is under way
  localparam int unsigned CW         = $clog2(Ctx);
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

  logic [CW:0] reserved_q;  // rows past their last beat whose root has not issued
  logic        fire, root_issue;
  assign in_ready_o = !last_i || (reserved_q < (CW + 1)'(Ctx));
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

  // --- Final sum: contexts and the tree's adder ------------------------------

  // Node j of level l (1 .. Levels; n-th of its level) adds entries n * 2^l
  // and n * 2^l + 2^(l-1) of its context and writes the first; its children
  // are nodes 2n and 2n + 1 of level l - 1.
  logic [AW-1:0]  left_of[Nodes], right_of[Nodes];
  logic [31:0]    ent_q[Ctx][A];
  logic [Ctx-1:0] busy_q;
  logic [Nodes-1:0] issued_q[Ctx];
  logic [Nodes-2:0] done_q[Ctx];  // the root's result leaves the context
  logic [CW-1:0]  snap_ptr_q, old_ptr_q;
  logic           out_room;
  logic [Nodes-1:0] ready[Ctx];

  function automatic logic [CW-1:0] next_ctx(logic [CW-1:0] c);
    return (c == CW'(Ctx - 1)) ? '0 : c + 1'b1;
  endfunction

  for (genvar l = 1; l <= Levels; l++) begin : gen_level
    localparam int unsigned Base = A - (A >> (l - 1));  // the level's first node
    for (genvar n = 0; n < (A >> l); n++) begin : gen_node
      localparam int unsigned J = Base + n;
      assign left_of[J]  = AW'(n << l);
      assign right_of[J] = AW'((n << l) + (1 << (l - 1)));
      for (genvar c = 0; c < Ctx; c++) begin : gen_ctx
        logic children_done;
        if (l == 1) begin : gen_leaf
          assign children_done = 1'b1;
        end else begin : gen_inner
          localparam int unsigned Child = A - (A >> (l - 2)) + 2 * n;
          assign children_done = done_q[c][Child] && done_q[c][Child+1];
        end
        if (l == Levels) begin : gen_root
          assign ready[c][J] = busy_q[c] && !issued_q[c][J] && children_done
                               && CW'(c) == old_ptr_q && out_room;
        end else begin : gen_other
          assign ready[c][J] = busy_q[c] && !issued_q[c][J] && children_done;
        end
      end
    end
  end

  // Oldest context first, then the lowest node: any order gives the same bits.
  logic          pick;
  logic [CW-1:0] pick_c, scan;
  logic [NW-1:0] pick_j;
  always_comb begin
    pick   = 1'b0;
    pick_c = '0;
    pick_j = '0;
    scan = old_ptr_q;
    for (int unsigned r = 0; r < Ctx; r++) begin
      for (int unsigned j = 0; j < Nodes; j++) begin
        if (!pick && ready[scan][j]) begin
          pick   = 1'b1;
          pick_c = scan;
          pick_j = NW'(j);
        end
      end
      scan = next_ctx(scan);
    end
  end
  assign root_issue = pick && pick_j == NW'(Nodes - 1);

  // Issue register: the operands, then the adder.
  typedef struct packed {
    logic          valid;
    logic          root;
    logic [CW-1:0] c;
    logic [NW-1:0] j;
  } tree_tag_t;

  tree_tag_t   iss_q;
  logic [31:0] op_a_q, op_b_q;
  always_ff @(posedge clk_i) begin
    iss_q  <= '{valid: pick, root: root_issue, c: pick_c, j: pick_j};
    op_a_q <= ent_q[pick_c][left_of[pick_j]];
    op_b_q <= ent_q[pick_c][right_of[pick_j]];
    if (!rst_ni) iss_q.valid <= 1'b0;
  end

  tree_tag_t   tree_tag_q[3];  // iss_q through fp_add's registers
  logic        tree_valid;
  logic [31:0] tree_sum;
  always_ff @(posedge clk_i) begin
    tree_tag_q[0] <= iss_q;
    tree_tag_q[1] <= tree_tag_q[0];
    tree_tag_q[2] <= tree_tag_q[1];
  end

  fp_add #(
    .W  (W),
    .Ftz(Ftz)
  ) u_tree (
    .clk_i,
    .rst_ni,
    .valid_i(iss_q.valid),
    .a_i    (decode_f32(op_a_q)),
    .b_i    (op_b_q),
    .valid_o(tree_valid),
    .y_o    (tree_sum)
  );

  logic tree_result, root_result;  // an inner node's sum, the row's result
  assign tree_result = tree_valid && !tree_tag_q[2].root;
  assign root_result = tree_valid && tree_tag_q[2].root;

  always_ff @(posedge clk_i) begin
    if (snap) begin
      for (int unsigned s = 0; s < S; s++) begin
        for (int unsigned e = 0; e < E; e++) begin
          ent_q[snap_ptr_q][s*E+e] <= written[e][s] ? acc[e][s] : 32'h0;
        end
      end
    end
    if (tree_result) ent_q[tree_tag_q[2].c][left_of[tree_tag_q[2].j]] <= tree_sum;
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      busy_q     <= '0;
      snap_ptr_q <= '0;
      old_ptr_q  <= '0;
      reserved_q <= '0;
      for (int unsigned c = 0; c < Ctx; c++) begin
        issued_q[c] <= '0;
        done_q[c]   <= '0;
      end
    end else begin
      // A copy goes to a free context, and a root frees a busy one: never the
      // same context in one cycle.
      if (snap) begin
        busy_q[snap_ptr_q]   <= 1'b1;
        issued_q[snap_ptr_q] <= '0;
        done_q[snap_ptr_q]   <= '0;
        snap_ptr_q           <= next_ctx(snap_ptr_q);
      end
      if (pick) issued_q[pick_c][pick_j] <= 1'b1;
      if (root_issue) begin
        busy_q[pick_c] <= 1'b0;
        old_ptr_q      <= next_ctx(old_ptr_q);
      end
      if (tree_result) done_q[tree_tag_q[2].c][tree_tag_q[2].j] <= 1'b1;
      reserved_q <= reserved_q + (CW + 1)'(fire && last_i) - (CW + 1)'(root_issue);
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
