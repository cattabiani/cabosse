// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The vector unit (D-030, D-048; docs/architecture.md, "Vector unit"): the
// on-chip buffers and the elementwise commands ADD, MUL, FMA, EXP, RECIP,
// RSQRT and ROTATE_HALF (model/commands.py, golden: vector_command). The
// reductions come next (D-050).
//
// Buffers: ids 1 to Bufs - 1 (0 is NONE), each of Len elements, FP32 or,
// where Bf16Mask has the id's bit, BF16. A buffer reads and writes up to
// Lanes consecutive elements a cycle from any element: Lanes banks, element
// k in bank k mod Lanes (tile_buffer), one read and one write a cycle. The
// unit's writes to a BF16 buffer are rounded (fp32_to_bf16); its sources
// are FP32 buffers.
//
// A command walks its rows in groups of up to Lanes elements that never
// cross a row: n rows of m, or with LEN_T n rows of t at stride (t and
// stride are the registers, taken with the command). An elementwise command
// without B_PER_ROW or B_ROW is one row of n * m; ROTATE_HALF walks a row's
// two halves (the first half of the output reads the second, negated). A
// group a cycle: a, b and c read, then Lanes fp32_chain slots, then the
// write to dst; done_o pulses at a command's last write, in order. The next
// command is taken while earlier ones drain: the unit does not check that it
// reads what one of them writes (the controller waits, D-045).
//
// The ext port reads and writes any buffer's raw bits (BF16 in the low
// half): the testbench now, LOAD and the engine later. It must not use a
// buffer's read or write port in a cycle the unit uses it: the unit's access
// wins and nothing flags it. Nothing enforces this yet; LOAD and the engine's
// ports settle how (M6's LOAD step, M7).

module vec_unit
  import fp_pkg::*;
#(
  parameter  int unsigned     Lanes    = 16,
  parameter  int unsigned     Bufs     = 22,
  parameter  int unsigned     Len      = 1024,
  parameter  logic [31:0]     Bf16Mask = 32'h4644,  // bit per id: X, QR, P, ATT, M (commands.buffers)
  parameter  int unsigned     AddrW    = 20,
  parameter  int unsigned     MulRegs  = 2,
  localparam int unsigned     BufW     = $clog2(Bufs),
  localparam int unsigned     LW       = $clog2(Lanes)
) (
  input  logic                         clk_i,
  input  logic                         rst_ni,
  // Commands (model/commands.py's fields), in order
  input  logic                         cmd_valid_i,
  output logic                         cmd_ready_o,
  input  logic [4:0]                   cmd_op_i,
  input  logic [BufW-1:0]              cmd_dst_i,
  input  logic [BufW-1:0]              cmd_a_i,
  input  logic [BufW-1:0]              cmd_b_i,
  input  logic [BufW-1:0]              cmd_c_i,
  input  logic [7:0]                   cmd_flags_i,
  input  logic [AddrW-1:0]             cmd_n_i,
  input  logic [AddrW-1:0]             cmd_m_i,
  input  logic [31:0]                  cmd_scalar_i,
  input  logic [AddrW-1:0]             t_i,          // positions attention reads
  input  logic [AddrW-1:0]             stride_i,     // the STRIDE register
  output logic                         done_o,       // a command's last write
  // ext: raw bits of any buffer
  input  logic                         ext_w_valid_i,
  input  logic [BufW-1:0]              ext_w_buf_i,
  input  logic [AddrW-1:0]             ext_w_addr_i,
  input  logic [Lanes-1:0]             ext_w_mask_i,
  input  logic [Lanes-1:0][31:0]       ext_w_data_i,
  input  logic                         ext_r_valid_i,
  input  logic [BufW-1:0]              ext_r_buf_i,
  input  logic [AddrW-1:0]             ext_r_addr_i,
  output logic [Lanes-1:0][31:0]       ext_r_data_o  // the cycle after
);

  // model/commands.py's opcodes and flags (docs/architecture.md, "Commands").
  localparam logic [4:0] OpAdd = 5'd7, OpMul = 5'd8, OpFma = 5'd9, OpExp = 5'd10;
  localparam logic [4:0] OpRecip = 5'd11, OpRsqrt = 5'd12, OpRotateHalf = 5'd16;
  localparam int unsigned NegA = 0, NegB = 1, BPerRow = 2, BRow = 3, LenT = 4;
  localparam int unsigned Depth = (Len + Lanes - 1) / Lanes;  // a bank's entries
  localparam int unsigned DW = $clog2(Depth);
  localparam int unsigned ChainLat = chain_latency(MulRegs);

  if (Lanes != 2 ** LW) begin : gen_lanes_check
    $error("Lanes must be a power of two");
  end
  if (Bufs > 32) begin : gen_bufs_check
    $error("Bufs must fit Bf16Mask's 32 bits");
  end

  // --- Walk: a command's groups ---------------------------------------------

  // The command being walked, with its walk.
  typedef struct packed {
    logic [4:0]       op;
    logic [BufW-1:0]  dst, a, b, c;
    logic [7:0]       flags;
    logic [31:0]      scalar;
    logic [AddrW-1:0] rows, len, row_stride;
  } walk_t;

  // What a group's operands, the slots and the write need of it, from its
  // reads on.
  typedef struct packed {
    logic [4:0]       op;
    logic [BufW-1:0]  dst, a, b, c;
    logic             neg_a, neg_b, b_per_row;
    logic [31:0]      scalar;
    logic [AddrW-1:0] addr;  // dst's and c's element (and a's but in ROTATE_HALF)
    logic [Lanes-1:0] mask;
    logic             last;  // the command's last group
  } issued_t;

  // One group: Lanes elements at most, within a row.
  typedef struct packed {
    issued_t          is;
    logic [AddrW-1:0] a_addr, b_addr;
  } group_t;

  function automatic logic elementwise(logic [4:0] op);
    return op inside {OpAdd, OpMul, OpFma, OpExp, OpRecip, OpRsqrt};
  endfunction

  function automatic walk_t take(logic [4:0] op, logic [BufW-1:0] dst, a, b, c, logic [7:0] flags,
                                 logic [AddrW-1:0] n, m, logic [31:0] scalar,
                                 logic [AddrW-1:0] t, stride);
    walk_t w;
    logic  flat;
    flat = elementwise(op) && !flags[BPerRow] && !flags[BRow] && !flags[LenT];
    w = '{op: op, dst: dst, a: a, b: b, c: c, flags: flags, scalar: scalar, default: '0};
    w.rows       = flat ? AddrW'(1) : n;
    w.len        = flags[LenT] ? t : flat ? AddrW'(n * m) : m;
    w.row_stride = flags[LenT] ? stride : m;
    return w;
  endfunction

  walk_t            cur_q;
  logic             active_q;
  logic [AddrW-1:0] row_q, base_q, g_q;  // row, its first element, the group's offset in it
  group_t           grp;
  logic             grp_fire, row_end;
  logic [AddrW-1:0] half, seg_end, count, g_next, addr;
  logic             rotate, first_half;

  always_comb begin
    half       = cur_q.len >> 1;
    rotate     = cur_q.op == OpRotateHalf;
    first_half = g_q < half;
    seg_end    = rotate && first_half ? half : cur_q.len;
    count      = seg_end - g_q > AddrW'(Lanes) ? AddrW'(Lanes) : seg_end - g_q;
    g_next     = g_q + count;
    row_end    = g_next == cur_q.len;
    addr       = base_q + g_q;

    grp.is = '{op: cur_q.op, dst: cur_q.dst, a: cur_q.a, b: cur_q.b, c: cur_q.c,
               neg_a: rotate ? first_half : cur_q.flags[NegA], neg_b: cur_q.flags[NegB],
               b_per_row: cur_q.flags[BPerRow], scalar: cur_q.scalar, addr: addr,
               mask: Lanes'((1 << count) - 1), last: row_end && row_q + 1'b1 == cur_q.rows};
    grp.a_addr = !rotate ? addr : first_half ? addr + half : addr - half;
    grp.b_addr = cur_q.flags[BPerRow] ? row_q : cur_q.flags[BRow] ? g_q : addr;
  end

  assign grp_fire    = active_q;
  assign cmd_ready_o = !active_q || grp.is.last;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      active_q <= 1'b0;
    end else if (cmd_valid_i && cmd_ready_o) begin
      active_q <= 1'b1;
      cur_q    <= take(cmd_op_i, cmd_dst_i, cmd_a_i, cmd_b_i, cmd_c_i, cmd_flags_i, cmd_n_i,
                       cmd_m_i, cmd_scalar_i, t_i, stride_i);
      {row_q, base_q, g_q} <= '0;
    end else if (grp_fire) begin
      if (grp.is.last) begin
        active_q <= 1'b0;
      end else if (row_end) begin
        row_q  <= row_q + 1'b1;
        base_q <= base_q + cur_q.row_stride;
        g_q    <= '0;
      end else begin
        g_q <= g_next;
      end
    end
  end

  // --- Buffers ----------------------------------------------------------------

  // The bank entry of element e + l: e's entry, or the next one where e's
  // place in its entry plus l passes the lanes ((e + l) >> LW without an
  // adder per lane).
  function automatic logic [DW-1:0] bank_addr(logic [AddrW-1:0] e, int unsigned l);
    return DW'(e >> LW) + DW'(32'(e[LW-1:0]) + l >= Lanes);
  endfunction

  logic [Bufs-1:0][AddrW-1:0]        rd_addr, wr_addr;
  logic [Bufs-1:0][Lanes-1:0]        wr_mask;
  logic [Bufs-1:0][Lanes-1:0][31:0]  wr_data, rd_data;

  // The unit's write (below) and the read of a group, else ext's.
  logic                    u_w_valid;
  logic [BufW-1:0]         u_w_buf;
  logic [AddrW-1:0]        u_w_addr;
  logic [Lanes-1:0]        u_w_mask;
  logic [Lanes-1:0][31:0]  u_w_data;

  // A buffer's read address is the group's where it is a source, else ext's
  // (the banks read every cycle; an unused read is harmless). Its write
  // mask is zero unless this cycle writes it.
  always_comb begin
    for (int unsigned i = 0; i < Bufs; i++) begin
      rd_addr[i] = ext_r_addr_i;
      if (grp_fire && grp.is.c == BufW'(i)) rd_addr[i] = grp.is.addr;
      if (grp_fire && grp.is.b == BufW'(i)) rd_addr[i] = grp.b_addr;
      if (grp_fire && grp.is.a == BufW'(i)) rd_addr[i] = grp.a_addr;
      wr_addr[i] = ext_w_addr_i;
      wr_mask[i] = ext_w_valid_i && ext_w_buf_i == BufW'(i) ? ext_w_mask_i : '0;
      wr_data[i] = ext_w_data_i;
      if (u_w_valid && u_w_buf == BufW'(i)) begin
        {wr_addr[i], wr_mask[i], wr_data[i]} = {u_w_addr, u_w_mask, u_w_data};
      end
    end
  end

  for (genvar i = 1; i < Bufs; i++) begin : gen_buf
    localparam int unsigned W = Bf16Mask[i] ? 16 : 32;
    logic [Lanes-1:0][DW-1:0] wa, ra;
    logic [Lanes-1:0][W-1:0]  wd, rd;
    for (genvar l = 0; l < Lanes; l++) begin : gen_lane
      assign wa[l] = bank_addr(wr_addr[i], l);
      assign ra[l] = bank_addr(rd_addr[i], l);
      assign wd[l] = W'(wr_data[i][l]);
      assign rd_data[i][l] = 32'(rd[l]);
    end
    tile_buffer #(
      .N    (Lanes),
      .Depth(Depth),
      .W    (W)
    ) u_banks (
      .clk_i,
      .wr_valid_i(wr_mask[i]),
      .wr_addr_i (wa),
      .wr_data_i (wd),
      .wr_rot_i  (LW'(wr_addr[i])),
      .rd_addr_i (ra),
      .rd_rot_i  (LW'(rd_addr[i])),
      .rd_data_o (rd)
    );
  end
  assign rd_data[0] = '0;  // NONE

  logic [BufW-1:0] ext_r_buf_q;
  always_ff @(posedge clk_i) ext_r_buf_q <= ext_r_buf_i;
  assign ext_r_data_o = rd_data[ext_r_buf_q];

  // --- Operands, the slots, the write -----------------------------------------

  // The group whose reads come back this cycle.
  issued_t grp_q;
  logic    grp_valid_q;
  always_ff @(posedge clk_i) begin
    grp_q       <= grp.is;
    grp_valid_q <= rst_ni && grp_fire;
  end

  function automatic chain_op_e chain_op(logic [4:0] op);
    case (op)
      OpMul:   return ChainMul;
      OpFma:   return ChainFma;
      OpExp:   return ChainExp;
      OpRecip: return ChainRecip;
      OpRsqrt: return ChainRsqrt;
      OpRotateHalf: return ChainPass;
      default: return ChainAdd;
    endcase
  endfunction

  // What the write needs, alongside the slots.
  typedef struct packed {
    logic [BufW-1:0]  dst;
    logic [AddrW-1:0] addr;
    logic [Lanes-1:0] mask;
    logic             last;
  } dest_t;

  dest_t                  dest_q[ChainLat];
  logic [Lanes-1:0]       slot_valid;
  logic [Lanes-1:0][31:0] y;

  for (genvar l = 0; l < Lanes; l++) begin : gen_slot
    logic [31:0] a, b, c;
    always_comb begin
      a = rd_data[grp_q.a][l];
      b = grp_q.b_per_row ? rd_data[grp_q.b][0] : rd_data[grp_q.b][l];
      if (grp_q.b == '0) b = grp_q.scalar;
      c = rd_data[grp_q.c][l];
      a[31] = a[31] ^ grp_q.neg_a;  // exact sign flips
      b[31] = b[31] ^ grp_q.neg_b;
    end
    fp32_chain #(
      .MulRegs(MulRegs)
    ) u_chain (
      .clk_i,
      .rst_ni,
      .valid_i(grp_valid_q),
      .op_i   (chain_op(grp_q.op)),
      .a_i    (a),
      .b_i    (b),
      .c_i    (c),
      .valid_o(slot_valid[l]),
      .y_o    (y[l])
    );
  end

  always_ff @(posedge clk_i) begin
    dest_q[0] <= '{dst: grp_q.dst, addr: grp_q.addr, mask: grp_q.mask, last: grp_q.last};
    for (int unsigned j = 1; j < ChainLat; j++) dest_q[j] <= dest_q[j-1];
  end

  // The write: dest_q[ChainLat - 1] belongs to the slots' result this cycle.
  dest_t d;
  assign d         = dest_q[ChainLat-1];
  assign u_w_valid = slot_valid[0];
  assign u_w_buf   = d.dst;
  assign u_w_addr  = d.addr;
  assign u_w_mask  = d.mask;
  assign done_o    = u_w_valid && d.last;
  for (genvar l = 0; l < Lanes; l++) begin : gen_round
    logic [15:0] r;
    fp32_to_bf16 u_round (
      .a_i(y[l]),
      .y_o(r)
    );
    assign u_w_data[l] = Bf16Mask[d.dst] ? {16'h0, r} : y[l];
  end

  // verilator lint_off UNUSEDSIGNAL
  logic unused;
  // ext_r_valid_i: the banks read every cycle, so a read needs no enable.
  assign unused = ^{slot_valid[Lanes-1:1], ext_r_valid_i, rd_addr[0], wr_addr[0], wr_mask[0],
                    wr_data[0]};
  // verilator lint_on UNUSEDSIGNAL

endmodule
