// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The vector unit (D-030, D-048, D-050; docs/architecture.md, "Vector unit"):
// the on-chip buffers, the elementwise commands ADD, MUL, FMA, EXP, RECIP,
// RSQRT and ROTATE_HALF, the reductions SUM, SUMSQ and MAX
// (model/commands.py, golden: vector_command), and LOAD, KV_STORE and
// V_STORE through its AXI port (golden: load_command, kv_store_command).
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
// A reduction (docs/numerics.md, sections 3 and 4): element k of a row goes
// to partial k mod Partials, so lane l of a row's group g to partial
// Lanes * (g mod Parts) + l, which slot l keeps in its file of Parts. SUM and
// SUMSQ accumulate with the chain's first FMA (add(x, p), fma(x, x, p)), its
// result back in the file MulRegs + 5 cycles later, before the partial's next
// group (Parts >= MulRegs + 6). A row's first Parts groups start from +0 and
// a missing element keeps its partial, so a partial no element reaches stays
// +0. MAX takes x through the same FMA (add(x, -0) = x; a NaN as the
// canonical NaN, which max_f32 drops as it does any NaN) and keeps
// max_f32(p, x), from the canonical NaN, which max_f32 leaves to any other
// value. At a row's end the tree unit (vec_tree) takes its partials, the
// cycle before the next row's first result lands, and writes the row's value
// to dst[row]; a row's last group waits until the tree is free, so the tree
// runs while the slots take the next row. After a reduction the next command
// is taken once its last row is written: only that row's tree shows.
//
// LOAD and the stores (PLAN.md, M6 answers of 2026-10-29) run in the same
// pipeline, so their writes keep the order of every other command's. AXI
// beats are DataWidth = Lanes * 16 bits: a group of BF16 values, half a group
// of FP32. LOAD (one row of m, the tensor at addr or its row token or
// position) reads m * 2 or m * 4 bytes from addr + row * those bytes
// (port_dma); a group's beats take the place of a's read (BF16 converted up
// exactly), pass the chain unchanged and are written like any result. Its
// rows must be whole beats: m a multiple of Lanes (BF16) or Lanes / 2 (FP32).
// KV_STORE and V_STORE (n heads of m, at position t - 1; m a power of two
// and a multiple of Lanes, V_STORE's at most L = KvPorts * LanesPerPort;
// cap a multiple of L; the cache's addr 128-byte aligned) write a K or a V
// cache in the order the engine's ports read it (D-062: KvPorts ports of
// LanesPerPort lanes, E values a lane a beat, commands.kv_lane_order): each
// KV head's region after the last, the same offset in every channel, at the
// global address channel * 2^ChanW + addr + offset. A group (Lanes values of
// a head's row) is read only when its beat has a place in the queue, and
// queued, rounded to BF16, as it leaves the slots; the writer turns it into
// strobed beats, a chunk's 8 bytes each:
// - K, position q: a group's LanesPerPort chunks of E dimensions are one
//   burst on channel (q / LanesPerPort) mod KvPorts, each chunk in the slot
//   q mod LanesPerPort of its beat; the row's m / E beats start at beat
//   (q / L) * m / E of the head's region of cap / L * m / E beats.
// - V (stored as Vᵀ, once per head of a VALUES pass: 2 copies if 2m <= L):
//   a group's E chunks of LanesPerPort dimensions each go in a one-beat
//   burst, dimensions d to d + LanesPerPort - 1 to channel d / LanesPerPort
//   (and the copy m / LanesPerPort channels on), at beat q / E of the head's
//   region of cap / E beats, dimension d + i in element E * i + q mod E.
// The walk queues each group's first burst address as it reads the group;
// the writer sends the bursts and their beats in that order. A KV_STORE or
// V_STORE is done once its last write is answered (D-044); the unit takes
// later commands meanwhile, and done_o stays one pulse per command in order:
// a done waits behind an earlier store's (a queue of the dones, one cycle
// later than the write that ends a command). err_o (sticky): a read or write
// answered with an error.
//
// The ext port reads and writes any buffer's raw bits (BF16 in the low
// half): the testbench now, the engine later. It must not use a buffer's read
// or write port in a cycle the unit uses it: the unit's access wins and
// nothing flags it. The engine's ports settle how (M7).

module vec_unit
  import fp_pkg::*;
#(
  parameter  int unsigned     Lanes    = 16,
  parameter  int unsigned     Bufs     = 22,
  parameter  int unsigned     Len      = 1024,
  parameter  logic [31:0]     Bf16Mask = 32'h4644,  // per id: X, QR, P, ATT, M (commands.buffers)
  parameter  int unsigned     AddrW    = 20,
  parameter  int unsigned     MulRegs  = 2,
  parameter  int unsigned     Partials = 128,  // a reduction's partial sums, S (D-030)
  parameter  int unsigned     MemAddrW = 34,   // F2's HBM: 16 GiB
  parameter  int unsigned     KvPorts  = 32,   // the engine's ports: the KV cache's channels (D-062)
  parameter  int unsigned     ChanW    = 29,   // log2 of a channel's bytes: 512 MiB
  parameter  int unsigned     MemDepth = 64,   // beats: port_dma's FIFO; groups: the stores' queue
  parameter  int unsigned     WrOutstanding = 128,  // write bursts sent, not yet answered (to verify on F2)
  parameter  int unsigned     Dones    = 64,   // commands taken and not yet done, at most
  localparam int unsigned     DataWidth = Lanes * 16,  // an AXI beat
  localparam int unsigned     BufW     = $clog2(Bufs),
  localparam int unsigned     LW       = $clog2(Lanes),
  localparam int unsigned     Parts    = Partials / Lanes,  // a slot's partials
  localparam int unsigned     PW       = $clog2(Parts)
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
  input  logic [AddrW-1:0]             cmd_cap_i,
  input  logic [MemAddrW-1:0]          cmd_addr_i,
  input  logic [AddrW-1:0]             token_i,
  input  logic [AddrW-1:0]             t_i,          // positions attention reads: position + 1
  input  logic [AddrW-1:0]             stride_i,     // the STRIDE register
  output logic                         done_o,       // a command done, in order
  output logic                         err_o,
  // AXI read channels (LOAD)
  output logic                         ar_valid_o,
  input  logic                         ar_ready_i,
  output logic [MemAddrW-1:0]          ar_addr_o,
  output logic [7:0]                   ar_len_o,
  output logic [2:0]                   ar_size_o,
  output logic [1:0]                   ar_burst_o,
  input  logic                         r_valid_i,
  output logic                         r_ready_o,
  input  logic [DataWidth-1:0]         r_data_i,
  input  logic [1:0]                   r_resp_i,
  // AXI write channels (KV_STORE, V_STORE)
  output logic                         aw_valid_o,
  input  logic                         aw_ready_i,
  output logic [MemAddrW-1:0]          aw_addr_o,
  output logic [7:0]                   aw_len_o,
  output logic [2:0]                   aw_size_o,
  output logic [1:0]                   aw_burst_o,
  output logic                         w_valid_o,
  input  logic                         w_ready_i,
  output logic [DataWidth-1:0]         w_data_o,
  output logic [DataWidth/8-1:0]       w_strb_o,
  output logic                         w_last_o,
  input  logic                         b_valid_i,
  output logic                         b_ready_o,
  input  logic [1:0]                   b_resp_i,
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
  localparam logic [4:0] OpLoad = 5'd1, OpKvStore = 5'd3, OpVStore = 5'd17;
  localparam logic [4:0] OpAdd = 5'd7, OpMul = 5'd8, OpFma = 5'd9, OpExp = 5'd10;
  localparam logic [4:0] OpRecip = 5'd11, OpRsqrt = 5'd12, OpSum = 5'd13, OpSumsq = 5'd14;
  localparam logic [4:0] OpMax = 5'd15, OpRotateHalf = 5'd16;
  localparam int unsigned NegA = 0, NegB = 1, BPerRow = 2, BRow = 3, LenT = 4;
  localparam int unsigned ByToken = 5, ByPosition = 6, SrcBf16 = 7;
  localparam int unsigned KvW = $clog2(KvPorts);
  localparam int unsigned BeatW = $clog2(Lanes * 2);  // log2 of a beat's bytes
  localparam int unsigned E = 4;                       // the engine's values a lane a beat (D-027)
  localparam int unsigned EW = $clog2(E);
  localparam int unsigned LanesPerPort = Lanes / E;    // the engine's lanes a port: a beat's chunks
  localparam int unsigned PpW = $clog2(LanesPerPort);
  localparam int unsigned KvBurst = LanesPerPort;      // K: a group's beats
  localparam int unsigned LenW = $clog2(KvBurst + 1);
  localparam int unsigned MW = $clog2(MemDepth + 1);
  localparam int unsigned DoneW = $clog2(Dones + 1);
  localparam int unsigned Depth = (Len + Lanes - 1) / Lanes;  // a bank's entries
  localparam int unsigned DW = $clog2(Depth);
  localparam int unsigned ChainLat = chain_latency(MulRegs);
  localparam int unsigned AccLat = mul_add_latency(MulRegs) + 1;  // chain in to s1_y_o
  localparam logic [31:0] NegZero = 32'h8000_0000;

  if (Lanes != 2 ** LW || Len < 2 * Lanes) begin : gen_lanes_check
    $error("Lanes must be a power of two, and Len at least two groups of them");
  end
  if (Bufs > 32) begin : gen_bufs_check
    $error("Bufs must fit Bf16Mask's 32 bits");
  end
  if (Partials != Parts * Lanes || Parts != 2 ** PW || Parts < AccLat + 1) begin : gen_parts_check
    $error("Partials must be a power-of-two multiple of Lanes, at least MulRegs + 6 a slot");
  end
  if (KvPorts != 2 ** KvW || MemAddrW < ChanW + KvW) begin : gen_mem_check
    $error("KvPorts must be a power of two, MemAddrW hold its channels");
  end
  if (Lanes < 2 * E || KvBurst > 16) begin : gen_kv_check
    $error("Lanes must be 2 * E to 16 * E: a beat's chunks, a K burst (AXI3)");
  end

  // --- Walk: a command's groups ---------------------------------------------

  // The command being walked, with its walk.
  typedef struct packed {
    logic [4:0]          op;
    logic [BufW-1:0]     dst, a, b, c;
    logic [7:0]          flags;
    logic [31:0]         scalar;
    logic [AddrW-1:0]    rows, len, row_stride;
    // LOAD and the stores: the tensor's address, the row LOAD reads (token,
    // position or 0) or the position a store writes, the cache's rows.
    logic [MemAddrW-1:0] addr;
    logic [AddrW-1:0]    mrow, cap;
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
    // A reduction's: the group's partial in each slot, whether it is the
    // row's first group, in its first Parts (which read no partial), or its
    // last, and the row.
    logic             row_first, first, row_last;
    logic [PW-1:0]    part;
    logic [AddrW-1:0] row;
  } issued_t;

  // One group: Lanes elements at most, within a row.
  typedef struct packed {
    issued_t          is;
    logic [AddrW-1:0] a_addr, b_addr;
  } group_t;

  function automatic logic elementwise(logic [4:0] op);
    return op inside {OpAdd, OpMul, OpFma, OpExp, OpRecip, OpRsqrt};
  endfunction

  function automatic logic store(logic [4:0] op);
    return op inside {OpKvStore, OpVStore};
  endfunction

  function automatic logic reduction(logic [4:0] op);
    return op inside {OpSum, OpSumsq, OpMax};
  endfunction

  // A partial before the row's first element reaches it: +0 for a sum, the
  // canonical NaN for a maximum (max_f32 leaves it to any other value).
  function automatic logic [31:0] red_start(logic rmax);
    return rmax ? CanonicalNaN : '0;
  endfunction

  function automatic walk_t take(logic [4:0] op, logic [BufW-1:0] dst, a, b, c, logic [7:0] flags,
                                 logic [AddrW-1:0] n, m, logic [31:0] scalar,
                                 logic [AddrW-1:0] cap, logic [MemAddrW-1:0] addr,
                                 logic [AddrW-1:0] token, t, stride);
    walk_t w;
    logic  flat;
    flat = elementwise(op) && !flags[BPerRow] && !flags[BRow] && !flags[LenT];
    w = '{op: op, dst: dst, a: a, b: b, c: c, flags: flags, scalar: scalar, addr: addr, cap: cap,
          default: '0};
    w.rows       = flat || op == OpLoad ? AddrW'(1) : n;
    w.len        = flags[LenT] ? t : flat ? AddrW'(n * m) : m;
    w.row_stride = flags[LenT] ? stride : m;
    w.mrow       = store(op) || flags[ByPosition] ? t - 1'b1 : flags[ByToken] ? token : '0;
    return w;
  endfunction

  walk_t            cur_q;
  logic             active_q;
  logic [AddrW-1:0] row_q, base_q, g_q;  // row, its first element, the group's offset in it
  group_t           grp;
  logic             grp_fire, row_end;
  logic [AddrW-1:0] half, seg_end, count, g_next, addr, gi;
  logic             rotate, first_half;
  logic             red_q;   // a reduction runs: no command is taken until its last row is written
  logic             pend_q;  // a row's last group went, its partials not yet in the tree
  logic             tree_busy, stall;
  logic             is_load, is_kv, is_v, ld_issue_q, ld_ready, kv_room;
  logic [MW-1:0]    kv_credit_q;  // the stores' queue: places no group read so far will take
  logic             prep_q;       // a LOAD or store taken last cycle: its addresses next
  logic             tags_ready;   // a done tag has a place: fewer than Dones commands open
  // LOAD's first byte or a store's first head's (below), a store's region
  // (a head's), the head's offset from the first's.
  logic [MemAddrW-1:0] mbase_q, region_q, kv_off_q;

  assign is_load = cur_q.op == OpLoad;
  assign is_kv   = store(cur_q.op);
  assign is_v    = cur_q.op == OpVStore;

  always_comb begin
    half       = cur_q.len >> 1;
    rotate     = cur_q.op == OpRotateHalf;
    first_half = g_q < half;
    seg_end    = rotate && first_half ? half : cur_q.len;
    count      = seg_end - g_q > AddrW'(Lanes) ? AddrW'(Lanes) : seg_end - g_q;
    g_next     = g_q + count;
    row_end    = g_next == cur_q.len;
    addr       = base_q + g_q;
    gi         = g_q >> LW;  // a reduction's groups start at multiples of Lanes

    grp.is = '{op: cur_q.op, dst: cur_q.dst, a: cur_q.a, b: cur_q.b, c: cur_q.c,
               neg_a: rotate ? first_half : cur_q.flags[NegA], neg_b: cur_q.flags[NegB],
               b_per_row: cur_q.flags[BPerRow], scalar: cur_q.scalar, addr: addr,
               mask: Lanes'((1 << count) - 1), last: row_end && row_q + 1'b1 == cur_q.rows,
               row_first: gi == '0, first: gi < AddrW'(Parts), row_last: row_end,
               part: gi[PW-1:0], row: row_q};
    grp.a_addr = !rotate ? addr : first_half ? addr + half : addr - half;
    grp.b_addr = cur_q.flags[BPerRow] ? row_q : cur_q.flags[BRow] ? g_q : addr;
  end

  // A reduction row's last group waits for the tree, a LOAD group for its
  // beats, a store's group for a place in the queue.
  assign stall = reduction(grp.is.op) && row_end && (pend_q || tree_busy)
              || is_load && !ld_ready || is_kv && (prep_q || !kv_room);
  assign grp_fire    = active_q && !stall;
  assign cmd_ready_o = !red_q && (!active_q || grp.is.last && grp_fire) && tags_ready;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      active_q <= 1'b0;
    end else if (cmd_valid_i && cmd_ready_o) begin
      active_q <= 1'b1;
      cur_q    <= take(cmd_op_i, cmd_dst_i, cmd_a_i, cmd_b_i, cmd_c_i, cmd_flags_i, cmd_n_i,
                       cmd_m_i, cmd_scalar_i, cmd_cap_i, cmd_addr_i, token_i, t_i, stride_i);
      {row_q, base_q, g_q} <= '0;
      kv_off_q <= '0;
    end else if (grp_fire) begin
      if (grp.is.last) begin
        active_q <= 1'b0;
      end else if (row_end) begin
        row_q    <= row_q + 1'b1;
        base_q   <= base_q + cur_q.row_stride;
        g_q      <= '0;
        kv_off_q <= kv_off_q + region_q;
      end else begin
        g_q <= g_next;
      end
    end
  end

  // --- LOAD and the stores: the AXI port -------------------------------------

  // LOAD's row of m BF16 or FP32 values, in beats. The cycle after the take,
  // the first byte: LOAD's of row mrow; a store's at position q = mrow
  // (header): K's channel and its row's first beat, V's beat; and a store's
  // region. By beats: one multiply for either first byte, one for K's
  // region.
  logic [AddrW-1:0] row_beats, k_cols, k_pass, mul_a, mul_b, region_beats;
  logic             ld_bf16;
  assign ld_bf16   = cur_q.flags[SrcBf16];
  assign row_beats = ld_bf16 ? cur_q.len >> LW : cur_q.len >> (LW - 1);
  assign k_cols    = cur_q.len >> EW;           // K: a row's beats
  assign k_pass    = cur_q.mrow >> (PpW + KvW);  // K: q / L
  assign mul_a     = is_kv ? k_pass : cur_q.mrow;
  assign mul_b     = is_kv ? k_cols : row_beats;
  assign region_beats = is_v ? cur_q.cap >> EW : (cur_q.cap >> (PpW + KvW)) * k_cols;

  always_ff @(posedge clk_i) begin
    prep_q <= rst_ni && cmd_valid_i && cmd_ready_o && (cmd_op_i == OpLoad || store(cmd_op_i));
    if (prep_q) begin
      mbase_q  <= (is_kv && !is_v ? MemAddrW'(cur_q.mrow[PpW+:KvW]) << ChanW : '0) + cur_q.addr
                + (is_v ? MemAddrW'(cur_q.mrow >> EW) << BeatW
                        : MemAddrW'(mul_a) * MemAddrW'(mul_b) << BeatW);
      region_q <= MemAddrW'(region_beats) << BeatW;
    end
  end

  // LOAD: its command goes to the reader the cycle after it is taken.
  logic                 dma_ready, beat_valid, beat_take;
  logic [DataWidth-1:0] beat;
  logic                 rd_err, wr_err;
  port_dma #(
    .AddrWidth (MemAddrW),
    .DataWidth (DataWidth),
    .CountWidth(AddrW),
    .Depth     (MemDepth)
  ) u_rd (
    .clk_i,
    .rst_ni,
    .cmd_valid_i(ld_issue_q && !prep_q),
    .cmd_ready_o(dma_ready),
    .cmd_addr_i (mbase_q),
    .cmd_beats_i(row_beats),
    .out_valid_o(beat_valid),
    .out_ready_i(beat_take),
    .out_data_o (beat),
    .ar_valid_o,
    .ar_ready_i,
    .ar_addr_o,
    .ar_len_o,
    .ar_size_o,
    .ar_burst_o,
    .r_valid_i,
    .r_ready_o,
    .r_data_i,
    .r_resp_i,
    .err_o      (rd_err),
    // verilator lint_off PINCONNECTEMPTY
    .out_count_o()
    // verilator lint_on PINCONNECTEMPTY
  );

  // The group's beats: one of BF16, one or two of FP32 (half a group or
  // less fits one). A beat is taken while the group lacks one, and as a
  // group goes when it is not the command's last (the next group's first).
  logic [1:0][DataWidth-1:0] gb_q;
  logic [1:0]                gb_have_q, need;
  assign need      = ld_bf16 || count <= AddrW'(Lanes / 2) ? 2'd1 : 2'd2;
  assign ld_ready  = !ld_issue_q && gb_have_q == need;
  assign beat_take = active_q && is_load && !ld_issue_q
                  && (gb_have_q < need || grp_fire && !grp.is.last);

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      ld_issue_q <= 1'b0;
      gb_have_q  <= '0;
    end else begin
      if (cmd_valid_i && cmd_ready_o) ld_issue_q <= cmd_op_i == OpLoad;
      else if (!prep_q && dma_ready) ld_issue_q <= 1'b0;
      if (grp_fire && is_load) begin
        gb_have_q <= {1'b0, beat_take && beat_valid};
        if (beat_take && beat_valid) gb_q[0] <= beat;
      end else if (beat_take && beat_valid) begin
        gb_q[gb_have_q[0]] <= beat;
        gb_have_q <= gb_have_q + 1'b1;
      end
    end
  end

  // LOAD's group as FP32 (BF16 converted up exactly, a NaN made canonical),
  // for the slots the cycle after it goes.
  logic [Lanes-1:0][31:0] ld_q, ld_up;
  for (genvar l = 0; l < Lanes; l++) begin : gen_up
    bf16_to_fp32 u_up (
      .a_i(gb_q[0][16*l+:16]),
      .y_o(ld_up[l])
    );
  end
  always_ff @(posedge clk_i) begin
    if (grp_fire && is_load) begin
      for (int unsigned l = 0; l < Lanes; l++) begin
        ld_q[l] <= ld_bf16 ? ld_up[l] : gb_q[l/(Lanes/2)][32*(l%(Lanes/2))+:32];
      end
    end
  end

  // The stores: a group's rounded values join the queue as they leave the
  // slots (kv_beat, below); a group is read only with a place for it, and
  // as it is read its first burst's address (K's channel and beat, V's
  // channel; header) and its beats' format join theirs.
  typedef struct packed {
    logic [MemAddrW-1:0] addr;
    logic                v, two, last;  // V, its 2 copies; the command's last group
    logic [KvW-1:0]      step;          // V: the copy's channels on
  } kv_burst_t;

  typedef struct packed {
    logic          v;
    logic          two;
    logic [LW-1:0] pos;  // q mod Lanes: K's slot, V's element in its chunk
  } kv_fmt_t;

  localparam int unsigned BiW = (PpW > EW ? PpW : EW) + 1;  // a beat or burst of a group
  kv_burst_t            kb_in, kb;
  kv_fmt_t              kf_in, kf;
  logic                 kv_push, kv_beat_valid, q_valid, q_take, kb_valid, kb_go, kb_last;
  logic                 wr_ready, in_ready, beat_go, wb_last, wr_done, wr_done_tag;
  logic [DataWidth-1:0] kv_beat, q_beat, kv_w_data;
  logic [Lanes*2-1:0]   kv_w_strb;
  logic [BiW-1:0]       kb_q, wb_q;  // the group's burst, beat: V's chunk, its copy at bit EW
  // verilator lint_off UNUSEDSIGNAL
  logic                 unused_ready, unused_kf_valid;
  // verilator lint_on UNUSEDSIGNAL
  assign kv_room = kv_credit_q != '0;
  assign kv_push = grp_fire && is_kv;
  assign kb_in   = '{addr: mbase_q + kv_off_q + (is_v ? MemAddrW'(g_q >> PpW) << ChanW
                                                      : MemAddrW'(g_q >> EW) << BeatW),
                     v: is_v, two: is_v && cur_q.len <= AddrW'(KvPorts * LanesPerPort / 2),
                     last: grp.is.last, step: KvW'(cur_q.len >> PpW)};
  assign kf_in   = '{v: is_v, two: kb_in.two, pos: cur_q.mrow[LW-1:0]};

  stream_fifo #(.Width(DataWidth), .Depth(MemDepth)) u_kv_queue (
    .clk_i,
    .rst_ni,
    .in_valid_i (kv_beat_valid),
    .in_ready_o (unused_ready),  // the credits keep a place, in each of the three
    .in_data_i  (kv_beat),
    .out_valid_o(q_valid),
    .out_ready_i(q_take),
    .out_data_o (q_beat),
    // verilator lint_off PINCONNECTEMPTY
    .count_o    ()
    // verilator lint_on PINCONNECTEMPTY
  );

  stream_fifo #(.Width($bits(kv_burst_t)), .Depth(MemDepth)) u_kv_bursts (
    .clk_i,
    .rst_ni,
    .in_valid_i (kv_push),
    // verilator lint_off PINCONNECTEMPTY
    .in_ready_o (),
    // verilator lint_on PINCONNECTEMPTY
    .in_data_i  (kb_in),
    .out_valid_o(kb_valid),
    .out_ready_i(kb_go && kb_last),
    .out_data_o (kb),
    // verilator lint_off PINCONNECTEMPTY
    .count_o    ()
    // verilator lint_on PINCONNECTEMPTY
  );

  stream_fifo #(.Width($bits(kv_fmt_t)), .Depth(MemDepth)) u_kv_fmts (
    .clk_i,
    .rst_ni,
    .in_valid_i (kv_push),
    // verilator lint_off PINCONNECTEMPTY
    .in_ready_o (),
    // verilator lint_on PINCONNECTEMPTY
    .in_data_i  (kf_in),
    .out_valid_o(unused_kf_valid),  // with the group's beat
    .out_ready_i(q_take),
    .out_data_o (kf),
    // verilator lint_off PINCONNECTEMPTY
    .count_o    ()
    // verilator lint_on PINCONNECTEMPTY
  );

  // A group's bursts: K's one of KvBurst beats; V's E of one beat, then the
  // copy's. Its last carries the command's last group's tag.
  assign kb_last = !kb.v || kb_q[EW-1:0] == EW'(E - 1) && kb_q[EW] == kb.two;
  assign kb_go   = kb_valid && wr_ready;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) kb_q <= '0;
    else if (kb_go) kb_q <= kb_last ? '0 : kb_q + 1'b1;
  end

  // A group's beats, from the queue: K's chunk wb_q of E dimensions in each
  // slot, V's chunk of LanesPerPort dimensions each E times (dimension i of
  // the chunk in elements E * i to E * i + E - 1); the strobes pick K's slot,
  // V's element of each.
  assign wb_last = kf.v ? wb_q[EW-1:0] == EW'(E - 1) && wb_q[EW] == kf.two
                        : wb_q[PpW-1:0] == PpW'(LanesPerPort - 1);
  assign beat_go = q_valid && in_ready;
  assign q_take  = beat_go && wb_last;

  always_comb begin
    for (int unsigned l = 0; l < Lanes; l++) begin
      logic [LW-1:0] el;
      logic          on;
      el = kf.v ? LW'(wb_q[EW-1:0] * LanesPerPort + l / E) : LW'(wb_q[PpW-1:0] * E + l % E);
      on = kf.v ? LW'(l % E) == LW'(kf.pos % E) : LW'(l / E) == LW'(kf.pos % LanesPerPort);
      kv_w_data[16*l+:16] = q_beat[16*el+:16];
      kv_w_strb[2*l+:2]   = {2{on}};
    end
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) wb_q <= '0;
    else if (beat_go) wb_q <= wb_last ? '0 : wb_q + 1'b1;
  end

  port_wdma #(
    .AddrWidth  (MemAddrW),
    .DataWidth  (DataWidth),
    .MaxBurst   (KvBurst),
    .Outstanding(WrOutstanding)
  ) u_wr (
    .clk_i,
    .rst_ni,
    .cmd_valid_i(kb_valid),
    .cmd_ready_o(wr_ready),
    .cmd_addr_i (kb.addr + (kb.v ? MemAddrW'(kb_q[EW-1:0]) + (kb_q[EW] ? MemAddrW'(kb.step) : '0)
                                   << ChanW : '0)),
    .cmd_beats_i(kb.v ? LenW'(1) : LenW'(KvBurst)),
    .cmd_tag_i  (kb.last && kb_last),
    .in_valid_i (q_valid),
    .in_ready_o (in_ready),
    .in_data_i  (kv_w_data),
    .in_strb_i  (kv_w_strb),
    .done_o     (wr_done),
    .done_tag_o (wr_done_tag),
    .aw_valid_o,
    .aw_ready_i,
    .aw_addr_o,
    .aw_len_o,
    .aw_size_o,
    .aw_burst_o,
    .w_valid_o,
    .w_ready_i,
    .w_data_o,
    .w_strb_o,
    .w_last_o,
    .b_valid_i,
    .b_ready_o,
    .b_resp_i,
    .err_o      (wr_err)
  );
  assign err_o = rd_err || wr_err;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) kv_credit_q <= MW'(MemDepth);
    else kv_credit_q <= kv_credit_q - MW'(kv_push) + MW'(q_take);
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

  // The tree's write of a row's value (below): one element.
  logic             t_w_valid;
  logic [BufW-1:0]  t_w_buf;
  logic [AddrW-1:0] t_w_addr;
  logic [31:0]      t_w_data;

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
      if (t_w_valid && t_w_buf == BufW'(i)) begin  // never with the slots' (t_w_valid)
        {wr_addr[i], wr_mask[i], wr_data[i]} = {t_w_addr, Lanes'(1), (Lanes * 32)'(t_w_data)};
      end
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
      OpFma, OpSumsq: return ChainFma;
      OpExp:   return ChainExp;
      OpRecip: return ChainRecip;
      OpRsqrt: return ChainRsqrt;
      OpRotateHalf, OpLoad, OpKvStore, OpVStore: return ChainPass;
      default: return ChainAdd;  // ADD; SUM; MAX's x through add(x, -0)
    endcase
  endfunction

  // Where a group's results go: a buffer, the partials (a reduction's, so
  // nowhere from here), the stores' queue.
  typedef enum logic [1:0] {
    SinkBuf,
    SinkNone,
    SinkKv
  } sink_e;

  // What the write needs, alongside the slots.
  typedef struct packed {
    logic [BufW-1:0]  dst;
    logic [AddrW-1:0] addr;
    logic [Lanes-1:0] mask;
    logic             last;
    sink_e            sink;
  } dest_t;

  dest_t                  dest_q[ChainLat];
  logic [Lanes-1:0]       slot_valid;
  logic [Lanes-1:0][31:0] y, s1_y;

  // A reduction group on its way to the partials, from the chain's input to
  // its first FMA's result: what the write-back and the tree need of it.
  typedef struct packed {
    logic             valid, rmax, row_first, row_last, last;
    logic [PW-1:0]    part;
    logic [Lanes-1:0] mask;
    logic [BufW-1:0]  dst;
    logic [AddrW-1:0] row;
  } acc_t;

  logic [Lanes-1:0][Parts-1:0][31:0] part_q;  // the slots' partials
  acc_t                              acc_q[AccLat], ad;
  logic [31:0]                       start;
  assign ad    = acc_q[AccLat-1];
  assign start = red_start(ad.rmax);

  for (genvar l = 0; l < Lanes; l++) begin : gen_slot
    logic [31:0] a, b, c, p;
    always_comb begin
      a = rd_data[grp_q.a][l];
      b = grp_q.b_per_row ? rd_data[grp_q.b][0] : rd_data[grp_q.b][l];
      if (grp_q.b == '0) b = grp_q.scalar;
      c = rd_data[grp_q.c][l];
      a[31] = a[31] ^ grp_q.neg_a;  // exact sign flips
      b[31] = b[31] ^ grp_q.neg_b;
      if (grp_q.op == OpLoad) a = ld_q[l];
      p = grp_q.first ? '0 : part_q[l][grp_q.part];  // a sum's
      if (reduction(grp_q.op)) begin  // SUM: add(x, p); SUMSQ: fma(x, x, p); MAX: add(x, -0)
        b = grp_q.op == OpMax ? NegZero : grp_q.op == OpSumsq ? a : p;
        c = p;
      end
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
      .y_o    (y[l]),
      .s1_y_o (s1_y[l])
    );

    // A reduction group's result into its partial: a sum's is the FMA's, a
    // maximum max_f32(p, x); a missing element leaves its partial. A row's
    // first group starts every partial of its slot: the last cycle the tree
    // takes the row before it, so a partial the row never reaches stays as
    // it starts.
    always_ff @(posedge clk_i) begin
      if (ad.valid && ad.row_first) begin
        for (int unsigned j = 0; j < Parts; j++) part_q[l][j] <= start;
      end
      if (ad.valid && ad.mask[l]) begin
        part_q[l][ad.part] <= !ad.rmax ? s1_y[l]
                            : max_f32(ad.row_first ? start : part_q[l][ad.part], s1_y[l]);
      end
    end
  end

  always_ff @(posedge clk_i) begin
    acc_q[0] <= '{valid: rst_ni && grp_valid_q && reduction(grp_q.op), rmax: grp_q.op == OpMax,
                  row_first: grp_q.row_first, row_last: grp_q.row_last, last: grp_q.last,
                  part: grp_q.part, mask: grp_q.mask, dst: grp_q.dst, row: grp_q.row};
    for (int unsigned j = 1; j < AccLat; j++) acc_q[j] <= acc_q[j-1];
  end

  // --- The tree: a row's partials to its value ---------------------------------

  // cap_q: the row's last result landed last cycle, so its partials are all
  // in; the next row's first lands this cycle, after the tree takes them.
  // tree_q: the row being reduced; its value waits in the tree (t_pend_q)
  // until no elementwise result is on its way: so its write never meets the
  // slots' and its done_o comes after every earlier command's, whatever the
  // parameters.
  typedef struct packed {
    logic [BufW-1:0]  dst;
    logic [AddrW-1:0] row;
    logic             last;
  } tree_t;

  logic                      cap_valid_q, t_pend_q, tree_run, tree_done;
  logic [$clog2(ChainLat+2)-1:0] slot_groups_q;  // groups in the slots, but a reduction's
  logic                      slot_out;  // one of them leaves the slots (below)
  logic                      cap_rmax_q;
  tree_t                     cap_q, tree_q;
  logic [Partials-1:0][31:0] parts;
  logic [31:0]               tree_y;

  for (genvar j = 0; j < Parts; j++) begin : gen_parts
    for (genvar l = 0; l < Lanes; l++) begin : gen_lane
      assign parts[j*Lanes+l] = part_q[l][j];
    end
  end

  always_ff @(posedge clk_i) begin
    cap_valid_q <= rst_ni && ad.valid && ad.row_last;
    cap_q       <= '{dst: ad.dst, row: ad.row, last: ad.last};
    cap_rmax_q  <= ad.rmax;
    if (cap_valid_q) tree_q <= cap_q;
  end

  vec_tree #(
    .Width(Partials)
  ) u_tree (
    .clk_i,
    .rst_ni,
    .start_i(cap_valid_q),
    .max_i  (cap_rmax_q),
    .parts_i(parts),
    .busy_o (tree_run),
    .done_o (tree_done),
    .y_o    (tree_y)
  );
  assign tree_busy = tree_run || t_pend_q;  // its value is not written yet

  logic [15:0] tree_r;
  fp32_to_bf16 u_tree_round (
    .a_i(tree_y),
    .y_o(tree_r)
  );
  assign t_w_valid = t_pend_q && slot_groups_q == '0;
  assign t_w_buf   = tree_q.dst;
  assign t_w_addr  = tree_q.row;
  assign t_w_data  = Bf16Mask[tree_q.dst] ? {16'h0, tree_r} : tree_y;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      {red_q, pend_q, t_pend_q} <= '0;
      slot_groups_q <= '0;
    end else begin
      slot_groups_q <= slot_groups_q + $bits(slot_groups_q)'(grp_fire && !reduction(grp.is.op))
                     - $bits(slot_groups_q)'(slot_out);
      if (cmd_valid_i && cmd_ready_o) red_q <= reduction(cmd_op_i);
      else if (t_w_valid && tree_q.last) red_q <= 1'b0;
      if (grp_fire && reduction(grp.is.op) && row_end) pend_q <= 1'b1;
      else if (cap_valid_q) pend_q <= 1'b0;
      if (tree_done) t_pend_q <= 1'b1;
      else if (t_w_valid) t_pend_q <= 1'b0;
    end
  end

  always_ff @(posedge clk_i) begin
    dest_q[0] <= '{dst: grp_q.dst, addr: grp_q.addr,
                   mask: grp_q.mask, last: grp_q.last,
                   sink: reduction(grp_q.op) ? SinkNone : store(grp_q.op) ? SinkKv : SinkBuf};
    for (int unsigned j = 1; j < ChainLat; j++) dest_q[j] <= dest_q[j-1];
  end

  // The write: dest_q[ChainLat - 1] belongs to the slots' result this cycle.
  // A reduction's groups write nothing here, a store's go to its queue.
  dest_t d;
  assign d             = dest_q[ChainLat-1];
  assign slot_out      = slot_valid[0] && d.sink != SinkNone;
  assign u_w_valid     = slot_valid[0] && d.sink == SinkBuf;
  assign u_w_buf       = d.dst;
  assign u_w_addr      = d.addr;
  assign u_w_mask      = d.mask;
  assign kv_beat_valid = slot_valid[0] && d.sink == SinkKv;
  for (genvar l = 0; l < Lanes; l++) begin : gen_round
    logic [15:0] r;
    fp32_to_bf16 u_round (
      .a_i(y[l]),
      .y_o(r)
    );
    assign u_w_data[l]        = Bf16Mask[d.dst] ? {16'h0, r} : y[l];
    assign kv_beat[16*l+:16] = r;
  end

  // --- Dones, in order ----------------------------------------------------------

  // Each command taken leaves a tag (a store or not); a command ends with
  // its last write (a store's last group leaving the slots). The oldest tag
  // leaves as done_o once its command has ended and, for a store, its last
  // burst is answered (the responses come in order). ended_q: commands
  // ended, not yet done; answered_q: stores answered, not yet done.
  logic             tag_valid, tag_go, tag_kv, end_valid;
  logic [DoneW-1:0] ended_q, answered_q;
  assign end_valid = (u_w_valid || kv_beat_valid) && d.last || t_w_valid && tree_q.last;

  stream_fifo #(.Width(1), .Depth(Dones)) u_tags (
    .clk_i,
    .rst_ni,
    .in_valid_i (cmd_valid_i && cmd_ready_o),
    .in_ready_o (tags_ready),
    .in_data_i  (store(cmd_op_i)),
    .out_valid_o(tag_valid),
    .out_ready_i(tag_go),
    .out_data_o (tag_kv),
    // verilator lint_off PINCONNECTEMPTY
    .count_o    ()
    // verilator lint_on PINCONNECTEMPTY
  );

  assign tag_go = tag_valid && ended_q != '0 && (!tag_kv || answered_q != '0);
  assign done_o = tag_go;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      answered_q <= '0;
      ended_q    <= '0;
    end else begin
      answered_q <= answered_q + DoneW'(wr_done && wr_done_tag) - DoneW'(tag_go && tag_kv);
      ended_q    <= ended_q + DoneW'(end_valid) - DoneW'(tag_go);
    end
  end

  // verilator lint_off UNUSEDSIGNAL
  logic unused;
  // ext_r_valid_i: the banks read every cycle, so a read needs no enable.
  assign unused = ^{slot_valid[Lanes-1:1], ext_r_valid_i, rd_addr[0], wr_addr[0], wr_mask[0],
                    wr_data[0]};
  // verilator lint_on UNUSEDSIGNAL

endmodule
