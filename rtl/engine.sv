// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The engine (D-027, docs/architecture.md): L = Ports * LanesPerPort lanes in
// lockstep, each port's DMA feeding its LanesPerPort lanes. Three commands:
// - MATVEC: dst[r] = dot(W[r], x) for the rows r < rows of a matrix of `cols`
//   columns, W stored in the lanes' order ("Weight layout") at the same
//   offset `addr` in every port's channel. Rows go in passes of L; in each
//   pass port p reads its group of LanesPerPort rows, ceil(cols / E) beats.
// - SCORES: dst[h * stride + i] = mul(dot(q[h], K[kv][i]), scale) for the heads
//   h < heads and positions i < t, q[h] the d = cols values of x at h * d,
//   kv = h div group, K the cache at `addr` (D-043: position i of KV head kv
//   in channel i mod Ports, row i div Ports of the head's region of
//   cap / Ports rows). A unit is one head and a tile of L positions: lane l
//   takes position j * L + l.
// - VALUES: dst[h * d + c] = dot over i < t of (p[h][i], V[kv][i][c]), p[h]
//   the t values of x at h * stride. A pass is up to Heads heads (one lane per
//   head and output dimension); a unit is a pass and a tile, and a lane's
//   row runs through all tiles.
// Ports is a multiple of 4 (the tile buffer's mapping), so a pass is whole
// write groups. Commands need rows >= 1 and cols >= 1 (cols = 0 hangs
// step). The attention commands need d a power of two from 16 to
// MaxD (VALUES: d <= L), cap a multiple of L, 1 <= t <= cap, stride (S and
// P's rows, the STRIDE register) a multiple of E with t <= stride, group
// dividing heads.
//
// Four stages, each with its own queue of the commands taken, so a command
// starts while earlier ones finish (PLAN.md, M5). Issue, step and drain
// each walk a command's units (unit_init, unit_next): a MATVEC's passes, an
// attention command's units as above; fill walks attention's only.
// - Issue: DMA commands, in order. MATVEC: per pass, one to each port with
//   rows in it. Attention: per unit and head, one to every port, the tile's
//   4 rows (d / 4 beats). It runs ahead as far as the DMAs' FIFOs allow.
// - Fill (attention): the ports' beats into the tile buffer, a unit at a
//   time into a free one of two slots (docs/architecture.md, "Tile buffer
//   banks"), a beat from every port per cycle. On a MATVEC it waits until
//   step has read it.
// - Step: one beat to every lane per cycle, when every lane is ready, and
//   for a MATVEC every port with rows in the pass has a beat, for attention
//   the unit's tile is full. x comes from the read ports, E values
//   broadcast (a VALUES lane takes its head's). Back-to-back commands leave
//   no gap. A unit's slot is freed when its last cycle steps.
// - Drain: the lanes' results, Writes lanes at a time, to the write port,
//   multiplied by scale (SCORES) or 1 (exact), then rounded to BF16 for a
//   BF16 destination. Lanes with no row (past the rows, past t, past the
//   heads) take no beat and have no result. done_o marks a command's last
//   write.
// The engine does not order a command after an earlier one whose output it
// reads; the controller does (D-041, D-042, D-044): it gives such a command
// `wait` and pulses go_i once its x is ready, so issue and fill read the
// command's memory ahead while step holds it until its go. Each go_i pulse
// is a credit for the oldest waiting command not yet given one, sent only
// after that command was taken (so at most CmdDepth + 1 are pending).
//
// Read ports: x_data_i[r] is the E values of group x_addr_o[r] (elements
// E * x_addr_o[r] ... + E - 1) of buffer x_buf_o as presented in the
// previous cycle (a synchronous RAM), zeros past the buffer's end. Port 1
// serves a VALUES pass's second head. Write port: when w_valid_o, slot s
// with w_mask_o[s] holds row w_row_o + s of buffer w_buf_o: FP32, or the
// BF16 bits in its low half when w_bf16_o. It never waits.

module engine
  import fp_pkg::*;
#(
  parameter  int unsigned Ports        = 32,
  parameter  int unsigned AddrWidth    = 34,   // F2's HBM: 16 GiB
  parameter  int unsigned DataWidth    = 256,  // an HBM port's beat
  parameter  int unsigned Depth        = 128,  // each port's FIFO, beats (PLAN.md, M5)
  parameter  int unsigned RowWidth     = 20,   // rows: Qwen2.5's classifier needs 18 bits
  parameter  int unsigned ColWidth     = 16,   // columns
  parameter  int unsigned HeadWidth    = 8,    // heads, group
  parameter  int unsigned BufWidth     = 8,    // buffer ids (docs/architecture.md, "Encoding")
  parameter  int unsigned Writes       = 8,    // results per cycle on the write port
  parameter  int unsigned CmdDepth     = 4,    // commands queued per stage
  parameter  int unsigned E            = 4,    // pairs per lane per cycle
  parameter  int unsigned A            = 16,   // partial sums per row (D-027)
  parameter  int unsigned MaxD         = 128,  // largest head dimension
  parameter  int unsigned MulRegs      = 2,    // registers in the scale's multiply (fp32_fma)
  localparam int unsigned LanesPerPort = DataWidth / (16 * E),  // a beat holds E weights each
  localparam int unsigned L            = Ports * LanesPerPort,
  localparam int unsigned Groups       = L / Writes,  // write groups per pass
  localparam int unsigned Heads        = 2     // heads per VALUES pass, at most
) (
  input  logic                                clk_i,
  input  logic                                rst_ni,
  // Command
  input  logic                                cmd_valid_i,
  output logic                                cmd_ready_o,
  input  logic [1:0]                          cmd_op_i,     // OpMatvec, OpScores, OpValues
  input  logic [RowWidth-1:0]                 cmd_rows_i,   // MATVEC
  input  logic [ColWidth-1:0]                 cmd_cols_i,   // MATVEC's columns; attention's d
  input  logic [AddrWidth-1:0]                cmd_addr_i,   // beat-aligned, in every channel
  input  logic [BufWidth-1:0]                 cmd_a_i,      // x
  input  logic [BufWidth-1:0]                 cmd_dst_i,
  input  logic                                cmd_bf16_i,   // round the results to BF16
  input  logic                                cmd_wait_i,   // step only after a go_i
  input  logic [HeadWidth-1:0]                cmd_heads_i,  // attention
  input  logic [HeadWidth-1:0]                cmd_group_i,  // query heads per KV head
  input  logic [RowWidth-1:0]                 cmd_t_i,      // positions read
  input  logic [RowWidth-1:0]                 cmd_cap_i,    // the cache's positions per head
  input  logic [RowWidth-1:0]                 cmd_stride_i, // S and P's rows: SCORES' dst, VALUES' x
  input  logic [31:0]                         cmd_scale_i,  // SCORES, FP32
  input  logic                                go_i,         // a waiting command's x is ready
  output logic                                done_o,       // a command's last write
  // x read ports
  output logic [BufWidth-1:0]                 x_buf_o,
  output logic [Heads-1:0][RowWidth-1:0]      x_addr_o,
  input  logic [Heads-1:0][E-1:0][15:0]       x_data_i,
  // Result write port
  output logic                                w_valid_o,
  output logic [BufWidth-1:0]                 w_buf_o,
  output logic [RowWidth-1:0]                 w_row_o,
  output logic [Writes-1:0]                   w_mask_o,
  output logic [Writes-1:0][31:0]             w_data_o,
  output logic                                w_bf16_o,
  // AXI read ports (port_dma.sv)
  output logic [Ports-1:0]                    ar_valid_o,
  input  logic [Ports-1:0]                    ar_ready_i,
  output logic [Ports-1:0][AddrWidth-1:0]     ar_addr_o,
  output logic [Ports-1:0][7:0]               ar_len_o,
  output logic [Ports-1:0][2:0]               ar_size_o,
  output logic [Ports-1:0][1:0]               ar_burst_o,
  input  logic [Ports-1:0]                    r_valid_i,
  output logic [Ports-1:0]                    r_ready_o,
  input  logic [Ports-1:0][DataWidth-1:0]     r_data_i,
  input  logic [Ports-1:0][1:0]               r_resp_i,
  output logic [Ports-1:0]                    err_o         // sticky, per port
);

  localparam logic [1:0]  OpMatvec  = 2'd0;
  localparam logic [1:0]  OpScores  = 2'd1;
  localparam logic [1:0]  OpValues  = 2'd2;
  localparam int unsigned BeatBytes = DataWidth / 8;
  localparam int unsigned GW        = Groups > 1 ? $clog2(Groups) : 1;
  localparam int unsigned CW        = $clog2(CmdDepth + 2);  // go credits: 0 .. CmdDepth + 1
  localparam int unsigned LW        = $clog2(L + 1);         // a count of lanes
  localparam int unsigned SW        = $clog2(L);             // a tile buffer rotation
  // The tile buffer: per slot, the entries of E values a unit fills: d / E
  // for one head, d <= MaxD; 2 * d / E for a VALUES pass of Heads heads,
  // which needs Heads * d <= L.
  localparam int unsigned SlotWidest = Heads * MaxD < L ? Heads * MaxD : L;
  localparam int unsigned SlotDepth = (MaxD > SlotWidest ? MaxD : SlotWidest) / E;
  localparam int unsigned TW        = $clog2(2 * SlotDepth);  // a tile buffer address
  localparam int unsigned EntryBits = E * 16;
  localparam int unsigned BeatValues = DataWidth / 16;     // BF16 values per beat
  localparam int unsigned TileRows  = LanesPerPort;        // a tile's rows per channel
  localparam int unsigned Entries   = DataWidth / EntryBits;  // a beat's entries
  localparam int unsigned DW        = $clog2(MaxD + 1);
  localparam int unsigned DrainLat  = MulRegs + 4;     // fp32_fma's latency (its header)

  if (DataWidth % (16 * E) != 0 || LanesPerPort == 0) begin : gen_bad_width
    $error("engine: a beat must hold E BF16 weights for a whole number of lanes");
  end
  if ((Writes & (Writes - 1)) != 0 || L % Writes != 0) begin : gen_bad_writes
    $error("engine: Writes must be a power of two dividing L");
  end
  if (Ports % 4 != 0) begin : gen_bad_ports
    $error("engine: Ports must be a multiple of 4");
  end
  if (ColWidth > RowWidth) begin : gen_bad_cols
    $error("engine: ColWidth must be at most RowWidth (a row's beats count in step's cycle)");
  end
  if (MaxD < 16 || MaxD > 128 || (MaxD & (MaxD - 1)) != 0) begin : gen_bad_maxd
    $error("engine: MaxD must be a power of two from 16 to 128 (unit_t.dlog)");
  end
  if (E != 4 || LanesPerPort != 4) begin : gen_bad_tile
    $error("engine: the tile buffer's mapping assumes 4 lanes per port and E = 4");
  end

  typedef struct packed {
    logic [1:0]           op;
    logic [RowWidth-1:0]  rows;
    logic [ColWidth-1:0]  cols;
    logic [AddrWidth-1:0] addr;
    logic [BufWidth-1:0]  a;
    logic [BufWidth-1:0]  dst;
    logic                 bf16;
    logic                 wait_go;
    logic [HeadWidth-1:0] heads;
    logic [HeadWidth-1:0] group;
    logic [RowWidth-1:0]  t;
    logic [RowWidth-1:0]  cap;
    logic [RowWidth-1:0]  stride;
    logic [31:0]          scale;
  } cmd_t;

  // --- Units -------------------------------------------------------------------
  //
  // A command is a sequence of units. MATVEC: its passes of L rows.
  // Attention: a pass's heads and a tile; SCORES for each head, for each
  // tile, VALUES for each pass of Heads heads (one when 2d > L), for each
  // tile. Every stage walks the same sequence with unit_init and unit_next.

  // A MATVEC pass is a SCORES unit of one head whose positions are rows:
  // t is the rows, a tile a pass of L of them.
  typedef struct packed {
    logic                  matvec;                   // a MATVEC, not attention
    logic [ColWidth-1:0]   beats;                    // each port's per unit and head
    logic                  values, heads2;           // VALUES; Heads heads a pass
    logic [HeadWidth-1:0]  n, g;                     // heads, group
    logic [2:0]            dlog;                     // log2(d)
    logic [RowWidth-1:0]   t, stride;                // positions (MATVEC: rows); S, P's rows
    logic [AddrWidth-1:0]  region;    // bytes of a KV head's region per channel
    logic [HeadWidth-1:0]  h, r;      // the unit's first head; h mod g
    logic [AddrWidth-1:0]  kv_off;    // (h div g) * region
    logic [RowWidth-1:0]   head_row;  // SCORES: h * stride; VALUES: h * d
    logic [RowWidth-1:0]   xh;        // head h's x group: SCORES h * d / E; VALUES h * stride / E
    logic [RowWidth-1:0]   pos;       // the tile's first position, j * L
    logic [AddrWidth-1:0]  unit_off;  // j * beats * BeatBytes (attention: 4 rows of d)
  } unit_t;

  // The helpers take a whole unit or command and read the fields they need.
  // verilator lint_off UNUSEDSIGNAL
  function automatic logic [DW-1:0] dim(unit_t u);
    return DW'(1) << u.dlog;
  endfunction
  function automatic logic [RowWidth-1:0] rem(unit_t u);  // positions from the tile on
    return u.t - u.pos;
  endfunction
  function automatic logic last_tile(unit_t u);
    return rem(u) <= RowWidth'(L);
  endfunction
  function automatic logic [1:0] unit_heads(unit_t u);
    return u.heads2 && 32'(u.h) + 1 < 32'(u.n) ? 2'd2 : 2'd1;
  endfunction
  function automatic logic last_unit(unit_t u);
    if (!last_tile(u)) return 1'b0;
    if (!u.values) return 32'(u.h) + 1 == 32'(u.n);
    return 32'(u.h) + (u.heads2 ? 2 : 1) >= 32'(u.n);
  endfunction
  // The byte offset of the unit's head a's KV head.
  function automatic logic [AddrWidth-1:0] kv_offset(unit_t u, logic a);
    return a && u.r + 1'b1 == u.g ? u.kv_off + u.region : u.kv_off;
  endfunction
  // Step cycles: MATVEC and SCORES a row's beats (d / E); VALUES L / E
  // positions' groups, fewer in the last tile.
  function automatic logic [RowWidth-1:0] unit_cycles(unit_t u);
    if (!u.values) return RowWidth'(u.beats);
    return last_tile(u) ? (rem(u) + RowWidth'(E - 1)) / RowWidth'(E) : RowWidth'(Ports);
  endfunction
  // A unit's results (drain): how many lanes, from lane 0, and lane 0's row.
  function automatic logic [LW-1:0] res_lanes(unit_t u);
    if (!u.values) return last_tile(u) ? LW'(rem(u)) : LW'(L);
    return LW'(unit_heads(u)) << u.dlog;
  endfunction
  function automatic logic [RowWidth-1:0] res_row(unit_t u);
    return u.values ? u.head_row : u.head_row + u.pos;
  endfunction
  // The ports a unit reads (MATVEC: those with rows in the pass).
  function automatic logic [Ports-1:0] unit_ports(unit_t u);
    logic [Ports-1:0] ports;
    for (int unsigned p = 0; p < Ports; p++) ports[p] = !u.matvec || p * LanesPerPort < 32'(rem(u));
    return ports;
  endfunction

  // Each port's beats per unit and head: a row's ceil(cols / E), for
  // attention TileRows * d / BeatValues = d / E.
  function automatic logic [ColWidth-1:0] n_beats(logic [ColWidth-1:0] cols);
    return cols / ColWidth'(E) + ColWidth'(cols % ColWidth'(E) != '0);
  endfunction
  // A MATVEC: one head of t = rows positions; its attention fields stay 0.
  function automatic unit_t unit_init(cmd_t c);
    unit_t u;
    u = '0;
    u.matvec = c.op == OpMatvec;
    u.beats  = n_beats(c.cols);
    if (u.matvec) begin
      u.n = HeadWidth'(1);
      u.t = c.rows;
      return u;
    end
    for (int unsigned i = 0; i < ColWidth; i++) if (c.cols[i]) u.dlog = 3'(i);  // d = 2^dlog
    u.values  = c.op == OpValues;
    u.heads2  = u.values && 2 * 32'(c.cols) <= L;
    u.n       = c.heads;
    u.g       = c.group;
    u.t       = c.t;
    u.stride  = c.stride;
    u.region  = AddrWidth'(c.cap / RowWidth'(Ports)) << (32'(u.dlog) + 1);  // * d * 2 bytes
    return u;
  endfunction
  // The next head: h + 1, and its KV head.
  function automatic unit_t next_head(unit_t u);
    u.h = u.h + 1'b1;
    if (u.r + 1'b1 == u.g) begin
      u.r      = '0;
      u.kv_off = u.kv_off + u.region;
    end else begin
      u.r = u.r + 1'b1;
    end
    return u;
  endfunction
  function automatic unit_t unit_next(unit_t u);
    if (!last_tile(u)) begin
      u.pos      = u.pos + RowWidth'(L);
      u.unit_off = u.unit_off + AddrWidth'(u.beats) * AddrWidth'(BeatBytes);
      return u;
    end
    u.pos      = '0;
    u.unit_off = '0;
    if (!u.values) begin
      u          = next_head(u);
      u.head_row = u.head_row + u.stride;
      u.xh       = u.xh + RowWidth'(dim(u)) / RowWidth'(E);
    end else begin
      u          = next_head(u);
      u.head_row = u.head_row + RowWidth'(dim(u));
      u.xh       = u.xh + u.stride / RowWidth'(E);
      if (u.heads2) begin
        u          = next_head(u);
        u.head_row = u.head_row + RowWidth'(dim(u));
        u.xh       = u.xh + u.stride / RowWidth'(E);
      end
    end
    return u;
  endfunction
  // The drain sees a VALUES pass once, at its last tile.
  function automatic unit_t to_last_tile(unit_t u);
    if (u.values) u.pos = (u.t - 1'b1) / RowWidth'(L) * RowWidth'(L);
    return u;
  endfunction
  // verilator lint_on UNUSEDSIGNAL

  // The tile buffer's mapping (docs/architecture.md, "Tile buffer banks"):
  // entry u (elements E*u ...) of position q sits in bank (q + E*alpha(u))
  // mod L.
  localparam int unsigned Quarter = Ports / 4;
  function automatic int unsigned alpha(int unsigned u);
    return (Quarter * (u % 4) + u / 4) % Ports;
  endfunction
  function automatic int unsigned alpha_inv(int unsigned g);  // g < Ports
    return 4 * (g % Quarter) + g / Quarter;
  endfunction

  // --- Command queues --------------------------------------------------------

  cmd_t cmd;
  assign cmd = '{op: cmd_op_i, rows: cmd_rows_i, cols: cmd_cols_i, addr: cmd_addr_i,
                 a: cmd_a_i, dst: cmd_dst_i, bf16: cmd_bf16_i, wait_go: cmd_wait_i,
                 heads: cmd_heads_i, group: cmd_group_i, t: cmd_t_i, cap: cmd_cap_i,
                 stride: cmd_stride_i, scale: cmd_scale_i};

  logic [3:0] q_in_ready, q_valid, q_pop;  // issue, fill, step, drain
  cmd_t       q_head[4];
  assign cmd_ready_o = &q_in_ready;

  for (genvar i = 0; i < 4; i++) begin : gen_queue
    stream_fifo #(
      .Width($bits(cmd_t)),
      .Depth(CmdDepth)
    ) u_queue (
      .clk_i,
      .rst_ni,
      .in_valid_i (cmd_valid_i && cmd_ready_o),
      .in_ready_o (q_in_ready[i]),
      .in_data_i  (cmd),
      .out_valid_o(q_valid[i]),
      .out_ready_i(q_pop[i]),
      .out_data_o (q_head[i])
    );
  end

  // --- DMAs --------------------------------------------------------------------

  logic [Ports-1:0]                dma_cmd_valid, dma_cmd_ready;
  logic [Ports-1:0]                dma_out_valid, dma_out_ready;
  logic [Ports-1:0][DataWidth-1:0] dma_out_data;
  logic [AddrWidth-1:0]            dma_addr;
  logic [ColWidth-1:0]             dma_beats;

  for (genvar p = 0; p < Ports; p++) begin : gen_dma
    port_dma #(
      .AddrWidth (AddrWidth),
      .DataWidth (DataWidth),
      .CountWidth(ColWidth),
      .Depth     (Depth)
    ) u_dma (
      .clk_i,
      .rst_ni,
      .cmd_valid_i(dma_cmd_valid[p]),
      .cmd_ready_o(dma_cmd_ready[p]),
      .cmd_addr_i (dma_addr),
      .cmd_beats_i(dma_beats),
      .out_valid_o(dma_out_valid[p]),
      .out_ready_i(dma_out_ready[p]),
      .out_data_o (dma_out_data[p]),
      .ar_valid_o (ar_valid_o[p]),
      .ar_ready_i (ar_ready_i[p]),
      .ar_addr_o  (ar_addr_o[p]),
      .ar_len_o   (ar_len_o[p]),
      .ar_size_o  (ar_size_o[p]),
      .ar_burst_o (ar_burst_o[p]),
      .r_valid_i  (r_valid_i[p]),
      .r_ready_o  (r_ready_o[p]),
      .r_data_i   (r_data_i[p]),
      .r_resp_i   (r_resp_i[p]),
      .err_o      (err_o[p])
    );
  end

  // --- Issue: DMA commands -------------------------------------------------------

  logic                 iss_busy_q, iss_head_q;  // iss_head_q: the unit's head
  logic [AddrWidth-1:0] iss_addr_q;   // the matrix's or the cache's
  logic [Ports-1:0]     iss_taken_q;  // this DMA command's ports already taken
  logic [Ports-1:0]     iss_active_q; // unit_ports(iss_unit_q), registered for timing
  unit_t                iss_unit_q;
  logic [Ports-1:0]     iss_taken;
  logic                 iss_done, iss_last_head;

  assign dma_cmd_valid = {Ports{iss_busy_q}} & iss_active_q & ~iss_taken_q;
  assign dma_addr      = iss_addr_q + kv_offset(iss_unit_q, iss_head_q) + iss_unit_q.unit_off;
  assign dma_beats     = iss_unit_q.beats;
  assign iss_taken     = iss_taken_q | dma_cmd_ready;
  assign iss_done      = iss_busy_q && &(iss_taken | ~iss_active_q);  // this DMA command's
  assign iss_last_head = 2'(iss_head_q) + 2'd1 == unit_heads(iss_unit_q);
  assign q_pop[0]      = !iss_busy_q && q_valid[0];

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      iss_busy_q  <= 1'b0;
      iss_taken_q <= '0;
    end else if (q_pop[0]) begin
      iss_busy_q   <= 1'b1;
      iss_addr_q   <= q_head[0].addr;
      iss_head_q   <= 1'b0;
      // The unit and its ports from each branch's own copy, in parallel: a
      // mux of the two units before unit_ports would lengthen the path.
      iss_unit_q   <= unit_init(q_head[0]);
      iss_active_q <= unit_ports(unit_init(q_head[0]));
    end else if (iss_done) begin
      iss_taken_q <= '0;
      if (!iss_last_head) begin
        iss_head_q <= 1'b1;
      end else begin
        iss_head_q   <= 1'b0;
        iss_unit_q   <= unit_next(iss_unit_q);
        iss_active_q <= unit_ports(unit_next(iss_unit_q));
        if (last_unit(iss_unit_q)) iss_busy_q <= 1'b0;
      end
    end else if (iss_busy_q) begin
      iss_taken_q <= iss_taken;
    end
  end

  // --- Fill: attention's beats into the tile buffer --------------------------

  logic [1:0]            tile_full_q;   // per slot: filled, not yet stepped
  logic [1:0]            tile_fill, tile_free;
  logic                  fill_busy_q, fill_slot_q, fill_head_q;
  logic [$clog2(TileRows)-1:0] fill_row_q;       // the tile's row
  logic [DW-1:0]         fill_beat_q;   // the row's beat
  unit_t                 fill_unit_q;
  logic                  fill_fire, fill_row_end, fill_tile_end, fill_unit_end, step_mv_end;
  logic                  fill_mv_q;     // a MATVEC whose beats step still reads
  logic                  step_load;     // step takes a command (below)
  logic [L-1:0]          fill_valid;
  logic [L-1:0][TW-1:0]  fill_addr;
  logic [L-1:0][EntryBits-1:0] fill_data;
  logic [SW-1:0]         fill_rot;

  // A slot freed this cycle takes writes at once: step's reads are in the other slot.
  assign fill_fire     = fill_busy_q && (!tile_full_q[fill_slot_q] || tile_free[fill_slot_q])
                         && &dma_out_valid;
  assign fill_row_end  = fill_fire && 32'(fill_beat_q) + 1 == 32'(dim(fill_unit_q)) / BeatValues;
  assign fill_tile_end = fill_row_end && 32'(fill_row_q) == TileRows - 1;  // a head's tile
  assign fill_unit_end = fill_tile_end && 2'(fill_head_q) + 2'd1 == unit_heads(fill_unit_q);
  assign tile_fill     = fill_unit_end ? 2'(1) << fill_slot_q : 2'b00;
  // A MATVEC's beats go to step: fill takes it off its queue when step takes
  // it, and starts the next attention command once step has read them.
  assign q_pop[1]      = q_valid[1] && !fill_busy_q
                         && (q_head[1].op == OpMatvec
                             ? step_load && q_head[2].op == OpMatvec  // the same one: in order
                             : !fill_mv_q || step_mv_end);

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      fill_busy_q  <= 1'b0;
      fill_mv_q    <= 1'b0;
      fill_slot_q  <= 1'b0;
      tile_full_q  <= '0;
    end else begin
      tile_full_q <= (tile_full_q | tile_fill) & ~tile_free;
      if (step_mv_end) fill_mv_q <= 1'b0;
      if (q_pop[1] && q_head[1].op == OpMatvec) fill_mv_q <= 1'b1;
      if (q_pop[1] && q_head[1].op != OpMatvec) begin
        fill_busy_q <= 1'b1;
        fill_unit_q <= unit_init(q_head[1]);
        fill_head_q <= 1'b0;
        fill_row_q  <= '0;
        fill_beat_q <= '0;
      end else begin
        if (fill_fire) fill_beat_q <= fill_row_end ? '0 : fill_beat_q + 1'b1;
        if (fill_row_end) fill_row_q <= fill_row_q + 1'b1;  // wraps at TileRows
        if (fill_tile_end) fill_head_q <= fill_unit_end ? 1'b0 : !fill_head_q;
        if (fill_unit_end) begin
          fill_slot_q <= !fill_slot_q;
          fill_unit_q <= unit_next(fill_unit_q);
          if (last_unit(fill_unit_q)) fill_busy_q <= 1'b0;
        end
      end
    end
  end

  // Port c's beat holds position c + row * Ports, entries u = head * d / E +
  // beat * 4 + w: logical entry c + Ports * w goes to bank (c + Ports * w +
  // rot) mod L, rot = row * Ports + E * alpha(u - w) (= row * Ports + u - w).
  always_comb begin
    int unsigned u0;
    u0       = 32'(fill_head_q) * 32'(dim(fill_unit_q)) / E + 4 * 32'(fill_beat_q);
    fill_rot = SW'((32'(fill_row_q) * Ports + u0) % L);
    for (int unsigned w = 0; w < Entries; w++) begin
      for (int unsigned c = 0; c < Ports; c++) begin
        fill_valid[c + Ports * w] = fill_fire;
        fill_addr[c + Ports * w]  = TW'(32'(fill_slot_q) * SlotDepth + u0 + w);
        fill_data[c + Ports * w]  = dma_out_data[c][w*EntryBits +: EntryBits];
      end
    end
  end

  // --- Step: one beat to every lane --------------------------------------------

  logic                step_busy_q;
  logic [E-1:0]        step_mask_q;    // MATVEC: the last beat's elements
  logic [BufWidth-1:0] step_buf_q;
  logic [Ports-1:0]    step_active_q;  // MATVEC: unit_ports(step_unit_q), registered for timing
  unit_t               step_unit_q;    // the unit stepped
  logic [RowWidth-1:0] step_cyc_q;     // the unit's cycle (MATVEC: the row's beat)
  logic [RowWidth-1:0] step_cycles_q;  // unit_cycles(step_unit_q), registered for timing
  logic                step_slot_q, tile_ok_q;
  logic [L-1:0]        lane_in_ready;
  logic                fire, step_end, step_go, unit_last_cyc, unit_end;
  logic [CW-1:0]       credits_q;      // go_i pulses not yet spent
  // The next cycle's state: the RAMs' addresses come from it.
  logic                slot_n;
  unit_t               unit_n;
  logic [RowWidth-1:0] cyc_n;
  logic [L-1:0][TW-1:0] tile_addr;
  logic [SW-1:0]       tile_rot;
  logic [L-1:0][EntryBits-1:0] tile_data;

  assign dma_out_ready = ({Ports{fire && step_unit_q.matvec}} & step_active_q)
                       | {Ports{fill_fire}};
  assign unit_last_cyc = step_cyc_q + 1'b1 == step_cycles_q;
  assign fire          = step_busy_q && &lane_in_ready
                         && (step_unit_q.matvec ? &(dma_out_valid | ~step_active_q) : tile_ok_q);
  assign unit_end      = fire && unit_last_cyc;
  assign step_end      = unit_end && last_unit(step_unit_q);
  assign step_mv_end   = step_end && step_unit_q.matvec;
  assign step_go       = !q_head[2].wait_go || credits_q != '0;
  assign step_load     = q_valid[2] && (!step_busy_q || step_end) && step_go;
  assign q_pop[2]      = step_load;
  assign tile_free     = unit_end && !step_unit_q.matvec ? 2'(1) << step_slot_q : 2'b00;
  assign x_buf_o       = step_load ? q_head[2].a : step_buf_q;

  always_comb begin
    slot_n = step_slot_q ^ (unit_end && !step_unit_q.matvec);
    unit_n = step_load ? unit_init(q_head[2]) : unit_end ? unit_next(step_unit_q) : step_unit_q;
    cyc_n  = step_load || unit_end ? '0 : fire ? step_cyc_q + 1'b1 : step_cyc_q;
    // The RAM's address is the next cycle's, so x_data_i is this one's.
    if (unit_n.matvec) begin
      x_addr_o[0] = cyc_n;
    end else if (!unit_n.values) begin
      x_addr_o[0] = unit_n.xh + cyc_n;
    end else begin
      x_addr_o[0] = unit_n.xh + unit_n.pos / RowWidth'(E) + cyc_n;
    end
    x_addr_o[1] = x_addr_o[0] + unit_n.stride / RowWidth'(E);
    // SCORES reads entry cyc of every position; VALUES the 4 positions
    // E*cyc ... of every entry.
    for (int unsigned i = 0; i < L; i++) begin
      tile_addr[i] = !unit_n.values
          ? TW'(32'(slot_n) * SlotDepth + 32'(cyc_n))
          : TW'(32'(slot_n) * SlotDepth + alpha_inv(i / E));
    end
    tile_rot = !unit_n.values ? SW'(E * alpha(32'(cyc_n))) : SW'(E * 32'(cyc_n));
  end

  tile_buffer #(
    .N    (L),
    .Depth(2 * SlotDepth),
    .W    (EntryBits)
  ) u_tile (
    .clk_i,
    .wr_valid_i(fill_valid),
    .wr_addr_i (fill_addr),
    .wr_data_i (fill_data),
    .wr_rot_i  (fill_rot),
    .rd_addr_i (tile_addr),
    .rd_rot_i  (tile_rot),
    .rd_data_o (tile_data)
  );

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      step_busy_q <= 1'b0;
      step_slot_q <= 1'b0;
      tile_ok_q   <= 1'b0;
      credits_q   <= '0;
    end else begin
      step_slot_q <= slot_n;
      step_unit_q <= unit_n;
      step_cyc_q  <= cyc_n;
      // The slot read next cycle was full this one, so the RAM's read sees it.
      tile_ok_q   <= tile_full_q[slot_n];
      credits_q   <= credits_q + CW'(go_i) - CW'(step_load && q_head[2].wait_go);
      step_cycles_q <= unit_cycles(unit_n);
      step_active_q <= unit_ports(unit_n);  // used for MATVEC only
      if (step_load) begin
        step_busy_q   <= 1'b1;
        step_buf_q    <= q_head[2].a;
        for (int unsigned e = 0; e < E; e++) begin
          step_mask_q[e] <= q_head[2].cols % ColWidth'(E) == '0
                            || ColWidth'(e) < q_head[2].cols % ColWidth'(E);
        end
      end else if (step_end) begin
        step_busy_q <= 1'b0;
      end
    end
  end

  // --- Lanes -------------------------------------------------------------------

  logic [L-1:0]                   lane_out_valid, lane_out_ready;
  logic [31:0]                    lane_out[L];
  logic [E-1:0]                   values_mask;     // positions of this cycle before t

  for (genvar e = 0; e < E; e++) begin : gen_values_mask
    assign values_mask[e] = E * step_cyc_q + e < rem(step_unit_q);
  end

  for (genvar l = 0; l < L; l++) begin : gen_lane
    localparam int unsigned P = l / LanesPerPort, I = l % LanesPerPort;
    localparam int unsigned G = E * alpha(l / E);  // VALUES: the lane's entries
    logic [E-1:0][15:0] w, x, vw;
    logic [E-1:0]       mask;
    logic               valid, last, second;
    for (genvar e = 0; e < E; e++) begin : gen_vw
      assign vw[e] = tile_data[G + e][(l % E)*16 +: 16];
    end
    always_comb begin
      second = (l >> step_unit_q.dlog) == 1;  // VALUES: the pass's second head
      valid = fire && l < 32'(res_lanes(step_unit_q));
      x     = x_data_i[0];
      last  = unit_last_cyc;
      if (step_unit_q.matvec) begin
        w     = dma_out_data[P][I*E*16 +: E*16];
        mask  = unit_last_cyc ? step_mask_q : '1;
      end else if (!step_unit_q.values) begin
        w     = tile_data[l];
        mask  = '1;
      end else begin
        w     = vw;
        x     = x_data_i[second];
        mask  = values_mask;
        last  = unit_last_cyc && last_tile(step_unit_q);
      end
    end
    dot_lane #(
      .E(E),
      .A(A)
    ) u_lane (
      .clk_i,
      .rst_ni,
      .in_valid_i (valid),
      .in_ready_o (lane_in_ready[l]),
      .w_i        (w),
      .x_i        (x),
      .mask_i     (mask),
      .last_i     (last),
      .out_valid_o(lane_out_valid[l]),
      .out_ready_i(lane_out_ready[l]),
      .out_o      (lane_out[l])
    );
  end

  // --- Drain: results to the write port, Writes lanes per cycle ----------------

  logic                     drain_busy_q, drain_bf16_q;
  logic [BufWidth-1:0]      drain_buf_q;
  logic [31:0]              drain_scale_q;
  logic [GW-1:0]            drain_grp_q;  // the write group
  unit_t                    drain_unit_q; // the unit whose results drain
  logic [RowWidth-1:0]      lanes_with_result, pass_row;
  logic [Writes-1:0]        has, slot_valid;
  logic [Writes-1:0][31:0]  slot_out;
  logic                     go, pass_end, drain_end, drain_load;

  assign lanes_with_result = RowWidth'(res_lanes(drain_unit_q));
  assign pass_row          = res_row(drain_unit_q);
  for (genvar s = 0; s < Writes; s++) begin : gen_slot
    logic [RowWidth-1:0] lane;
    assign lane          = RowWidth'(drain_grp_q) * RowWidth'(Writes) + RowWidth'(s);
    assign has[s]        = lane < lanes_with_result;
    assign slot_out[s]   = lane_out[drain_grp_q*Writes+s];
    assign slot_valid[s] = lane_out_valid[drain_grp_q*Writes+s];
  end
  assign go     = drain_busy_q && &(slot_valid | ~has);
  always_comb begin
    lane_out_ready = '0;
    for (int unsigned s = 0; s < Writes; s++) begin
      lane_out_ready[drain_grp_q*Writes+s] = go && has[s];
    end
  end
  assign pass_end   = go && RowWidth'(drain_grp_q + 1'b1) * RowWidth'(Writes) >= lanes_with_result;
  assign drain_end  = pass_end && last_unit(drain_unit_q);
  assign drain_load = q_valid[3] && (!drain_busy_q || drain_end);
  assign q_pop[3]   = drain_load;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      drain_busy_q <= 1'b0;
      drain_grp_q  <= '0;
    end else if (drain_load) begin
      drain_busy_q   <= 1'b1;
      drain_grp_q    <= '0;
      drain_unit_q   <= to_last_tile(unit_init(q_head[3]));
      drain_buf_q    <= q_head[3].dst;
      drain_bf16_q   <= q_head[3].bf16;
      drain_scale_q  <= q_head[3].op == OpScores ? q_head[3].scale : Fp32One;
    end else if (drain_end) begin
      drain_busy_q <= 1'b0;
    end else if (pass_end) begin
      drain_grp_q    <= '0;
      drain_unit_q   <= to_last_tile(unit_next(drain_unit_q));
    end else if (go) begin
      drain_grp_q <= drain_grp_q + 1'b1;
    end
  end

  // The scale's multiply (exact for 1: x * 1 + -0 = x), then the write port,
  // registered; the write's fields wait for the multiply.
  typedef struct packed {
    logic                valid, done, bf16;
    logic [BufWidth-1:0] buf_id;
    logic [RowWidth-1:0] row;
    logic [Writes-1:0]   mask;
  } wr_t;

  wr_t                     wr_q[DrainLat];
  logic [Writes-1:0][31:0] scaled;
  logic [Writes-1:0][15:0] rounded;

  for (genvar s = 0; s < Writes; s++) begin : gen_scale
    fp32_fma #(
      .MulRegs(MulRegs)
    ) u_mul (
      .clk_i,
      .rst_ni,
      .valid_i(go),
      .op_i   (Fp32Mul),
      .a_i    (slot_out[s]),
      .b_i    (drain_scale_q),
      .c_i    ('0),
      // verilator lint_off PINCONNECTEMPTY
      .valid_o(),
      // verilator lint_on PINCONNECTEMPTY
      .y_o    (scaled[s])
    );
    fp32_to_bf16 u_round (
      .a_i(scaled[s]),
      .y_o(rounded[s])
    );
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      for (int unsigned i = 0; i < DrainLat; i++) wr_q[i] <= '0;
      w_valid_o <= 1'b0;
      done_o    <= 1'b0;
    end else begin
      wr_q[0] <= '{valid: go && |has, done: drain_end, bf16: drain_bf16_q,
                   buf_id: drain_buf_q,
                   row: pass_row + RowWidth'(drain_grp_q) * RowWidth'(Writes),
                   mask: has};
      for (int unsigned i = 1; i < DrainLat; i++) wr_q[i] <= wr_q[i-1];
      w_valid_o <= wr_q[DrainLat-1].valid;
      done_o    <= wr_q[DrainLat-1].done;
    end
    w_buf_o  <= wr_q[DrainLat-1].buf_id;
    w_row_o  <= wr_q[DrainLat-1].row;
    w_mask_o <= wr_q[DrainLat-1].mask;
    w_bf16_o <= wr_q[DrainLat-1].bf16;
    for (int unsigned s = 0; s < Writes; s++) begin
      w_data_o[s] <= wr_q[DrainLat-1].bf16 ? {16'h0, rounded[s]} : scaled[s];
    end
  end

endmodule
