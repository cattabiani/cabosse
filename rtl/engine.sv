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
//   time into a free one of three slots (docs/architecture.md, "Tile
//   buffer banks"), a beat from every port per cycle. On a MATVEC it waits
//   until step has read it.
// - Step: one beat to every lane per cycle, when the lanes have room for
//   it, and for a MATVEC every port with rows in the pass has a beat, for
//   attention the unit's tile is full. x comes from the read ports, E
//   values broadcast (a VALUES lane takes its head's). Back-to-back
//   commands leave no gap. A unit's slot is freed when its last cycle is
//   decided.
// Fill and step decide a beat PopLat cycles before it reaches the DMAs,
// the lanes and the tile buffer, from estimates ("Beats" below, D-051).
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
  localparam int unsigned Slots     = 3;  // a unit filling, one full, one stepping (D-051)
  localparam int unsigned TW        = $clog2(Slots * SlotDepth);  // a tile buffer address
  localparam int unsigned EntryBits = E * 16;
  localparam int unsigned BeatValues = DataWidth / 16;     // BF16 values per beat
  localparam int unsigned TileRows  = LanesPerPort;        // a tile's rows per channel
  localparam int unsigned DW        = $clog2(MaxD + 1);
  localparam int unsigned DrainLat  = mul_add_latency(MulRegs);  // fp32_fma's
  localparam int unsigned CrW       = $clog2(Depth + 1);   // a FIFO count

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
  function automatic logic [1:0] next_slot(logic [1:0] s);
    return s == 2'(Slots - 1) ? '0 : s + 1'b1;
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
      .out_data_o (q_head[i]),
      // verilator lint_off PINCONNECTEMPTY
      .count_o    ()
      // verilator lint_on PINCONNECTEMPTY
    );
  end

  // --- DMAs --------------------------------------------------------------------

  logic [Ports-1:0]                dma_cmd_valid, dma_cmd_ready;
  logic [Ports-1:0]                dma_out_valid, dma_out_ready;
  logic [Ports-1:0][DataWidth-1:0] dma_out_data;
  logic [AddrWidth-1:0]            dma_addr;
  logic [ColWidth-1:0]             dma_beats;
  logic [Ports-1:0][CrW-1:0]       dma_out_count;  // beats in each port's FIFO

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
      .out_count_o(dma_out_count[p]),
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

  // --- Beats: decided ahead, delivered through registers (D-051) ---------------
  //
  // Fill and step decide their beats from registered state only: the DMAs'
  // FIFO counts and the lanes' reservations reach the decision through
  // registers, so each decision assumes the worst of the beats it has not
  // seen yet (Window of them). A beat's descriptor then goes through a
  // central register and PopLat - 1 more in copies per group of 4 ports (16
  // lanes, 16 banks): PopLat cycles after its decision the beat pops the
  // DMAs and reaches the lanes, which never refuse it (a lane's ready drops
  // only on a last beat, and a last beat is decided only with room for it).
  // The tile buffer's read addresses are presented at stage RdStage, from a
  // register per bank; its writes enter at PopLat.

  localparam int unsigned NG       = Ports / 4;  // groups
  localparam int unsigned GL       = L / NG;     // lanes and banks per group
  localparam int unsigned PopLat   = 6;
  localparam int unsigned RotRegs  = 2;          // the tile buffer's rotations
  localparam int unsigned RotCopies = GL / E;    // per group: a copy of the rotation per E banks
  localparam int unsigned RdStage  = PopLat - 1 - RotRegs;  // read data at PopLat
  localparam int unsigned XLat     = 4;  // x_data at stage 3, its central register at 4
  localparam int unsigned SeenRegs = 2;  // a group's register and the central one
  // Decisions not yet in what fill and step see: PopLat to the pop or the
  // lane, one into the FIFO count or the reservation, SeenRegs, and the
  // estimate's own register.
  localparam int unsigned Window   = PopLat + SeenRegs + 1;
  // A unit's last write lands this many cycles after its decision, counted
  // so that step's first read of the slot comes after it (a read sees the
  // writes of earlier cycles only).
  localparam int unsigned LandLat  = PopLat + RotRegs - RdStage - 1;
  localparam int unsigned LaneBanks = 2;  // dot_lane's level-1 buffers
  localparam int unsigned RW       = $clog2(LaneBanks + Window + 1);
  localparam int unsigned CycW     = $clog2(SlotDepth > Ports ? SlotDepth : Ports);

  if (PopLat <= XLat || RdStage < 3) begin : gen_bad_lat
    $error("engine: PopLat must leave the x path and the read addresses a group register");
  end

  typedef struct packed {
    logic                 valid;    // a beat to the lanes
    logic                 matvec, values, last;
    logic [E-1:0]         mask;
    logic [LW-1:0]        lanes;    // the lanes with a row
    logic [2:0]           dlog;
    logic [Ports-1:0]     active;   // MATVEC: the ports popped
    logic [1:0]           slot;     // the tile read
    logic [CycW-1:0]      cyc;
    logic [SW-1:0]        rot;
    logic                 fill;     // a fill beat, from every port
    logic [TW-1:0]        f_base;   // its slot's and the row's first entry
    logic [SW-1:0]        f_shift;  // bank B takes logical entry (B + f_shift) mod L
  } beat_t;

  // --- Fill: attention's beats into the tile buffer --------------------------

  logic [Slots-1:0]      tile_full_q;   // per slot: being filled or full, not yet stepped
  logic [Slots-1:0]      tile_ready_q;  // per slot: its writes landed, not yet stepped
  logic [Slots-1:0]      tile_fill, tile_free;
  logic [Slots-1:0]      land_q[LandLat];
  logic [1:0]            fill_slot_q;
  logic                  fill_busy_q, fill_head_q;
  logic [$clog2(TileRows)-1:0] fill_row_q;       // the tile's row
  logic [DW-1:0]         fill_beat_q;   // the row's beat
  unit_t                 fill_unit_q;
  logic                  fill_fire, fill_row_end, fill_tile_end, fill_unit_end, step_mv_end;
  logic                  fill_mv_q;     // a MATVEC whose beats step still reads
  logic                  step_load;     // step takes a command (below)
  logic                  dma_ok;        // a beat from every port checked (below)
  logic [TW-1:0]         fill_base;
  logic [SW-1:0]         fill_shift;

  // A slot freed this cycle takes writes at once: its last reads come first.
  assign fill_fire     = fill_busy_q && (!tile_full_q[fill_slot_q] || tile_free[fill_slot_q])
                         && dma_ok;
  assign fill_row_end  = fill_fire && 32'(fill_beat_q) + 1 == 32'(dim(fill_unit_q)) / BeatValues;
  assign fill_tile_end = fill_row_end && 32'(fill_row_q) == TileRows - 1;  // a head's tile
  assign fill_unit_end = fill_tile_end && 2'(fill_head_q) + 2'd1 == unit_heads(fill_unit_q);
  assign tile_fill     = fill_unit_end ? Slots'(1) << fill_slot_q : '0;
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
      fill_slot_q  <= '0;
      tile_full_q  <= '0;
      tile_ready_q <= '0;
      for (int unsigned i = 0; i < LandLat; i++) land_q[i] <= '0;
    end else begin
      tile_full_q  <= (tile_full_q | tile_fill) & ~tile_free;
      land_q[0]    <= tile_fill;
      for (int unsigned i = 1; i < LandLat; i++) land_q[i] <= land_q[i-1];
      tile_ready_q <= (tile_ready_q | land_q[LandLat-1]) & ~tile_free;
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
          fill_slot_q <= next_slot(fill_slot_q);
          fill_unit_q <= unit_next(fill_unit_q);
          if (last_unit(fill_unit_q)) fill_busy_q <= 1'b0;
        end
      end
    end
  end

  // Port c's beat holds position c + row * Ports, entries u = head * d / E +
  // beat * 4 + w: logical entry c + Ports * w goes to bank (c + Ports * w +
  // rot) mod L, rot = row * Ports + E * alpha(u - w) (= row * Ports + u - w),
  // at fill_base + w; bank B takes logical entry (B + L - rot) mod L.
  always_comb begin
    int unsigned u0, rot;
    u0         = 32'(fill_head_q) * 32'(dim(fill_unit_q)) / E + 4 * 32'(fill_beat_q);
    rot        = (32'(fill_row_q) * Ports + u0) % L;
    fill_shift = SW'((L - rot) % L);
    fill_base  = TW'(32'(fill_slot_q) * SlotDepth + u0);
  end

  // --- Step: one beat to every lane --------------------------------------------
  //
  // What step's decision reads of its unit is registered: for the next
  // unit, from each candidate's own copy (the command taken, the unit after
  // this one), muxed at the end, as issue does.

  logic                step_busy_q;
  logic [E-1:0]        step_mask_q;    // MATVEC: the last beat's elements
  logic [BufWidth-1:0] step_buf_q;
  logic [Ports-1:0]    step_active_q;  // MATVEC: unit_ports(step_unit_q)
  unit_t               step_unit_q;    // the unit stepped
  logic [RowWidth-1:0] step_cyc_q;     // the unit's cycle (MATVEC: the row's beat)
  logic [RowWidth-1:0] step_cycles_q;  // unit_cycles(step_unit_q)
  logic                step_last_cyc_q;   // step_cyc_q + 1 == step_cycles_q
  logic                step_last_tile_q;  // last_tile(step_unit_q)
  logic                step_last_unit_q;  // last_unit(step_unit_q)
  logic [1:0]          step_slot_q;
  logic                tile_ok_q;
  logic                fire, step_end, step_go, unit_end, last_beat, room_ok;
  logic [CW-1:0]       credits_q;      // go_i pulses not yet spent
  logic [1:0]          slot_n;
  unit_t               unit_init_c, unit_next_c, unit_n;
  logic [RowWidth-1:0] cyc_n;
  logic [E-1:0]        values_mask;    // positions of this cycle before t

  assign last_beat     = step_last_cyc_q && (!step_unit_q.values || step_last_tile_q);
  assign fire          = step_busy_q && (step_unit_q.matvec ? dma_ok : tile_ok_q)
                         && (!last_beat || room_ok);
  assign unit_end      = fire && step_last_cyc_q;
  assign step_end      = unit_end && step_last_unit_q;
  assign step_mv_end   = step_end && step_unit_q.matvec;
  assign step_go       = !q_head[2].wait_go || credits_q != '0;
  assign step_load     = q_valid[2] && (!step_busy_q || step_end) && step_go;
  assign q_pop[2]      = step_load;
  // The slot's reads are all decided; fill's writes to it land after them.
  assign tile_free     = unit_end && !step_unit_q.matvec ? Slots'(1) << step_slot_q : '0;
  assign unit_init_c   = unit_init(q_head[2]);
  assign unit_next_c   = unit_next(step_unit_q);

  for (genvar e = 0; e < E; e++) begin : gen_values_mask
    assign values_mask[e] = E * step_cyc_q + e < rem(step_unit_q);
  end

  always_comb begin
    slot_n = unit_end && !step_unit_q.matvec ? next_slot(step_slot_q) : step_slot_q;
    unit_n = step_load ? unit_init_c : unit_end ? unit_next_c : step_unit_q;
    cyc_n  = step_load || unit_end ? '0 : fire ? step_cyc_q + 1'b1 : step_cyc_q;
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      step_busy_q <= 1'b0;
      step_slot_q <= '0;
      tile_ok_q   <= 1'b0;
      credits_q   <= '0;
    end else begin
      step_slot_q <= slot_n;
      step_unit_q <= unit_n;
      step_cyc_q  <= cyc_n;
      // The slot stepped next cycle: its writes landed by this one.
      tile_ok_q   <= tile_ready_q[slot_n];
      credits_q   <= credits_q + CW'(go_i) - CW'(step_load && q_head[2].wait_go);
      if (step_load) begin
        step_cycles_q    <= unit_cycles(unit_init_c);
        step_active_q    <= unit_ports(unit_init_c);
        step_last_cyc_q  <= unit_cycles(unit_init_c) == RowWidth'(1);
        step_last_tile_q <= last_tile(unit_init_c);
        step_last_unit_q <= last_unit(unit_init_c);
      end else if (unit_end) begin
        step_cycles_q    <= unit_cycles(unit_next_c);
        step_active_q    <= unit_ports(unit_next_c);
        step_last_cyc_q  <= unit_cycles(unit_next_c) == RowWidth'(1);
        step_last_tile_q <= last_tile(unit_next_c);
        step_last_unit_q <= last_unit(unit_next_c);
      end else if (fire) begin
        step_last_cyc_q  <= step_cyc_q + RowWidth'(2) == step_cycles_q;
      end
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

  // --- What fill and step have not seen: estimates ----------------------------
  //
  // DMA beats: each port's FIFO count, of the ports checked (chk; a
  // MATVEC's last pass leaves ports out), as a min per group and over the
  // groups, less the pops of the last Window decisions. The ports an
  // estimate covers go down with it (cover); a pop needs every port
  // checked now among them.
  // Lane room: each lane's reservations as thermometer bits, ORed per group
  // and over the groups (the most any lane has), plus the last beats of the
  // last Window decisions; a last beat needs fewer than LaneBanks.

  // The min of n counts, as a balanced tree.
  function automatic logic [CrW-1:0] min_tree(logic [NG*4-1:0][CrW-1:0] v, int unsigned n);
    for (int unsigned w = 1; w < n; w *= 2) begin
      for (int unsigned i = 0; i + w < n; i += 2 * w) if (v[i+w] < v[i]) v[i] = v[i+w];
    end
    return v[0];
  endfunction

  logic [L-1:0][$clog2(LaneBanks):0] lane_reserved;
  logic [Ports-1:0]             chk_d, chk_q;
  logic [Ports-1:0]             cover_q[SeenRegs+2];  // chk_q's, as each stage's min has them
  logic                         pop_now, last_now;
  logic [NG-1:0][CrW-1:0]       grp_min_q;
  logic [NG-1:0][LaneBanks-1:0] grp_res_q;
  logic [CrW-1:0]               min_q;
  logic [LaneBanks-1:0]         res_q;
  logic [Window-2:0]            pop_win_q, last_win_q;  // the decisions before this one
  logic signed [CrW+1:0]        est_q;   // beats certainly in every port covered
  logic [RW-1:0]                room_q;  // reservations a lane may have, at most

  assign chk_d    = fill_busy_q ? '1 : step_active_q;
  assign dma_ok   = est_q > 0 && !(|(chk_d & ~cover_q[SeenRegs+1]));
  assign room_ok  = room_q < RW'(LaneBanks);
  assign pop_now  = (fire && step_unit_q.matvec) || fill_fire;
  assign last_now = fire && last_beat;

  for (genvar g = 0; g < NG; g++) begin : gen_seen
    logic [3:0]                 mask_q;  // chk_q's, for this group's ports
    logic [NG*4-1:0][CrW-1:0]   counts;
    logic [LaneBanks-1:0]       r;
    always_comb begin
      counts = '1;
      for (int unsigned p = 0; p < 4; p++) if (mask_q[p]) counts[p] = dma_out_count[4*g+p];
      r = '0;
      for (int unsigned l = GL * g; l < GL * (g + 1); l++) begin
        for (int unsigned k = 0; k < LaneBanks; k++) r[k] = r[k] | (32'(lane_reserved[l]) > k);
      end
    end
    always_ff @(posedge clk_i) begin
      mask_q       <= chk_q[4*g +: 4];
      grp_min_q[g] <= min_tree(counts, 4);
      grp_res_q[g] <= r;
    end
  end

  logic [NG*4-1:0][CrW-1:0] grp_mins;
  logic [LaneBanks-1:0]     all_res;
  always_comb begin
    grp_mins = '1;
    all_res  = '0;
    for (int unsigned g = 0; g < NG; g++) begin
      grp_mins[g] = grp_min_q[g];
      all_res     = all_res | grp_res_q[g];
    end
  end

  always_ff @(posedge clk_i) begin
    min_q      <= min_tree(grp_mins, NG);
    res_q      <= all_res;
    chk_q      <= chk_d;
    cover_q[0] <= chk_q;
    for (int unsigned i = 1; i < SeenRegs + 2; i++) cover_q[i] <= cover_q[i-1];
    if (!rst_ni) begin
      pop_win_q  <= '0;
      last_win_q <= '0;
      est_q      <= '0;
      room_q     <= '0;
    end else begin
      est_q      <= $signed({2'b00, min_q}) - $signed((CrW + 2)'($countones(pop_win_q)))
                    - $signed((CrW + 2)'(pop_now));
      room_q     <= RW'($countones(res_q)) + RW'($countones(last_win_q)) + RW'(last_now);
      pop_win_q  <= {pop_win_q[Window-3:0], pop_now};
      last_win_q <= {last_win_q[Window-3:0], last_now};
    end
  end

  // --- The beat's way: descriptor, x, tile buffer, DMAs, lanes ----------------

  beat_t beat_d, beat_q;  // the decision, its central register

  always_comb begin
    beat_d         = '0;
    beat_d.valid   = fire;
    beat_d.matvec  = step_unit_q.matvec;
    beat_d.values  = step_unit_q.values;
    beat_d.last    = last_beat;
    beat_d.mask    = step_unit_q.matvec ? (step_last_cyc_q ? step_mask_q : '1)
                   : step_unit_q.values ? values_mask : '1;
    beat_d.lanes   = res_lanes(step_unit_q);
    beat_d.dlog    = step_unit_q.dlog;
    beat_d.active  = step_active_q;
    beat_d.slot    = step_slot_q;
    beat_d.cyc     = CycW'(step_cyc_q);
    // SCORES reads entry cyc of every position; VALUES the 4 positions
    // E*cyc ... of every entry.
    beat_d.rot     = !step_unit_q.values ? SW'(E * alpha(32'(step_cyc_q)))
                                         : SW'(E * 32'(step_cyc_q));
    beat_d.fill    = fill_fire;
    beat_d.f_base  = fill_base;
    beat_d.f_shift = fill_shift;
  end

  // x: the address from the decision, presented at stage 2, so x_data_i
  // comes at 3; a central register (XLat) and the group's take it to PopLat.
  logic [Heads-1:0][RowWidth-1:0] x_addr_d, x_addr_q;
  logic [BufWidth-1:0]            x_buf_q;
  logic [Heads-1:0][E-1:0][15:0]  x_data_q;

  always_comb begin
    if (step_unit_q.matvec) begin
      x_addr_d[0] = step_cyc_q;
    end else if (!step_unit_q.values) begin
      x_addr_d[0] = step_unit_q.xh + step_cyc_q;
    end else begin
      x_addr_d[0] = step_unit_q.xh + step_unit_q.pos / RowWidth'(E) + step_cyc_q;
    end
    x_addr_d[1] = x_addr_d[0] + step_unit_q.stride / RowWidth'(E);
  end

  always_ff @(posedge clk_i) begin
    beat_q   <= beat_d;
    x_addr_q <= x_addr_d;
    x_buf_q  <= step_buf_q;
    x_addr_o <= x_addr_q;
    x_buf_o  <= x_buf_q;
    x_data_q <= x_data_i;
    if (!rst_ni) begin
      beat_q.valid <= 1'b0;
      beat_q.fill  <= 1'b0;
    end
  end

  logic [NG*RotCopies-1:0][SW-1:0] rd_rot, wr_shift;
  logic [L-1:0][TW-1:0]        tile_addr, fill_addr;
  logic [L-1:0]                fill_valid;
  logic [L-1:0][EntryBits-1:0] fill_data;
  logic [L-1:0][EntryBits-1:0] tile_data;
  beat_t [NG-1:0]              beat_at;   // each group's beat at PopLat
  logic [NG-1:0][Heads-1:0][E-1:0][15:0] x_at;

  for (genvar g = 0; g < NG; g++) begin : gen_group
    beat_t pipe_q[PopLat-1];  // stages 2 .. PopLat
    logic [Heads-1:0][E-1:0][15:0] x_q[PopLat-XLat];  // stages XLat + 1 .. PopLat
    logic [RotCopies-1:0][SW-1:0]  shift_q;           // the fill's, at PopLat
    always_ff @(posedge clk_i) begin
      pipe_q[0] <= beat_q;
      for (int unsigned s = 1; s < PopLat - 1; s++) pipe_q[s] <= pipe_q[s-1];
      x_q[0] <= x_data_q;
      for (int unsigned s = 1; s < PopLat - XLat; s++) x_q[s] <= x_q[s-1];
      for (int unsigned k = 0; k < RotCopies; k++) shift_q[k] <= pipe_q[PopLat-3].f_shift;
      if (!rst_ni) begin
        for (int unsigned s = 0; s < PopLat - 1; s++) begin
          pipe_q[s].valid <= 1'b0;
          pipe_q[s].fill  <= 1'b0;
        end
      end
    end
    assign beat_at[g] = pipe_q[PopLat-2];
    assign x_at[g]    = x_q[PopLat-XLat-1];
    for (genvar k = 0; k < RotCopies; k++) begin : gen_rot
      assign rd_rot[g*RotCopies+k]   = pipe_q[RdStage-2].rot;
      assign wr_shift[g*RotCopies+k] = shift_q[k];
    end
    // Each bank's read address, a stage ahead, in its own register: physical
    // bank B holds logical entry (B - rot) mod L. SCORES: entry cyc in every
    // bank; VALUES: logical entry i's alpha_inv(i / E), i / E = (B / E -
    // cyc) mod Ports.
    for (genvar b = GL * g; b < GL * (g + 1); b++) begin : gen_bank
      logic [1:0]      slot;
      logic [CycW-1:0] cyc;
      logic            values;
      int unsigned     q;
      logic [TW-1:0]   addr_q;
      assign slot   = pipe_q[RdStage-3].slot;
      assign cyc    = pipe_q[RdStage-3].cyc;
      assign values = pipe_q[RdStage-3].values;
      assign q      = (b / E + Ports - 32'(cyc) % Ports) % Ports;
      always_ff @(posedge clk_i) begin
        addr_q <= !values ? TW'(32'(slot) * SlotDepth + 32'(cyc))
                          : TW'(32'(slot) * SlotDepth + alpha_inv(q));
      end
      assign tile_addr[b] = addr_q;
      // The fill's write: bank B takes logical entry i = (B + f_shift) mod
      // L, port i mod Ports's word i / Ports, at f_base + i / Ports.
      assign fill_valid[b] = beat_at[g].fill;
      assign fill_addr[b]  = beat_at[g].f_base
                             + TW'((b + 32'(beat_at[g].f_shift)) % L / Ports);
    end
    // The fill's logical entries c + Ports * w in this group: port c's word w.
    for (genvar i = GL * g; i < GL * (g + 1); i++) begin : gen_fill
      assign fill_data[i] = dma_out_data[i % Ports][(i / Ports)*EntryBits +: EntryBits];
    end
    for (genvar p = 4 * g; p < 4 * g + 4; p++) begin : gen_pop
      assign dma_out_ready[p] = (beat_at[g].valid && beat_at[g].matvec && beat_at[g].active[p])
                                || beat_at[g].fill;
    end
  end

  tile_buffer_pipe #(
    .N     (L),
    .Depth (Slots * SlotDepth),
    .W     (EntryBits),
    .Regs  (RotRegs),
    .Groups(NG * RotCopies)
  ) u_tile (
    .clk_i,
    .wr_valid_i(fill_valid),
    .wr_addr_i (fill_addr),
    .wr_data_i (fill_data),
    .wr_shift_i(wr_shift),
    .rd_addr_i (tile_addr),
    .rd_rot_i  (rd_rot),
    .rd_data_o (tile_data)
  );

  // --- Lanes -------------------------------------------------------------------

  logic [L-1:0]                   lane_in_ready, lane_in_valid;
  logic [L-1:0]                   lane_out_valid, lane_out_ready;
  logic [31:0]                    lane_out[L];

  for (genvar l = 0; l < L; l++) begin : gen_lane
    localparam int unsigned P = l / LanesPerPort, I = l % LanesPerPort;
    localparam int unsigned G = E * alpha(l / E);  // VALUES: the lane's entries
    // verilator lint_off UNUSEDSIGNAL
    beat_t              bt;  // a lane takes the fields of its beat, not the tile's or the fill's
    // verilator lint_on UNUSEDSIGNAL
    logic [E-1:0][15:0] w, x, vw;
    logic               second;
    assign bt = beat_at[l/GL];
    for (genvar e = 0; e < E; e++) begin : gen_vw
      assign vw[e] = tile_data[G + e][(l % E)*16 +: 16];
    end
    always_comb begin
      second = (l >> bt.dlog) == 1;  // VALUES: the pass's second head
      lane_in_valid[l] = bt.valid && l < 32'(bt.lanes);
      if (bt.matvec) begin
        w = dma_out_data[P][I*E*16 +: E*16];
        x = x_at[l/GL][0];
      end else if (!bt.values) begin
        w = tile_data[l];
        x = x_at[l/GL][0];
      end else begin
        w = vw;
        x = x_at[l/GL][second];
      end
    end
    dot_lane #(
      .E    (E),
      .A    (A),
      .Banks(LaneBanks)
    ) u_lane (
      .clk_i,
      .rst_ni,
      .in_valid_i (lane_in_valid[l]),
      .in_ready_o (lane_in_ready[l]),
      .w_i        (w),
      .x_i        (x),
      .mask_i     (bt.mask),
      .last_i     (bt.last),
      .out_valid_o(lane_out_valid[l]),
      .out_ready_i(lane_out_ready[l]),
      .out_o      (lane_out[l]),
      .reserved_o (lane_reserved[l])
    );
  end

`ifndef SYNTHESIS
  // The estimates' promises: no pop from an empty FIFO, no beat refused.
  always_ff @(posedge clk_i) begin
    if (rst_ni) begin
      if (|(dma_out_ready & ~dma_out_valid)) $error("engine: a pop from an empty DMA FIFO");
      if (|(lane_in_valid & ~lane_in_ready)) $error("engine: a lane refused a beat");
      if ((fire && step_unit_q.matvec) && fill_fire) $error("engine: fill and step pop at once");
    end
  end
`endif

  // --- Drain: results to the write port, Writes lanes per cycle ----------------
  //
  // No one-cycle loop across the lanes (D-051): each write group's "every
  // lane with a result has one" is registered at the group, for the unit
  // draining and the one after it (the cycle after a unit's end reads that
  // one); a group's lanes pop the cycle after its go, so the group waits two
  // cycles before its next go (its registered state is the pop's by then);
  // its results go to the multiply through a register.

  localparam int unsigned WrLat = DrainLat + 1;  // the results' register, then fp32_fma

  logic                     drain_busy_q, drain_bf16_q;
  logic [BufWidth-1:0]      drain_buf_q;
  logic [31:0]              drain_scale_q;
  logic [GW-1:0]            drain_grp_q;  // the write group
  unit_t                    drain_unit_q; // the unit whose results drain
  unit_t                    drain_next;   // the unit after it
  logic [RowWidth-1:0]      lanes_with_result, next_lanes, pass_row;
  logic [Writes-1:0]        has;
  logic [Groups-1:0]        ok_q, ok_next_q;  // per write group: its results are there
  logic                     unit_new_q;       // drain_unit_q changed last cycle
  logic [1:0]               gone_q;           // a go in each of the last two cycles
  logic [GW-1:0]            gone_grp_q[2];
  logic                     go, pass_end, drain_end, drain_load, ok, recent;

  assign lanes_with_result = RowWidth'(res_lanes(drain_unit_q));
  assign pass_row          = res_row(drain_unit_q);
  assign drain_next        = !drain_busy_q || drain_end ? to_last_tile(unit_init(q_head[3]))
                                                        : to_last_tile(unit_next(drain_unit_q));
  assign next_lanes        = RowWidth'(res_lanes(drain_next));
  for (genvar s = 0; s < Writes; s++) begin : gen_slot
    assign has[s] = RowWidth'(drain_grp_q) * RowWidth'(Writes) + RowWidth'(s) < lanes_with_result;
  end
  for (genvar k = 0; k < Groups; k++) begin : gen_wgroup
    logic [Writes-1:0] cur, nxt;  // per lane: has its result, or has none to give
    for (genvar s = 0; s < Writes; s++) begin : gen_lane_ok
      assign cur[s] = lane_out_valid[k*Writes+s] || k * Writes + s >= 32'(lanes_with_result);
      assign nxt[s] = lane_out_valid[k*Writes+s] || k * Writes + s >= 32'(next_lanes);
    end
    always_ff @(posedge clk_i) begin
      ok_q[k]      <= &cur;
      ok_next_q[k] <= &nxt;
    end
  end
  assign ok     = unit_new_q ? ok_next_q[drain_grp_q] : ok_q[drain_grp_q];
  assign recent = (gone_q[0] && gone_grp_q[0] == drain_grp_q)
                  || (gone_q[1] && gone_grp_q[1] == drain_grp_q);
  assign go     = drain_busy_q && ok && !recent;
  assign pass_end   = go && RowWidth'(drain_grp_q + 1'b1) * RowWidth'(Writes) >= lanes_with_result;
  assign drain_end  = pass_end && last_unit(drain_unit_q);
  assign drain_load = q_valid[3] && (!drain_busy_q || drain_end);
  assign q_pop[3]   = drain_load;

  always_ff @(posedge clk_i) begin
    for (int unsigned l = 0; l < L; l++) begin
      lane_out_ready[l] <= go && 32'(drain_grp_q) == l / Writes && has[l%Writes];
    end
    gone_grp_q[0] <= drain_grp_q;
    gone_grp_q[1] <= gone_grp_q[0];
    if (!rst_ni) begin
      lane_out_ready <= '0;
      gone_q         <= '0;
      unit_new_q     <= 1'b0;
    end else begin
      gone_q     <= {gone_q[0], go};
      unit_new_q <= drain_load || pass_end;
    end
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      drain_busy_q <= 1'b0;
      drain_grp_q  <= '0;
    end else if (drain_load) begin
      drain_busy_q   <= 1'b1;
      drain_grp_q    <= '0;
      drain_unit_q   <= drain_next;
      drain_buf_q    <= q_head[3].dst;
      drain_bf16_q   <= q_head[3].bf16;
      drain_scale_q  <= q_head[3].op == OpScores ? q_head[3].scale : Fp32One;
    end else if (drain_end) begin
      drain_busy_q <= 1'b0;
    end else if (pass_end) begin
      drain_grp_q    <= '0;
      drain_unit_q   <= drain_next;
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

  wr_t                     wr_q[WrLat];
  logic                    mul_valid_q;
  logic [31:0]             mul_scale_q;
  logic [Writes-1:0][31:0] slot_q, scaled;
  logic [Writes-1:0][15:0] rounded;

  always_ff @(posedge clk_i) begin
    mul_valid_q <= rst_ni && go;
    mul_scale_q <= drain_scale_q;
    for (int unsigned s = 0; s < Writes; s++) slot_q[s] <= lane_out[drain_grp_q*Writes+s];
  end

  for (genvar s = 0; s < Writes; s++) begin : gen_scale
    fp32_fma #(
      .MulRegs(MulRegs)
    ) u_mul (
      .clk_i,
      .rst_ni,
      .valid_i(mul_valid_q),
      .op_i   (Fp32Mul),
      .a_i    (slot_q[s]),
      .b_i    (mul_scale_q),
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
      for (int unsigned i = 0; i < WrLat; i++) wr_q[i] <= '0;
      w_valid_o <= 1'b0;
      done_o    <= 1'b0;
    end else begin
      wr_q[0] <= '{valid: go && |has, done: drain_end, bf16: drain_bf16_q,
                   buf_id: drain_buf_q,
                   row: pass_row + RowWidth'(drain_grp_q) * RowWidth'(Writes),
                   mask: has};
      for (int unsigned i = 1; i < WrLat; i++) wr_q[i] <= wr_q[i-1];
      w_valid_o <= wr_q[WrLat-1].valid;
      done_o    <= wr_q[WrLat-1].done;
    end
    w_buf_o  <= wr_q[WrLat-1].buf_id;
    w_row_o  <= wr_q[WrLat-1].row;
    w_mask_o <= wr_q[WrLat-1].mask;
    w_bf16_o <= wr_q[WrLat-1].bf16;
    for (int unsigned s = 0; s < Writes; s++) begin
      w_data_o[s] <= wr_q[WrLat-1].bf16 ? {16'h0, rounded[s]} : scaled[s];
    end
  end

endmodule
