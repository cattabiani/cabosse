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
//   kv = h div group. K is stored as a MATVEC's matrix of cap rows (the
//   positions) and d columns, one per KV head, at `addr` (D-062: in each
//   tile of L positions port p holds positions 4p ... 4p+3). A unit is one
//   head and a tile: lane l takes position j * L + l.
// - VALUES: dst[h * d + c] = dot over i < t of (p[h][i], V[kv][i][c]), p[h]
//   the t values of x at h * stride. V is stored transposed, as a matrix of
//   d rows (the dimensions) and cap columns, once per head of a pass (D-062:
//   head a of a pass reads ports a * d / 4 ...). A pass is up to Heads heads
//   (one lane per head and output dimension); a unit is a pass and a tile
//   of L positions, and a lane's row runs through all tiles.
// Every command streams its port's beats straight to the port's lanes.
// Ports is a multiple of 4 (groups of 4 ports, below). Commands need rows
// >= 1 and cols >= 1 (cols = 0 hangs step). The attention commands need d
// a power of two from 16 to MaxD (VALUES: d <= L), cap a multiple of L,
// 1 <= t <= cap, stride (S and P's rows, the STRIDE register) a multiple
// of E with t <= stride, group dividing heads.
//
// Three stages, each with its own queue of the commands taken, so a
// command starts while earlier ones finish (PLAN.md, M5). Each walks a
// command's units (unit_init, unit_next): a MATVEC's passes, an attention
// command's units as above.
// - Issue: DMA commands, in order, per unit and head to each port with
//   rows in it: MATVEC and SCORES ceil(cols / E) beats, VALUES the tile's
//   positions / E. It runs ahead as far as the DMAs' FIFOs allow.
// - Step: one beat to every lane per cycle, when the lanes have room for
//   it and every port with rows in the unit has a beat. x comes from the
//   read ports, E values broadcast (a VALUES lane takes its head's).
//   Back-to-back commands leave no gap. Step decides a beat PopLat cycles
//   before it reaches the DMAs and the lanes, from estimates ("Beats"
//   below, D-051).
// - Drain: the lanes' results, Writes lanes at a time, to the write port,
//   multiplied by scale (SCORES) or 1 (exact), then rounded to BF16 for a
//   BF16 destination. Lanes with no row (past the rows, past t, past the
//   heads) take no beat and have no result. done_o marks a command's last
//   write.
// The engine does not order a command after an earlier one whose output it
// reads; the controller does (D-041, D-042, D-044): it gives such a command
// `wait` and pulses go_i once its x is ready, so issue reads the
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
  localparam int unsigned LPW       = $clog2(LanesPerPort);
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
  if ((LanesPerPort & (LanesPerPort - 1)) != 0 || 16 % LanesPerPort != 0) begin : gen_bad_lpp
    $error("engine: LanesPerPort must be a power of two dividing 16 (a head's ports, d >= 16)");
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
  // Each port reads its unit's beats at addr + its head's KV offset +
  // unit_off: a KV head's region per channel holds SCORES' cap / L tiles of
  // d / E beats, or VALUES' cap / E beats.

  // A MATVEC pass is a SCORES unit of one head whose positions are rows:
  // t is the rows, a tile a pass of L of them.
  typedef struct packed {
    logic                  matvec;                   // a MATVEC, not attention
    logic [ColWidth-1:0]   beats;                    // a port's per unit (VALUES: a whole tile's)
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
    logic [AddrWidth-1:0]  unit_off;  // j * beats * BeatBytes
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
  // The ports head a of a unit reads: MATVEC and SCORES those with rows
  // (positions) in it; VALUES the d / LanesPerPort ports of head a's lanes.
  function automatic logic [Ports-1:0] head_ports(unit_t u, logic a);
    logic [Ports-1:0] ports;
    for (int unsigned p = 0; p < Ports; p++) begin
      ports[p] = u.values ? (p >> (32'(u.dlog) - LPW)) == 32'(a)
                          : !a && p * LanesPerPort < 32'(rem(u));
    end
    return ports;
  endfunction
  function automatic logic [Ports-1:0] unit_ports(unit_t u);  // every head's
    return head_ports(u, 1'b0) | (unit_heads(u) == 2'd2 ? head_ports(u, 1'b1) : '0);
  endfunction

  // A row's beats: ceil(cols / E).
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
    if (u.values) begin
      u.beats  = ColWidth'(L / E);
      u.region = AddrWidth'(c.cap / RowWidth'(E)) * AddrWidth'(BeatBytes);
    end else begin
      u.region = AddrWidth'(c.cap / RowWidth'(Ports)) << (32'(u.dlog) + 1);  // * d * 2 bytes
    end
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

  // --- Command queues --------------------------------------------------------

  cmd_t cmd;
  assign cmd = '{op: cmd_op_i, rows: cmd_rows_i, cols: cmd_cols_i, addr: cmd_addr_i,
                 a: cmd_a_i, dst: cmd_dst_i, bf16: cmd_bf16_i, wait_go: cmd_wait_i,
                 heads: cmd_heads_i, group: cmd_group_i, t: cmd_t_i, cap: cmd_cap_i,
                 stride: cmd_stride_i, scale: cmd_scale_i};

  logic [2:0] q_in_ready, q_valid, q_pop;  // issue, step, drain
  cmd_t       q_head[3];
  assign cmd_ready_o = &q_in_ready;

  for (genvar i = 0; i < 3; i++) begin : gen_queue
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
      .Depth     (Depth),
      .Slices    (LanesPerPort)
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
  logic [Ports-1:0]     iss_ports_q[Heads];  // head_ports(iss_unit_q, a), registered for timing
  logic [ColWidth-1:0]  iss_beats_q;  // unit_cycles(iss_unit_q): each port's beats
  unit_t                iss_unit_q;
  logic [Ports-1:0]     iss_taken, iss_active;
  logic                 iss_done, iss_last_head;

  assign iss_active    = iss_ports_q[iss_head_q];
  assign dma_cmd_valid = {Ports{iss_busy_q}} & iss_active & ~iss_taken_q;
  assign dma_addr      = iss_addr_q + kv_offset(iss_unit_q, iss_head_q) + iss_unit_q.unit_off;
  assign dma_beats     = iss_beats_q;
  assign iss_taken     = iss_taken_q | dma_cmd_ready;
  assign iss_done      = iss_busy_q && &(iss_taken | ~iss_active);  // this DMA command's
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
      iss_beats_q  <= ColWidth'(unit_cycles(unit_init(q_head[0])));
      for (int unsigned a = 0; a < Heads; a++) iss_ports_q[a] <= head_ports(unit_init(q_head[0]), 1'(a));
    end else if (iss_done) begin
      iss_taken_q <= '0;
      if (!iss_last_head) begin
        iss_head_q <= 1'b1;
      end else begin
        iss_head_q   <= 1'b0;
        iss_unit_q   <= unit_next(iss_unit_q);
        iss_beats_q  <= ColWidth'(unit_cycles(unit_next(iss_unit_q)));
        for (int unsigned a = 0; a < Heads; a++) iss_ports_q[a] <= head_ports(unit_next(iss_unit_q), 1'(a));
        if (last_unit(iss_unit_q)) iss_busy_q <= 1'b0;
      end
    end else if (iss_busy_q) begin
      iss_taken_q <= iss_taken;
    end
  end

  // --- Beats: decided ahead, delivered through registers (D-051) ---------------
  //
  // Step decides its beats from registered state only: the DMAs' FIFO
  // counts and the lanes' reservations reach the decision through
  // registers, so each decision assumes the worst of the beats it has not
  // seen yet (Window of them). A beat's descriptor then goes through a
  // central register and PopLat - 1 more in copies per group of 4 ports (16
  // lanes): PopLat cycles after its decision the beat pops the DMAs and
  // reaches the lanes, which never refuse it (a lane's ready drops only on
  // a last beat, and a last beat is decided only with room for it).

  localparam int unsigned GP       = 4;          // ports per group
  localparam int unsigned NG       = Ports / GP; // groups
  localparam int unsigned GL       = L / NG;     // lanes per group
  localparam int unsigned PopLat   = 6;
  localparam int unsigned XLat     = 4;  // x_data at stage 3, its central register at 4
  localparam int unsigned SeenRegs = 2;  // a group's register and the central one
  // Decisions not yet in what step sees: PopLat to the pop or the lane, one
  // into the FIFO count or the reservation, SeenRegs, and the estimate's own
  // register.
  localparam int unsigned Window   = PopLat + SeenRegs + 1;
  localparam int unsigned LaneBanks = 2;  // dot_lane's level-1 buffers
  localparam int unsigned RW       = $clog2(LaneBanks + Window + 1);

  if (PopLat <= XLat) begin : gen_bad_lat
    $error("engine: PopLat must leave the x path a group register");
  end

  typedef struct packed {
    logic                 valid;    // a beat to the lanes
    logic                 values, last;
    logic [E-1:0]         mask;
    logic [LW-1:0]        lanes;    // the lanes with a row
    logic [2:0]           dlog;
    logic [Ports-1:0]     active;   // the ports popped
  } beat_t;

  logic dma_ok;  // a beat from every port checked (below)

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
  logic                fire, step_end, step_go, step_load, unit_end, last_beat, room_ok;
  logic [CW-1:0]       credits_q;      // go_i pulses not yet spent
  unit_t               unit_init_c, unit_next_c, unit_n;
  logic [RowWidth-1:0] cyc_n;
  logic [E-1:0]        values_mask;    // positions of this cycle before t

  assign last_beat     = step_last_cyc_q && (!step_unit_q.values || step_last_tile_q);
  assign fire          = step_busy_q && dma_ok && (!last_beat || room_ok);
  assign unit_end      = fire && step_last_cyc_q;
  assign step_end      = unit_end && step_last_unit_q;
  assign step_go       = !q_head[1].wait_go || credits_q != '0;
  assign step_load     = q_valid[1] && (!step_busy_q || step_end) && step_go;
  assign q_pop[1]      = step_load;
  assign unit_init_c   = unit_init(q_head[1]);
  assign unit_next_c   = unit_next(step_unit_q);

  for (genvar e = 0; e < E; e++) begin : gen_values_mask
    assign values_mask[e] = E * step_cyc_q + e < rem(step_unit_q);
  end

  always_comb begin
    unit_n = step_load ? unit_init_c : unit_end ? unit_next_c : step_unit_q;
    cyc_n  = step_load || unit_end ? '0 : fire ? step_cyc_q + 1'b1 : step_cyc_q;
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      step_busy_q <= 1'b0;
      credits_q   <= '0;
    end else begin
      step_unit_q <= unit_n;
      step_cyc_q  <= cyc_n;
      credits_q   <= credits_q + CW'(go_i) - CW'(step_load && q_head[1].wait_go);
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
        step_buf_q    <= q_head[1].a;
        for (int unsigned e = 0; e < E; e++) begin
          step_mask_q[e] <= q_head[1].cols % ColWidth'(E) == '0
                            || ColWidth'(e) < q_head[1].cols % ColWidth'(E);
        end
      end else if (step_end) begin
        step_busy_q <= 1'b0;
      end
    end
  end

  // --- What step has not seen: estimates ---------------------------------------
  //
  // DMA beats: each port's FIFO count, of the ports checked (chk; a
  // MATVEC's last pass leaves ports out), as a min per group and over the
  // groups, less the pops of the last Window decisions. The ports an
  // estimate covers go down with it (cover); a pop needs every port
  // checked now among them.
  // Lane room: each lane's reservations as thermometer bits, ORed per group
  // and over the groups (the most any lane has), plus the last beats of the
  // last Window decisions; a last beat needs fewer than LaneBanks.

  // The min of v[0 .. n-1], as a balanced tree.
  function automatic logic [CrW-1:0] min_tree(logic [Ports-1:0][CrW-1:0] v, int unsigned n);
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

  assign chk_d    = step_active_q;
  assign dma_ok   = est_q > 0 && !(|(chk_d & ~cover_q[SeenRegs+1]));
  assign room_ok  = room_q < RW'(LaneBanks);
  assign pop_now  = fire;
  assign last_now = fire && last_beat;

  for (genvar g = 0; g < NG; g++) begin : gen_seen
    logic [GP-1:0]              mask_q;  // chk_q's, for this group's ports
    logic [Ports-1:0][CrW-1:0]  counts;
    logic [LaneBanks-1:0]       r;
    always_comb begin
      counts = '1;
      for (int unsigned p = 0; p < GP; p++) if (mask_q[p]) counts[p] = dma_out_count[GP*g+p];
      r = '0;
      for (int unsigned l = GL * g; l < GL * (g + 1); l++) begin
        for (int unsigned k = 0; k < LaneBanks; k++) r[k] = r[k] | (32'(lane_reserved[l]) > k);
      end
    end
    always_ff @(posedge clk_i) begin
      mask_q       <= chk_q[GP*g +: GP];
      grp_min_q[g] <= min_tree(counts, GP);
      grp_res_q[g] <= r;
    end
  end

  logic [Ports-1:0][CrW-1:0] grp_mins;
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

  // --- The beat's way: descriptor, x, DMAs, lanes ------------------------------

  beat_t beat_d, beat_q;  // the decision, its central register

  always_comb begin
    beat_d         = '0;
    beat_d.valid   = fire;
    beat_d.values  = step_unit_q.values;
    beat_d.last    = last_beat;
    beat_d.mask    = step_unit_q.matvec ? (step_last_cyc_q ? step_mask_q : '1)
                   : step_unit_q.values ? values_mask : '1;
    beat_d.lanes   = res_lanes(step_unit_q);
    beat_d.dlog    = step_unit_q.dlog;
    beat_d.active  = step_active_q;
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
    if (!rst_ni) beat_q.valid <= 1'b0;
  end

  beat_t [NG-1:0]                        beat_at;  // each group's beat at PopLat
  logic [NG-1:0][Heads-1:0][E-1:0][15:0] x_at;

  for (genvar g = 0; g < NG; g++) begin : gen_group
    beat_t pipe_q[PopLat-1];  // stages 2 .. PopLat
    logic [Heads-1:0][E-1:0][15:0] x_q[PopLat-XLat];  // stages XLat + 1 .. PopLat
    always_ff @(posedge clk_i) begin
      pipe_q[0] <= beat_q;
      for (int unsigned s = 1; s < PopLat - 1; s++) pipe_q[s] <= pipe_q[s-1];
      x_q[0] <= x_data_q;
      for (int unsigned s = 1; s < PopLat - XLat; s++) x_q[s] <= x_q[s-1];
      if (!rst_ni) begin
        for (int unsigned s = 0; s < PopLat - 1; s++) pipe_q[s].valid <= 1'b0;
      end
    end
    assign beat_at[g] = pipe_q[PopLat-2];
    assign x_at[g]    = x_q[PopLat-XLat-1];
    for (genvar p = GP * g; p < GP * (g + 1); p++) begin : gen_pop
      assign dma_out_ready[p] = beat_at[g].valid && beat_at[g].active[p];
    end
  end

  // --- Lanes -------------------------------------------------------------------

  logic [L-1:0]                   lane_in_ready, lane_in_valid;
  logic [L-1:0]                   lane_out_valid, lane_out_ready;
  logic [31:0]                    lane_out[L];

  for (genvar l = 0; l < L; l++) begin : gen_lane
    localparam int unsigned P = l / LanesPerPort, I = l % LanesPerPort;
    // verilator lint_off UNUSEDSIGNAL
    beat_t              bt;  // a lane takes the fields of its beat, not the ports'
    // verilator lint_on UNUSEDSIGNAL
    logic [E-1:0][15:0] w, x;
    logic               second;
    assign bt = beat_at[l/GL];
    assign w  = dma_out_data[P][I*E*16 +: E*16];
    always_comb begin
      second = (l >> bt.dlog) == 1;  // VALUES: the pass's second head
      lane_in_valid[l] = bt.valid && l < 32'(bt.lanes);
      x = x_at[l/GL][bt.values && second];
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
  assign drain_next        = !drain_busy_q || drain_end ? to_last_tile(unit_init(q_head[2]))
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
  assign drain_load = q_valid[2] && (!drain_busy_q || drain_end);
  assign q_pop[2]   = drain_load;

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
      drain_buf_q    <= q_head[2].dst;
      drain_bf16_q   <= q_head[2].bf16;
      drain_scale_q  <= q_head[2].op == OpScores ? q_head[2].scale : Fp32One;
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
