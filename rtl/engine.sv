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
//   head a of a pass reads ports a * d / 4 ...). A unit is a pass of up to
//   Heads heads (one lane per head and output dimension), each row t
//   positions long, like a MATVEC pass of t columns.
// Every command streams its port's beats straight to the port's lanes.
// Ports is a multiple of 4 (groups of 4 ports, below). Commands need rows
// >= 1 and cols >= 1 (cols = 0 hangs step). The attention commands need d
// a power of two from 16 to MaxD (VALUES: d <= L), cap a multiple of L,
// 1 <= t <= cap (VALUES: t / E beats fit ColWidth), stride (S and P's
// rows, the STRIDE register) a multiple of E with t <= stride, group
// dividing heads.
//
// Three stages, each with its own queue of the commands taken, so a
// command starts while earlier ones finish (PLAN.md, M5). Each walks a
// command's units (unit_init, unit_next): a MATVEC's passes, an attention
// command's units as above.
// - Issue: DMA commands, in order, per unit and head to each port with
//   rows in it, a row's ceil(cols / E) beats (VALUES: ceil(t / E)). It runs
//   ahead as far as the DMAs' FIFOs allow.
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
// E * x_addr_o[r] ... + E - 1) of buffer x_buf_o as presented XRdLat
// cycles before (rtl/core.sv: the vector unit's 4 and 2 registers between the
// units); a command must not read past the
// buffer's end (the vector unit's banks wrap; commands.validate checks the
// buffers' length). Port 1
// serves a VALUES pass's second head; on other commands the lanes past d may
// read it too, and it must give port 0's data there (rtl/vec_unit.sv). Write
// port: when w_valid_o, slot s with w_mask_o[s] holds row w_row_o + s of
// buffer w_buf_o: FP32, or the BF16 bits in its low half when w_bf16_o. It
// never waits.

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
  parameter  int unsigned XRdLat       = 4,    // x_addr_o to x_data_i, cycles (vec_unit's RdLat)
  parameter  int unsigned E            = 4,    // pairs per lane per cycle
  parameter  int unsigned A            = 16,   // partial sums per row (D-027)
  parameter  int unsigned MaxD         = 128,  // largest head dimension
  parameter  int unsigned MulRegs      = 5,    // registers in the scale's multiply (fp32_fma)
  // The regions (below): the first group of region 1, of region 2.
  parameter  int unsigned Region1      = Ports / 8,
  parameter  int unsigned Region2      = Ports / 4,
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

  // The registered reset; a copy per DMA port and per lane below, each from
  // rst_ni, so all take it in the same cycle.
  logic rst_nq;
  always_ff @(posedge clk_i) rst_nq <= rst_ni;

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
  if ((4 * LanesPerPort) % Writes != 0) begin : gen_bad_drain
    $error("engine: a write group must fit in a group of 4 ports' lanes (the drain's registers)");
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
  // SCORES: a head and a tile of L positions, for each head, for each tile.
  // VALUES: a pass of Heads heads (one when 2d > L). Every stage walks the
  // same sequence with unit_init and unit_next. Each port reads its unit's
  // beats at addr + its head's KV offset + unit_off: a KV head's region per
  // channel holds SCORES' cap / L tiles of d / E beats, or VALUES' cap / E
  // beats.

  // A MATVEC pass is a SCORES unit of one head whose positions are rows:
  // t is the rows, a tile a pass of L of them.
  typedef struct packed {
    logic [ColWidth-1:0]   beats;                    // a row's: each port's, and step's cycles
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
  function automatic logic last_tile(unit_t u);  // VALUES: no tiles
    return u.values || rem(u) <= RowWidth'(L);
  endfunction
  function automatic logic [1:0] unit_heads(unit_t u);
    return u.heads2 && 32'(u.h) + 1 < 32'(u.n) ? 2'd2 : 2'd1;
  endfunction
  function automatic logic last_unit(unit_t u);
    return last_tile(u) && 32'(u.h) + 32'(unit_heads(u)) >= 32'(u.n);
  endfunction
  // The byte offset of the unit's head a's KV head.
  function automatic logic [AddrWidth-1:0] kv_offset(unit_t u, logic a);
    return a && u.r + 1'b1 == u.g ? u.kv_off + u.region : u.kv_off;
  endfunction
  // Where head a of unit u reads, for a command at addr.
  function automatic logic [AddrWidth-1:0] head_addr(logic [AddrWidth-1:0] addr, unit_t u, logic a);
    return addr + kv_offset(u, a) + u.unit_off;
  endfunction
  // A unit's results (drain): how many lanes, from lane 0, and lane 0's row.
  function automatic logic [LW-1:0] res_lanes(unit_t u);
    if (!u.values) return last_tile(u) ? LW'(rem(u)) : LW'(L);
    return LW'(unit_heads(u)) << u.dlog;
  endfunction
  function automatic logic [RowWidth-1:0] res_row(unit_t u);
    return u.values ? u.head_row : u.head_row + u.pos;
  endfunction
  // The ports a unit reads: those with lanes with a row.
  function automatic logic [Ports-1:0] unit_ports(unit_t u);
    logic [Ports-1:0] ports;
    for (int unsigned p = 0; p < Ports; p++) ports[p] = p * LanesPerPort < 32'(res_lanes(u));
    return ports;
  endfunction
  // Those of head a (issue): VALUES the d / LanesPerPort ports of head a's
  // lanes, l >> dlog == a; the others have one head.
  function automatic logic [Ports-1:0] head_ports(unit_t u, logic a);
    logic [Ports-1:0] ports;
    if (!u.values) return a ? '0 : unit_ports(u);
    for (int unsigned p = 0; p < Ports; p++) ports[p] = (p >> (32'(u.dlog) - LPW)) == 32'(a);
    return ports & unit_ports(u);
  endfunction

  // A row's beats: ceil(cols / E).
  function automatic logic [ColWidth-1:0] n_beats(logic [ColWidth-1:0] cols);
    return cols / ColWidth'(E) + ColWidth'(cols % ColWidth'(E) != '0);
  endfunction
  // A MATVEC: one head of t = rows positions; its attention fields stay 0.
  function automatic unit_t unit_init(cmd_t c);
    unit_t u;
    u = '0;
    u.beats  = n_beats(c.cols);
    if (c.op == OpMatvec) begin
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
      u.beats  = ColWidth'((c.t + RowWidth'(E - 1)) / RowWidth'(E));
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
  // ... and its rows: SCORES' S row and q, VALUES' outputs and p.
  function automatic unit_t next_head_rows(unit_t u);
    u          = next_head(u);
    u.head_row = u.head_row + (u.values ? RowWidth'(dim(u)) : u.stride);
    u.xh       = u.xh + (u.values ? u.stride : RowWidth'(dim(u))) / RowWidth'(E);
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
    u          = next_head_rows(u);
    if (u.heads2) u = next_head_rows(u);
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
  assign cmd_ready_o = &q_in_ready && rst_nq;  // none taken while the reset comes

  for (genvar i = 0; i < 3; i++) begin : gen_queue
    stream_fifo #(
      .Width($bits(cmd_t)),
      .Depth(CmdDepth)
    ) u_queue (
      .clk_i,
      .rst_ni     (rst_nq),
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

  // --- Groups and regions ------------------------------------------------------
  //
  // Ports go in groups of GP (each group's lanes, beat registers, estimates
  // and drain registers), groups in up to three regions: those before
  // Region1 in region 0, those before Region2 in region 1, the rest in
  // region 2. On F2 a region is an SLR (scripts/f2/floorplan.py), the
  // central logic (issue, step, drain) in region 1's. A signal between the
  // central logic and a region or a group goes sender register, a register
  // with one load and no logic (named `_fq`: in no pblock, free to float to
  // the SLR's edge), receiver register, then logic (M10 step 6). Every
  // region has the same registers, so the groups stay in lockstep.

  localparam int unsigned GP = 4;           // ports per group
  localparam int unsigned NG = Ports / GP;  // groups
  localparam int unsigned GL = L / NG;      // lanes per group

  if (Region1 > Region2 || Region2 > NG) begin : gen_bad_regions
    $error("engine: the regions need Region1 <= Region2 <= Ports / 4");
  end

  function automatic int unsigned region_lo(int unsigned r);
    return r == 0 ? 0 : r == 1 ? Region1 : Region2;
  endfunction
  function automatic int unsigned region_hi(int unsigned r);
    return r == 0 ? Region1 : r == 1 ? Region2 : NG;
  endfunction
  function automatic int unsigned region_of(int g);  // signed: Region1 may be 0
    return g < int'(Region1) ? 0 : g < int'(Region2) ? 1 : 2;
  endfunction

  // --- DMAs --------------------------------------------------------------------
  //
  // Issue pushes a DMA command into each port's FIFO of DmaCmds (u_cmds)
  // through its region's copies and holds a credit per place: a push takes
  // one, the port's take (pop_q) gives it back through its region's float.
  // A credit's round trip is 8 cycles (push, dma_cmd_q, the float, the
  // region's copy, the FIFO, the take, pop_q, the float, ret_q), so 4
  // places let a port take a command every 2 cycles, port_dma's rate.

  localparam int unsigned DmaCmds = 4;
  localparam int unsigned DcW     = $clog2(DmaCmds + 1);

  typedef struct packed {
    logic [AddrWidth-1:0] addr;
    logic [ColWidth-1:0]  beats;
  } dma_cmd_t;

  dma_cmd_t                        dma_cmd_q;      // issue's command (central)
  logic [Ports-1:0]                dma_push_q;     // ... pushed to these ports
  dma_cmd_t                        dma_cmd_at[3];  // the command, each region's copy
  logic [Ports-1:0]                dma_push_at;    // the pushes, each port's region's copy
  logic [Ports-1:0]                dma_pop_q;      // each port took a command a cycle ago
  logic [Ports-1:0]                dma_ret_at;     // ... through its region's float
  logic [Ports-1:0]                dma_out_valid, dma_out_ready;
  logic [Ports-1:0][DataWidth-1:0] dma_out_data;
  logic [Ports-1:0][CrW-1:0]       dma_out_count;  // beats in each port's FIFO

  for (genvar p = 0; p < Ports; p++) begin : gen_dma
    localparam int unsigned R = region_of(p / GP);
    logic     rst_nq_p;  // the port's copy of the reset
    logic     cmd_valid, cmd_ready, cmds_ready;
    dma_cmd_t dcmd;
    always_ff @(posedge clk_i) begin
      rst_nq_p     <= rst_ni;
      dma_pop_q[p] <= rst_nq_p && cmd_valid && cmd_ready;
    end
    stream_fifo #(
      .Width($bits(dma_cmd_t)),
      .Depth(DmaCmds)
    ) u_cmds (
      .clk_i,
      .rst_ni     (rst_nq_p),
      .in_valid_i (dma_push_at[p]),
      .in_ready_o (cmds_ready),
      .in_data_i  (dma_cmd_at[R]),
      .out_valid_o(cmd_valid),
      .out_ready_i(cmd_ready),
      .out_data_o (dcmd),
      // verilator lint_off PINCONNECTEMPTY
      .count_o    ()
      // verilator lint_on PINCONNECTEMPTY
    );
`ifndef SYNTHESIS
    always_ff @(posedge clk_i) begin
      if (rst_nq_p && dma_push_at[p] && !cmds_ready) $error("engine: a DMA command without a credit");
    end
`endif
    port_dma #(
      .AddrWidth (AddrWidth),
      .DataWidth (DataWidth),
      .CountWidth(ColWidth),
      .Depth     (Depth),
      .Slices    (LanesPerPort)
    ) u_dma (
      .clk_i,
      .rst_ni     (rst_nq_p),
      .cmd_valid_i(cmd_valid),
      .cmd_ready_o(cmd_ready),
      .cmd_addr_i (dcmd.addr),
      .cmd_beats_i(dcmd.beats),
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
  //
  // Issue pushes a unit's DMA commands, a head a cycle, to every port with
  // rows in the head once each of them has a credit (dma_cr_ok_q, one AND over
  // registers). The next unit's fields come from registers a cycle and two
  // after the unit changed (iss_next_*), so a unit's last head waits for
  // them (iss_age_q 2) unless it is the command's last: a unit every 3
  // cycles at most.

  logic                      iss_busy_q, iss_head_q;  // iss_head_q: the unit's head
  logic                      iss_two_q;     // unit_heads(iss_unit_q) == 2
  logic                      iss_last_q;    // last_unit(iss_unit_q)
  logic [1:0]                iss_age_q;     // cycles since iss_unit_q changed, to 2
  logic [AddrWidth-1:0]      iss_addr_q;    // the matrix's or the cache's
  logic [Ports-1:0]          iss_act_q;     // head_ports(iss_unit_q, iss_head_q)
  logic [Ports-1:0]          iss_ports1_q;  // head_ports(iss_unit_q, 1)
  unit_t                     iss_unit_q;
  unit_t                     iss_next_q;    // unit_next(iss_unit_q), a cycle late
  logic [Ports-1:0]          iss_next_ports_q[Heads];  // head_ports(iss_next_q, a), a cycle later
  logic                      iss_next_two_q, iss_next_last_q;  // ... its heads, its last flag
  // Each head's DMA address, registered: head 0's there in its unit's first
  // cycle (the next unit's two cycles after the unit changed, as its
  // ports), head 1's a cycle after the unit changed, before head 1 pushes.
  logic [AddrWidth-1:0]      iss_addr0_q, iss_addr1_q;
  logic [AddrWidth-1:0]      iss_next_addr0_q;  // head_addr(iss_addr_q, iss_next_q, 0)
  logic [Ports-1:0][DcW-1:0] dma_cr_q;  // each port's free places, as issue knows them
  logic [Ports-1:0]          dma_cr_ok_q;    // dma_cr_q != 0
  logic [Ports-1:0]          ret_q;      // a port's take, back through its region
  logic                      iss_last_head, iss_push;

  assign iss_last_head = iss_head_q || !iss_two_q;
  assign iss_push      = iss_busy_q && &(dma_cr_ok_q | ~iss_act_q)
                         && (!iss_last_head || iss_last_q || iss_age_q == 2'd2);
  assign q_pop[0]      = !iss_busy_q && q_valid[0];

  always_ff @(posedge clk_i) begin
    iss_next_q <= unit_next(iss_unit_q);
    for (int unsigned a = 0; a < Heads; a++) iss_next_ports_q[a] <= head_ports(iss_next_q, 1'(a));
    iss_next_two_q   <= unit_heads(iss_next_q) == 2'd2;
    iss_next_last_q  <= last_unit(iss_next_q);
    iss_next_addr0_q <= head_addr(iss_addr_q, iss_next_q, 1'b0);
    iss_addr1_q      <= head_addr(iss_addr_q, iss_unit_q, 1'b1);
    dma_cmd_q        <= '{addr: iss_head_q ? iss_addr1_q : iss_addr0_q, beats: iss_unit_q.beats};
  end

  always_ff @(posedge clk_i) begin
    if (!rst_nq) begin
      iss_busy_q <= 1'b0;
      dma_push_q <= '0;
      ret_q      <= '0;
      dma_cr_q  <= {Ports{DcW'(DmaCmds)}};
      dma_cr_ok_q    <= '1;
    end else begin
      dma_push_q <= iss_push ? iss_act_q : '0;
      ret_q      <= dma_ret_at;
      for (int unsigned p = 0; p < Ports; p++) begin
        dma_cr_q[p] <= dma_cr_q[p] - DcW'(iss_push && iss_act_q[p]) + DcW'(ret_q[p]);
        dma_cr_ok_q[p]   <= dma_cr_q[p] - DcW'(iss_push && iss_act_q[p]) + DcW'(ret_q[p]) != '0;
      end
      if (q_pop[0]) begin
        iss_busy_q   <= 1'b1;
        iss_addr_q   <= q_head[0].addr;
        iss_head_q   <= 1'b0;
        // The unit and its fields from the command's own copy, in parallel:
        // a mux of the two units before head_ports would lengthen the path.
        iss_unit_q   <= unit_init(q_head[0]);
        iss_act_q    <= head_ports(unit_init(q_head[0]), 1'b0);
        iss_ports1_q <= head_ports(unit_init(q_head[0]), 1'b1);
        iss_two_q    <= unit_heads(unit_init(q_head[0])) == 2'd2;
        iss_last_q   <= last_unit(unit_init(q_head[0]));
        iss_addr0_q  <= q_head[0].addr;  // a first unit's head 0: no offsets
      end else if (iss_push && !iss_last_head) begin
        iss_head_q <= 1'b1;
        iss_act_q  <= iss_ports1_q;
      end else if (iss_push) begin
        iss_head_q   <= 1'b0;
        iss_unit_q   <= iss_next_q;
        iss_act_q    <= iss_next_ports_q[0];
        iss_ports1_q <= iss_next_ports_q[1];
        iss_two_q    <= iss_next_two_q;
        iss_last_q   <= iss_next_last_q;
        iss_addr0_q  <= iss_next_addr0_q;
        if (iss_last_q) iss_busy_q <= 1'b0;
      end
      if (q_pop[0] || (iss_push && iss_last_head)) iss_age_q <= '0;
      else if (iss_age_q != 2'd2) iss_age_q <= iss_age_q + 1'b1;
    end
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk_i) begin
    if (rst_nq && iss_push) begin
      if ((iss_head_q ? iss_addr1_q : iss_addr0_q) != head_addr(iss_addr_q, iss_unit_q, iss_head_q)) begin
        $error("engine: a DMA command's address not ready");
      end
      if (iss_act_q != head_ports(iss_unit_q, iss_head_q) || iss_last_q != last_unit(iss_unit_q)
          || iss_two_q != (unit_heads(iss_unit_q) == 2'd2)) begin
        $error("engine: issue's unit fields not ready");
      end
      if (iss_last_head && !iss_last_q) begin
        if (iss_next_q != unit_next(iss_unit_q)) $error("engine: issue's next unit not ready");
        for (int unsigned a = 0; a < Heads; a++) begin
          if (iss_next_ports_q[a] != head_ports(unit_next(iss_unit_q), 1'(a))) begin
            $error("engine: issue's next unit's ports not ready");
          end
        end
        if (iss_next_addr0_q != head_addr(iss_addr_q, unit_next(iss_unit_q), 1'b0)) begin
          $error("engine: issue's next address not ready");
        end
      end
    end
  end
`endif

  // --- Beats: decided ahead, delivered through registers (D-051) ---------------
  //
  // Step decides its beats from registered state only: the DMAs' FIFO
  // counts and the lanes' reservations reach the decision through
  // registers, so each decision assumes the worst of the beats it has not
  // seen yet (Window of them). A beat's descriptor then goes through a
  // central register, its region's float and copy, and PopLat - 3 more in
  // copies per group of 4 ports (16 lanes): PopLat cycles after its
  // decision the beat pops the DMAs and reaches the lanes, which never
  // refuse it (a lane's ready drops only on a last beat, and a last beat is
  // decided only with room for it). x goes the same way, its last register
  // a copy per port holding the head its lanes take.

  localparam int unsigned XLat     = 3 + XRdLat;  // x_data at stage 2 + XRdLat, its central register
  localparam int unsigned PopLat   = XLat + 4;    // the region's float and copy, the group's, the port's
  // A group's register, its region's, the float, the central copy, the
  // central one.
  localparam int unsigned SeenRegs = 5;
  localparam int unsigned MaskRegs = 2;  // chk_q to a group's mask: its region's float and copy
  // Decisions not yet in what step sees: PopLat to the pop or the lane, one
  // into the FIFO count or the reservation, SeenRegs, and the estimate's own
  // register.
  localparam int unsigned Window   = PopLat + SeenRegs + 1;
  // dot_lane's level-1 buffers: 3 so that short rows (SCORES' d / E beats)
  // do not wait on the reservations of the rows in Window (M10 step 6).
  localparam int unsigned LaneBanks = 3;
  localparam int unsigned LaneMulRegs = 2;  // registers in a lane's product (fp_product)
  localparam int unsigned RW       = $clog2(LaneBanks + Window + 1);

  typedef struct packed {
    logic                 valid;    // a beat to the lanes
    logic                 last;
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
  // this one), muxed at the end, as issue does. The next unit's ports come
  // from registers, two cycles after the unit changed: a unit's last beat
  // waits for them (units of fewer than 3 beats only).

  logic                step_busy_q;
  logic [E-1:0]        step_mask_q;    // a row's last beat's elements
  logic [BufWidth-1:0] step_buf_q;
  logic [Ports-1:0]    step_active_q;  // unit_ports(step_unit_q)
  unit_t               step_unit_q;    // the unit stepped
  logic [RowWidth-1:0] step_cyc_q;     // the unit's cycle: the row's beat
  logic                step_last_cyc_q;   // step_cyc_q + 1 == step_unit_q.beats
  logic                step_last_unit_q;  // last_unit(step_unit_q)
  unit_t               step_next_q;       // unit_next(step_unit_q), a cycle late
  logic [Ports-1:0]    step_next_ports_q; // unit_ports(step_next_q), a cycle late
  logic [1:0]          step_age_q;        // cycles since step_unit_q changed, to 2
  logic                next_ready;
  logic                fire, step_end, step_go, last_beat, room_ok;
  // The selects of step_unit_q's ~200 bits (core build 6: -0.26 ns from
  // step_last_cyc_q and step_age_q through them): max_fanout 32, a hint
  // (platforms/f2/timing/audit.tcl reports it).
  (* max_fanout = 32 *) logic step_load, unit_end;
  logic [CW-1:0]       credits_q;      // go_i pulses not yet spent
  unit_t               unit_init_c, unit_next_c, unit_n;
  logic [RowWidth-1:0] cyc_n;
  logic [ColWidth-1:0] row_len;        // the command's row length: cols, VALUES t

  assign last_beat     = step_last_cyc_q;  // every unit ends its lanes' rows
  assign next_ready    = step_last_unit_q || step_age_q == 2'd2;
  assign fire          = step_busy_q && dma_ok && (!last_beat || (room_ok && next_ready));
  assign unit_end      = fire && step_last_cyc_q;
  assign step_end      = unit_end && step_last_unit_q;
  assign step_go       = !q_head[1].wait_go || credits_q != '0;
  assign step_load     = q_valid[1] && (!step_busy_q || step_end) && step_go;
  assign q_pop[1]      = step_load;
  assign unit_init_c   = unit_init(q_head[1]);
  assign unit_next_c   = unit_next(step_unit_q);

  assign row_len       = q_head[1].op == OpValues ? ColWidth'(q_head[1].t) : q_head[1].cols;

  always_comb begin
    unit_n = step_load ? unit_init_c : unit_end ? unit_next_c : step_unit_q;
    cyc_n  = step_load || unit_end ? '0 : fire ? step_cyc_q + 1'b1 : step_cyc_q;
  end

  always_ff @(posedge clk_i) begin
    if (!rst_nq) begin
      step_busy_q <= 1'b0;
      credits_q   <= '0;
    end else begin
      step_unit_q <= unit_n;
      step_cyc_q  <= cyc_n;
      credits_q   <= credits_q + CW'(go_i) - CW'(step_load && q_head[1].wait_go);
      if (step_load) begin
        step_active_q    <= unit_ports(unit_init_c);
        step_last_cyc_q  <= unit_init_c.beats == ColWidth'(1);
        step_last_unit_q <= last_unit(unit_init_c);
      end else if (unit_end) begin
        step_active_q    <= step_next_ports_q;
        step_last_cyc_q  <= unit_next_c.beats == ColWidth'(1);
        step_last_unit_q <= last_unit(unit_next_c);
      end else if (fire) begin
        step_last_cyc_q  <= step_cyc_q + RowWidth'(2) == RowWidth'(step_unit_q.beats);
      end
      if (step_load) begin
        step_busy_q   <= 1'b1;
        step_buf_q    <= q_head[1].a;
        for (int unsigned e = 0; e < E; e++) begin
          step_mask_q[e] <= row_len % ColWidth'(E) == '0 || ColWidth'(e) < row_len % ColWidth'(E);
        end
      end else if (step_end) begin
        step_busy_q <= 1'b0;
      end
    end
  end

  always_ff @(posedge clk_i) begin
    step_next_q       <= unit_next_c;
    step_next_ports_q <= unit_ports(step_next_q);
    if (!rst_nq || step_load || unit_end) step_age_q <= '0;
    else if (step_age_q != 2'd2) step_age_q <= step_age_q + 1'b1;
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk_i) begin
    if (rst_nq && unit_end && !step_last_unit_q && step_next_ports_q != unit_ports(unit_next_c)) begin
      $error("engine: step's next unit's ports not ready at the end of its unit");
    end
  end
`endif

  // --- What step has not seen: estimates ---------------------------------------
  //
  // DMA beats: each port's FIFO count, of the ports checked (chk; a
  // MATVEC's last pass leaves ports out), as a min per group, per region
  // and over the regions, less the pops of the last Window decisions. The
  // ports an estimate covers go down with it (cover); a pop needs every
  // port checked now among them.
  // Lane room: each lane's reservations as thermometer bits, ORed per
  // group, per region and over the regions (the most any lane has), plus
  // the last beats of the last Window decisions; a last beat needs fewer
  // than LaneBanks.

  // The min of v[0 .. n-1], as a balanced tree.
  function automatic logic [CrW-1:0] min_tree(logic [Ports-1:0][CrW-1:0] v, int unsigned n);
    for (int unsigned w = 1; w < n; w *= 2) begin
      for (int unsigned i = 0; i + w < n; i += 2 * w) if (v[i+w] < v[i]) v[i] = v[i+w];
    end
    return v[0];
  endfunction

  logic [L-1:0][$clog2(LaneBanks):0] lane_reserved;
  logic [Ports-1:0]             chk_q;
  logic [Ports-1:0]             chk_at;  // chk_q through each port's region
  // chk_q's, as each stage's min has them: the mask's MaskRegs + 1, then SeenRegs + 1
  logic [Ports-1:0]             cover_q[MaskRegs+SeenRegs+2];
  logic                         last_now;
  logic [NG-1:0][CrW-1:0]       grp_min_q;
  logic [NG-1:0][LaneBanks-1:0] grp_res_q;
  logic [CrW-1:0]               seen_min[3], seen_min_q[3];  // each region's, its central copy
  logic [LaneBanks-1:0]         seen_res[3], seen_res_q[3];
  logic [CrW-1:0]               min_q;
  logic [LaneBanks-1:0]         res_q;
  logic [Window-2:0]            pop_win_q, last_win_q;  // the decisions before this one
  logic signed [CrW+1:0]        est_q;   // beats certainly in every port covered
  logic [RW-1:0]                room_q;  // reservations a lane may have, at most

  assign dma_ok   = est_q > 0 && !(|(step_active_q & ~cover_q[MaskRegs+SeenRegs+1]));
  assign room_ok  = room_q < RW'(LaneBanks);
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
      mask_q       <= chk_at[GP*g +: GP];
      grp_min_q[g] <= min_tree(counts, GP);
      grp_res_q[g] <= r;
    end
  end

  logic [Ports-1:0][CrW-1:0] reg_mins;
  logic [LaneBanks-1:0]      all_res;
  always_comb begin
    reg_mins = '1;
    all_res  = '0;
    for (int unsigned r = 0; r < 3; r++) begin
      reg_mins[r] = seen_min_q[r];
      all_res     = all_res | seen_res_q[r];
    end
  end

  always_ff @(posedge clk_i) begin
    seen_min_q <= seen_min;
    seen_res_q <= seen_res;
    min_q      <= min_tree(reg_mins, 3);
    res_q      <= all_res;
    chk_q      <= step_active_q;
    cover_q[0] <= chk_q;
    for (int unsigned i = 1; i < MaskRegs + SeenRegs + 2; i++) cover_q[i] <= cover_q[i-1];
    if (!rst_nq) begin
      pop_win_q  <= '0;
      last_win_q <= '0;
      est_q      <= '0;
      room_q     <= '0;
    end else begin
      est_q      <= $signed({2'b00, min_q}) - $signed((CrW + 2)'($countones(pop_win_q)))
                    - $signed((CrW + 2)'(fire));
      room_q     <= RW'($countones(res_q)) + RW'($countones(last_win_q)) + RW'(last_now);
      pop_win_q  <= {pop_win_q[Window-3:0], fire};
      last_win_q <= {last_win_q[Window-3:0], last_now};
    end
  end

  // --- The beat's way: descriptor, x, DMAs, lanes ------------------------------

  beat_t beat_d, beat_q;  // the decision, its central register

  always_comb begin
    beat_d         = '0;
    beat_d.valid   = fire;
    beat_d.last    = last_beat;
    beat_d.mask    = step_last_cyc_q ? step_mask_q : '1;
    beat_d.lanes   = res_lanes(step_unit_q);
    beat_d.dlog    = step_unit_q.dlog;
    beat_d.active  = step_active_q;
  end

  // x: the address from the decision, presented at stage 2, so x_data_i
  // comes at 2 + XRdLat; a central register (XLat), the region's float and
  // copy, the group's and the port's take it to PopLat.
  logic [Heads-1:0][RowWidth-1:0] x_addr_d, x_addr_q;
  logic [BufWidth-1:0]            x_buf_q;
  logic [Heads-1:0][E-1:0][15:0]  x_data_q;

  // Port 1 reads port 0's group but for VALUES (its second head), so a
  // lane may take port 1 whatever the command (MATVEC: xh = 0).
  always_comb begin
    x_addr_d[0] = step_unit_q.xh + step_cyc_q;
    x_addr_d[1] = x_addr_d[0] + (step_unit_q.values ? step_unit_q.stride / RowWidth'(E) : '0);
  end

  always_ff @(posedge clk_i) begin
    beat_q   <= beat_d;
    x_addr_q <= x_addr_d;
    x_buf_q  <= step_buf_q;
    x_addr_o <= x_addr_q;
    x_buf_o  <= x_buf_q;
    x_data_q <= x_data_i;
    if (!rst_nq) beat_q.valid <= 1'b0;
  end

  // --- Regions: the copies between the central logic and the groups ------------
  //
  // Out: the beat (stages 2 and 3), x (XLat + 1 and + 2), chk_q, the DMA
  // command and its pushes, each through a float, then the region's copy.
  // Back: the ports' takes through a float (pop_q is the sender), and the
  // estimates' min and OR over the region's groups, then a float.
  beat_t                         beat_r[3];  // each region's beat, at stage 3
  logic [Heads-1:0][E-1:0][15:0] x_r[3];     // ... its x, at XLat + 2

  for (genvar r = 0; r < 3; r++) begin : gen_region
    localparam int unsigned Lo = region_lo(r), Hi = region_hi(r);
    localparam int unsigned P0 = GP * Lo, NP = GP * (Hi - Lo);  // its ports
    if (Lo < Hi) begin : gen_on
      logic                          rst_nq_r;  // the region's copy of the reset
      beat_t                         beat_fq, beat_rq;
      logic [Heads-1:0][E-1:0][15:0] x_fq, x_rq;
      logic [NP-1:0]                 chk_fq, chk_rq, push_fq, push_rq, pop_fq;
      dma_cmd_t                      cmd_fq, cmd_rq;
      logic [Ports-1:0][CrW-1:0]     mins;
      logic [LaneBanks-1:0]          res, rres_q, rres_fq;
      logic [CrW-1:0]                rmin_q, rmin_fq;
      always_comb begin
        mins = '1;
        res  = '0;
        for (int unsigned g = Lo; g < Hi; g++) begin
          mins[g-Lo] = grp_min_q[g];
          res        = res | grp_res_q[g];
        end
      end
      always_ff @(posedge clk_i) begin
        rst_nq_r <= rst_ni;
        beat_fq  <= beat_q;
        beat_rq  <= beat_fq;
        x_fq     <= x_data_q;
        x_rq     <= x_fq;
        chk_fq   <= chk_q[P0 +: NP];
        chk_rq   <= chk_fq;
        cmd_fq   <= dma_cmd_q;
        cmd_rq   <= cmd_fq;
        push_fq  <= dma_push_q[P0 +: NP];
        push_rq  <= push_fq;
        pop_fq   <= dma_pop_q[P0 +: NP];
        rmin_q   <= min_tree(mins, Hi - Lo);
        rres_q    <= res;
        rmin_fq  <= rmin_q;
        rres_fq   <= rres_q;
        if (!rst_nq_r) begin
          beat_fq.valid <= 1'b0;
          beat_rq.valid <= 1'b0;
          push_fq       <= '0;
          push_rq       <= '0;
          pop_fq        <= '0;
        end
      end
      assign beat_r[r]               = beat_rq;
      assign x_r[r]                  = x_rq;
      assign chk_at[P0 +: NP]        = chk_rq;
      assign dma_cmd_at[r]           = cmd_rq;
      assign dma_push_at[P0 +: NP]   = push_rq;
      assign dma_ret_at[P0 +: NP]    = pop_fq;
      assign seen_min[r]             = rmin_fq;
      assign seen_res[r]             = rres_fq;
    end else begin : gen_off
      assign seen_min[r] = '1;
      assign seen_res[r] = '0;
    end
  end

  beat_t [NG-1:0]              beat_at;  // each group's beat at PopLat
  logic [L-1:0]                lane_row_q;  // l < beat_at[l / GL].lanes
  logic [Ports-1:0][E-1:0][15:0] x_at;    // each port's x at PopLat

  // A port's lanes take one head: VALUES' lane l takes the pass's second
  // head when l >> dlog == 1, the same for a port's lanes since d >= 16
  // (the attention commands' precondition, above) >= LanesPerPort
  // (gen_bad_lpp); the other commands read the same x on both ports.
  for (genvar g = 0; g < NG; g++) begin : gen_group
    localparam int unsigned R = region_of(g);
    logic  rst_nq_g;  // the group's copy of the reset
    beat_t pipe_q[PopLat-3];  // stages 4 .. PopLat
    logic [Heads-1:0][E-1:0][15:0] x_q[PopLat-XLat-3];  // stages XLat + 3 .. PopLat - 1
    always_ff @(posedge clk_i) begin
      rst_nq_g  <= rst_ni;
      pipe_q[0] <= beat_r[R];
      for (int unsigned s = 1; s < PopLat - 3; s++) pipe_q[s] <= pipe_q[s-1];
      x_q[0] <= x_r[R];
      for (int unsigned s = 1; s < PopLat - XLat - 3; s++) x_q[s] <= x_q[s-1];
      if (!rst_nq_g) begin
        for (int unsigned s = 0; s < PopLat - 3; s++) pipe_q[s].valid <= 1'b0;
      end
    end
    assign beat_at[g] = pipe_q[PopLat-4];
    // Which of the group's lanes have a row, from the beat a stage before.
    always_ff @(posedge clk_i) begin
      for (int unsigned i = 0; i < GL; i++) lane_row_q[g*GL+i] <= g * GL + i < 32'(pipe_q[PopLat-5].lanes);
    end
    for (genvar p = GP * g; p < GP * (g + 1); p++) begin : gen_port
      logic second_q;  // at PopLat - 1, from the beat at PopLat - 2 (pipe_q[s] is stage s + 4)
      always_ff @(posedge clk_i) begin
        second_q <= (p * LanesPerPort) >> pipe_q[(PopLat-2)-4].dlog == 1;
        x_at[p]  <= x_q[PopLat-XLat-4][second_q];
      end
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
    logic [E-1:0][15:0] w;
    assign bt = beat_at[l/GL];
    assign w  = dma_out_data[P][I*E*16 +: E*16];
    assign lane_in_valid[l] = bt.valid && lane_row_q[l];
    logic rst_nq_l;  // the lane's copy of the reset
    always_ff @(posedge clk_i) rst_nq_l <= rst_ni;
    dot_lane #(
      .E      (E),
      .A      (A),
      .MulRegs(LaneMulRegs),
      .Banks  (LaneBanks)
    ) u_lane (
      .clk_i,
      .rst_ni     (rst_nq_l),
      .in_valid_i (lane_in_valid[l]),
      .in_ready_o (lane_in_ready[l]),
      .w_i        (w),
      .x_i        (x_at[P]),
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
    if (rst_nq) begin
      if (|(dma_out_ready & ~dma_out_valid)) $error("engine: a pop from an empty DMA FIFO");
      if (|(lane_in_valid & ~lane_in_ready)) $error("engine: a lane refused a beat");
    end
  end
`endif

  // --- Drain: results to the write port, Writes lanes per cycle ----------------
  //
  // No one-cycle loop across the lanes (D-051): the lanes' out_valid come
  // to the drain through a register in their group, a float and out_valid_q,
  // then each write group's "every lane with a result has one" is
  // registered for the next cycle's group (the same one, or the next after
  // a go: ok_same_q, ok_inc_q) and for the unit after the one draining and
  // the queue's next command's first unit (a unit starts at group 0). A go
  // reaches its lanes' pops through a central register per group, a float
  // and the group's copy (ReadyLat), so the group waits Recent cycles
  // before its next go (its registered state is the pop's by then); its
  // results go to the multiply through registers.
  //
  // Nothing in a go is computed from a unit in the same cycle: the unit
  // after the one draining (aft) and the queue head's first unit (ini) are
  // registered, then their lane counts and last flags, two cycles after
  // their source changes. A pass ends, or a command loads, only once its
  // successor's are there; a unit takes at least four cycles anyway.

  // A go's results: each group registers its lanes' outputs (lo_q), selects
  // the go's write group into out_q in the pop's cycle, then a float, a
  // copy on the drain's side (grp_out_q), the central register (slot_q) and
  // fp32_fma. The lanes pop ReadyLat cycles after the go, so their results
  // are still in lo_q for out_q.
  localparam int unsigned WG       = GL / Writes;   // write groups per group
  localparam int unsigned WGW      = WG > 1 ? $clog2(WG) : 1;
  localparam int unsigned ReadyLat = 3;  // a go to the pop: rdy_go_q, the float, the group's copy
  // A go to slot_q's load: ReadyLat, out_q, its float, grp_out_q.
  localparam int unsigned SlotLat  = ReadyLat + 3;
  localparam int unsigned WrLat    = DrainLat + SlotLat + 1;  // ... slot_q, fp32_fma
  // Cycles a write group waits after its go: ReadyLat, the lanes'
  // out_valid, the group's register, its float, out_valid_q.
  localparam int unsigned Recent   = ReadyLat + 4;

  typedef struct packed {
    unit_t         unit;
    logic [LW-1:0] lanes;  // res_lanes(unit)
    logic          last;   // last_unit(unit)
  } drain_unit_t;

  logic                     drain_busy_q, drain_bf16_q;
  logic [BufWidth-1:0]      drain_buf_q;
  logic [31:0]              drain_scale_q;
  logic [GW-1:0]            drain_grp_q;  // the write group
  drain_unit_t              cur_q;        // the unit whose results drain
  drain_unit_t              aft_q, ini_q; // unit_next(cur_q.unit); unit_init(q_head[2])
  logic                     end_q;        // drain_grp_q is the pass's last group
  logic [1:0]               aft_age_q, ini_age_q;  // cycles since their source changed, to 2
  logic [RowWidth-1:0]      pass_row;
  logic [Writes-1:0]        has, aft_has, ini_has;
  logic [Groups-1:0]        ok_d;                // per write group: its results are there
  logic                     ok_same_q, ok_inc_q; // drain_grp_q's and the next one's, for the next cycle
  logic                     ok_aft_q, ok_ini_q;  // ... group 0's, for the successors
  logic                     went_q;              // a go last cycle
  logic                     unit_new_q;          // cur_q changed last cycle
  logic                     from_ini_q;          // ... to a command's first unit
  logic [L-1:0]             out_valid_at;        // lane_out_valid through its group's registers
  logic [L-1:0]             out_valid_q;         // ... on the drain's side
  logic [Recent-1:0]        gone_q;              // a go in each of the last Recent cycles
  logic [GW-1:0]            gone_grp_q[Recent];
  logic                     rec_same, rec_inc, rec_0;  // a go in Recent - 1 cycles, to each
  logic                     go, pass_end, drain_end, drain_load, unit_load, ok;
  logic                     aft_ready, ini_ready;
  logic [GW-1:0]            grp_inc;
  logic [NG-1:0]            rdy_go_q;   // a go to group g's lanes
  logic [WGW-1:0]           rdy_wg_q;   // ... to its write group
  logic [Writes-1:0]        rdy_has_q;  // ... lanes with a result

  assign pass_row = res_row(cur_q.unit);
  assign grp_inc  = drain_grp_q + 1'b1;
  for (genvar s = 0; s < Writes; s++) begin : gen_slot
    assign has[s]     = RowWidth'(drain_grp_q) * RowWidth'(Writes) + RowWidth'(s) < RowWidth'(cur_q.lanes);
    assign aft_has[s] = !out_valid_q[s] && s < 32'(aft_q.lanes);  // a result missing
    assign ini_has[s] = !out_valid_q[s] && s < 32'(ini_q.lanes);
  end
  for (genvar k = 0; k < Groups; k++) begin : gen_wgroup
    logic [Writes-1:0] cur;  // per lane: has its result, or has none to give
    for (genvar s = 0; s < Writes; s++) begin : gen_lane_ok
      assign cur[s] = out_valid_q[k*Writes+s] || k * Writes + s >= 32'(cur_q.lanes);
    end
    assign ok_d[k] = &cur;
  end
  // A go to the group in the last Recent - 1 cycles: with this cycle's
  // go, the next cycle's "recent".
  always_comb begin
    rec_same = 1'b0;
    rec_inc  = 1'b0;
    rec_0    = 1'b0;
    for (int unsigned i = 0; i + 1 < Recent; i++) begin
      rec_same = rec_same || (gone_q[i] && gone_grp_q[i] == drain_grp_q);
      rec_inc  = rec_inc || (gone_q[i] && gone_grp_q[i] == grp_inc);
      rec_0    = rec_0 || (gone_q[i] && gone_grp_q[i] == '0);
    end
  end
  always_ff @(posedge clk_i) begin
    out_valid_q <= out_valid_at;
    // No go this cycle: the same group; a go: the next (a pass's end is unit_new_q).
    ok_same_q   <= ok_d[drain_grp_q] && !rec_same;
    ok_inc_q    <= ok_d[grp_inc] && !rec_inc;
    ok_aft_q    <= !(|aft_has) && !rec_0 && !(go && drain_grp_q == '0);
    ok_ini_q    <= !(|ini_has) && !rec_0 && !(go && drain_grp_q == '0);
  end
  assign ok         = unit_new_q ? (from_ini_q ? ok_ini_q : ok_aft_q) : went_q ? ok_inc_q : ok_same_q;
  assign aft_ready  = aft_age_q == 2'd2;
  assign ini_ready  = ini_age_q == 2'd2;
  assign go         = drain_busy_q && ok && (!end_q || cur_q.last || aft_ready);
  assign pass_end   = go && end_q;
  assign drain_end  = pass_end && cur_q.last;
  assign drain_load = q_valid[2] && ini_ready && (!drain_busy_q || drain_end);
  assign unit_load  = drain_load || pass_end;
  assign q_pop[2]   = drain_load;

  // The successors: the unit, then its count and flag from the register.
  always_ff @(posedge clk_i) begin
    aft_q.unit  <= unit_next(cur_q.unit);
    aft_q.lanes <= res_lanes(aft_q.unit);
    aft_q.last  <= last_unit(aft_q.unit);
    ini_q.unit  <= unit_init(q_head[2]);
    ini_q.lanes <= res_lanes(ini_q.unit);
    ini_q.last  <= last_unit(ini_q.unit);
    if (!rst_nq || unit_load) aft_age_q <= '0;
    else if (!aft_ready) aft_age_q <= aft_age_q + 1'b1;
    if (!rst_nq || q_pop[2] || !q_valid[2]) ini_age_q <= '0;
    else if (!ini_ready) ini_age_q <= ini_age_q + 1'b1;
  end

  always_ff @(posedge clk_i) begin
    for (int unsigned g = 0; g < NG; g++) rdy_go_q[g] <= go && 32'(drain_grp_q) / WG == g;
    rdy_wg_q      <= WGW'(32'(drain_grp_q) % WG);
    rdy_has_q     <= has;
    gone_grp_q[0] <= drain_grp_q;
    for (int unsigned i = 1; i < Recent; i++) gone_grp_q[i] <= gone_grp_q[i-1];
    from_ini_q    <= drain_load;
    if (!rst_nq) begin
      rdy_go_q   <= '0;
      gone_q     <= '0;
      went_q     <= 1'b0;
      unit_new_q <= 1'b0;
    end else begin
      gone_q     <= {gone_q[Recent-2:0], go};
      went_q     <= go;
      unit_new_q <= unit_load;
    end
  end

  always_ff @(posedge clk_i) begin
    if (!rst_nq) begin
      drain_busy_q <= 1'b0;
      drain_grp_q  <= '0;
    end else if (drain_load) begin
      drain_busy_q   <= 1'b1;
      drain_grp_q    <= '0;
      cur_q          <= ini_q;
      end_q          <= 32'(Writes) >= 32'(ini_q.lanes);
      drain_buf_q    <= q_head[2].dst;
      drain_bf16_q   <= q_head[2].bf16;
      drain_scale_q  <= q_head[2].op == OpScores ? q_head[2].scale : Fp32One;
    end else if (drain_end) begin
      drain_busy_q <= 1'b0;
    end else if (pass_end) begin
      drain_grp_q <= '0;
      cur_q       <= aft_q;
      end_q       <= 32'(Writes) >= 32'(aft_q.lanes);
    end else if (go) begin
      drain_grp_q <= grp_inc;
      end_q       <= (32'(drain_grp_q) + 2) * Writes >= 32'(cur_q.lanes);
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

  logic [NG-1:0][Writes-1:0][31:0] grp_out, grp_out_q;  // each group's float, its copy
  logic [SlotLat-1:0]              go_q;
  logic [31:0]                     scale_q[SlotLat];
  logic [GW-1:0]                   grp_q[SlotLat];
  for (genvar g = 0; g < NG; g++) begin : gen_drain
    logic                    rst_nq_g;  // the group's copy of the reset
    logic                    go_fq, go_rq;
    logic [WGW-1:0]          wg_fq, wg_rq;
    logic [Writes-1:0]       has_fq, has_rq;
    logic [GL-1:0]           vld_q, vld_fq;
    logic [GL-1:0][31:0]     lo_q;
    logic [Writes-1:0][31:0] out_q, out_fq;
    always_ff @(posedge clk_i) begin
      rst_nq_g <= rst_ni;
      go_fq    <= rst_nq_g && rdy_go_q[g];
      go_rq    <= rst_nq_g && go_fq;
      wg_fq    <= rdy_wg_q;
      wg_rq    <= wg_fq;
      has_fq   <= rdy_has_q;
      has_rq   <= has_fq;
      vld_q    <= lane_out_valid[g*GL +: GL];
      vld_fq   <= vld_q;
      for (int unsigned i = 0; i < GL; i++) lo_q[i] <= lane_out[g*GL+i];
      out_q    <= lo_q[32'(wg_rq)*Writes +: Writes];
      out_fq   <= out_q;
    end
    for (genvar i = 0; i < GL; i++) begin : gen_ready
      assign lane_out_ready[g*GL+i] = go_rq && 32'(wg_rq) == i / Writes && has_rq[i%Writes];
    end
    assign out_valid_at[g*GL +: GL] = vld_fq;
    assign grp_out[g]               = out_fq;
  end

  always_ff @(posedge clk_i) begin
    grp_out_q   <= grp_out;
    go_q        <= {go_q[SlotLat-2:0], rst_nq && go};
    scale_q[0]  <= drain_scale_q;
    grp_q[0]    <= drain_grp_q;
    for (int unsigned i = 1; i < SlotLat; i++) begin
      scale_q[i] <= scale_q[i-1];
      grp_q[i]   <= grp_q[i-1];
    end
    mul_valid_q <= rst_nq && go_q[SlotLat-1];
    mul_scale_q <= scale_q[SlotLat-1];
    slot_q      <= grp_out_q[32'(grp_q[SlotLat-1])/WG];
    if (!rst_nq) go_q <= '0;
  end

  for (genvar s = 0; s < Writes; s++) begin : gen_scale
    fp32_fma #(
      .MulRegs(MulRegs)
    ) u_mul (
      .clk_i,
      .rst_ni (rst_nq),
      .valid_i(mul_valid_q),
      .op_i   (Fp32Mul),
      .a_i    (slot_q[s]),
      .b_i    (mul_scale_q),
      .c_i    ('0),
      .late_i  (1'b0),
      .c_late_i('0),
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
    if (!rst_nq) begin
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
