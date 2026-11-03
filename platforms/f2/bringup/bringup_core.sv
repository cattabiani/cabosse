// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The bring-up design's core (platforms/f2/README.md): the registers on an
// AXI-Lite port and Ports test engines (bringup_port.sv), one per HBM port,
// all on one clock. Vendor-free; the CL top (cabosse_bringup.sv) adds the
// shell, the clock crossings and the HBM. HBM port 0 is shared with the
// host's port (PCIS) through bringup_host.sv, which HOST_PORT drives.
//
// AXI-Lite: one access at a time; a write takes its address and data in the
// same cycle and ignores the strobes (whole words). A read answers 4 cycles
// after its address: the ports' counters are a register stage in each port,
// then a stage per group of 8 ports and one here.
//
// A start is refused (STATUS.refused) while busy, and when the settings
// break README.md's rules, which would hang a port or end a run with bursts
// in flight: a burst not 1, 2, 4, 8 or 16 beats, a limit of 0, BEATS not a
// multiple of both bursts or past CountWidth, BASE or STRIDE not 4 KB-aligned.
// Port 0 does not start while it serves the host, and HOST_PORT does not
// change during a run.

module bringup_core #(
  parameter  int unsigned Ports      = 32,
  parameter  logic [31:0] Build      = 32'h0,  // the commit's first 8 hex digits
  localparam int unsigned AddrWidth  = 34,
  localparam int unsigned DataWidth  = 256,
  localparam int unsigned BeatBytes  = DataWidth / 8,
  localparam int unsigned CountWidth = 24,
  localparam int unsigned LenW       = 5,  // a burst's beats, 1 .. 16 (bringup_port's)
  localparam int unsigned LimitW     = 8,  // bursts unanswered (bringup_port's)
  localparam int unsigned GroupPorts = 8,  // ports a group register drives (Ports, below)
  localparam int unsigned Groups     = (Ports + GroupPorts - 1) / GroupPorts,
  localparam int unsigned SlotW      = $clog2(GroupPorts),           // a port's place in its group
  localparam int unsigned GroupW     = Groups > 1 ? $clog2(Groups) : 1
) (
  input  logic                                 clk_i,
  input  logic                                 rst_ni,
  input  logic                                 hbm_ready_i,  // both stacks initialized, synchronized to clk_i
  output logic                                 host_sel_o,   // HBM port 0 serves the host now
  // AXI-Lite (registers)
  input  logic                                 s_aw_valid_i,
  output logic                                 s_aw_ready_o,
  input  logic [11:0]                          s_aw_addr_i,
  input  logic                                 s_w_valid_i,
  output logic                                 s_w_ready_o,
  input  logic [31:0]                          s_w_data_i,
  output logic                                 s_b_valid_o,
  input  logic                                 s_b_ready_i,
  output logic [1:0]                           s_b_resp_o,
  input  logic                                 s_ar_valid_i,
  output logic                                 s_ar_ready_o,
  input  logic [11:0]                          s_ar_addr_i,
  output logic                                 s_r_valid_o,
  input  logic                                 s_r_ready_i,
  output logic [31:0]                          s_r_data_o,
  output logic [1:0]                           s_r_resp_o,
  // The host's AXI (PCIS, through the CL top), to HBM port 0 (bringup_host.sv)
  input  logic                                 h_ar_valid_i,
  output logic                                 h_ar_ready_o,
  input  logic [AddrWidth-1:0]                 h_ar_addr_i,
  input  logic [7:0]                           h_ar_len_i,
  input  logic [1:0]                           h_ar_burst_i,
  output logic                                 h_r_valid_o,
  input  logic                                 h_r_ready_i,
  output logic [DataWidth-1:0]                 h_r_data_o,
  output logic [1:0]                           h_r_resp_o,
  output logic                                 h_r_last_o,
  input  logic                                 h_aw_valid_i,
  output logic                                 h_aw_ready_o,
  input  logic [AddrWidth-1:0]                 h_aw_addr_i,
  input  logic [7:0]                           h_aw_len_i,
  input  logic [1:0]                           h_aw_burst_i,
  input  logic                                 h_w_valid_i,
  output logic                                 h_w_ready_o,
  input  logic [DataWidth-1:0]                 h_w_data_i,
  input  logic [BeatBytes-1:0]                 h_w_strb_i,
  input  logic                                 h_w_last_i,
  output logic                                 h_b_valid_o,
  input  logic                                 h_b_ready_i,
  output logic [1:0]                           h_b_resp_o,
  // AXI, a port each (sizes: always the full beat)
  output logic [Ports-1:0]                     ar_valid_o,
  input  logic [Ports-1:0]                     ar_ready_i,
  output logic [Ports-1:0][AddrWidth-1:0]      ar_addr_o,
  output logic [Ports-1:0][7:0]                ar_len_o,
  output logic [Ports-1:0][2:0]                ar_size_o,
  output logic [Ports-1:0][1:0]                ar_burst_o,
  input  logic [Ports-1:0]                     r_valid_i,
  output logic [Ports-1:0]                     r_ready_o,
  input  logic [Ports-1:0][DataWidth-1:0]      r_data_i,
  input  logic [Ports-1:0][1:0]                r_resp_i,
  input  logic [Ports-1:0]                     r_last_i,
  output logic [Ports-1:0]                     aw_valid_o,
  input  logic [Ports-1:0]                     aw_ready_i,
  output logic [Ports-1:0][AddrWidth-1:0]      aw_addr_o,
  output logic [Ports-1:0][7:0]                aw_len_o,
  output logic [Ports-1:0][2:0]                aw_size_o,
  output logic [Ports-1:0][1:0]                aw_burst_o,
  output logic [Ports-1:0]                     w_valid_o,
  input  logic [Ports-1:0]                     w_ready_i,
  output logic [Ports-1:0][DataWidth-1:0]      w_data_o,
  output logic [Ports-1:0][BeatBytes-1:0]      w_strb_o,
  output logic [Ports-1:0]                     w_last_o,
  input  logic [Ports-1:0]                     b_valid_i,
  output logic [Ports-1:0]                     b_ready_o,
  input  logic [Ports-1:0][1:0]                b_resp_i
);

  localparam logic [31:0] Id = 32'hCAB0_B001;
  // Register offsets (README.md)
  localparam logic [11:0] RegId = 12'h000, RegBuild = 12'h004, RegScratch = 12'h008,
                          RegStatus = 12'h00C, RegCyclesLo = 12'h010, RegCyclesHi = 12'h014,
                          RegCtrl = 12'h020, RegPortEn = 12'h024, RegMode = 12'h028,
                          RegBaseLo = 12'h02C, RegBaseHi = 12'h030, RegStride = 12'h034,
                          RegBeats = 12'h038, RegRdBurst = 12'h03C, RegWrBurst = 12'h040,
                          RegRdLimit = 12'h044, RegWrLimit = 12'h048, RegWrStrb = 12'h04C,
                          RegSeed = 12'h050, RegHostPort = 12'h054;
  localparam logic [11:0] RegPorts = 12'h100;  // port p's counters at RegPorts + 0x40 p

  if (Ports < 1 || Ports > 32) begin : gen_bad_ports
    $error("bringup_core: Ports must be 1 to 32");
  end

  // --- Registers ----------------------------------------------------------------

  logic                  start_q, start_q1;
  logic [31:0]           port_en_q, base_lo_q, base_hi_q, stride_q, beats_q, seed_q, strb_q;
  logic [2:0]            mode_q;  // check, write, read
  logic [LenW-1:0]       rd_burst_q, wr_burst_q;
  logic [LimitW-1:0]     rd_limit_q, wr_limit_q;
  logic [31:0]           scratch_q;
  logic                  host_port_q;
  logic [63:0]           cycles_q;
  logic [31:0]           cycles_hi_q;  // CYCLES_HI as of the last CYCLES_LO read
  logic                  busy_q, hbm_ready_q, refused_q;
  logic [7:0]            starts_q;  // start requests, taken or refused: the host sees its own arrive
  logic [Ports-1:0]      busy;
  logic                  start_wr, settings_ok;
  logic [2:0]            rd_stage_q;  // reads: 0 idle, then 1, 2, 3, 4 (answer held until taken)
  logic [11:0]           rd_addr_q;   // the read's address
  typedef struct packed {
    logic [2:0]            mode;
    logic [AddrWidth-1:0]  base;
    logic [31:0]           stride;
    logic [CountWidth-1:0] beats;
    logic [LenW-1:0]       rd_burst, wr_burst;
    logic [LimitW-1:0]     rd_limit, wr_limit;
    logic [BeatBytes-1:0]  strb;
    logic [31:0]           seed;
    logic                  start;
  } settings_t;
  settings_t [Groups-1:0] set_q;  // a copy per group (Ports, below)
  typedef struct packed {
    logic [3:0]       counter;  // the read's counter in its port
    logic [SlotW-1:0] port;  // its port's place in its group
  } rd_sel_t;
  rd_sel_t [Groups-1:0] rd_sel_q;  // a copy per group

  // A burst of 1, 2, 4, 8 or 16 beats: a power of two up to 16.
  function automatic logic burst_ok(logic [LenW-1:0] beats);
    return beats != '0 && (beats & (beats - 1'b1)) == '0 && beats <= LenW'(16);
  endfunction

  // README.md's rules for a run (the module's header).
  assign settings_ok = burst_ok(rd_burst_q) && burst_ok(wr_burst_q)
                    && rd_limit_q != '0 && wr_limit_q != '0
                    && beats_q[31:CountWidth] == '0
                    && (beats_q[LenW-1:0] & (rd_burst_q - 1'b1)) == '0
                    && (beats_q[LenW-1:0] & (wr_burst_q - 1'b1)) == '0
                    && base_lo_q[11:0] == '0 && stride_q[11:0] == '0;

  logic                  wr_fire, b_pend_q;
  assign s_aw_ready_o = s_aw_valid_i && s_w_valid_i && !b_pend_q;
  assign s_w_ready_o  = s_aw_ready_o;
  assign wr_fire      = s_aw_ready_o;
  assign s_b_valid_o  = b_pend_q;
  assign s_b_resp_o   = 2'b00;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      b_pend_q    <= 1'b0;
      start_q     <= 1'b0;
      port_en_q   <= '0;
      mode_q      <= '0;
      base_lo_q   <= '0;
      base_hi_q   <= '0;
      stride_q    <= '0;
      beats_q     <= '0;
      rd_burst_q  <= LenW'(16);
      wr_burst_q  <= LenW'(16);
      rd_limit_q  <= LimitW'(8);
      wr_limit_q  <= LimitW'(8);
      strb_q      <= '1;
      seed_q      <= '0;
      scratch_q   <= '0;
      host_port_q <= 1'b1;
      refused_q   <= 1'b0;
      starts_q    <= '0;
    end else begin
      if (wr_fire) b_pend_q <= 1'b1;
      else if (s_b_ready_i) b_pend_q <= 1'b0;
      start_q <= start_wr && !busy_q && settings_ok;
      if (start_wr) begin
        refused_q <= busy_q || !settings_ok;
        starts_q  <= starts_q + 1'b1;
      end
      if (wr_fire) begin
        unique case (s_aw_addr_i)
          RegScratch:  scratch_q   <= s_w_data_i;
          RegPortEn:   port_en_q   <= s_w_data_i;
          RegMode:     mode_q      <= s_w_data_i[2:0];
          RegBaseLo:   base_lo_q   <= s_w_data_i;
          RegBaseHi:   base_hi_q   <= s_w_data_i;
          RegStride:   stride_q    <= s_w_data_i;
          RegBeats:    beats_q     <= s_w_data_i;
          RegRdBurst:  rd_burst_q  <= s_w_data_i[LenW-1:0];
          RegWrBurst:  wr_burst_q  <= s_w_data_i[LenW-1:0];
          RegRdLimit:  rd_limit_q  <= s_w_data_i[LimitW-1:0];
          RegWrLimit:  wr_limit_q  <= s_w_data_i[LimitW-1:0];
          RegWrStrb:   strb_q      <= s_w_data_i;
          RegSeed:     seed_q      <= s_w_data_i;
          RegHostPort: if (!busy_q) host_port_q <= s_w_data_i[0];
          default: ;
        endcase
      end
    end
  end

  assign start_wr    = wr_fire && s_aw_addr_i == RegCtrl && s_w_data_i[0];

  always_ff @(posedge clk_i) begin
    if (!rst_ni) cycles_q <= '0;
    else cycles_q <= cycles_q + 1'b1;
    if (!rst_ni) cycles_hi_q <= '0;
    else if (rd_stage_q == 3'd3 && rd_addr_q == RegCyclesLo) cycles_hi_q <= cycles_q[63:32];
    start_q1    <= start_q;  // the groups' start, here
    busy_q      <= |busy || start_q || start_q1;  // a port is busy from its start, two cycles after start_q
    hbm_ready_q <= hbm_ready_i;
  end

  // --- Ports ----------------------------------------------------------------------

  logic [AddrWidth-1:0]        base;
  logic [Ports-1:0][31:0]      stat;
  // The engines' AXI: straight out, but for port 0's (bringup_host.sv).
  logic [Ports-1:0]                e_ar_valid, e_ar_ready, e_r_valid, e_r_ready;
  logic [Ports-1:0]                e_aw_valid, e_aw_ready, e_w_valid, e_w_ready, e_w_last;
  logic [Ports-1:0]                e_b_valid, e_b_ready;
  logic [Ports-1:0][AddrWidth-1:0] e_ar_addr, e_aw_addr;
  logic [Ports-1:0][7:0]           e_ar_len, e_aw_len;
  logic [Ports-1:0][1:0]           e_ar_burst, e_aw_burst;
  logic [Ports-1:0][DataWidth-1:0] e_w_data;
  logic [Ports-1:0][BeatBytes-1:0] e_w_strb;
  assign base = {base_hi_q[AddrWidth-33:0], base_lo_q};

  // The settings, the start and a read's counter reach the ports through a
  // register per group of GroupPorts ports, each driving its group only, and
  // each port has its own reset register: short wires (bring-up build 3's
  // 250 MHz failures). The settings change only between runs, a write apart.
  settings_t set_d;
  assign set_d = '{mode: mode_q, base: base, stride: stride_q, beats: beats_q[CountWidth-1:0],
                   rd_burst: rd_burst_q, wr_burst: wr_burst_q, rd_limit: rd_limit_q,
                   wr_limit: wr_limit_q, strb: strb_q, seed: seed_q, start: start_q};
  always_ff @(posedge clk_i) set_q <= {Groups{set_d}};

  for (genvar p = 0; p < Ports; p++) begin : gen_port
    // The port's address, two register stages after its group's (its offset,
    // then the sum), settled long before a start; its start, one.
    settings_t            set;
    logic [AddrWidth-1:0] offset_q, addr_q;
    logic                 start_q2, rst_nq;
    assign set = set_q[p / GroupPorts];
    always_ff @(posedge clk_i) begin
      offset_q <= AddrWidth'(p) * AddrWidth'(set.stride);
      addr_q   <= set.base + offset_q;
      start_q2 <= set.start && port_en_q[p] && !(p == 0 && host_port_q);
      rst_nq   <= rst_ni;
    end

    bringup_port #(
      .AddrWidth (AddrWidth),
      .DataWidth (DataWidth),
      .CountWidth(CountWidth)
    ) u_port (
      .clk_i,
      .rst_ni    (rst_nq),
      .start_i   (start_q2),
      .rd_en_i   (set.mode[0]),
      .wr_en_i   (set.mode[1]),
      .check_i   (set.mode[2]),
      .addr_i    (addr_q),
      .beats_i   (set.beats),
      .rd_burst_i(set.rd_burst),
      .wr_burst_i(set.wr_burst),
      .rd_limit_i(set.rd_limit),
      .wr_limit_i(set.wr_limit),
      .wr_strb_i (set.strb),
      .seed_i    (set.seed),
      .busy_o    (busy[p]),
      .stat_sel_i(rd_sel_q[p / GroupPorts].counter),
      .stat_o    (stat[p]),
      .ar_valid_o(e_ar_valid[p]),
      .ar_ready_i(e_ar_ready[p]),
      .ar_addr_o (e_ar_addr[p]),
      .ar_len_o  (e_ar_len[p]),
      .ar_size_o (ar_size_o[p]),
      .ar_burst_o(e_ar_burst[p]),
      .r_valid_i (e_r_valid[p]),
      .r_ready_o (e_r_ready[p]),
      .r_data_i  (r_data_i[p]),
      .r_resp_i  (r_resp_i[p]),
      .r_last_i  (r_last_i[p]),
      .aw_valid_o(e_aw_valid[p]),
      .aw_ready_i(e_aw_ready[p]),
      .aw_addr_o (e_aw_addr[p]),
      .aw_len_o  (e_aw_len[p]),
      .aw_size_o (aw_size_o[p]),
      .aw_burst_o(e_aw_burst[p]),
      .w_valid_o (e_w_valid[p]),
      .w_ready_i (e_w_ready[p]),
      .w_data_o  (e_w_data[p]),
      .w_strb_o  (e_w_strb[p]),
      .w_last_o  (e_w_last[p]),
      .b_valid_i (e_b_valid[p]),
      .b_ready_o (e_b_ready[p]),
      .b_resp_i  (b_resp_i[p])
    );

    if (p > 0) begin : gen_direct
      assign ar_valid_o[p] = e_ar_valid[p];
      assign e_ar_ready[p] = ar_ready_i[p];
      assign ar_addr_o[p]  = e_ar_addr[p];
      assign ar_len_o[p]   = e_ar_len[p];
      assign ar_burst_o[p] = e_ar_burst[p];
      assign e_r_valid[p]  = r_valid_i[p];
      assign r_ready_o[p]  = e_r_ready[p];
      assign aw_valid_o[p] = e_aw_valid[p];
      assign e_aw_ready[p] = aw_ready_i[p];
      assign aw_addr_o[p]  = e_aw_addr[p];
      assign aw_len_o[p]   = e_aw_len[p];
      assign aw_burst_o[p] = e_aw_burst[p];
      assign w_valid_o[p]  = e_w_valid[p];
      assign e_w_ready[p]  = w_ready_i[p];
      assign w_data_o[p]   = e_w_data[p];
      assign w_strb_o[p]   = e_w_strb[p];
      assign w_last_o[p]   = e_w_last[p];
      assign e_b_valid[p]  = b_valid_i[p];
      assign b_ready_o[p]  = e_b_ready[p];
    end
  end

  // Port 0, shared with the host; the engine reads R and B data from the
  // memory's side, the host from its FIFOs.
  bringup_host #(.AddrWidth(AddrWidth), .DataWidth(DataWidth)) u_host (
    .clk_i,
    .rst_ni,
    .sel_i       (host_port_q),
    .sel_o       (host_sel_o),
    .h_ar_valid_i,
    .h_ar_ready_o,
    .h_ar_i      ({h_ar_addr_i, h_ar_len_i, h_ar_burst_i}),
    .h_r_valid_o,
    .h_r_ready_i,
    .h_r_o       ({h_r_data_o, h_r_resp_o, h_r_last_o}),
    .h_aw_valid_i,
    .h_aw_ready_o,
    .h_aw_i      ({h_aw_addr_i, h_aw_len_i, h_aw_burst_i}),
    .h_w_valid_i,
    .h_w_ready_o,
    .h_w_i       ({h_w_data_i, h_w_strb_i, h_w_last_i}),
    .h_b_valid_o,
    .h_b_ready_i,
    .h_b_resp_o,
    .e_ar_valid_i(e_ar_valid[0]),
    .e_ar_ready_o(e_ar_ready[0]),
    .e_ar_i      ({e_ar_addr[0], e_ar_len[0], e_ar_burst[0]}),
    .e_r_valid_o (e_r_valid[0]),
    .e_r_ready_i (e_r_ready[0]),
    .e_aw_valid_i(e_aw_valid[0]),
    .e_aw_ready_o(e_aw_ready[0]),
    .e_aw_i      ({e_aw_addr[0], e_aw_len[0], e_aw_burst[0]}),
    .e_w_valid_i (e_w_valid[0]),
    .e_w_ready_o (e_w_ready[0]),
    .e_w_i       ({e_w_data[0], e_w_strb[0], e_w_last[0]}),
    .e_b_valid_o (e_b_valid[0]),
    .e_b_ready_i (e_b_ready[0]),
    .m_ar_valid_o(ar_valid_o[0]),
    .m_ar_ready_i(ar_ready_i[0]),
    .m_ar_o      ({ar_addr_o[0], ar_len_o[0], ar_burst_o[0]}),
    .m_r_valid_i (r_valid_i[0]),
    .m_r_ready_o (r_ready_o[0]),
    .m_r_i       ({r_data_i[0], r_resp_i[0], r_last_i[0]}),
    .m_aw_valid_o(aw_valid_o[0]),
    .m_aw_ready_i(aw_ready_i[0]),
    .m_aw_o      ({aw_addr_o[0], aw_len_o[0], aw_burst_o[0]}),
    .m_w_valid_o (w_valid_o[0]),
    .m_w_ready_i (w_ready_i[0]),
    .m_w_o       ({w_data_o[0], w_strb_o[0], w_last_o[0]}),
    .m_b_valid_i (b_valid_i[0]),
    .m_b_ready_o (b_ready_o[0]),
    .m_b_resp_i  (b_resp_i[0])
  );

  // --- Reads: address taken (stage 1), the port's counter (2), its group's (3),
  // the answer (4) -----------------------------------------------------------------

  logic [Groups-1:0][31:0] group_q;  // the read's counter of its port's place in each group

  logic        rd_is_port_q;  // the address is a port's counter
  logic [GroupW-1:0] rd_group_q;  // the read's port's group
  logic [31:0] r_data_q;
  logic [5:0]  ar_port;      // the port the address would be, 0 .. 59

  assign s_ar_ready_o = rd_stage_q == 3'd0;
  assign s_r_valid_o  = rd_stage_q == 3'd4;
  assign s_r_data_o   = r_data_q;
  assign s_r_resp_o   = 2'b00;
  assign ar_port      = s_ar_addr_i[11:6] - 6'(RegPorts >> 6);

  always_ff @(posedge clk_i) begin
    if (!rst_ni) rd_stage_q <= '0;
    else if (s_ar_valid_i && s_ar_ready_o) rd_stage_q <= 3'd1;
    else if (rd_stage_q != 3'd0 && rd_stage_q != 3'd4) rd_stage_q <= rd_stage_q + 1'b1;
    else if (s_r_valid_o && s_r_ready_i) rd_stage_q <= 3'd0;
  end

  always_ff @(posedge clk_i) begin
    if (s_ar_valid_i && s_ar_ready_o) begin
      rd_addr_q    <= s_ar_addr_i;
      rd_is_port_q <= s_ar_addr_i >= RegPorts && 32'(ar_port) < 32'(Ports);
      rd_group_q   <= GroupW'(ar_port / GroupPorts);
      rd_sel_q     <= {Groups{rd_sel_t'{counter: s_ar_addr_i[5:2], port: SlotW'(ar_port)}}};
    end
    // Every cycle: in stage 3 it holds stage 2's counters.
    for (int g = 0; g < Groups; g++) begin
      automatic int unsigned p = g * GroupPorts + int'(rd_sel_q[g].port);
      group_q[g] <= p < Ports ? stat[p] : '0;
    end
    if (rd_stage_q == 3'd3) begin
      if (rd_is_port_q) begin
        r_data_q <= group_q[rd_group_q];
      end else begin
        unique case (rd_addr_q)
          RegId:       r_data_q <= Id;
          RegBuild:    r_data_q <= Build;
          RegScratch:  r_data_q <= scratch_q;
          RegStatus:   r_data_q <= {16'd0, starts_q, 4'd0, host_sel_o, refused_q, hbm_ready_q, busy_q};
          RegCyclesLo: r_data_q <= cycles_q[31:0];
          RegCyclesHi: r_data_q <= cycles_hi_q;
          RegPortEn:   r_data_q <= port_en_q;
          RegMode:     r_data_q <= {29'd0, mode_q};
          RegBaseLo:   r_data_q <= base_lo_q;
          RegBaseHi:   r_data_q <= base_hi_q;
          RegStride:   r_data_q <= stride_q;
          RegBeats:    r_data_q <= beats_q;
          RegRdBurst:  r_data_q <= 32'(rd_burst_q);
          RegWrBurst:  r_data_q <= 32'(wr_burst_q);
          RegRdLimit:  r_data_q <= 32'(rd_limit_q);
          RegWrLimit:  r_data_q <= 32'(wr_limit_q);
          RegWrStrb:   r_data_q <= strb_q;
          RegSeed:     r_data_q <= seed_q;
          RegHostPort: r_data_q <= {31'd0, host_port_q};
          default: r_data_q <= 32'd0;
        endcase
      end
    end
  end

endmodule
