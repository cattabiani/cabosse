// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The accelerator's core (docs/architecture.md, "Blocks"): the controller,
// the engine and the vector unit, with one AXI-Lite port for the host's
// registers, the engine's Ports AXI read ports (port p on channel p) and the
// vector unit's AXI port (global HBM addresses: channel * 2^ChanW + offset).
// Platform code (platforms/) connects them to the HBM and the shell.
//
// The engine reads its x and writes its results through the vector unit's
// ext port, on the buffers the vector unit holds: x_addr_o[0] (a group of E
// values) as ext_r_addr_i = E * x_addr_o[0], x_addr_o[1] (a VALUES pass's
// second head, on P) as ext_r2_addr_i, and each write's Writes slots as the
// first Writes lanes of an ext write. Only P has a second read (D-069); the
// vector unit gives port 0's data on port 1 for any other buffer, which the
// engine needs. The command list keeps the two units off each other's buffer
// ports (commands.hazards).
//
// host_r_*: a read of a buffer's Lanes elements from host_r_addr_i, the data
// RdLat = 4 cycles after, in place of the engine's read: the simulation's view of
// any buffer (M7; the logits come back through HBM, OUTPUT: D-071). It must
// not be used while a token runs.
//
// The engine and the vector unit sit in different SLRs on F2 (platforms/f2,
// the floorplan): every signal between them goes through a register here,
// on the vector unit's side (the xv_ registers), and the engine's own on
// its side. The engine's x reads take RdLat + 2 cycles (its XRdLat); its
// writes, and its done to the controller, one more cycle (the same for
// both, so a command after the done still sees every write). The
// controller (the engine's SLR) gives the vector unit its commands through
// two queues, one on each side, and takes its done and errors registered.

module core #(
  parameter  int unsigned Ports    = 32,
  parameter  int unsigned Lanes    = 16,      // the vector unit's (D-030)
  parameter  int unsigned Bufs     = 22,
  parameter  int unsigned Len      = 1024,    // elements in each buffer
  parameter  logic [31:0] Bf16Mask = 32'h4644,  // per id: X, QR, P, ATT, M (commands.buffers)
  parameter  int unsigned BufP     = 9,       // P's id (commands.Buf)
  parameter  int unsigned ChanW    = 29,      // log2 of a channel's bytes: 512 MiB
  parameter  int unsigned Commands = 2048,
  localparam int unsigned MemAddrW = 34,      // F2's HBM: 16 GiB
  localparam int unsigned AddrW    = 20,
  localparam int unsigned DataW    = 256,     // an HBM port's beat
  localparam int unsigned BufW     = $clog2(Bufs),
  localparam int unsigned E        = 4,       // the engine's pairs a lane a cycle (D-027)
  localparam int unsigned Writes   = 8,       // the engine's results a cycle
  localparam int unsigned XHeads   = 2,       // the engine's x reads (rtl/engine.sv's Heads)
  localparam int unsigned RdLat    = 4,       // a buffer read's cycles (tile_buffer; engine's XRdLat)
  localparam int unsigned AxilW    = 17
) (
  input  logic                          clk_i,
  input  logic                          rst_ni,

  input  logic                          s_aw_valid_i,
  output logic                          s_aw_ready_o,
  input  logic [AxilW-1:0]              s_aw_addr_i,
  input  logic                          s_w_valid_i,
  output logic                          s_w_ready_o,
  input  logic [31:0]                   s_w_data_i,
  output logic                          s_b_valid_o,
  input  logic                          s_b_ready_i,
  output logic [1:0]                    s_b_resp_o,
  input  logic                          s_ar_valid_i,
  output logic                          s_ar_ready_o,
  input  logic [AxilW-1:0]              s_ar_addr_i,
  output logic                          s_r_valid_o,
  input  logic                          s_r_ready_i,
  output logic [31:0]                   s_r_data_o,
  output logic [1:0]                    s_r_resp_o,
  output logic                          irq_o,

  output logic [Ports-1:0]              e_ar_valid_o,
  input  logic [Ports-1:0]              e_ar_ready_i,
  output logic [Ports-1:0][MemAddrW-1:0] e_ar_addr_o,
  output logic [Ports-1:0][7:0]         e_ar_len_o,
  output logic [Ports-1:0][2:0]         e_ar_size_o,
  output logic [Ports-1:0][1:0]         e_ar_burst_o,
  input  logic [Ports-1:0]              e_r_valid_i,
  output logic [Ports-1:0]              e_r_ready_o,
  input  logic [Ports-1:0][DataW-1:0]   e_r_data_i,
  input  logic [Ports-1:0][1:0]         e_r_resp_i,

  output logic                          v_ar_valid_o,
  input  logic                          v_ar_ready_i,
  output logic [MemAddrW-1:0]           v_ar_addr_o,
  output logic [7:0]                    v_ar_len_o,
  output logic [2:0]                    v_ar_size_o,
  output logic [1:0]                    v_ar_burst_o,
  input  logic                          v_r_valid_i,
  output logic                          v_r_ready_o,
  input  logic [DataW-1:0]              v_r_data_i,
  input  logic [1:0]                    v_r_resp_i,
  output logic                          v_aw_valid_o,
  input  logic                          v_aw_ready_i,
  output logic [MemAddrW-1:0]           v_aw_addr_o,
  output logic [7:0]                    v_aw_len_o,
  output logic [2:0]                    v_aw_size_o,
  output logic [1:0]                    v_aw_burst_o,
  output logic                          v_w_valid_o,
  input  logic                          v_w_ready_i,
  output logic [DataW-1:0]              v_w_data_o,
  output logic [DataW/8-1:0]            v_w_strb_o,
  output logic                          v_w_last_o,
  input  logic                          v_b_valid_i,
  output logic                          v_b_ready_o,
  input  logic [1:0]                    v_b_resp_i,

  input  logic                          host_r_valid_i,
  input  logic [BufW-1:0]               host_r_buf_i,
  input  logic [AddrW-1:0]              host_r_addr_i,
  output logic [Lanes-1:0][31:0]        host_r_data_o
);

  if (Lanes * 16 != DataW) begin : gen_bad_lanes
    $error("core: the vector unit's beat (Lanes * 16 bits) must be an HBM port's");
  end
  if (Writes > Lanes || E > Lanes) begin : gen_bad_writes
    $error("core: the engine's writes and x reads must fit the vector unit's lanes");
  end

  localparam int unsigned HeadWidth = 8;
  localparam int unsigned EBufW     = 8;  // the engine's buffer ids

  logic                 eng_rst_nq, vec_rst_nq, units_reset;
  logic [4:0]           cmd_op;
  logic [BufW-1:0]      cmd_dst, cmd_a, cmd_b, cmd_c;
  logic [7:0]           cmd_flags;
  logic [AddrW-1:0]     cmd_n, cmd_m, cmd_cap, token, t, stride;
  logic [MemAddrW-1:0]  cmd_addr;
  logic [31:0]          cmd_scalar;
  logic [HeadWidth-1:0] cmd_group;
  logic                 cmd_bf16;
  logic                 eng_wait, vec_err;
  // The handshakes between the controller and the units, readable by the
  // simulation (verif/bulk/core.cpp's trace: each command's cycles, M7);
  // synthesis ignores the pragma.
  logic eng_valid /* verilator public_flat_rd */;
  logic eng_ready /* verilator public_flat_rd */;
  logic eng_go    /* verilator public_flat_rd */;
  logic eng_done  /* verilator public_flat_rd */;
  logic vec_valid /* verilator public_flat_rd */;
  logic vec_ready /* verilator public_flat_rd */;
  logic vec_done  /* verilator public_flat_rd */;
  logic [1:0]           eng_op;
  logic [Ports-1:0]     eng_err;
  logic                 eng_done_q, eng_err_q;  // to the controller, registered
  logic                 vec_done_q, vec_err_q;

  // The units' reset (the core's, or CONTROL's), a register per unit here
  // and again inside each unit, per lane, port and slot: UnitsResetDelay
  // cycles after the controller's, which holds the AXI-Lite writes and its
  // own state until it is through (controller.sv).
  localparam int unsigned UnitsResetDelay = 2;
  always_ff @(posedge clk_i) begin
    eng_rst_nq <= rst_ni && !units_reset;
    vec_rst_nq <= rst_ni && !units_reset;
  end

`ifndef SYNTHESIS
  // UnitsResetDelay matches the registers: past the controller's window,
  // every unit is out of its reset.
  always_ff @(posedge clk_i) begin
    if (rst_ni && !u_controller.settling && !(u_engine.rst_nq && u_vec.rst_nq)) begin
      $error("core: a unit still in reset after the controller's window");
    end
  end
`endif

  controller #(
    .Bufs     (Bufs),
    .Bf16Mask (Bf16Mask),
    .AddrW    (AddrW),
    .MemAddrW (MemAddrW),
    .HeadWidth(HeadWidth),
    .Commands (Commands),
    .AxilAddrW(AxilW),
    .UnitsResetDelay(UnitsResetDelay)
  ) u_controller (
    .clk_i,
    .rst_ni,
    .s_aw_valid_i,
    .s_aw_ready_o,
    .s_aw_addr_i,
    .s_w_valid_i,
    .s_w_ready_o,
    .s_w_data_i,
    .s_b_valid_o,
    .s_b_ready_i,
    .s_b_resp_o,
    .s_ar_valid_i,
    .s_ar_ready_o,
    .s_ar_addr_i,
    .s_r_valid_o,
    .s_r_ready_i,
    .s_r_data_o,
    .s_r_resp_o,
    .irq_o,
    .units_reset_o(units_reset),
    .cmd_op_o     (cmd_op),
    .cmd_dst_o    (cmd_dst),
    .cmd_a_o      (cmd_a),
    .cmd_b_o      (cmd_b),
    .cmd_c_o      (cmd_c),
    .cmd_flags_o  (cmd_flags),
    .cmd_n_o      (cmd_n),
    .cmd_m_o      (cmd_m),
    .cmd_cap_o    (cmd_cap),
    .cmd_addr_o   (cmd_addr),
    .cmd_scalar_o (cmd_scalar),
    .cmd_group_o  (cmd_group),
    .cmd_bf16_o   (cmd_bf16),
    .token_o      (token),
    .t_o          (t),
    .stride_o     (stride),
    .eng_valid_o  (eng_valid),
    .eng_ready_i  (eng_ready),
    .eng_op_o     (eng_op),
    .eng_wait_o   (eng_wait),
    .eng_go_o     (eng_go),
    .eng_done_i   (eng_done_q),
    .eng_err_i    (eng_err_q),
    .vec_valid_o  (vec_valid),
    .vec_ready_i  (vec_ready),
    .vec_done_i   (vec_done_q),
    .vec_err_i    (vec_err_q)
  );

  // The engine's x reads and result writes, on the vector unit's ext port.
  logic [EBufW-1:0]                     x_buf, w_buf;
  logic [XHeads-1:0][AddrW-1:0]         x_addr;
  logic [XHeads-1:0][E-1:0][15:0]       x_data;
  logic                                 w_valid, w_bf16;
  logic [AddrW-1:0]                     w_row;
  logic [Writes-1:0]                    w_mask;
  logic [Writes-1:0][31:0]              w_data;
  logic [Lanes-1:0][31:0]               ext_r_data, ext_r2_data, ext_w_data;
  logic [BufW-1:0]                      ext_r_buf;
  logic [AddrW-1:0]                     ext_r_addr;

  // The crossing's registers, on the vector unit's side (the floorplan puts
  // the xv_ registers in its SLR): the engine's read addresses and writes
  // in, the read data out.
  logic [EBufW-1:0]               xv_r_buf_q, xv_w_buf_q;
  logic [XHeads-1:0][AddrW-1:0]   xv_r_addr_q;
  logic [XHeads-1:0][E-1:0][15:0] xv_x_data_q;
  logic                           xv_w_valid_q;
  logic [AddrW-1:0]               xv_w_row_q;
  logic [Writes-1:0]              xv_w_mask_q;
  logic [Writes-1:0][31:0]        xv_w_data_q;
  always_ff @(posedge clk_i) begin
    xv_r_buf_q   <= x_buf;
    xv_r_addr_q  <= x_addr;
    xv_w_valid_q <= eng_rst_nq && w_valid;
    xv_w_buf_q   <= w_buf;
    xv_w_row_q   <= w_row;
    xv_w_mask_q  <= w_mask;
    xv_w_data_q  <= w_data;
    for (int unsigned e = 0; e < E; e++) begin
      xv_x_data_q[0][e] <= ext_r_data[e][15:0];   // x is BF16: the low half
      xv_x_data_q[1][e] <= ext_r2_data[e][15:0];
    end
  end
  assign x_data = xv_x_data_q;

  // The engine's done with its last write (above); its DMAs' errors. The
  // vector unit's, as the engine's.
  always_ff @(posedge clk_i) begin
    eng_done_q <= eng_rst_nq && eng_done;
    eng_err_q  <= |eng_err;
    vec_done_q <= vec_rst_nq && vec_done;
    vec_err_q  <= vec_err;
  end

  // The vector unit's commands: a queue of 2 on the controller's side
  // (u_vec_cmd0) and one on the vector unit's (u_vec_cmd1), so only
  // registers face the crossing (stream_fifo's ready and valid are
  // registers). A command reaches the vector unit 2 cycles after the
  // controller gives it; token, t and stride go with it.
  typedef struct packed {
    logic [4:0]          op;
    logic [BufW-1:0]     dst, a, b, c;
    logic [7:0]          flags;
    logic [AddrW-1:0]    n, m, cap;
    logic [31:0]         scalar;
    logic [MemAddrW-1:0] addr;
    logic [AddrW-1:0]    token, t, stride;
  } vec_cmd_t;
  vec_cmd_t vc_in, vc_mid, vc;
  logic     vc_mid_valid, vc_mid_ready, vc_valid, vc_ready;
  assign vc_in = '{op: cmd_op, dst: cmd_dst, a: cmd_a, b: cmd_b, c: cmd_c, flags: cmd_flags,
                   n: cmd_n, m: cmd_m, cap: cmd_cap, scalar: cmd_scalar, addr: cmd_addr,
                   token: token, t: t, stride: stride};
  // verilator lint_off PINCONNECTEMPTY
  stream_fifo #(
    .Width($bits(vec_cmd_t)),
    .Depth(2)
  ) u_vec_cmd0 (
    .clk_i,
    .rst_ni     (vec_rst_nq),
    .in_valid_i (vec_valid),
    .in_ready_o (vec_ready),
    .in_data_i  (vc_in),
    .out_valid_o(vc_mid_valid),
    .out_ready_i(vc_mid_ready),
    .out_data_o (vc_mid),
    .count_o    ()
  );
  stream_fifo #(
    .Width($bits(vec_cmd_t)),
    .Depth(2)
  ) u_vec_cmd1 (
    .clk_i,
    .rst_ni     (vec_rst_nq),
    .in_valid_i (vc_mid_valid),
    .in_ready_o (vc_mid_ready),
    .in_data_i  (vc_mid),
    .out_valid_o(vc_valid),
    .out_ready_i(vc_ready),
    .out_data_o (vc),
    .count_o    ()
  );
  // verilator lint_on PINCONNECTEMPTY

  assign ext_w_data    = (Lanes * 32)'(xv_w_data_q);
  assign ext_r_buf     = host_r_valid_i ? host_r_buf_i : BufW'(xv_r_buf_q);
  assign ext_r_addr    = host_r_valid_i ? host_r_addr_i : AddrW'(E * xv_r_addr_q[0]);
  assign host_r_data_o = ext_r_data;

  engine #(
    .XRdLat   (RdLat + 2),  // the xv registers: the address's, the data's
    .Ports    (Ports),
    .AddrWidth(MemAddrW),
    .DataWidth(DataW),
    .RowWidth (AddrW),
    .BufWidth (EBufW),
    .Writes   (Writes),
    .E        (E)
  ) u_engine (
    .clk_i,
    .rst_ni       (eng_rst_nq),
    .cmd_valid_i  (eng_valid),
    .cmd_ready_o  (eng_ready),
    .cmd_op_i     (eng_op),
    .cmd_rows_i   (cmd_n),
    .cmd_cols_i   (16'(cmd_m)),
    .cmd_addr_i   (cmd_addr),
    .cmd_a_i      (EBufW'(cmd_a)),
    .cmd_dst_i    (EBufW'(cmd_dst)),
    .cmd_bf16_i   (cmd_bf16),
    .cmd_wait_i   (eng_wait),
    .cmd_heads_i  (HeadWidth'(cmd_n)),
    .cmd_group_i  (cmd_group),
    .cmd_t_i      (t),
    .cmd_cap_i    (cmd_cap),
    .cmd_stride_i (stride),
    .cmd_scale_i  (cmd_scalar),
    .go_i         (eng_go),
    .done_o       (eng_done),
    .x_buf_o      (x_buf),
    .x_addr_o     (x_addr),
    .x_data_i     (x_data),
    .w_valid_o    (w_valid),
    .w_buf_o      (w_buf),
    .w_row_o      (w_row),
    .w_mask_o     (w_mask),
    .w_data_o     (w_data),
    .w_bf16_o     (w_bf16),
    .ar_valid_o   (e_ar_valid_o),
    .ar_ready_i   (e_ar_ready_i),
    .ar_addr_o    (e_ar_addr_o),
    .ar_len_o     (e_ar_len_o),
    .ar_size_o    (e_ar_size_o),
    .ar_burst_o   (e_ar_burst_o),
    .r_valid_i    (e_r_valid_i),
    .r_ready_o    (e_r_ready_o),
    .r_data_i     (e_r_data_i),
    .r_resp_i     (e_r_resp_i),
    .err_o        (eng_err)
  );

  vec_unit #(
    .RdLat    (RdLat),
    .Lanes    (Lanes),
    .Bufs     (Bufs),
    .Len      (Len),
    .Bf16Mask (Bf16Mask),
    .Read2Mask(32'(1) << BufP),
    .AddrW    (AddrW),
    .MemAddrW (MemAddrW),
    .KvPorts  (Ports),
    .ChanW    (ChanW)
  ) u_vec (
    .clk_i,
    .rst_ni        (vec_rst_nq),
    .cmd_valid_i   (vc_valid),
    .cmd_ready_o   (vc_ready),
    .cmd_op_i      (vc.op),
    .cmd_dst_i     (vc.dst),
    .cmd_a_i       (vc.a),
    .cmd_b_i       (vc.b),
    .cmd_c_i       (vc.c),
    .cmd_flags_i   (vc.flags),
    .cmd_n_i       (vc.n),
    .cmd_m_i       (vc.m),
    .cmd_scalar_i  (vc.scalar),
    .cmd_cap_i     (vc.cap),
    .cmd_addr_i    (vc.addr),
    .token_i       (vc.token),
    .t_i           (vc.t),
    .stride_i      (vc.stride),
    .done_o        (vec_done),
    .err_o         (vec_err),
    .ar_valid_o    (v_ar_valid_o),
    .ar_ready_i    (v_ar_ready_i),
    .ar_addr_o     (v_ar_addr_o),
    .ar_len_o      (v_ar_len_o),
    .ar_size_o     (v_ar_size_o),
    .ar_burst_o    (v_ar_burst_o),
    .r_valid_i     (v_r_valid_i),
    .r_ready_o     (v_r_ready_o),
    .r_data_i      (v_r_data_i),
    .r_resp_i      (v_r_resp_i),
    .aw_valid_o    (v_aw_valid_o),
    .aw_ready_i    (v_aw_ready_i),
    .aw_addr_o     (v_aw_addr_o),
    .aw_len_o      (v_aw_len_o),
    .aw_size_o     (v_aw_size_o),
    .aw_burst_o    (v_aw_burst_o),
    .w_valid_o     (v_w_valid_o),
    .w_ready_i     (v_w_ready_i),
    .w_data_o      (v_w_data_o),
    .w_strb_o      (v_w_strb_o),
    .w_last_o      (v_w_last_o),
    .b_valid_i     (v_b_valid_i),
    .b_ready_o     (v_b_ready_o),
    .b_resp_i      (v_b_resp_i),
    .ext_w_valid_i (xv_w_valid_q),
    .ext_w_buf_i   (BufW'(xv_w_buf_q)),
    .ext_w_addr_i  (xv_w_row_q),
    .ext_w_mask_i  (Lanes'(xv_w_mask_q)),
    .ext_w_data_i  (ext_w_data),
    .ext_r_valid_i (1'b1),
    .ext_r_buf_i   (ext_r_buf),
    .ext_r_addr_i  (ext_r_addr),
    .ext_r_data_o  (ext_r_data),
    .ext_r2_addr_i (AddrW'(E * xv_r_addr_q[1])),
    .ext_r2_data_o (ext_r2_data)
  );

  // verilator lint_off UNUSEDSIGNAL
  logic unused;
  // w_bf16: the buffer's format decides (a BF16 buffer keeps the low half);
  // x reads past the lanes E uses; the ids' high bits (Bufs <= 2^BufW).
  assign unused = ^{w_bf16, ext_r_data[Lanes-1:E], ext_r2_data[Lanes-1:E], xv_r_buf_q, xv_w_buf_q};
  // verilator lint_on UNUSEDSIGNAL

endmodule
