// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Timing harness for the whole engine (rtl/engine.sv) at its default size,
// 32 ports and 128 lanes (fewer ports only for the Yosys check): the placed
// build of D-051 and D-059. Every input and output registered, so every path is register to register, the
// engine's global ones included (`fire`, the x broadcast). Out of context,
// these registers go wherever the placer likes; on F2 the read ports sit
// next to the HBM. Not part of the design.

module engine_path
  import fp_pkg::*;
#(
  parameter  int unsigned MulRegs   = 2,
  parameter  int unsigned Ports     = 32,
  localparam int unsigned AddrWidth = 34,
  localparam int unsigned DataWidth = 256,
  localparam int unsigned RowWidth  = 20,
  localparam int unsigned ColWidth  = 16,
  localparam int unsigned HeadWidth = 8,
  localparam int unsigned BufWidth  = 8,
  localparam int unsigned Writes    = 8,
  localparam int unsigned E         = 4,
  localparam int unsigned Heads     = 2
) (
  input  logic                            clk_i,
  input  logic                            rst_ni,
  input  logic                            cmd_valid_i,
  output logic                            cmd_ready_o,
  input  logic [1:0]                      cmd_op_i,
  input  logic [RowWidth-1:0]             cmd_rows_i,
  input  logic [ColWidth-1:0]             cmd_cols_i,
  input  logic [AddrWidth-1:0]            cmd_addr_i,
  input  logic [BufWidth-1:0]             cmd_a_i,
  input  logic [BufWidth-1:0]             cmd_dst_i,
  input  logic                            cmd_bf16_i,
  input  logic                            cmd_wait_i,
  input  logic [HeadWidth-1:0]            cmd_heads_i,
  input  logic [HeadWidth-1:0]            cmd_group_i,
  input  logic [RowWidth-1:0]             cmd_t_i,
  input  logic [RowWidth-1:0]             cmd_cap_i,
  input  logic [RowWidth-1:0]             cmd_stride_i,
  input  logic [31:0]                     cmd_scale_i,
  input  logic                            go_i,
  output logic                            done_o,
  output logic [BufWidth-1:0]             x_buf_o,
  output logic [Heads-1:0][RowWidth-1:0]  x_addr_o,
  input  logic [Heads-1:0][E-1:0][15:0]   x_data_i,
  output logic                            w_valid_o,
  output logic [BufWidth-1:0]             w_buf_o,
  output logic [RowWidth-1:0]             w_row_o,
  output logic [Writes-1:0]               w_mask_o,
  output logic [Writes-1:0][31:0]         w_data_o,
  output logic                            w_bf16_o,
  output logic [Ports-1:0]                ar_valid_o,
  input  logic [Ports-1:0]                ar_ready_i,
  output logic [Ports-1:0][AddrWidth-1:0] ar_addr_o,
  output logic [Ports-1:0][7:0]           ar_len_o,
  output logic [Ports-1:0][2:0]           ar_size_o,
  output logic [Ports-1:0][1:0]           ar_burst_o,
  input  logic [Ports-1:0]                r_valid_i,
  output logic [Ports-1:0]                r_ready_o,
  input  logic [Ports-1:0][DataWidth-1:0] r_data_i,
  input  logic [Ports-1:0][1:0]           r_resp_i,
  output logic [Ports-1:0]                err_o
);

  logic                            rst_nq, cmd_valid_q, cmd_bf16_q, cmd_wait_q, go_q;
  logic [1:0]                      cmd_op_q;
  logic [RowWidth-1:0]             cmd_rows_q, cmd_t_q, cmd_cap_q, cmd_stride_q;
  logic [ColWidth-1:0]             cmd_cols_q;
  logic [AddrWidth-1:0]            cmd_addr_q;
  logic [BufWidth-1:0]             cmd_a_q, cmd_dst_q;
  logic [HeadWidth-1:0]            cmd_heads_q, cmd_group_q;
  logic [31:0]                     cmd_scale_q;
  logic [Heads-1:0][E-1:0][15:0]   x_data_q;
  logic [Ports-1:0]                ar_ready_q, r_valid_q;
  logic [Ports-1:0][DataWidth-1:0] r_data_q;
  logic [Ports-1:0][1:0]           r_resp_q;
  always_ff @(posedge clk_i) begin
    rst_nq       <= rst_ni;
    cmd_valid_q  <= cmd_valid_i;
    cmd_op_q     <= cmd_op_i;
    cmd_rows_q   <= cmd_rows_i;
    cmd_cols_q   <= cmd_cols_i;
    cmd_addr_q   <= cmd_addr_i;
    cmd_a_q      <= cmd_a_i;
    cmd_dst_q    <= cmd_dst_i;
    cmd_bf16_q   <= cmd_bf16_i;
    cmd_wait_q   <= cmd_wait_i;
    cmd_heads_q  <= cmd_heads_i;
    cmd_group_q  <= cmd_group_i;
    cmd_t_q      <= cmd_t_i;
    cmd_cap_q    <= cmd_cap_i;
    cmd_stride_q <= cmd_stride_i;
    cmd_scale_q  <= cmd_scale_i;
    go_q         <= go_i;
    x_data_q     <= x_data_i;
    ar_ready_q   <= ar_ready_i;
    r_valid_q    <= r_valid_i;
    r_data_q     <= r_data_i;
    r_resp_q     <= r_resp_i;
  end

  logic                            cmd_ready, done, w_valid, w_bf16;
  logic [BufWidth-1:0]             x_buf, w_buf;
  logic [Heads-1:0][RowWidth-1:0]  x_addr;
  logic [RowWidth-1:0]             w_row;
  logic [Writes-1:0]               w_mask;
  logic [Writes-1:0][31:0]         w_data;
  logic [Ports-1:0]                ar_valid, r_ready, err;
  logic [Ports-1:0][AddrWidth-1:0] ar_addr;
  logic [Ports-1:0][7:0]           ar_len;
  logic [Ports-1:0][2:0]           ar_size;
  logic [Ports-1:0][1:0]           ar_burst;
  engine #(
    .Ports    (Ports),
    .AddrWidth(AddrWidth),
    .DataWidth(DataWidth),
    .RowWidth (RowWidth),
    .ColWidth (ColWidth),
    .HeadWidth(HeadWidth),
    .BufWidth (BufWidth),
    .Writes   (Writes),
    .E        (E),
    .MulRegs  (MulRegs)
  ) u_engine (
    .clk_i,
    .rst_ni      (rst_nq),
    .cmd_valid_i (cmd_valid_q),
    .cmd_ready_o (cmd_ready),
    .cmd_op_i    (cmd_op_q),
    .cmd_rows_i  (cmd_rows_q),
    .cmd_cols_i  (cmd_cols_q),
    .cmd_addr_i  (cmd_addr_q),
    .cmd_a_i     (cmd_a_q),
    .cmd_dst_i   (cmd_dst_q),
    .cmd_bf16_i  (cmd_bf16_q),
    .cmd_wait_i  (cmd_wait_q),
    .cmd_heads_i (cmd_heads_q),
    .cmd_group_i (cmd_group_q),
    .cmd_t_i     (cmd_t_q),
    .cmd_cap_i   (cmd_cap_q),
    .cmd_stride_i(cmd_stride_q),
    .cmd_scale_i (cmd_scale_q),
    .go_i        (go_q),
    .done_o      (done),
    .x_buf_o     (x_buf),
    .x_addr_o    (x_addr),
    .x_data_i    (x_data_q),
    .w_valid_o   (w_valid),
    .w_buf_o     (w_buf),
    .w_row_o     (w_row),
    .w_mask_o    (w_mask),
    .w_data_o    (w_data),
    .w_bf16_o    (w_bf16),
    .ar_valid_o  (ar_valid),
    .ar_ready_i  (ar_ready_q),
    .ar_addr_o   (ar_addr),
    .ar_len_o    (ar_len),
    .ar_size_o   (ar_size),
    .ar_burst_o  (ar_burst),
    .r_valid_i   (r_valid_q),
    .r_ready_o   (r_ready),
    .r_data_i    (r_data_q),
    .r_resp_i    (r_resp_q),
    .err_o       (err)
  );

  always_ff @(posedge clk_i) begin
    cmd_ready_o <= cmd_ready;
    done_o      <= done;
    x_buf_o     <= x_buf;
    x_addr_o    <= x_addr;
    w_valid_o   <= w_valid;
    w_buf_o     <= w_buf;
    w_row_o     <= w_row;
    w_mask_o    <= w_mask;
    w_data_o    <= w_data;
    w_bf16_o    <= w_bf16;
    ar_valid_o  <= ar_valid;
    ar_addr_o   <= ar_addr;
    ar_len_o    <= ar_len;
    ar_size_o   <= ar_size;
    ar_burst_o  <= ar_burst;
    r_ready_o   <= r_ready;
    err_o       <= err;
  end

endmodule
