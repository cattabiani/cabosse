// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Timing harness for the whole vector unit (rtl/vec_unit.sv) at the core's
// shape: 16 lanes, 22 buffers of 1024 elements, P with a second read port
// (core.sv). Fewer lanes and a shorter Len only for the Yosys check. Every
// input and output registered, so every path is register to register:
// M10 step 2, the unit at 250 MHz before a full build. Out of context, these
// registers go wherever the placer likes; in the core the engine's writes
// and x reads come from across the chip. Not part of the design.

module vec_path #(
  parameter  int unsigned Lanes     = 16,
  parameter  int unsigned Len       = 1024,
  localparam int unsigned Bufs      = 22,
  localparam logic [31:0] Bf16Mask  = 32'h4644,
  localparam logic [31:0] Read2Mask = 32'h200,  // P (core.sv's BufP = 9)
  localparam int unsigned AddrW     = 20,
  localparam int unsigned MemAddrW  = 34,
  localparam int unsigned KvPorts   = 32,
  localparam int unsigned ChanW     = 29,
  localparam int unsigned RdLat     = 4,
  localparam int unsigned DataWidth = Lanes * 16,
  localparam int unsigned BufW      = $clog2(Bufs)
) (
  input  logic                         clk_i,
  input  logic                         rst_ni,
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
  input  logic [AddrW-1:0]             t_i,
  input  logic [AddrW-1:0]             stride_i,
  output logic                         done_o,
  output logic                         err_o,
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
  input  logic                         ext_w_valid_i,
  input  logic [BufW-1:0]              ext_w_buf_i,
  input  logic [AddrW-1:0]             ext_w_addr_i,
  input  logic [Lanes-1:0]             ext_w_mask_i,
  input  logic [Lanes-1:0][31:0]       ext_w_data_i,
  input  logic [BufW-1:0]              ext_r_buf_i,
  input  logic [AddrW-1:0]             ext_r_addr_i,
  output logic [Lanes-1:0][31:0]       ext_r_data_o,
  input  logic [AddrW-1:0]             ext_r2_addr_i,
  output logic [Lanes-1:0][31:0]       ext_r2_data_o
);

  logic                   rst_nq, cmd_valid_q, ar_ready_q, r_valid_q, aw_ready_q, w_ready_q;
  logic                   b_valid_q, ext_w_valid_q;
  logic [4:0]             cmd_op_q;
  logic [BufW-1:0]        cmd_dst_q, cmd_a_q, cmd_b_q, cmd_c_q, ext_w_buf_q, ext_r_buf_q;
  logic [7:0]             cmd_flags_q;
  logic [AddrW-1:0]       cmd_n_q, cmd_m_q, cmd_cap_q, token_q, t_q, stride_q;
  logic [AddrW-1:0]       ext_w_addr_q, ext_r_addr_q, ext_r2_addr_q;
  logic [31:0]            cmd_scalar_q;
  logic [MemAddrW-1:0]    cmd_addr_q;
  logic [DataWidth-1:0]   r_data_q;
  logic [1:0]             r_resp_q, b_resp_q;
  logic [Lanes-1:0]       ext_w_mask_q;
  logic [Lanes-1:0][31:0] ext_w_data_q;
  always_ff @(posedge clk_i) begin
    rst_nq        <= rst_ni;
    cmd_valid_q   <= cmd_valid_i;
    cmd_op_q      <= cmd_op_i;
    cmd_dst_q     <= cmd_dst_i;
    cmd_a_q       <= cmd_a_i;
    cmd_b_q       <= cmd_b_i;
    cmd_c_q       <= cmd_c_i;
    cmd_flags_q   <= cmd_flags_i;
    cmd_n_q       <= cmd_n_i;
    cmd_m_q       <= cmd_m_i;
    cmd_scalar_q  <= cmd_scalar_i;
    cmd_cap_q     <= cmd_cap_i;
    cmd_addr_q    <= cmd_addr_i;
    token_q       <= token_i;
    t_q           <= t_i;
    stride_q      <= stride_i;
    ar_ready_q    <= ar_ready_i;
    r_valid_q     <= r_valid_i;
    r_data_q      <= r_data_i;
    r_resp_q      <= r_resp_i;
    aw_ready_q    <= aw_ready_i;
    w_ready_q     <= w_ready_i;
    b_valid_q     <= b_valid_i;
    b_resp_q      <= b_resp_i;
    ext_w_valid_q <= ext_w_valid_i;
    ext_w_buf_q   <= ext_w_buf_i;
    ext_w_addr_q  <= ext_w_addr_i;
    ext_w_mask_q  <= ext_w_mask_i;
    ext_w_data_q  <= ext_w_data_i;
    ext_r_buf_q   <= ext_r_buf_i;
    ext_r_addr_q  <= ext_r_addr_i;
    ext_r2_addr_q <= ext_r2_addr_i;
  end

  logic                   cmd_ready, done, err, ar_valid, r_ready, aw_valid, w_valid;
  logic                   w_last, b_ready;
  logic [MemAddrW-1:0]    ar_addr, aw_addr;
  logic [7:0]             ar_len, aw_len;
  logic [2:0]             ar_size, aw_size;
  logic [1:0]             ar_burst, aw_burst;
  logic [DataWidth-1:0]   w_data;
  logic [DataWidth/8-1:0] w_strb;
  logic [Lanes-1:0][31:0] ext_r_data, ext_r2_data;
  vec_unit #(
    .RdLat    (RdLat),
    .Lanes    (Lanes),
    .Bufs     (Bufs),
    .Len      (Len),
    .Bf16Mask (Bf16Mask),
    .Read2Mask(Read2Mask),
    .AddrW    (AddrW),
    .MemAddrW (MemAddrW),
    .KvPorts  (KvPorts),
    .ChanW    (ChanW)
  ) u_vec (
    .clk_i,
    .rst_ni        (rst_nq),
    .cmd_valid_i   (cmd_valid_q),
    .cmd_ready_o   (cmd_ready),
    .cmd_op_i      (cmd_op_q),
    .cmd_dst_i     (cmd_dst_q),
    .cmd_a_i       (cmd_a_q),
    .cmd_b_i       (cmd_b_q),
    .cmd_c_i       (cmd_c_q),
    .cmd_flags_i   (cmd_flags_q),
    .cmd_n_i       (cmd_n_q),
    .cmd_m_i       (cmd_m_q),
    .cmd_scalar_i  (cmd_scalar_q),
    .cmd_cap_i     (cmd_cap_q),
    .cmd_addr_i    (cmd_addr_q),
    .token_i       (token_q),
    .t_i           (t_q),
    .stride_i      (stride_q),
    .done_o        (done),
    .err_o         (err),
    .ar_valid_o    (ar_valid),
    .ar_ready_i    (ar_ready_q),
    .ar_addr_o     (ar_addr),
    .ar_len_o      (ar_len),
    .ar_size_o     (ar_size),
    .ar_burst_o    (ar_burst),
    .r_valid_i     (r_valid_q),
    .r_ready_o     (r_ready),
    .r_data_i      (r_data_q),
    .r_resp_i      (r_resp_q),
    .aw_valid_o    (aw_valid),
    .aw_ready_i    (aw_ready_q),
    .aw_addr_o     (aw_addr),
    .aw_len_o      (aw_len),
    .aw_size_o     (aw_size),
    .aw_burst_o    (aw_burst),
    .w_valid_o     (w_valid),
    .w_ready_i     (w_ready_q),
    .w_data_o      (w_data),
    .w_strb_o      (w_strb),
    .w_last_o      (w_last),
    .b_valid_i     (b_valid_q),
    .b_ready_o     (b_ready),
    .b_resp_i      (b_resp_q),
    .ext_w_valid_i (ext_w_valid_q),
    .ext_w_buf_i   (ext_w_buf_q),
    .ext_w_addr_i  (ext_w_addr_q),
    .ext_w_mask_i  (ext_w_mask_q),
    .ext_w_data_i  (ext_w_data_q),
    .ext_r_valid_i (1'b1),  // as in core.sv
    .ext_r_buf_i   (ext_r_buf_q),
    .ext_r_addr_i  (ext_r_addr_q),
    .ext_r_data_o  (ext_r_data),
    .ext_r2_addr_i (ext_r2_addr_q),
    .ext_r2_data_o (ext_r2_data)
  );

  always_ff @(posedge clk_i) begin
    cmd_ready_o   <= cmd_ready;
    done_o        <= done;
    err_o         <= err;
    ar_valid_o    <= ar_valid;
    ar_addr_o     <= ar_addr;
    ar_len_o      <= ar_len;
    ar_size_o     <= ar_size;
    ar_burst_o    <= ar_burst;
    r_ready_o     <= r_ready;
    aw_valid_o    <= aw_valid;
    aw_addr_o     <= aw_addr;
    aw_len_o      <= aw_len;
    aw_size_o     <= aw_size;
    aw_burst_o    <= aw_burst;
    w_valid_o     <= w_valid;
    w_data_o      <= w_data;
    w_strb_o      <= w_strb;
    w_last_o      <= w_last;
    b_ready_o     <= b_ready;
    ext_r_data_o  <= ext_r_data;
    ext_r2_data_o <= ext_r2_data;
  end

endmodule
