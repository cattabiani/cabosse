// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// N banks of Depth entries of W bits, one write and one read per bank and
// cycle, whose data cross the banks through rotations of Regs registers
// (rotate_pipe.sv, D-051): the engine's tile buffer (docs/architecture.md,
// "Tile buffer banks"), and with Regs = 0 the vector unit's (tile_buffer.sv).
// Physical bank B holds logical entry (B - rot) mod N. Addresses and write
// enables are per physical bank, from the caller, so no rotation carries
// them: each bank's come from a register of its own when Regs > 0.
//
// Write: bank B writes wr_data_i[(B + wr_shift) mod N] (wr_shift = N - rot
// mod N), Regs cycles after the data and the shift, with the wr_valid_i[B]
// and wr_addr_i[B] given that cycle (Regs > 0: registered here first, so
// given with the data). Read: bank B reads rd_addr_i[B]; 1 + Regs cycles
// later rd_data_o[i] is what bank (i + rd_rot_i) mod N read. A read sees the
// writes of earlier cycles only. Each shift comes as Groups copies.

module tile_buffer_pipe #(
  parameter  int unsigned N      = 8,
  parameter  int unsigned Depth  = 16,
  parameter  int unsigned W      = 64,
  parameter  int unsigned Regs   = 2,
  parameter  int unsigned Groups = 1,
  localparam int unsigned AW     = Depth > 1 ? $clog2(Depth) : 1,
  localparam int unsigned SW     = N > 1 ? $clog2(N) : 1
) (
  input  logic                      clk_i,
  input  logic [N-1:0]              wr_valid_i,
  input  logic [N-1:0][AW-1:0]      wr_addr_i,
  input  logic [N-1:0][W-1:0]       wr_data_i,
  input  logic [Groups-1:0][SW-1:0] wr_shift_i,
  input  logic [N-1:0][AW-1:0]      rd_addr_i,
  input  logic [Groups-1:0][SW-1:0] rd_rot_i,
  output logic [N-1:0][W-1:0]       rd_data_o
);

  logic [N-1:0]              wr_valid;
  logic [N-1:0][AW-1:0]      wr_addr;
  logic [N-1:0][W-1:0]       wr_phys, rd_phys_q;
  logic [Groups-1:0][SW-1:0] rd_rot_q;

  rotate_pipe #(
    .N     (N),
    .W     (W),
    .Regs  (Regs),
    .Groups(Groups)
  ) u_wr_rot (
    .clk_i,
    .in_i   (wr_data_i),
    .shift_i(wr_shift_i),
    .out_o  (wr_phys)
  );

  // A bank's enable and address wait for its data, in its own registers.
  if (Regs == 0) begin : gen_wr_now
    assign wr_valid = wr_valid_i;
    assign wr_addr  = wr_addr_i;
  end else begin : gen_wr_later
    logic [N-1:0]         valid_q[Regs];
    logic [N-1:0][AW-1:0] addr_q[Regs];
    always_ff @(posedge clk_i) begin
      valid_q[0] <= wr_valid_i;
      addr_q[0]  <= wr_addr_i;
      for (int unsigned r = 1; r < Regs; r++) begin
        valid_q[r] <= valid_q[r-1];
        addr_q[r]  <= addr_q[r-1];
      end
    end
    assign wr_valid = valid_q[Regs-1];
    assign wr_addr  = addr_q[Regs-1];
  end

  for (genvar b = 0; b < N; b++) begin : gen_bank
    logic [W-1:0] mem_q[Depth];
    always_ff @(posedge clk_i) begin
      if (wr_valid[b]) mem_q[wr_addr[b]] <= wr_phys[b];
      rd_phys_q[b] <= mem_q[rd_addr_i[b]];
    end
  end

  always_ff @(posedge clk_i) rd_rot_q <= rd_rot_i;

  rotate_pipe #(
    .N     (N),
    .W     (W),
    .Regs  (Regs),
    .Groups(Groups)
  ) u_rd_rot (
    .clk_i,
    .in_i   (rd_phys_q),
    .shift_i(rd_rot_q),
    .out_o  (rd_data_o)
  );

endmodule
