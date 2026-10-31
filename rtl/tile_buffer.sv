// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The vector unit's banked buffer (docs/architecture.md, "Tile buffer
// banks"): tile_buffer_pipe.sv without registers, on logical vectors. N
// banks of Depth entries of W bits, one write and one read per bank and
// cycle; physical bank B holds logical entry (B - rot) mod N, so a mapping
// bank = (index + rot) mod N is one rotation each way.
//
// Write: logical entry i (wr_valid_i[i], wr_addr_i[i], wr_data_i[i]) goes to
// bank (i + wr_rot_i) mod N. Read: bank (i + rd_rot_i) mod N reads
// rd_addr_i[i]; the cycle after, rd_data_o[i] is what it read (a
// synchronous RAM: a write in the same cycle to the same entry is not seen).

module tile_buffer #(
  parameter  int unsigned N     = 8,
  parameter  int unsigned Depth = 16,
  parameter  int unsigned W     = 64,
  localparam int unsigned AW    = Depth > 1 ? $clog2(Depth) : 1,
  localparam int unsigned SW    = N > 1 ? $clog2(N) : 1
) (
  input  logic                 clk_i,
  input  logic [N-1:0]         wr_valid_i,
  input  logic [N-1:0][AW-1:0] wr_addr_i,
  input  logic [N-1:0][W-1:0]  wr_data_i,
  input  logic [SW-1:0]        wr_rot_i,
  input  logic [N-1:0][AW-1:0] rd_addr_i,
  input  logic [SW-1:0]        rd_rot_i,
  output logic [N-1:0][W-1:0]  rd_data_o
);

  typedef struct packed {
    logic          valid;
    logic [AW-1:0] addr;
  } entry_t;

  // Physical bank B takes logical entry (B - rot) mod N: a rotation by N - rot.
  function automatic logic [SW-1:0] back(logic [SW-1:0] rot);
    return rot == '0 ? '0 : SW'(N - 32'(rot));
  endfunction

  entry_t [N-1:0]         wr_log, wr_phys;
  logic   [N-1:0]         wr_valid;
  logic   [N-1:0][AW-1:0] wr_addr, rd_addr;

  for (genvar i = 0; i < N; i++) begin : gen_entry
    assign wr_log[i]   = '{valid: wr_valid_i[i], addr: wr_addr_i[i]};
    assign wr_valid[i] = wr_phys[i].valid;
    assign wr_addr[i]  = wr_phys[i].addr;
  end

  rotate_pipe #(
    .N(N),
    .W($bits(entry_t))
  ) u_wr_rot (
    .clk_i,
    .in_i   (wr_log),
    .shift_i(back(wr_rot_i)),
    .out_o  (wr_phys)
  );

  rotate_pipe #(
    .N(N),
    .W(AW)
  ) u_rd_addr_rot (
    .clk_i,
    .in_i   (rd_addr_i),
    .shift_i(back(rd_rot_i)),
    .out_o  (rd_addr)
  );

  tile_buffer_pipe #(
    .N    (N),
    .Depth(Depth),
    .W    (W),
    .Regs (0)
  ) u_banks (
    .clk_i,
    .wr_valid_i(wr_valid),
    .wr_addr_i (wr_addr),
    .wr_data_i (wr_data_i),
    .wr_shift_i(back(wr_rot_i)),
    .rd_addr_i (rd_addr),
    .rd_rot_i  (rd_rot_i),
    .rd_data_o
  );

endmodule
