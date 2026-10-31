// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The engine's tile buffer (docs/architecture.md, "Tile buffer banks"):
// tile_buffer.sv (the vector unit's) with its rotations through Regs
// registers (rotate_pipe.sv, D-051), and each bank's read address from the
// caller. N banks of Depth
// entries of W bits, one write and one read per bank and cycle. The caller
// works on "logical" vectors; physical bank B holds logical entry (B - rot)
// mod N, so a mapping bank = (index + rot) mod N is one rotation each way.
//
// Write: logical entry i (wr_valid_i[i], wr_addr_i[i], wr_data_i[i]) goes to
// bank (i + wr_rot_i) mod N, written at the end of the cycle Regs cycles
// later. Read: bank B reads rd_addr_i[B] (physical: the caller computes each
// bank's address); 1 + Regs cycles later rd_data_o[i] is what bank (i +
// rd_rot_i) mod N read. A read sees the writes of earlier cycles only. The
// rotations come as Groups copies (rotate_pipe.sv), for the fan-out.

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
  input  logic [Groups-1:0][SW-1:0] wr_rot_i,
  input  logic [N-1:0][AW-1:0]      rd_addr_i,
  input  logic [Groups-1:0][SW-1:0] rd_rot_i,
  output logic [N-1:0][W-1:0]       rd_data_o
);

  typedef struct packed {
    logic          valid;
    logic [AW-1:0] addr;
    logic [W-1:0]  data;
  } entry_t;

  // Physical bank B takes logical entry (B - rot) mod N: a rotation by N - rot.
  function automatic logic [SW-1:0] back(logic [SW-1:0] rot);
    return rot == '0 ? '0 : SW'(N - 32'(rot));
  endfunction

  entry_t [N-1:0]            wr_log, wr_phys;
  logic   [Groups-1:0][SW-1:0] wr_back, rd_rot_q;
  logic   [N-1:0][W-1:0]     rd_phys_q;

  for (genvar i = 0; i < N; i++) begin : gen_entry
    assign wr_log[i] = '{valid: wr_valid_i[i], addr: wr_addr_i[i], data: wr_data_i[i]};
  end
  for (genvar g = 0; g < Groups; g++) begin : gen_back
    assign wr_back[g] = back(wr_rot_i[g]);
  end

  rotate_pipe #(
    .N     (N),
    .W     ($bits(entry_t)),
    .Regs  (Regs),
    .Groups(Groups)
  ) u_wr_rot (
    .clk_i,
    .in_i   (wr_log),
    .shift_i(wr_back),
    .out_o  (wr_phys)
  );

  for (genvar b = 0; b < N; b++) begin : gen_bank
    logic [W-1:0] mem_q[Depth];
    always_ff @(posedge clk_i) begin
      if (wr_phys[b].valid) mem_q[wr_phys[b].addr] <= wr_phys[b].data;
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
