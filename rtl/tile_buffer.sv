// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The vector unit's banked buffer, on logical vectors: N banks of Depth
// entries of W bits, one write and one read per bank and cycle. Element e
// is in bank e mod N, entry e / N. A vector of N lanes from element e
// (wr_addr_i, rd_addr_i) puts lane i in bank (i + e) mod N; bank B's entry
// is e / N + (B < e mod N), so no rotation carries the addresses.
//
// Registered for 250 MHz (M10): every input in a register, the write's
// banks (data rotated, enables, entries) in a register of their own, the
// read's data in the RAM's output register (URAM OREG, BRAM DOREG) and the
// rotated data in a register.
// - Write: wr_mask_i[i], wr_data_i[i] given in cycle c are in the RAM from
//   cycle c + 3 on (written at the end of c + 2).
// - Read: rd_data_o[i] is element rd_addr_i + i, RdLat = 4 cycles after
//   rd_addr_i. It sees the writes given 2 or more cycles before rd_addr_i,
//   not the later ones. RdLat is fixed by the registers; the parameter lets
//   the users name it (core.sv passes it to vec_unit and the engine).
module tile_buffer #(
  parameter  int unsigned RdLat = 4,
  parameter  int unsigned N     = 8,
  parameter  int unsigned Depth = 16,
  parameter  int unsigned W     = 64,
  localparam int unsigned AW    = Depth > 1 ? $clog2(Depth) : 1,
  localparam int unsigned SW    = N > 1 ? $clog2(N) : 1,
  localparam int unsigned EW    = AW + SW  // an element's address
) (
  input  logic                clk_i,
  input  logic [N-1:0]        wr_mask_i,
  input  logic [EW-1:0]       wr_addr_i,
  input  logic [N-1:0][W-1:0] wr_data_i,
  input  logic [EW-1:0]       rd_addr_i,
  output logic [N-1:0][W-1:0] rd_data_o
);

  if (RdLat != 4) begin : gen_lat_check
    $error("tile_buffer: RdLat is 4 (its registers)");
  end

  // Bank b's entry for a vector from element e.
  function automatic logic [AW-1:0] entry(logic [EW-1:0] e, int unsigned b);
    return e[EW-1:SW] + AW'(32'(b) < 32'(e[SW-1:0]));
  endfunction

  // Physical bank B takes logical lane (B - rot) mod N: a rotation by N - rot.
  function automatic logic [SW-1:0] back(logic [SW-1:0] rot);
    return rot == '0 ? '0 : SW'(N - 32'(rot));
  endfunction

  // Write: the inputs' register, then the banks'.
  logic [N-1:0]        wr_mask_q, wr_valid, wr_valid_q;
  logic [EW-1:0]       wr_addr_q;
  logic [N-1:0][W-1:0] wr_data_q, wr_phys, wr_phys_q;
  logic [N-1:0][AW-1:0] wr_entry_q;
  always_ff @(posedge clk_i) begin
    wr_mask_q <= wr_mask_i;
    wr_addr_q <= wr_addr_i;
    wr_data_q <= wr_data_i;
  end
  rotate_pipe #(
    .N   (N),
    .W   (W),
    .Regs(0)
  ) u_wr_rot (
    .clk_i,
    .in_i   (wr_data_q),
    .shift_i(back(wr_addr_q[SW-1:0])),
    .out_o  (wr_phys)
  );
  rotate_pipe #(
    .N   (N),
    .W   (1),
    .Regs(0)
  ) u_wr_mask_rot (
    .clk_i,
    .in_i   (wr_mask_q),
    .shift_i(back(wr_addr_q[SW-1:0])),
    .out_o  (wr_valid)
  );
  always_ff @(posedge clk_i) begin
    wr_valid_q <= wr_valid;
    wr_phys_q  <= wr_phys;
    for (int unsigned b = 0; b < N; b++) wr_entry_q[b] <= entry(wr_addr_q, b);
  end

  // Read: each bank's entry in a register, the RAM's, its output register,
  // the rotation's. The rotation's shift follows the data.
  logic [N-1:0][AW-1:0] rd_entry_q;
  logic [SW-1:0]        rd_rot_q[3];  // with rd_entry_q, rd_ram_q, rd_out_q
  logic [N-1:0][W-1:0]  rd_ram_q, rd_out_q, rd_rot;
  always_ff @(posedge clk_i) begin
    for (int unsigned b = 0; b < N; b++) rd_entry_q[b] <= entry(rd_addr_i, b);
    rd_rot_q[0] <= rd_addr_i[SW-1:0];
    rd_rot_q[1] <= rd_rot_q[0];
    rd_rot_q[2] <= rd_rot_q[1];
  end
  for (genvar b = 0; b < N; b++) begin : gen_bank
    logic [W-1:0] mem_q[Depth];
    always_ff @(posedge clk_i) begin
      if (wr_valid_q[b]) mem_q[wr_entry_q[b]] <= wr_phys_q[b];
      rd_ram_q[b] <= mem_q[rd_entry_q[b]];
      rd_out_q[b] <= rd_ram_q[b];  // the RAM's output register
    end
  end
  rotate_pipe #(
    .N   (N),
    .W   (W),
    .Regs(0)
  ) u_rd_rot (
    .clk_i,
    .in_i   (rd_out_q),
    .shift_i(rd_rot_q[2]),
    .out_o  (rd_rot)
  );
  always_ff @(posedge clk_i) rd_data_o <= rd_rot;
endmodule
