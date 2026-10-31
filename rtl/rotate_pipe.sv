// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Rotation of N entries of W bits through Regs registers (D-051):
// out_o[i] = in_i[(i + shift) mod N], Regs cycles after in_i and shift_i,
// for shift < N. log2(N) stages, stage b rotating by 2^b mod N when bit b
// of the shift is set (rotations add up, so any N works). The last stages
// move entries farthest across the banks, so the registers go before each
// of the last Regs stages: the first stages stay local, and each long hop
// starts from a register. The shift comes as Groups copies, copy g for
// entries g * N / Groups ... (each copy drives only its own entries'
// multiplexers), registered along with the data. Regs = 0 is
// combinational (clk_i unused).

module rotate_pipe #(
  parameter  int unsigned N      = 8,
  parameter  int unsigned W      = 1,
  parameter  int unsigned Regs   = 0,
  parameter  int unsigned Groups = 1,  // dividing N
  localparam int unsigned SW     = N > 1 ? $clog2(N) : 1
) (
  // verilator lint_off UNUSEDSIGNAL
  input  logic                      clk_i,  // Regs = 0: unused
  // verilator lint_on UNUSEDSIGNAL
  input  logic [N-1:0][W-1:0]       in_i,
  input  logic [Groups-1:0][SW-1:0] shift_i,
  output logic [N-1:0][W-1:0]       out_o
);

  localparam int unsigned GS = N / Groups;  // entries per copy of the shift

  if (N % Groups != 0 || SW <= Regs) begin : gen_bad_params
    $error("rotate_pipe: Groups must divide N, and N needs more than Regs stages");
  end

  for (genvar b = 0; b < SW; b++) begin : gen_stage
    localparam int unsigned Amount = (1 << b) % N;
    logic [N-1:0][W-1:0]       in, rotated, out;
    // verilator lint_off UNUSEDSIGNAL
    logic [Groups-1:0][SW-1:0] sh_in, sh_out;  // the last stage's sh_out drives nothing
    // verilator lint_on UNUSEDSIGNAL
    if (b == 0) begin : gen_first
      assign in    = in_i;
      assign sh_in = shift_i;
    end else begin : gen_next
      assign in    = gen_stage[b-1].out;
      assign sh_in = gen_stage[b-1].sh_out;
    end
    for (genvar i = 0; i < N; i++) begin : gen_entry
      assign rotated[i] = sh_in[i/GS][b] ? in[(i + Amount) % N] : in[i];
    end
    if (b + 1 < SW && b + 1 + Regs >= SW) begin : gen_reg
      always_ff @(posedge clk_i) begin
        out    <= rotated;
        sh_out <= sh_in;
      end
    end else begin : gen_wire
      assign out    = rotated;
      assign sh_out = sh_in;
    end
  end
  assign out_o = gen_stage[SW-1].out;

endmodule
