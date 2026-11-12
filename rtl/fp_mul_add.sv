// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// y = a * b + c with one rounding to FP32 (D-038, D-039), for a and b in
// fp_product's M + 8-bit format (BF16 for M = 8, FP32 for M = 24) and an
// FP32 value c: fp_product with MulRegs registers inside (0 to 6), a
// register, then fp_add with AddRegs (3 or 4). A result leaves
// mul_add_latency(MulRegs, AddRegs) = MulRegs + 1 + AddRegs cycles after its
// operands enter; valid_o marks it. bf16_mac (M = 8) and fp32_fma (M = 24)
// are this unit. Ftz = 1 is the flush-to-zero variant (fp_product, fp_add).
// late_i (with the operands) takes c from c_late_i instead, MulRegs cycles
// after the operands enter, when c meets the product: the vector unit's
// reductions read their partial there (vec_unit.sv), so the partial's loop
// does not go through the multiply.

module fp_mul_add
  import fp_pkg::*;
#(
  parameter  int unsigned M       = 24,
  parameter  int unsigned MulRegs = 2,
  parameter  int unsigned AddRegs = AddLatency,
  parameter  bit          Ftz     = 1'b0,
  localparam int unsigned W       = prod_sig_w(M)
) (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         valid_i,
  input  logic [M+7:0] a_i,
  input  logic [M+7:0] b_i,
  input  logic [31:0]  c_i,
  input  logic         late_i,
  input  logic [31:0]  c_late_i,  // MulRegs cycles after valid_i, for late_i
  output logic         valid_o,
  output logic [31:0]  y_o
);

  logic [W+11:0] p_d, p_q;
  fp_product #(
    .M   (M),
    .Regs(MulRegs),
    .Ftz (Ftz)
  ) u_mul (
    .clk_i,
    .a_i,
    .b_i,
    .p_o(p_d)
  );
  always_ff @(posedge clk_i) p_q <= p_d;

  // The addend and valid, delayed MulRegs + 1 cycles to meet the product;
  // the last register takes c_late_i instead for a late c.
  logic [31:0] c_q[MulRegs+1];
  logic        valid_q[MulRegs+1];
  // late_q[MulRegs] goes unread (all of it at MulRegs = 0)
  // verilator lint_off UNUSEDSIGNAL
  logic        late_q[MulRegs+1];
  // verilator lint_on UNUSEDSIGNAL
  always_ff @(posedge clk_i) begin
    c_q[0]     <= MulRegs == 0 && late_i ? c_late_i : c_i;
    valid_q[0] <= rst_ni && valid_i;
    late_q[0]  <= late_i;
  end
  for (genvar i = 0; i < MulRegs; i++) begin : gen_delay
    always_ff @(posedge clk_i) begin
      c_q[i+1]     <= i + 1 == MulRegs && late_q[i] ? c_late_i : c_q[i];
      valid_q[i+1] <= rst_ni && valid_q[i];
      late_q[i+1]  <= late_q[i];
    end
  end

  fp_add #(
    .W   (W),
    .Ftz (Ftz),
    .Regs(AddRegs)
  ) u_add (
    .clk_i,
    .rst_ni,
    .valid_i(valid_q[MulRegs]),
    .a_i    (p_q),
    .b_i    (c_q[MulRegs]),
    .valid_o,
    .y_o
  );

endmodule
