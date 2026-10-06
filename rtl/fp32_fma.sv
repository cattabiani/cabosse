// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// FP32 fused multiply-add, add and multiply, round to nearest even: CVFPU's
// fpnew_fma (D-033), with our operand order. One rounding per operation
// (docs/numerics.md, section 2). Golden model: golden.arith.fma, add, mul.
//   OpAdd: y = a + b
//   OpMul: y = a * b
//   any other op_i (fma): y = a * b + c
// A result leaves NumPipeRegs cycles after its operands enter; valid_o marks it.

`default_nettype none

module fp32_fma #(
  parameter int unsigned NumPipeRegs = 0
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  logic [1:0]  op_i,
  input  logic [31:0] a_i,
  input  logic [31:0] b_i,
  input  logic [31:0] c_i,
  output logic        valid_o,
  output logic [31:0] y_o
);

  localparam logic [1:0] OpAdd = 2'd1;
  localparam logic [1:0] OpMul = 2'd2;

  // fpnew_fma computes operands[0] * operands[1] + operands[2]; its ADD
  // replaces operands[0] with +1, its MUL operands[2] with -0.
  logic [2:0][31:0]      operands;
  fpnew_pkg::operation_e fpnew_op;
  always_comb begin
    operands = {c_i, b_i, a_i};
    fpnew_op = fpnew_pkg::FMADD;
    case (op_i)
      OpAdd: begin
        operands = {b_i, a_i, 32'h0000_0000};
        fpnew_op = fpnew_pkg::ADD;
      end
      OpMul: begin
        operands = {32'h0000_0000, b_i, a_i};
        fpnew_op = fpnew_pkg::MUL;
      end
      default: ;
    endcase
  end

  // Outputs of fpnew_fma we do not use: flags, tags, handshakes (it is always
  // ready, and its output is always accepted).
  fpnew_pkg::status_t unused_status;
  logic unused_in_ready, unused_extension_bit, unused_tag, unused_mask, unused_aux;
  logic unused_busy, unused_early_out_valid;

  fpnew_fma #(
    .FpFormat   (fpnew_pkg::FP32),
    .NumPipeRegs(NumPipeRegs),
    .PipeConfig (fpnew_pkg::DISTRIBUTED)
  ) u_fma (
    .clk_i,
    .rst_ni,
    .operands_i       (operands),
    .is_boxed_i       (3'b111),
    .rnd_mode_i       (fpnew_pkg::RNE),
    .op_i             (fpnew_op),
    .op_mod_i         (1'b0),
    .tag_i            (1'b0),
    .mask_i           (1'b1),
    .aux_i            (1'b0),
    .in_valid_i       (valid_i),
    .in_ready_o       (unused_in_ready),
    .flush_i          (1'b0),
    .result_o         (y_o),
    .status_o         (unused_status),
    .extension_bit_o  (unused_extension_bit),
    .tag_o            (unused_tag),
    .mask_o           (unused_mask),
    .aux_o            (unused_aux),
    .out_valid_o      (valid_o),
    .out_ready_i      (1'b1),
    .busy_o           (unused_busy),
    .reg_ena_i        ('0),  // no external register-enable override
    .early_out_valid_o(unused_early_out_valid)
  );

endmodule

`default_nettype wire
