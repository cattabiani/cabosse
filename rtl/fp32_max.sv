// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// FP32 max(a, b): CVFPU's fpnew_noncomp as MAX (D-033), IEEE 754-2019
// maximumNumber (D-037): a NaN operand gives the other, two NaNs give the
// canonical NaN, max(-0, +0) = +0. Exact. Golden model: golden.arith.maximum.
// A result leaves NumPipeRegs cycles after its operands enter; valid_o marks it.

module fp32_max #(
  parameter int unsigned NumPipeRegs = 0
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  logic [31:0] a_i,
  input  logic [31:0] b_i,
  output logic        valid_o,
  output logic [31:0] y_o
);

  // Outputs of fpnew_noncomp we do not use: flags, class, tags, handshakes
  // (it is always ready, and its output is always accepted).
  fpnew_pkg::status_t    unused_status;
  fpnew_pkg::classmask_e unused_class_mask;
  logic unused_in_ready, unused_extension_bit, unused_is_class, unused_tag, unused_mask;
  logic unused_aux, unused_busy, unused_early_out_valid;

  fpnew_noncomp #(
    .FpFormat   (fpnew_pkg::FP32),
    .NumPipeRegs(NumPipeRegs),
    .PipeConfig (fpnew_pkg::DISTRIBUTED)
  ) u_noncomp (
    .clk_i,
    .rst_ni,
    .operands_i       ({b_i, a_i}),
    .is_boxed_i       (2'b11),
    .rnd_mode_i       (fpnew_pkg::RTZ),  // with MINMAX: RTZ selects MAX
    .op_i             (fpnew_pkg::MINMAX),
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
    .class_mask_o     (unused_class_mask),
    .is_class_o       (unused_is_class),
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
