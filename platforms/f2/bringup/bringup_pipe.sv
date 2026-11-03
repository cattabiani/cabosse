// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Stages 2-entry FIFOs (rtl/stream_fifo.sv) in a row: a valid/ready stream
// with every signal registered at each stage, at full rate, for a long wire
// (bringup_host.sv's way to the shell's PCIS).

module bringup_pipe #(
  parameter int unsigned Width  = 32,
  parameter int unsigned Stages = 2
) (
  input  logic             clk_i,
  input  logic             rst_ni,
  input  logic             in_valid_i,
  output logic             in_ready_o,
  input  logic [Width-1:0] in_data_i,
  output logic             out_valid_o,
  input  logic             out_ready_i,
  output logic [Width-1:0] out_data_o
);

  if (Stages == 0) begin : gen_bad_stages
    $error("bringup_pipe: Stages must be at least 1");
  end

  logic [Stages:0]            valid, ready;
  logic [Stages:0][Width-1:0] data;
  assign valid[0]      = in_valid_i;
  assign in_ready_o    = ready[0];
  assign data[0]       = in_data_i;
  assign out_valid_o   = valid[Stages];
  assign ready[Stages] = out_ready_i;
  assign out_data_o    = data[Stages];

  for (genvar s = 0; s < Stages; s++) begin : gen_stage
    // verilator lint_off PINCONNECTEMPTY
    // The count goes unread.
    stream_fifo #(.Width(Width), .Depth(2)) u_fifo (
      .clk_i, .rst_ni,
      .in_valid_i (valid[s]), .in_ready_o (ready[s]), .in_data_i (data[s]),
      .out_valid_o(valid[s+1]), .out_ready_i(ready[s+1]), .out_data_o(data[s+1]), .count_o()
    );
    // verilator lint_on PINCONNECTEMPTY
  end

endmodule
