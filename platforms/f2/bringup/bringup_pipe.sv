// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Stages register slices in a row: a valid/ready stream at full rate with
// every signal registered at each stage, in flops only, for a long wire
// (bringup_host.sv's way to the shell's PCIS, across an SLR). A stage is a
// skid buffer: its output register, and a second register that takes the
// beat arriving while the output waits; its ready is that register empty.
// busy_o: a beat inside, in any stage (registered).

module bringup_pipe #(
  parameter int unsigned Width  = 32,
  parameter int unsigned Stages = 1
) (
  input  logic             clk_i,
  input  logic             rst_ni,
  input  logic             in_valid_i,
  output logic             in_ready_o,
  input  logic [Width-1:0] in_data_i,
  output logic             busy_o,
  output logic             out_valid_o,
  input  logic             out_ready_i,
  output logic [Width-1:0] out_data_o
);

  if (Stages == 0) begin : gen_bad_stages
    $error("bringup_pipe: Stages must be at least 1");
  end

  logic [Stages:0]            valid, ready;
  logic [Stages:0][Width-1:0] data;
  logic [Stages-1:0]          held;  // a stage holds a beat
  assign busy_o        = |held;
  assign valid[0]      = in_valid_i;
  assign in_ready_o    = ready[0];
  assign data[0]       = in_data_i;
  assign out_valid_o   = valid[Stages];
  assign ready[Stages] = out_ready_i;
  assign out_data_o    = data[Stages];

  for (genvar s = 0; s < Stages; s++) begin : gen_stage
    logic             out_valid_q, skid_valid_q;
    logic [Width-1:0] out_q, skid_q;
    logic             take;  // the output register loads this cycle
    assign take       = ready[s+1] || !out_valid_q;
    assign ready[s]   = !skid_valid_q;
    assign valid[s+1] = out_valid_q;
    assign data[s+1]  = out_q;
    assign held[s]    = out_valid_q || skid_valid_q;

    always_ff @(posedge clk_i) begin
      if (!rst_ni) begin
        out_valid_q  <= 1'b0;
        skid_valid_q <= 1'b0;
      end else if (take) begin
        out_valid_q  <= skid_valid_q || valid[s];
        skid_valid_q <= 1'b0;
      end else if (valid[s] && !skid_valid_q) begin
        skid_valid_q <= 1'b1;
      end
    end

    always_ff @(posedge clk_i) begin
      if (take) out_q <= skid_valid_q ? skid_q : data[s];
      if (!take && !skid_valid_q) skid_q <= data[s];
    end
  end

endmodule
