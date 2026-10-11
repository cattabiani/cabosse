// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The engine (D-027, docs/architecture.md): L = Ports * LanesPerPort lanes in
// lockstep, each port's DMA feeding its LanesPerPort lanes. This step runs
// MATVEC: dst[r] = dot(W[r], x) for the rows r < rows of a matrix of `cols`
// columns, W stored in the lanes' order ("Weight layout") at the same
// offset `addr` in every port's channel. Rows go in passes of L; in each pass
// port p reads its group of LanesPerPort rows, ceil(cols / E) beats.
//
// Three stages, each with its own queue of the commands taken, so a command
// starts while earlier ones finish (PLAN.md, M5):
// - Issue: per pass, one DMA command to each port with rows in that pass.
//   It runs ahead of the lanes as far as the DMAs' FIFOs allow, so the next
//   matrix's reads start during the current one.
// - Step: one beat to every lane per cycle, when every port with rows in the
//   pass has a beat and every lane is ready; x comes from the read port, E
//   values broadcast. Back-to-back commands leave no gap.
// - Drain: the lanes' results, Writes lanes at a time, to the write port,
//   rounded to BF16 for a BF16 destination. Lanes past the matrix's last row
//   (a partial last group) are popped and not written; lanes of ports with no
//   rows in a pass have no result. done_o marks a command's last write.
// The engine does not order a command after an earlier one whose output it
// reads: the controller waits for done_o (M7).
//
// Commands need rows >= 1 and cols >= 1.
//
// Read port: x_data_i is the E values of group x_addr_o (elements
// E * x_addr_o ... + E - 1) of buffer x_buf_o as presented in the previous
// cycle (a synchronous RAM), zeros past the buffer's end. Write port: when
// w_valid_o, slot s with w_mask_o[s] holds row w_row_o + s of buffer w_buf_o:
// FP32, or the BF16 bits in its low half when w_bf16_o. It never waits.

module engine
  import fp_pkg::*;
#(
  parameter  int unsigned Ports        = 32,
  parameter  int unsigned AddrWidth    = 34,   // F2's HBM: 16 GiB
  parameter  int unsigned DataWidth    = 256,  // an HBM port's beat
  parameter  int unsigned Depth        = 64,   // each port's FIFO, beats
  parameter  int unsigned RowWidth     = 20,   // rows: Qwen2.5's classifier needs 18 bits
  parameter  int unsigned ColWidth     = 16,   // columns
  parameter  int unsigned BufWidth     = 8,    // buffer ids (docs/architecture.md, "Encoding")
  parameter  int unsigned Writes       = 8,    // results per cycle on the write port
  parameter  int unsigned CmdDepth     = 4,    // commands queued per stage
  parameter  int unsigned E            = 4,    // pairs per lane per cycle
  parameter  int unsigned A            = 16,   // partial sums per row (D-027)
  localparam int unsigned LanesPerPort = DataWidth / (16 * E),  // a beat holds E weights each
  localparam int unsigned L            = Ports * LanesPerPort,
  localparam int unsigned Groups       = (L + Writes - 1) / Writes  // write groups per pass
) (
  input  logic                              clk_i,
  input  logic                              rst_ni,
  // Command
  input  logic                              cmd_valid_i,
  output logic                              cmd_ready_o,
  input  logic [RowWidth-1:0]               cmd_rows_i,
  input  logic [ColWidth-1:0]               cmd_cols_i,
  input  logic [AddrWidth-1:0]              cmd_addr_i,  // beat-aligned, in every channel
  input  logic [BufWidth-1:0]               cmd_a_i,     // x
  input  logic [BufWidth-1:0]               cmd_dst_i,
  input  logic                              cmd_bf16_i,  // round the results to BF16
  output logic                              done_o,      // a command's last write
  // x read port
  output logic [BufWidth-1:0]               x_buf_o,
  output logic [ColWidth-1:0]               x_addr_o,
  input  logic [E-1:0][15:0]                x_data_i,
  // Result write port
  output logic                              w_valid_o,
  output logic [BufWidth-1:0]               w_buf_o,
  output logic [RowWidth-1:0]               w_row_o,
  output logic [Writes-1:0]                 w_mask_o,
  output logic [Writes-1:0][31:0]           w_data_o,
  output logic                              w_bf16_o,
  // AXI read ports (port_dma.sv)
  output logic [Ports-1:0]                  ar_valid_o,
  input  logic [Ports-1:0]                  ar_ready_i,
  output logic [Ports-1:0][AddrWidth-1:0]   ar_addr_o,
  output logic [Ports-1:0][7:0]             ar_len_o,
  output logic [Ports-1:0][2:0]             ar_size_o,
  output logic [Ports-1:0][1:0]             ar_burst_o,
  input  logic [Ports-1:0]                  r_valid_i,
  output logic [Ports-1:0]                  r_ready_o,
  input  logic [Ports-1:0][DataWidth-1:0]   r_data_i,
  input  logic [Ports-1:0][1:0]             r_resp_i,
  output logic [Ports-1:0]                  err_o        // sticky, per port
);

  localparam int unsigned BeatBytes = DataWidth / 8;
  localparam int unsigned PW        = $clog2(Ports + 1);  // a count of ports
  localparam int unsigned GW        = Groups > 1 ? $clog2(Groups) : 1;

  if (DataWidth % (16 * E) != 0 || LanesPerPort == 0) begin : gen_bad_width
    $error("engine: a beat must hold E BF16 weights for a whole number of lanes");
  end
  if ((Writes & (Writes - 1)) != 0) begin : gen_bad_writes
    $error("engine: Writes must be a power of two");
  end

  typedef struct packed {
    logic [RowWidth-1:0]  rows;
    logic [ColWidth-1:0]  cols;
    logic [AddrWidth-1:0] addr;
    logic [BufWidth-1:0]  a;
    logic [BufWidth-1:0]  dst;
    logic                 bf16;
  } cmd_t;

  // Groups of LanesPerPort rows; beats per row.
  function automatic logic [RowWidth-1:0] n_groups(logic [RowWidth-1:0] rows);
    return rows / RowWidth'(LanesPerPort) + RowWidth'(rows % RowWidth'(LanesPerPort) != '0);
  endfunction
  function automatic logic [ColWidth-1:0] n_beats(logic [ColWidth-1:0] cols);
    return cols / ColWidth'(E) + ColWidth'(cols % ColWidth'(E) != '0);
  endfunction
  // Groups of this pass: the ports with rows in it.
  function automatic logic [PW-1:0] pass_ports(logic [RowWidth-1:0] groups);
    return groups > RowWidth'(Ports) ? PW'(Ports) : PW'(groups);
  endfunction

  // --- Command queues --------------------------------------------------------

  cmd_t cmd;
  assign cmd = '{rows: cmd_rows_i, cols: cmd_cols_i, addr: cmd_addr_i, a: cmd_a_i,
                 dst: cmd_dst_i, bf16: cmd_bf16_i};

  logic [2:0] q_in_ready, q_valid, q_pop;  // issue, step, drain
  cmd_t       q_head[3];
  assign cmd_ready_o = &q_in_ready;

  for (genvar i = 0; i < 3; i++) begin : gen_queue
    stream_fifo #(
      .Width($bits(cmd_t)),
      .Depth(CmdDepth)
    ) u_queue (
      .clk_i,
      .rst_ni,
      .in_valid_i (cmd_valid_i && cmd_ready_o),
      .in_ready_o (q_in_ready[i]),
      .in_data_i  (cmd),
      .out_valid_o(q_valid[i]),
      .out_ready_i(q_pop[i]),
      .out_data_o (q_head[i])
    );
  end

  // --- DMAs --------------------------------------------------------------------

  logic [Ports-1:0]                dma_cmd_valid, dma_cmd_ready;
  logic [Ports-1:0]                dma_out_valid, dma_out_ready;
  logic [Ports-1:0][DataWidth-1:0] dma_out_data;
  logic [AddrWidth-1:0]            iss_addr_q;
  logic [ColWidth-1:0]             iss_beats_q;

  for (genvar p = 0; p < Ports; p++) begin : gen_dma
    port_dma #(
      .AddrWidth (AddrWidth),
      .DataWidth (DataWidth),
      .CountWidth(ColWidth),
      .Depth     (Depth)
    ) u_dma (
      .clk_i,
      .rst_ni,
      .cmd_valid_i(dma_cmd_valid[p]),
      .cmd_ready_o(dma_cmd_ready[p]),
      .cmd_addr_i (iss_addr_q),
      .cmd_beats_i(iss_beats_q),
      .out_valid_o(dma_out_valid[p]),
      .out_ready_i(dma_out_ready[p]),
      .out_data_o (dma_out_data[p]),
      .ar_valid_o (ar_valid_o[p]),
      .ar_ready_i (ar_ready_i[p]),
      .ar_addr_o  (ar_addr_o[p]),
      .ar_len_o   (ar_len_o[p]),
      .ar_size_o  (ar_size_o[p]),
      .ar_burst_o (ar_burst_o[p]),
      .r_valid_i  (r_valid_i[p]),
      .r_ready_o  (r_ready_o[p]),
      .r_data_i   (r_data_i[p]),
      .r_resp_i   (r_resp_i[p]),
      .err_o      (err_o[p])
    );
  end

  // --- Issue: per pass, a DMA command to each port with rows in it ------------

  logic                iss_busy_q;
  logic [RowWidth-1:0] iss_groups_q;  // groups not yet issued, this pass's included
  logic [Ports-1:0]    iss_taken_q;   // this pass's commands already taken
  logic [Ports-1:0]    iss_active, iss_taken;
  logic                iss_pass_done, iss_last_pass;

  for (genvar p = 0; p < Ports; p++) begin : gen_issue
    assign iss_active[p]    = RowWidth'(p) < iss_groups_q;
    assign dma_cmd_valid[p] = iss_busy_q && iss_active[p] && !iss_taken_q[p];
    assign iss_taken[p]     = iss_taken_q[p] || dma_cmd_ready[p];
  end
  assign iss_pass_done = iss_busy_q && &(iss_taken | ~iss_active);
  assign iss_last_pass = iss_groups_q <= RowWidth'(Ports);
  assign q_pop[0]      = !iss_busy_q && q_valid[0];

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      iss_busy_q  <= 1'b0;
      iss_taken_q <= '0;
    end else if (q_pop[0]) begin
      iss_busy_q   <= 1'b1;
      iss_addr_q   <= q_head[0].addr;
      iss_groups_q <= n_groups(q_head[0].rows);
      iss_beats_q  <= n_beats(q_head[0].cols);
    end else if (iss_pass_done) begin
      iss_taken_q  <= '0;
      iss_addr_q   <= iss_addr_q + AddrWidth'(iss_beats_q) * AddrWidth'(BeatBytes);
      iss_groups_q <= iss_groups_q - RowWidth'(Ports);
      if (iss_last_pass) iss_busy_q <= 1'b0;
    end else if (iss_busy_q) begin
      iss_taken_q <= iss_taken;
    end
  end

  // --- Step: one beat to every lane --------------------------------------------

  logic                step_busy_q;
  logic [RowWidth-1:0] step_groups_q;  // groups of this pass and the later ones
  logic [ColWidth-1:0] step_beats_q, step_col_q;
  logic [E-1:0]        step_mask_q;    // the last beat's elements
  logic [BufWidth-1:0] step_buf_q;
  logic [Ports-1:0]    step_active;
  logic [L-1:0]        lane_in_ready;
  logic                step_last, fire, step_end, step_load;
  logic [ColWidth-1:0] next_col;

  for (genvar p = 0; p < Ports; p++) begin : gen_step
    assign step_active[p]   = RowWidth'(p) < step_groups_q;
    assign dma_out_ready[p] = fire && step_active[p];
  end
  assign step_last = step_col_q == step_beats_q - 1'b1;
  assign fire      = step_busy_q && &(dma_out_valid | ~step_active) && &lane_in_ready;
  assign step_end  = fire && step_last && step_groups_q <= RowWidth'(Ports);
  assign step_load = q_valid[1] && (!step_busy_q || step_end);
  assign q_pop[1]  = step_load;
  assign next_col  = step_load || (fire && step_last) ? '0
                   : fire ? step_col_q + 1'b1 : step_col_q;
  // The RAM's address is the next cycle's column, so x_data_i is this one's.
  assign x_addr_o  = next_col;
  assign x_buf_o   = step_load ? q_head[1].a : step_buf_q;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      step_busy_q <= 1'b0;
      step_col_q  <= '0;
    end else begin
      step_col_q <= next_col;
      if (step_load) begin
        step_busy_q   <= 1'b1;
        step_groups_q <= n_groups(q_head[1].rows);
        step_beats_q  <= n_beats(q_head[1].cols);
        step_buf_q    <= q_head[1].a;
        for (int unsigned e = 0; e < E; e++) begin
          step_mask_q[e] <= q_head[1].cols % ColWidth'(E) == '0
                            || ColWidth'(e) < q_head[1].cols % ColWidth'(E);
        end
      end else if (step_end) begin
        step_busy_q <= 1'b0;
      end else if (fire && step_last) begin
        step_groups_q <= step_groups_q - RowWidth'(Ports);
      end
    end
  end

  // --- Lanes -------------------------------------------------------------------

  // Padded to whole write groups: lanes past L never have a result.
  logic [Groups*Writes-1:0]       lane_out_valid;
  // verilator lint_off UNUSEDSIGNAL
  logic [Groups*Writes-1:0]       lane_out_ready;  // the padding's bits go nowhere
  // verilator lint_on UNUSEDSIGNAL
  logic [31:0]                    lane_out[Groups*Writes];

  for (genvar l = 0; l < Groups * Writes; l++) begin : gen_lane
    if (l < L) begin : gen_real
      localparam int unsigned P = l / LanesPerPort, I = l % LanesPerPort;
      logic [E-1:0][15:0] w;
      assign w = dma_out_data[P][I*E*16 +: E*16];
      dot_lane #(
        .E(E),
        .A(A)
      ) u_lane (
        .clk_i,
        .rst_ni,
        .in_valid_i (fire && step_active[P]),
        .in_ready_o (lane_in_ready[l]),
        .w_i        (w),
        .x_i        (x_data_i),
        .mask_i     (step_last ? step_mask_q : '1),
        .last_i     (step_last),
        .out_valid_o(lane_out_valid[l]),
        .out_ready_i(lane_out_ready[l]),
        .out_o      (lane_out[l])
      );
    end else begin : gen_pad
      assign lane_out_valid[l] = 1'b0;
      assign lane_out[l]       = '0;
    end
  end

  // --- Drain: results to the write port, Writes lanes per cycle ----------------

  logic                     drain_busy_q, drain_bf16_q;
  logic [RowWidth-1:0]      drain_rows_q, drain_groups_q, drain_base_q;  // base: lane 0's row
  logic [BufWidth-1:0]      drain_buf_q;
  logic [GW-1:0]            drain_grp_q;  // the write group
  logic [RowWidth-1:0]      lanes_with_result;
  logic [Writes-1:0]        has, row_in;
  logic [Writes-1:0][31:0]  slot_out;
  logic                     go, pass_end, drain_end, drain_load;

  assign lanes_with_result = RowWidth'(pass_ports(drain_groups_q)) * RowWidth'(LanesPerPort);
  for (genvar s = 0; s < Writes; s++) begin : gen_slot
    logic [RowWidth-1:0] lane;
    assign lane        = RowWidth'(drain_grp_q) * RowWidth'(Writes) + RowWidth'(s);
    assign has[s]      = lane < lanes_with_result;
    assign row_in[s]   = drain_base_q + lane < drain_rows_q;
    assign slot_out[s] = lane_out[drain_grp_q*Writes+s];
  end
  always_comb begin
    go = drain_busy_q;
    for (int unsigned s = 0; s < Writes; s++) begin
      if (has[s] && !lane_out_valid[drain_grp_q*Writes+s]) go = 1'b0;
    end
    lane_out_ready = '0;
    for (int unsigned s = 0; s < Writes; s++) begin
      lane_out_ready[drain_grp_q*Writes+s] = go && has[s];
    end
  end
  assign pass_end   = go && RowWidth'(drain_grp_q + 1'b1) * RowWidth'(Writes) >= lanes_with_result;
  assign drain_end  = pass_end && drain_groups_q <= RowWidth'(Ports);
  assign drain_load = q_valid[2] && (!drain_busy_q || drain_end);
  assign q_pop[2]   = drain_load;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      drain_busy_q <= 1'b0;
      drain_grp_q  <= '0;
    end else if (drain_load) begin
      drain_busy_q   <= 1'b1;
      drain_grp_q    <= '0;
      drain_base_q   <= '0;
      drain_rows_q   <= q_head[2].rows;
      drain_groups_q <= n_groups(q_head[2].rows);
      drain_buf_q    <= q_head[2].dst;
      drain_bf16_q   <= q_head[2].bf16;
    end else if (drain_end) begin
      drain_busy_q <= 1'b0;
    end else if (pass_end) begin
      drain_grp_q    <= '0;
      drain_base_q   <= drain_base_q + RowWidth'(L);
      drain_groups_q <= drain_groups_q - RowWidth'(Ports);
    end else if (go) begin
      drain_grp_q <= drain_grp_q + 1'b1;
    end
  end

  // The write port, registered.
  logic [Writes-1:0][15:0] rounded;
  for (genvar s = 0; s < Writes; s++) begin : gen_round
    fp32_to_bf16 u_round (
      .a_i(slot_out[s]),
      .y_o(rounded[s])
    );
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      w_valid_o <= 1'b0;
      done_o    <= 1'b0;
    end else begin
      w_valid_o <= go && |(has & row_in);
      done_o    <= drain_end;
    end
    w_buf_o  <= drain_buf_q;
    w_row_o  <= drain_base_q + RowWidth'(drain_grp_q) * RowWidth'(Writes);
    w_mask_o <= has & row_in;
    w_bf16_o <= drain_bf16_q;
    for (int unsigned s = 0; s < Writes; s++) begin
      w_data_o[s] <= drain_bf16_q ? {16'h0, rounded[s]} : slot_out[s];
    end
  end

endmodule
