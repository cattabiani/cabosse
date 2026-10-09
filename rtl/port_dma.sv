// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// One HBM port's reader (PLAN.md, M5): a command names a beat-aligned start
// address and a number of beats; the reader fetches them with AXI read bursts
// and streams them out in order, valid/ready, with out_last_o on each
// command's final beat. A command of 0 beats is taken and does nothing.
//
// AXI: read channels only, in the subset AXI3 and AXI4 share: incrementing
// bursts of at most MaxBurst (<= 16) beats of the full data width, none
// crossing a 4 KB boundary, one ID (so data returns in order; no ID ports).
//
// Credits: a burst is requested only when the data FIFO has room for all of
// it, counting the beats already requested; so r_ready_o stays high and the
// memory never waits for the engine. Full rate needs Depth to cover the beats
// in flight: about the memory's latency plus one burst.
//
// The next command is taken as soon as the current one's bursts are all
// requested, while its data may still be on the way; up to Commands
// commands can have data not yet out.

module port_dma #(
  parameter  int unsigned AddrWidth  = 34,   // F2's HBM: 16 GiB
  parameter  int unsigned DataWidth  = 256,  // an HBM port's beat
  parameter  int unsigned CountWidth = 32,   // beats per command
  parameter  int unsigned MaxBurst   = 16,   // beats per burst: AXI3's longest
  parameter  int unsigned Depth      = 64,   // data FIFO, beats
  parameter  int unsigned Commands   = 4,    // commands with data not yet out
  localparam int unsigned BeatBytes  = DataWidth / 8
) (
  input  logic                  clk_i,
  input  logic                  rst_ni,
  // Command
  input  logic                  cmd_valid_i,
  output logic                  cmd_ready_o,
  input  logic [AddrWidth-1:0]  cmd_addr_i,   // beat-aligned
  input  logic [CountWidth-1:0] cmd_beats_i,
  // Beats out
  output logic                  out_valid_o,
  input  logic                  out_ready_i,
  output logic [DataWidth-1:0]  out_data_o,
  output logic                  out_last_o,
  // AXI read address channel
  output logic                  ar_valid_o,
  input  logic                  ar_ready_i,
  output logic [AddrWidth-1:0]  ar_addr_o,
  output logic [7:0]            ar_len_o,     // beats - 1
  output logic [2:0]            ar_size_o,    // log2(bytes per beat)
  output logic [1:0]            ar_burst_o,   // INCR
  // AXI read data channel
  input  logic                  r_valid_i,
  output logic                  r_ready_o,
  input  logic [DataWidth-1:0]  r_data_i,
  input  logic [1:0]            r_resp_i,
  output logic                  err_o         // sticky: a beat came with an error response
);

  localparam int unsigned PageBytes = 4096;  // AXI: no burst crosses it
  localparam int unsigned OffW      = $clog2(BeatBytes);
  localparam int unsigned PageW     = $clog2(PageBytes);
  localparam int unsigned CrW       = $clog2(Depth + 1);
  localparam logic [1:0]  Incr      = 2'b01;

  if (MaxBurst < 1 || MaxBurst > 16) begin : gen_bad_burst
    $error("port_dma: MaxBurst must be 1 to 16 (AXI3)");
  end
  if (Depth < MaxBurst) begin : gen_bad_depth
    $error("port_dma: Depth must hold a whole burst");
  end
  if (DataWidth < 8 || (DataWidth & (DataWidth - 1)) != 0 || BeatBytes > 128) begin : gen_bad_w
    $error("port_dma: DataWidth must be a power of two, 8 to 1024 bits");
  end

  // --- Address side: one command's bursts -------------------------------------

  logic                  busy_q;      // a command has bursts left to request
  logic [AddrWidth-1:0]  addr_q;
  logic [CountWidth-1:0] left_q;      // beats not yet requested
  logic [CrW-1:0]        reserved_q;  // beats requested and not yet out
  logic [CountWidth-1:0] to_page, cap, len;
  logic                  cmd_fire, ar_fire, r_fire, out_fire, lens_ready;

  // Beats to the next 4 KB boundary, then the burst: the fewest of those, the
  // beats left and MaxBurst.
  assign to_page = CountWidth'((PageBytes - int'(addr_q[PageW-1:0])) / BeatBytes);
  assign cap     = to_page < CountWidth'(MaxBurst) ? to_page : CountWidth'(MaxBurst);
  assign len     = left_q < cap ? left_q : cap;

  assign cmd_ready_o = !busy_q && lens_ready;
  assign cmd_fire    = cmd_valid_i && cmd_ready_o;
  assign ar_valid_o  = busy_q && (CrW'(Depth) - reserved_q >= CrW'(len));
  assign ar_fire     = ar_valid_o && ar_ready_i;
  assign ar_addr_o   = addr_q;
  assign ar_len_o    = 8'(len - 1'b1);
  assign ar_size_o   = 3'(OffW);
  assign ar_burst_o  = Incr;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      busy_q <= 1'b0;
      addr_q <= '0;
      left_q <= '0;
    end else if (cmd_fire) begin
      busy_q <= cmd_beats_i != '0;
      addr_q <= cmd_addr_i;
      left_q <= cmd_beats_i;
    end else if (ar_fire) begin
      busy_q <= left_q != len;
      addr_q <= addr_q + AddrWidth'(len * BeatBytes);
      left_q <= left_q - len;
    end
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) reserved_q <= '0;
    else reserved_q <= reserved_q + (ar_fire ? CrW'(len) : '0) - CrW'(out_fire);
  end

  // --- Data side: FIFO, command boundaries, errors ----------------------------

  stream_fifo #(.Width(DataWidth), .Depth(Depth)) u_data (
    .clk_i,
    .rst_ni,
    .in_valid_i (r_valid_i),
    .in_ready_o (r_ready_o),
    .in_data_i  (r_data_i),
    .out_valid_o,
    .out_ready_i,
    .out_data_o
  );

  // Each taken command's length, so the last beat out can be marked.
  logic                  lens_valid;
  logic [CountWidth-1:0] head_len, sent_q;  // beats of the head command already out
  stream_fifo #(.Width(CountWidth), .Depth(Commands)) u_lens (
    .clk_i,
    .rst_ni,
    .in_valid_i  (cmd_fire && cmd_beats_i != '0),
    .in_ready_o  (lens_ready),
    .in_data_i   (cmd_beats_i),
    .out_valid_o (lens_valid),
    .out_ready_i (out_fire && out_last_o),
    .out_data_o  (head_len)
  );

  assign out_fire   = out_valid_o && out_ready_i;
  assign out_last_o = lens_valid && sent_q == head_len - 1'b1;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) sent_q <= '0;
    else if (out_fire) sent_q <= out_last_o ? '0 : sent_q + 1'b1;
  end

  assign r_fire = r_valid_i && r_ready_o;
  always_ff @(posedge clk_i) begin
    if (!rst_ni) err_o <= 1'b0;
    else if (r_fire && r_resp_i != 2'b00) err_o <= 1'b1;
  end

endmodule
