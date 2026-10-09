// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// One HBM port's reader (PLAN.md, M5): a command names a beat-aligned start
// address and a number of beats; the reader fetches them with AXI read bursts
// and streams them out in order, valid/ready. Command boundaries are not
// marked: the engine, which issued the commands, counts the beats. A command
// of 0 beats is taken and does nothing.
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
// The next command is taken the cycle after the current one's last burst is
// requested, while its data may still be on the way. So a command costs at
// least 2 cycles of requests: commands of 1 beat run at half rate (the
// engine's are hundreds of beats per port).

module port_dma #(
  parameter  int unsigned AddrWidth  = 34,   // F2's HBM: 16 GiB
  parameter  int unsigned DataWidth  = 256,  // an HBM port's beat
  parameter  int unsigned CountWidth = 24,   // beats per command: Qwen2.5's classifier needs 19
  parameter  int unsigned MaxBurst   = 16,   // beats per burst: AXI3's longest
  parameter  int unsigned Depth      = 64,   // data FIFO, beats
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
  localparam int unsigned PageBeats = PageBytes / BeatBytes;
  localparam int unsigned OffW      = $clog2(BeatBytes);
  localparam int unsigned PageW     = $clog2(PageBytes);
  localparam int unsigned IdxW      = PageW - OffW;          // a beat's index in its page
  localparam int unsigned LenW      = $clog2(MaxBurst + 1);  // a burst's beats
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
  if (AddrWidth < PageW || CountWidth < LenW) begin : gen_bad_count
    $error("port_dma: AddrWidth must hold a 4 KB page, CountWidth a burst");
  end

  // --- Address side: one command's bursts -------------------------------------

  logic [AddrWidth-1:0]  addr_q;
  logic [CountWidth-1:0] left_q;      // beats not yet requested
  logic [CrW-1:0]        reserved_q;  // beats requested and not yet out
  logic [IdxW:0]         to_page;     // beats to the next 4 KB boundary, 1 .. PageBeats
  logic [LenW-1:0]       cap, left, len;
  logic                  cmd_fire, ar_fire, out_fire;

  // The burst: the fewest of the beats to the page's end, MaxBurst and the
  // beats left, each narrowed to a burst's width before comparing.
  assign to_page = (IdxW + 1)'(PageBeats) - (IdxW + 1)'(addr_q[PageW-1:OffW]);
  assign cap     = to_page < (IdxW + 1)'(MaxBurst) ? LenW'(to_page) : LenW'(MaxBurst);
  assign left    = left_q < CountWidth'(MaxBurst) ? LenW'(left_q) : LenW'(MaxBurst);
  assign len     = left < cap ? left : cap;

  assign cmd_ready_o = left_q == '0;
  assign cmd_fire    = cmd_valid_i && cmd_ready_o;
  assign ar_valid_o  = left_q != '0 && (CrW'(Depth) - reserved_q >= CrW'(len));
  assign ar_fire     = ar_valid_o && ar_ready_i;
  assign ar_addr_o   = addr_q;
  assign ar_len_o    = 8'(len - 1'b1);
  assign ar_size_o   = 3'(OffW);
  assign ar_burst_o  = Incr;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      addr_q <= '0;
      left_q <= '0;
    end else if (cmd_fire) begin
      addr_q <= cmd_addr_i;
      left_q <= cmd_beats_i;
    end else if (ar_fire) begin
      addr_q <= addr_q + (AddrWidth'(len) << OffW);
      left_q <= left_q - CountWidth'(len);
    end
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) reserved_q <= '0;
    else reserved_q <= reserved_q + (ar_fire ? CrW'(len) : '0) - CrW'(out_fire);
  end

  // --- Data side: FIFO, errors -------------------------------------------------

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

  assign out_fire = out_valid_o && out_ready_i;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) err_o <= 1'b0;
    else if (r_valid_i && r_ready_o && r_resp_i != 2'b00) err_o <= 1'b1;
  end

endmodule
