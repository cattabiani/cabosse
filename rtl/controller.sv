// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// The controller (D-029, D-068; docs/architecture.md, "Controller" and
// "Registers"): the host's registers on an AXI-Lite port, the command
// buffer, and a sequencer that runs the command list once per token, handing
// each command to the engine or the vector unit in list order.
//
// Registers (model/commands.py, REGISTERS): 32-bit, at their offsets; the
// command buffer from COMMAND_BUFFER (Commands commands of 32 bytes, written
// a word at a time, not readable). Writes take whole words (WSTRB is not
// looked at); a read of an offset with no register gives 0; every response
// is OKAY. CONTROL bit 0 starts a token (when not busy), bit 1 resets the
// sequencer and the units (units_reset_o; bit 1 wins over bit 0). The host must
// not write the command buffer while a token runs.
//
// A token: pc runs from 0; the command at pc is read from the buffer (a
// synchronous RAM, read at the next pc so it is there the cycle after a
// handover) and handed over when its unit takes it and these hold (D-068,
// model/commands.py: stops, engine_waits, kv_waits):
// - stop: a command off the engine (the vector unit's, END) that reads a
//   buffer written since the last stop waits until every command handed over
//   is done; its handover clears the written set (D-045).
// - wait: an engine command whose input buffer's last writer no earlier
//   engine command waited for goes with wait (a buffer no command wrote this
//   token, the host's, needs none), and a go is queued for it: the go is
//   sent once that writer's unit has counted its done (D-042). Gos leave in
//   order, each after its command was taken.
// - SCORES (VALUES) is handed over after the last KV_STORE (V_STORE) is
//   done. kv_waits asks for the last store to the cache it reads; the last
//   store of its kind is that one or a later one, whose done comes later
//   (the vector unit's dones are in order).
// - OUTPUT goes to the vector unit like its other commands (D-071: it
//   writes the logits to HBM, done once its writes are answered).
// - END waits until every command is done and every go sent, then sets
//   STATUS done, pulses irq_o and keeps the token's cycles in CYCLES.
// Every command handed over writes its dst into the written set and becomes
// its last writer. Errors stop the sequencer with STATUS error and ERROR =
// {reason, 16 bits of pc}: an unknown opcode (ErrOpcode), a unit's memory
// error (ErrMemory: err_i, sticky in the units; pc is where the sequencer
// was, not the failing command), or pc reaching COMMANDS (or the buffer's
// end) without END (ErrNoEnd). Nothing else is checked: the runtime validates the
// list (commands.validate).
//
// The command's fields go to both units on one bus (cmd_*); eng_valid_o or
// vec_valid_o says whose it is. cmd_group_o (n / kv_heads, attention) is
// found by subtraction in the cycles after the command reaches pc, so an
// attention command waits a cycle more than its group. token_o, t_o
// (position + 1) and stride_o are TOKEN, POSITION and STRIDE as they were at
// the token's start. units_reset_o pulses with CONTROL's reset, for the
// units' own reset (their memory errors are sticky). It reaches the units
// UnitsResetDelay cycles later (core.sv's registers); for SettleCycles
// after a reset (CONTROL's or rst_ni) the controller keeps its sequencer
// reset, so no done from before it counts, and takes no write, so no token
// starts before the units are out of their reset.

module controller #(
  parameter  int unsigned Bufs      = 22,
  parameter  logic [31:0] Bf16Mask  = 32'h4644,  // per id: X, QR, P, ATT, M (commands.buffers)
  parameter  int unsigned AddrW     = 20,        // n, m, cap, stride, token, position
  parameter  int unsigned MemAddrW  = 34,        // F2's HBM: 16 GiB
  parameter  int unsigned HeadWidth = 8,
  parameter  int unsigned Commands  = 2048,      // the command buffer: 64 KiB
  parameter  int unsigned GoDepth   = 8,         // gos queued, at most
  parameter  int unsigned AxilAddrW = 17,
  parameter  int unsigned UnitsResetDelay = 2,  // registers from units_reset_o to the units' flops
  localparam int unsigned BufW      = $clog2(Bufs),
  localparam int unsigned PcW       = $clog2(Commands + 1),
  localparam int unsigned SeqW      = PcW                     // a unit's commands of a token
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,

  input  logic                 s_aw_valid_i,
  output logic                 s_aw_ready_o,
  input  logic [AxilAddrW-1:0] s_aw_addr_i,
  input  logic                 s_w_valid_i,
  output logic                 s_w_ready_o,
  input  logic [31:0]          s_w_data_i,
  output logic                 s_b_valid_o,
  input  logic                 s_b_ready_i,
  output logic [1:0]           s_b_resp_o,
  input  logic                 s_ar_valid_i,
  output logic                 s_ar_ready_o,
  input  logic [AxilAddrW-1:0] s_ar_addr_i,
  output logic                 s_r_valid_o,
  input  logic                 s_r_ready_i,
  output logic [31:0]          s_r_data_o,
  output logic [1:0]           s_r_resp_o,
  output logic                 irq_o,        // a pulse: the token is done
  output logic                 units_reset_o,  // a pulse: CONTROL's reset

  output logic [4:0]           cmd_op_o,
  output logic [BufW-1:0]      cmd_dst_o,
  output logic [BufW-1:0]      cmd_a_o,
  output logic [BufW-1:0]      cmd_b_o,
  output logic [BufW-1:0]      cmd_c_o,
  output logic [7:0]           cmd_flags_o,
  output logic [AddrW-1:0]     cmd_n_o,
  output logic [AddrW-1:0]     cmd_m_o,
  output logic [AddrW-1:0]     cmd_cap_o,
  output logic [MemAddrW-1:0]  cmd_addr_o,
  output logic [31:0]          cmd_scalar_o,
  output logic [HeadWidth-1:0] cmd_group_o,  // attention: query heads per KV head
  output logic                 cmd_bf16_o,   // dst is a BF16 buffer
  output logic [AddrW-1:0]     token_o,
  output logic [AddrW-1:0]     t_o,
  output logic [AddrW-1:0]     stride_o,

  output logic                 eng_valid_o,
  input  logic                 eng_ready_i,
  output logic [1:0]           eng_op_o,     // rtl/engine.sv's OpMatvec, OpScores, OpValues
  output logic                 eng_wait_o,
  output logic                 eng_go_o,
  input  logic                 eng_done_i,
  input  logic                 eng_err_i,

  output logic                 vec_valid_o,
  input  logic                 vec_ready_i,
  input  logic                 vec_done_i,
  input  logic                 vec_err_i
);

  if (Bufs > 32 || Bufs < 2) begin : gen_bad_bufs
    $error("controller: Bufs must be 2 to 32 (Bf16Mask's bits)");
  end
  if (Commands > 65536 || 2 ** $clog2(Commands) != Commands) begin : gen_bad_commands
    $error("controller: Commands must be a power of two, at most 65536 (ERROR's 16 bits)");
  end
  if (AxilAddrW < 17 || 2 ** (AxilAddrW - 1) < 32 * Commands) begin : gen_bad_axil
    $error("controller: AxilAddrW must hold COMMAND_BUFFER (0x10000) and the buffer");
  end

  // The registers' offsets (commands.REGISTERS) and values.
  localparam logic [15:0] RegId        = 16'h00;
  localparam logic [15:0] RegVersion   = 16'h04;
  localparam logic [15:0] RegControl   = 16'h08;
  localparam logic [15:0] RegStatus    = 16'h0C;
  localparam logic [15:0] RegToken     = 16'h10;
  localparam logic [15:0] RegPosition  = 16'h14;
  localparam logic [15:0] RegCommands  = 16'h18;
  localparam logic [15:0] RegError     = 16'h1C;
  localparam logic [15:0] RegLogitsLo  = 16'h20;
  localparam logic [15:0] RegLogitsHi  = 16'h24;
  localparam logic [15:0] RegCyclesLo  = 16'h28;
  localparam logic [15:0] RegCyclesHi  = 16'h2C;
  localparam logic [15:0] RegStride    = 16'h30;
  localparam logic [31:0] IdValue      = 32'h4342_4F53;  // 'CBOS' (commands.ID)
  localparam logic [31:0] VersionValue = 32'd1;          // commands.VERSION

  // Opcodes (commands.Op).
  localparam logic [7:0] OpEnd     = 8'd0;
  localparam logic [7:0] OpMatvec  = 8'd2;
  localparam logic [7:0] OpKvStore = 8'd3;
  localparam logic [7:0] OpScores  = 8'd4;
  localparam logic [7:0] OpValues  = 8'd5;
  localparam logic [7:0] OpVStore  = 8'd17;  // the last opcode

  localparam logic [3:0] ErrOpcode = 4'd1;
  localparam logic [3:0] ErrMemory = 4'd2;
  localparam logic [3:0] ErrNoEnd  = 4'd3;

  // ---------------------------------------------------------------- AXI-Lite

  logic [31:0] token_q, position_q, commands_q, stride_q, logits_lo_q, logits_hi_q;
  logic [31:0] error_q;
  logic [63:0] cycles_q;
  logic        busy_q, done_q, err_q;
  logic        bvalid_q, rvalid_q;
  logic [31:0] rdata_q;
  logic        wr, rd, start, ctl_reset;
  logic [15:0] waddr, raddr;
  logic        cb_we;

  // The reset's way: units_reset_o's register, UnitsResetDelay registers
  // to the units' flops, their registered done and err outputs (1),
  // core.sv's registers on them (1), and 2 cycles spare.
  localparam int unsigned SettleMargin = 1 + 1 + 1 + 2;
  localparam int unsigned SettleCycles = UnitsResetDelay + SettleMargin;
  localparam int unsigned SettleW      = $clog2(SettleCycles + 1);

  logic [SettleW-1:0] settle_q;
  logic               settling;
  assign settling = settle_q != '0;

  assign wr          = s_aw_valid_i && s_w_valid_i && !bvalid_q && !settling;
  assign s_aw_ready_o = wr;
  assign s_w_ready_o  = wr;
  assign s_b_valid_o  = bvalid_q;
  assign s_b_resp_o   = 2'b00;
  assign rd          = s_ar_valid_i && !rvalid_q;
  assign s_ar_ready_o = rd;
  assign s_r_valid_o  = rvalid_q;
  assign s_r_data_o   = rdata_q;
  assign s_r_resp_o   = 2'b00;
  assign waddr       = s_aw_addr_i[15:0];
  assign raddr       = s_ar_addr_i[15:0];
  assign cb_we       = wr && s_aw_addr_i[16];
  assign start       = wr && !s_aw_addr_i[16] && waddr == RegControl && s_w_data_i[0] && !busy_q
                    && !s_w_data_i[1];
  assign ctl_reset   = wr && !s_aw_addr_i[16] && waddr == RegControl && s_w_data_i[1];

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      token_q     <= '0;
      position_q  <= '0;
      commands_q  <= '0;
      stride_q    <= '0;
      logits_lo_q <= '0;
      logits_hi_q <= '0;
      bvalid_q    <= 1'b0;
      rvalid_q    <= 1'b0;
      rdata_q     <= '0;
    end else begin
      if (wr) bvalid_q <= 1'b1;
      else if (s_b_ready_i) bvalid_q <= 1'b0;
      if (wr && !s_aw_addr_i[16]) begin
        unique case (waddr)
          RegToken:    token_q     <= s_w_data_i;
          RegPosition: position_q  <= s_w_data_i;
          RegCommands: commands_q  <= s_w_data_i;
          RegLogitsLo: logits_lo_q <= s_w_data_i;
          RegLogitsHi: logits_hi_q <= s_w_data_i;
          RegStride:   stride_q    <= s_w_data_i;
          default: ;
        endcase
      end
      if (rd) begin
        rvalid_q <= 1'b1;
        rdata_q  <= '0;
        if (!s_ar_addr_i[16]) begin
          unique case (raddr)
            RegId:       rdata_q <= IdValue;
            RegVersion:  rdata_q <= VersionValue;
            RegStatus:   rdata_q <= {29'd0, err_q, done_q, busy_q};
            RegToken:    rdata_q <= token_q;
            RegPosition: rdata_q <= position_q;
            RegCommands: rdata_q <= commands_q;
            RegError:    rdata_q <= error_q;
            RegLogitsLo: rdata_q <= logits_lo_q;
            RegLogitsHi: rdata_q <= logits_hi_q;
            RegCyclesLo: rdata_q <= cycles_q[31:0];
            RegCyclesHi: rdata_q <= cycles_q[63:32];
            RegStride:   rdata_q <= stride_q;
            default: ;
          endcase
        end
      end else if (s_r_ready_i) begin
        rvalid_q <= 1'b0;
      end
    end
  end

  // ---------------------------------------------------------- command buffer

  localparam int unsigned CmdAW = $clog2(Commands);

  logic [7:0][31:0] cmd_q;  // the command at pc
  logic [AddrW-1:0] run_token_q, run_t_q, run_stride_q;  // the registers at start

  always_ff @(posedge clk_i) begin
    units_reset_o <= rst_ni && ctl_reset;
    if (!rst_ni || ctl_reset) settle_q <= SettleW'(SettleCycles);
    else if (settling) settle_q <= settle_q - 1'b1;
    if (start) begin
      run_token_q  <= AddrW'(token_q);
      run_t_q      <= AddrW'(position_q + 32'd1);
      run_stride_q <= AddrW'(stride_q);
    end
  end
  logic [PcW-1:0]   pc_q, pc_d;

  // A RAM per word of a command (a plain simple dual-port RAM each, for
  // Vivado to infer), read together.
  for (genvar w = 0; w < 8; w++) begin : gen_bank
    logic [31:0] bank[Commands];

    always_ff @(posedge clk_i) begin
      if (cb_we && s_aw_addr_i[4:2] == 3'(w)) bank[s_aw_addr_i[CmdAW+4:5]] <= s_w_data_i;
      cmd_q[w] <= bank[pc_d[CmdAW-1:0]];
    end
  end

  // Fields (commands.ENCODING): op, dst, a, b, c, flags, kv_heads, n, m, cap,
  // addr, scalar.
  logic [7:0]  op;
  logic [7:0]  f_dst, f_a, f_b, f_c;
  logic [HeadWidth-1:0] f_kv_heads;
  logic [AddrW-1:0]     f_n;  // the encoding's 32 bits; validate keeps n below 2^AddrW

  assign op           = cmd_q[0][7:0];
  assign f_dst        = cmd_q[0][15:8];
  assign f_a          = cmd_q[0][23:16];
  assign f_b          = cmd_q[0][31:24];
  assign f_c          = cmd_q[1][7:0];
  assign f_kv_heads   = cmd_q[1][16+:HeadWidth];
  assign f_n          = cmd_q[2][AddrW-1:0];
  assign cmd_op_o     = op[4:0];
  assign cmd_dst_o    = BufW'(f_dst);
  assign cmd_a_o      = BufW'(f_a);
  assign cmd_b_o      = BufW'(f_b);
  assign cmd_c_o      = BufW'(f_c);
  assign cmd_flags_o  = cmd_q[1][15:8];
  assign cmd_n_o      = f_n;
  assign cmd_m_o      = AddrW'(cmd_q[3]);
  assign cmd_cap_o    = AddrW'(cmd_q[4]);
  assign cmd_addr_o   = MemAddrW'({cmd_q[6], cmd_q[5]});
  assign cmd_scalar_o = cmd_q[7];
  assign cmd_bf16_o   = Bf16Mask[f_dst[4:0]];
  assign token_o      = run_token_q;
  assign t_o          = run_t_q;
  assign stride_o     = run_stride_q;

  // ---------------------------------------------------------- the sequencer

  logic            is_eng, is_vec, known;
  logic [Bufs-1:0] reads, dst_bit;

  assign known  = op <= OpVStore;
  assign is_eng = op == OpMatvec || op == OpScores || op == OpValues;
  assign is_vec = known && !is_eng && op != OpEnd;  // OUTPUT too (D-071)

  always_comb begin
    reads   = '0;
    dst_bit = '0;
    if (f_a != 0) reads[BufW'(f_a)] = 1'b1;
    if (f_b != 0) reads[BufW'(f_b)] = 1'b1;
    if (f_c != 0) reads[BufW'(f_c)] = 1'b1;
    if (f_dst != 0) dst_bit[BufW'(f_dst)] = 1'b1;
  end

  // The scoreboard: per buffer its last writer this token (its unit and its
  // count among that unit's commands) and whether an engine command waited
  // for it (set for a buffer no command wrote: the host's); the buffers
  // written since the last stop; the vector unit's dones the last stores
  // need.
  logic [Bufs-1:0]           written_q, waited_q, wunit_q;  // wunit: the engine's
  logic [Bufs-1:0][SeqW-1:0] wseq_q;
  logic [SeqW-1:0]           eng_issued_q, eng_done_q, vec_issued_q, vec_done_q;
  logic [SeqW-1:0]           kstore_q, vstore_q;

  // The group (query heads per KV head) of an attention command, by
  // subtraction, from the cycle after it reaches pc: no divider on the path
  // from the command buffer (2 such commands a layer, a few cycles each).
  logic [HeadWidth-1:0] group_q, rem_q;
  logic                 group_ok_q, fresh_q;  // fresh: cmd_q is a new command

  assign cmd_group_o = group_q;

  logic all_done, stop, stop_ok, kv_ready, need_wait, writer_done, active;
  logic go_in_ready, go_out_valid, go_out_unit;
  logic [SeqW-1:0] go_out_seq;
  logic issuing, fire, fail_opcode, fail_mem, fail_end, end_now;

  assign all_done  = eng_issued_q == eng_done_q && vec_issued_q == vec_done_q && !go_out_valid;
  assign stop      = !is_eng && |(reads & written_q);
  assign stop_ok   = !stop || all_done;
  assign kv_ready  = op == OpScores ? vec_done_q >= kstore_q && group_ok_q
                   : op == OpValues ? vec_done_q >= vstore_q && group_ok_q
                   : 1'b1;
  assign need_wait = is_eng && !waited_q[BufW'(f_a)];

  assign issuing     = busy_q && !ctl_reset;
  assign fail_end    = issuing && (32'(pc_q) >= commands_q || pc_q >= PcW'(Commands));
  assign active      = issuing && !fail_end;
  assign fail_opcode = active && !known;
  assign fail_mem    = issuing && (eng_err_i || vec_err_i);
  assign end_now     = active && op == OpEnd && all_done;

  assign eng_valid_o = active && is_eng && kv_ready && (!need_wait || go_in_ready);
  assign vec_valid_o = active && is_vec && stop_ok;
  assign eng_op_o    = op == OpScores ? 2'd1 : op == OpValues ? 2'd2 : 2'd0;
  assign eng_wait_o  = need_wait;
  assign fire        = (eng_valid_o && eng_ready_i) || (vec_valid_o && vec_ready_i);

  always_ff @(posedge clk_i) begin
    fresh_q <= start || fire;
    if (start || fire) begin  // the next command's group is not known yet
      group_ok_q <= 1'b0;
    end else if (fresh_q) begin
      group_q <= '0;
      rem_q   <= f_n[HeadWidth-1:0];
    end else if (!group_ok_q) begin
      if (f_kv_heads != 0 && rem_q >= f_kv_heads) begin
        rem_q   <= rem_q - f_kv_heads;
        group_q <= group_q + 1'b1;
      end else begin
        group_ok_q <= 1'b1;
      end
    end
  end

  // Gos, in the order of their commands: each waits for its writer's done.
  stream_fifo #(.Width(1 + SeqW), .Depth(GoDepth)) u_gos (
    .clk_i,
    .rst_ni      (rst_ni && !start && !ctl_reset && !settling),  // a token starts with none
    .in_valid_i  (eng_valid_o && eng_ready_i && need_wait),
    .in_ready_o  (go_in_ready),
    .in_data_i   ({wunit_q[BufW'(f_a)], wseq_q[BufW'(f_a)]}),
    .out_valid_o (go_out_valid),
    .out_ready_i (writer_done),
    .out_data_o  ({go_out_unit, go_out_seq}),
    // verilator lint_off PINCONNECTEMPTY
    .count_o     ()
    // verilator lint_on PINCONNECTEMPTY
  );
  assign writer_done = go_out_valid && (go_out_unit ? eng_done_q > go_out_seq
                                                    : vec_done_q > go_out_seq);
  assign eng_go_o    = writer_done;

  always_comb begin
    pc_d = pc_q;
    if (start) pc_d = '0;
    else if (fire) pc_d = pc_q + 1'b1;
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni || ctl_reset || settling) begin
      pc_q           <= '0;
      busy_q         <= 1'b0;
      done_q         <= 1'b0;
      err_q          <= 1'b0;
      error_q        <= '0;
      cycles_q       <= '0;
      irq_o          <= 1'b0;
      written_q      <= '0;
      waited_q       <= '1;
      wunit_q        <= '0;
      wseq_q         <= '0;
      eng_issued_q   <= '0;
      eng_done_q     <= '0;
      vec_issued_q   <= '0;
      vec_done_q     <= '0;
      kstore_q       <= '0;
      vstore_q       <= '0;
    end else begin
      pc_q  <= pc_d;
      irq_o <= 1'b0;
      if (busy_q) cycles_q <= cycles_q + 1'b1;
      if (eng_done_i) eng_done_q <= eng_done_q + 1'b1;
      if (vec_done_i) vec_done_q <= vec_done_q + 1'b1;
      if (start) begin
        busy_q         <= 1'b1;
        done_q         <= 1'b0;
        err_q          <= 1'b0;
        error_q        <= '0;
        cycles_q       <= '0;
        written_q      <= '0;
        waited_q       <= '1;
        eng_issued_q   <= '0;
        eng_done_q     <= '0;
        vec_issued_q   <= '0;
        vec_done_q     <= '0;
        kstore_q       <= '0;
        vstore_q       <= '0;
      end else if (fail_end || fail_opcode || fail_mem) begin
        busy_q  <= 1'b0;
        err_q   <= 1'b1;
        error_q <= {fail_mem ? ErrMemory : fail_end ? ErrNoEnd : ErrOpcode, 12'd0, 16'(pc_q)};
      end else if (end_now) begin
        busy_q <= 1'b0;
        done_q <= 1'b1;
        irq_o  <= 1'b1;
      end else if (fire) begin
        if (stop) written_q <= dst_bit;
        else written_q <= written_q | dst_bit;
        if (need_wait) waited_q[BufW'(f_a)] <= 1'b1;
        if (f_dst != 0) begin
          waited_q[BufW'(f_dst)] <= 1'b0;
          wunit_q[BufW'(f_dst)]  <= is_eng;
          wseq_q[BufW'(f_dst)]   <= is_eng ? eng_issued_q : vec_issued_q;
        end
        if (is_eng) eng_issued_q <= eng_issued_q + 1'b1;
        if (is_vec) vec_issued_q <= vec_issued_q + 1'b1;
        if (op == OpKvStore) kstore_q <= vec_issued_q + 1'b1;  // its done
        if (op == OpVStore) vstore_q <= vec_issued_q + 1'b1;
      end
    end
  end

endmodule
