// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief Driver for rtl/vec_unit.sv (verif/vector.py): fills buffers through
/// the ext port and a memory behind the AXI port, runs commands, reads both
/// back.
///
/// Input (little-endian uint32 on stdin): the memory's size in words, then
/// its words; the number of writes, then per write buffer, first element,
/// lane mask and Lanes words (raw bits, BF16 in the low half); the number of
/// commands, then per command op, dst, a, b, c, flags, n, m, scalar, cap,
/// addr (low, high), token, t, stride and wait; the number of reads, then per
/// read buffer and first element. The writes go one per cycle after reset.
/// Commands are offered in order, each held until taken; one with wait is
/// offered only after every earlier command is done (the controller's stop,
/// D-045), the others as soon as the port is free, or at random cycles
/// (+offer, permille; +seed). After the last done_o and kDrainCycles more,
/// the reads go one per cycle.
///
/// The memory starts at address 0, a read port (axi_bench.h's plusargs) and a
/// write port (the same with a w in front: +wlatency, +woutstanding, +wrate,
/// +wpause, +wpause_max, +wbad: the burst answered with SLVERR).
///
/// Output: per command the cycle it was taken and the cycle of its done_o
/// (counted from the first command's offer); per read its Lanes words; the
/// memory's words; the number of write responses, then the cycle of each;
/// err_o; the cycles from the first offer to the last done_o.
/// Checks, each ending the run with a message: no done_o past the last
/// command, no command left without one, the AXI rules (axi_mem.h), and no
/// burst left unanswered at the end.

#include <array>
#include <bit>
#include <random>

#include "Vvec_unit.h"
#include "Vvec_unit___024root.h"
#include "axi_bench.h"
#include "axi_mem.h"
#include "stream.h"

namespace {

using Params = Vvec_unit___024root;  // the build's parameters (verif/bulk.py: --public-params)
constexpr size_t kLanes = Params::vec_unit__DOT__Lanes;
constexpr size_t kAddrBits = Params::vec_unit__DOT__MemAddrW;
constexpr size_t kBeatWords = kLanes / 2;  ///< DataWidth = Lanes * 16 bits
constexpr size_t kCmdWords = 16;  ///< op ... scalar, cap, addr low, addr high, token, t, stride, wait
constexpr int kExitBadDone = 4;
constexpr int kExitAxi = 5;

/// @brief One ext write: raw bits into a buffer.
struct Write {
  uint32_t buf, addr, mask;
  std::span<const uint32_t> data;  ///< kLanes words
};

/// @brief One command and when it ran.
struct Command {
  std::span<const uint32_t> fields;  ///< kCmdWords
  bool wait() const { return fields[15] != 0; }
  uint64_t taken = 0, done = 0;  ///< cycles from the first offer
};

/// @brief A done_o that belongs to no command.
struct DoneError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// @brief Word i (little-endian) of a run of bytes.
uint32_t word_at(std::span<const std::byte> bytes, size_t i) {
  std::array<std::byte, 4> w;
  std::ranges::copy(bytes.subspan(4 * i, 4), w.begin());
  return std::bit_cast<uint32_t>(w);
}

/// @brief Set word i (little-endian) of a run of bytes.
void put_word(std::span<std::byte> bytes, size_t i, uint32_t word) {
  std::ranges::copy(std::bit_cast<std::array<std::byte, 4>>(word), bytes.subspan(4 * i, 4).begin());
}

/// @brief The write port's settings: axi_bench.h's plusargs with a w in front.
AxiMemSettings write_settings(VerilatedContext& context) {
  AxiMemSettings s;
  s.latency = plusarg(context, "wlatency", s.latency);
  s.outstanding = plusarg(context, "woutstanding", s.outstanding);
  s.rate_permille = plusarg(context, "wrate", s.rate_permille);
  s.pause_permille = plusarg(context, "wpause", s.pause_permille);
  s.pause_max = plusarg(context, "wpause_max", s.pause_max);
  s.bad_beat = plusarg(context, "wbad", s.bad_beat);
  s.beat_bytes = 4 * kBeatWords;
  return s;
}

/// @brief The unit, its memory, and the test's side of its command port.
struct Bench {
  Vvec_unit& dut;
  AxiReadPort& rd;
  AxiWritePort& wr;
  std::vector<Command>& commands;
  uint64_t offer_permille;
  std::mt19937_64 rng;     ///< draws the offers
  size_t next_cmd = 0;     ///< the first command not yet taken
  size_t oldest = 0;       ///< the first command not yet done
  bool offering = false;   ///< a command is on the port, waiting to be taken
  bool running = false;    ///< commands may be offered
  uint64_t cycles = 0;     ///< since the first offer
  std::vector<uint64_t> answers;  ///< the cycle of each write response
  std::vector<std::byte> w_beat = std::vector<std::byte>(4 * kBeatWords);

  /// @brief Whether the next command may be offered this cycle.
  bool may_offer() {
    if (!running || offering || next_cmd >= commands.size()) return false;
    if (commands[next_cmd].wait() && oldest < next_cmd) return false;
    return rng() % 1000 < offer_permille;
  }

  /// @brief Put a command's fields on the port.
  void put(const Command& c) {
    const auto f = c.fields;
    dut.cmd_op_i = f[0];
    dut.cmd_dst_i = f[1];
    dut.cmd_a_i = f[2];
    dut.cmd_b_i = f[3];
    dut.cmd_c_i = f[4];
    dut.cmd_flags_i = f[5];
    dut.cmd_n_i = f[6];
    dut.cmd_m_i = f[7];
    dut.cmd_scalar_i = f[8];
    dut.cmd_cap_i = f[9];
    dut.cmd_addr_i = (static_cast<uint64_t>(f[11]) << 32 | f[10]) & ((1ull << kAddrBits) - 1);
    dut.token_i = f[12];
    dut.t_i = f[13];
    dut.stride_i = f[14];
  }

  /// @brief The memory's outputs onto the unit's AXI inputs.
  void drive_memory() {
    dut.ar_ready_i = rd.ar_ready();
    dut.r_valid_i = rd.r_valid();
    dut.r_resp_i = rd.r_resp();
    if (const auto data = rd.r_data(); !data.empty()) {
      for (size_t i = 0; i < kBeatWords; ++i) set_bits(dut.r_data_i, 32 * i, 32, word_at(data, i));
    }
    dut.aw_ready_i = wr.aw_ready();
    dut.w_ready_i = wr.w_ready();
    dut.b_valid_i = wr.b_valid();
    dut.b_resp_i = wr.b_resp();
  }

  /// @brief The clock edge for the memory: the unit's AXI outputs this cycle.
  void tick_memory() {
    rd.tick({static_cast<bool>(dut.ar_valid_o), dut.ar_addr_o, dut.ar_len_o, dut.ar_size_o, dut.ar_burst_o},
            dut.r_ready_o);
    for (size_t i = 0; i < kBeatWords; ++i) put_word(w_beat, i, static_cast<uint32_t>(get_bits(dut.w_data_o, 32 * i, 32)));
    wr.tick({static_cast<bool>(dut.aw_valid_o), dut.aw_addr_o, dut.aw_len_o, dut.aw_size_o, dut.aw_burst_o},
            {static_cast<bool>(dut.w_valid_o), w_beat, get_bits(dut.w_strb_o, 0, 4 * kBeatWords),
             static_cast<bool>(dut.w_last_o)},
            dut.b_ready_o);
  }

  /// @brief One clock cycle: offer a command, settle, take it, a done_o and a
  /// write response, then the rising edge. In reset the memory does not tick.
  /// @throws DoneError on a done_o with no command left; AxiError on a
  ///   broken AXI rule.
  void cycle() {
    if (may_offer()) offering = true;
    dut.cmd_valid_i = offering;
    if (offering) put(commands[next_cmd]);
    drive_memory();
    dut.clk_i = 0;
    dut.eval();
    if (offering && dut.cmd_ready_o) {
      commands[next_cmd++].taken = cycles;
      offering = false;
    }
    if (dut.rst_ni && dut.done_o) {
      if (oldest >= commands.size()) throw DoneError("done_o with no command left");
      commands[oldest++].done = cycles;
    }
    if (dut.rst_ni && dut.b_valid_i && dut.b_ready_o) answers.push_back(cycles);
    if (dut.rst_ni) tick_memory();
    dut.clk_i = 1;
    dut.eval();
    if (running) ++cycles;
  }

  /// @brief One ext write, in one cycle.
  void ext_write(const Write& w) {
    dut.ext_w_valid_i = 1;
    dut.ext_w_buf_i = w.buf;
    dut.ext_w_addr_i = w.addr;
    dut.ext_w_mask_i = w.mask;
    for (size_t l = 0; l < kLanes; ++l) set_bits(dut.ext_w_data_i, 32 * l, 32, w.data[l]);
    cycle();
    dut.ext_w_valid_i = 0;
  }

  /// @brief One ext read: kLanes words, from the cycle after.
  std::vector<uint32_t> ext_read(uint32_t buf, uint32_t addr) {
    dut.ext_r_valid_i = 1;
    dut.ext_r_buf_i = buf;
    dut.ext_r_addr_i = addr;
    cycle();
    dut.ext_r_valid_i = 0;
    dut.clk_i = 0;
    dut.eval();
    std::vector<uint32_t> out(kLanes);
    for (size_t l = 0; l < kLanes; ++l) out[l] = static_cast<uint32_t>(get_bits(dut.ext_r_data_o, 32 * l, 32));
    return out;
  }
};

}  // namespace

/// @brief Run the protocol of this file's header.
/// @return 0; 2 if the input is malformed; 3 if the unit hangs or leaves a
///   burst unanswered; 4 on a done_o with no command; 5 on a broken AXI rule
///   (each with a message on stderr).
int main(int argc, char** argv) {
  VerilatedContext context;
  context.commandArgs(argc, argv);
  auto dut = std::make_unique<Vvec_unit>(&context);

  const std::vector<uint32_t> words = read_words(stdin);
  Reader in{words};
  std::vector<uint32_t> mem;
  std::vector<Write> writes;
  std::vector<Command> commands;
  std::vector<std::pair<uint32_t, uint32_t>> reads;
  try {
    const auto image = in.take(in.next());
    mem.assign(image.begin(), image.end());
    for (size_t i = 0, n = in.next(); i < n; ++i) {
      const auto f = in.take(3);
      writes.push_back({f[0], f[1], f[2], in.take(kLanes)});
    }
    for (size_t i = 0, n = in.next(); i < n; ++i) commands.push_back({in.take(kCmdWords)});
    for (size_t i = 0, n = in.next(); i < n; ++i) {
      const auto f = in.take(2);
      reads.emplace_back(f[0], f[1]);
    }
    if (in.at != words.size()) throw std::out_of_range("words left after the reads");
  } catch (const std::out_of_range& e) {
    std::println(stderr, "bad input: {}", e.what());
    return kExitBadInput;
  }

  const uint64_t seed = plusarg(context, "seed", 1);
  const std::span<std::byte> bytes = std::as_writable_bytes(std::span(mem));
  AxiReadPort rd(bytes, axi_settings(context, 4 * kBeatWords, 0), seed << 16);
  AxiWritePort wr(bytes, write_settings(context), (seed << 16) + 1);
  Bench bench{.dut = *dut,
              .rd = rd,
              .wr = wr,
              .commands = commands,
              .offer_permille = plusarg(context, "offer", 1000),
              .rng = std::mt19937_64(seed)};
  try {
    reset(bench);
    for (const Write& w : writes) bench.ext_write(w);
    bench.running = true;
    const auto cycles = run(
        bench, [&] { return bench.oldest == commands.size(); },
        [&] { return bench.next_cmd + bench.oldest; });
    if (!cycles) {
      std::println(stderr, "hung: {} of {} commands taken, {} done, after {} cycles", bench.next_cmd,
                   commands.size(), bench.oldest, bench.cycles);
      return kExitHang;
    }
    bench.running = false;
    for (int i = 0; i < kDrainCycles; ++i) bench.cycle();
    if (!rd.idle() || !wr.idle()) {
      std::println(stderr, "a burst still open after the last done ({} written, {} answered)", wr.n_bursts(),
                   wr.n_answered());
      return kExitHang;
    }
    std::vector<uint32_t> out;
    for (const Command& c : commands) {
      out.push_back(static_cast<uint32_t>(c.taken));
      out.push_back(static_cast<uint32_t>(c.done));
    }
    for (const auto& [buf, addr] : reads) std::ranges::copy(bench.ext_read(buf, addr), std::back_inserter(out));
    std::ranges::copy(mem, std::back_inserter(out));
    out.push_back(static_cast<uint32_t>(bench.answers.size()));
    for (const uint64_t a : bench.answers) out.push_back(static_cast<uint32_t>(a));
    out.push_back(dut->err_o);
    out.push_back(static_cast<uint32_t>(*cycles));
    std::fwrite(out.data(), sizeof(uint32_t), out.size(), stdout);
  } catch (const DoneError& e) {
    std::println(stderr, "cycle {}: {}", bench.cycles, e.what());
    return kExitBadDone;
  } catch (const AxiError& e) {
    std::println(stderr, "AXI: {}", e.what());
    return kExitAxi;
  }
  dut->final();
  return 0;
}
