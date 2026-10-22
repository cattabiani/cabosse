// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief Driver for rtl/vec_unit.sv (verif/vector.py): fills buffers through
/// the ext port, runs commands, reads buffers back.
///
/// Input (little-endian uint32 on stdin): the number of writes, then per write
/// buffer, first element, lane mask and Lanes words (raw bits, BF16 in the low
/// half); the number of commands, then per command op, dst, a, b, c, flags, n,
/// m, scalar, t, stride and wait; the number of reads, then per read buffer
/// and first element. The writes go one per cycle after reset. Commands are
/// offered in order, each held until taken; one with wait is offered only
/// after every earlier command is done (the controller's stop, D-045), the
/// others as soon as the port is free, or at random cycles (+offer, permille;
/// +seed). After the last done_o and kDrainCycles more, the reads go one per
/// cycle.
/// Output: per command the cycle it was taken and the cycle of its done_o
/// (counted from the first command's offer); per read its Lanes words; the
/// cycles from the first offer to the last done_o.
/// Checks, each ending the run with a message: no done_o past the last command,
/// and no command left without one.

#include <random>

#include "Vvec_unit.h"
#include "Vvec_unit___024root.h"
#include "stream.h"

namespace {

using Params = Vvec_unit___024root;  // the build's parameters (verif/bulk.py: --public-params)
constexpr size_t kLanes = Params::vec_unit__DOT__Lanes;
constexpr size_t kCmdWords = 12;  ///< op, dst, a, b, c, flags, n, m, scalar, t, stride, wait
constexpr int kExitBadInput = 2, kExitHang = 3, kExitBadDone = 4;

/// @brief One ext write: raw bits into a buffer.
struct Write {
  uint32_t buf, addr, mask;
  std::span<const uint32_t> data;  ///< kLanes words
};

/// @brief One command and when it ran.
struct Command {
  std::span<const uint32_t> fields;  ///< kCmdWords: op ... stride, wait
  bool wait() const { return fields[11] != 0; }
  uint64_t taken = 0, done = 0;  ///< cycles from the first offer
};

/// @brief A done_o that belongs to no command.
struct DoneError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// @brief The unit and the test's side of its command port.
struct Bench {
  Vvec_unit& dut;
  std::vector<Command>& commands;
  uint64_t offer_permille;
  std::mt19937_64 rng;     ///< draws the offers
  size_t next_cmd = 0;     ///< the first command not yet taken
  size_t oldest = 0;       ///< the first command not yet done
  bool offering = false;   ///< a command is on the port, waiting to be taken
  bool running = false;    ///< commands may be offered
  uint64_t cycles = 0;     ///< since the first offer

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
    dut.t_i = f[9];
    dut.stride_i = f[10];
  }

  /// @brief One clock cycle: offer a command, settle, take it and a done_o,
  /// then the rising edge.
  /// @throws DoneError on a done_o with no command left.
  void cycle() {
    if (may_offer()) offering = true;
    dut.cmd_valid_i = offering;
    if (offering) put(commands[next_cmd]);
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

/// @brief Reads the input words of this file's header in order.
struct Reader {
  std::span<const uint32_t> words;
  size_t at = 0;

  /// @brief The next n words.
  /// @throws std::out_of_range if the input ends first.
  std::span<const uint32_t> take(size_t n) {
    if (n > words.size() - at) throw std::out_of_range("input too short");
    at += n;
    return words.subspan(at - n, n);
  }
  uint32_t next() { return take(1)[0]; }
};

}  // namespace

/// @brief Run the protocol of this file's header.
/// @return 0; 2 if the input is malformed; 3 if the unit hangs; 4 on a
///   done_o with no command (each with a message on stderr).
int main(int argc, char** argv) {
  VerilatedContext context;
  context.commandArgs(argc, argv);
  auto dut = std::make_unique<Vvec_unit>(&context);

  const std::vector<uint32_t> words = read_words(stdin);
  Reader in{words};
  std::vector<Write> writes;
  std::vector<Command> commands;
  std::vector<std::pair<uint32_t, uint32_t>> reads;
  try {
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

  Bench bench{.dut = *dut,
              .commands = commands,
              .offer_permille = plusarg(context, "offer", 1000),
              .rng = std::mt19937_64(plusarg(context, "seed", 1))};
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
    std::vector<uint32_t> out;
    for (const Command& c : commands) {
      out.push_back(static_cast<uint32_t>(c.taken));
      out.push_back(static_cast<uint32_t>(c.done));
    }
    for (const auto& [buf, addr] : reads) std::ranges::copy(bench.ext_read(buf, addr), std::back_inserter(out));
    out.push_back(static_cast<uint32_t>(*cycles));
    std::fwrite(out.data(), sizeof(uint32_t), out.size(), stdout);
  } catch (const DoneError& e) {
    std::println(stderr, "cycle {}: {}", bench.cycles, e.what());
    return kExitBadDone;
  }
  dut->final();
  return 0;
}
