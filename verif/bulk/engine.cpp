// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief Bulk driver for rtl/engine.sv (verif/bulk.py builds it): one
/// memory model of axi_mem.h per port, the x buffers as a synchronous RAM,
/// and the write port checked as it goes.
///
/// Input (little-endian uint32 on stdin): the words per channel, then each
/// port's channel image; the number of x buffers, then per buffer its id,
/// its BF16 elements and the elements two per word (low half first); the
/// number of commands, then per command op (0 MATVEC, 1 SCORES, 2 VALUES),
/// rows, cols, address (low word, high word), a, dst, bf16, wait, heads,
/// group, t, cap, stride and scale (FP32 bits). Commands are offered in order, each
/// held until taken. A command with wait gets its go_i as the controller may give
/// it (D-042), in order, once it is taken: after every earlier command is
/// done (its dependency is the one before), or at once (an earlier one, done
/// already), then after a random delay. Until then its x buffer reads as NaN,
/// so a step before the go writes wrong bits: a waiting command must be its
/// buffer's first reader, and the only waiting one (later commands may share
/// it, as k and v share q's).
/// Plusargs: the memories' (axi_bench.h: +latency, +outstanding, +rate,
/// +pause, +pause_max, +bad with +bad_port, +slow_rate with +slow_port),
/// +offer (permille of cycles a new command is offered), +go_early (permille
/// of waiting commands whose go does not wait for the earlier ones' done),
/// +go_delay (a go comes 0 to go_delay cycles late, uniformly), +seed (port p's memory draws
/// with (seed << 16) + p, so no two runs or ports share a stream).
/// Output: per command its rows' results in order (FP32, or BF16 in the low
/// half): rows for a MATVEC, heads * stride for SCORES (0 where i >= t), heads
/// * cols for VALUES; the cycle it was taken and the cycle of its done_o; then the
/// cycles, the cycles any memory waited on RREADY, the ports with err_o, and
/// the run's wall time in microseconds (reset to the last drain cycle).
/// Checks, each ending the run with a message: writes go to the oldest
/// command not done, to its dst, to its rows (for SCORES, h * stride + i with
/// i < t), each row once, with its bf16; done_o comes after all of its rows; nothing is written and every
/// memory is idle for kDrainCycles cycles after the last done_o.

#include <algorithm>
#include <array>
#include <chrono>
#include <optional>

#include "Vengine.h"
#include "Vengine___024root.h"
#include "axi_bench.h"

namespace {

using Params = Vengine___024root;  // the build's parameters (verif/bulk.py: --public-params)
constexpr size_t kBeatWords = Params::engine__DOT__DataWidth / 32;
constexpr size_t kAddrBits = Params::engine__DOT__AddrWidth;
constexpr size_t kXValues = Params::engine__DOT__E;     ///< BF16 values per read
constexpr size_t kXReads = Params::engine__DOT__Heads;  ///< x read ports
constexpr size_t kXAddrBits = Params::engine__DOT__RowWidth;
constexpr size_t kWrites = Params::engine__DOT__Writes;  ///< the write port's slots
constexpr size_t kBufs = size_t{1} << Params::engine__DOT__BufWidth;
constexpr size_t kCmdWords = 15;  ///< op, rows, cols, addr lo, addr hi, a, dst, bf16, wait, heads, group, t, cap, stride, scale
constexpr uint32_t kOpScores = Params::engine__DOT__OpScores, kOpValues = Params::engine__DOT__OpValues;
constexpr uint16_t kNan = 0x7FC0;  ///< what a waiting command's x holds before its go
constexpr int kExitBadInput = 2, kExitHang = 3, kExitBadWrite = 4, kExitAxi = 5;

/// @brief One engine command and what it wrote.
struct Command {
  uint32_t op, rows, cols;
  uint64_t addr;
  uint32_t a, dst;
  bool bf16;
  bool wait;                      ///< steps only after its go
  uint32_t heads, group, t, cap, stride;  ///< attention; stride: S and P's rows
  uint32_t scale;                 ///< SCORES, FP32 bits
  std::vector<uint32_t> out;      ///< results by row
  std::vector<bool> written;      ///< rows written
  uint64_t taken = 0, done = 0;   ///< cycles

  /// @brief The rows of dst the command's output spans.
  size_t n_rows() const {
    return op == kOpScores ? size_t{heads} * stride : op == kOpValues ? size_t{heads} * cols : rows;
  }
  /// @brief Whether the command writes this row of dst.
  bool writes(size_t row) const { return row < n_rows() && (op != kOpScores || row % stride < t); }
};

/// @brief A write the engine must not make, with what it did.
struct WriteError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// @brief The engine, its memories and buffers, and the test's side of its ports.
struct Bench {
  Vengine& dut;
  std::vector<AxiReadPort>& ports;
  std::vector<std::vector<uint16_t>>& x_bufs;  ///< by buffer id
  std::vector<Command>& commands;
  uint64_t offer_permille;
  uint64_t go_early;               ///< permille of gos that do not wait for the earlier dones
  uint64_t go_delay;               ///< a go comes up to this many cycles late
  std::mt19937_64 rng;            ///< draws the command offers and go delays
  std::vector<bool> poisoned;     ///< by buffer id: a waiting reader has no go yet
  size_t next_cmd = 0;            ///< the first command not yet taken
  size_t next_go = 0;             ///< the first waiting command without its go
  /// @brief next_go's go, drawn once the command is taken.
  struct GoDraw {
    bool early;     ///< does not wait for the earlier commands' done
    uint64_t left;  ///< cycles still to wait once due
  };
  std::optional<GoDraw> go_draw;
  size_t oldest = 0;              ///< the first command not yet done
  bool offering = false;          ///< a command is on the port, waiting to be taken
  std::array<std::array<uint16_t, kXValues>, kXReads> x_q{};  ///< the RAM's output registers
  uint64_t cycles = 0, writes = 0;

  /// @brief The x RAM's read: E values of a group, zeros past the buffer.
  std::array<uint16_t, kXValues> read_x(uint32_t buf, uint64_t group) const {
    std::array<uint16_t, kXValues> v{};
    const std::vector<uint16_t>& x = x_bufs[buf];  // empty while no command reads it
    const bool poison = poisoned[buf];
    for (size_t e = 0; e < kXValues; ++e) {
      const size_t k = group * kXValues + e;
      if (k < x.size()) v[e] = poison ? kNan : x[k];
    }
    return v;
  }

  /// @brief go_i for this cycle: next_go's go once it is taken, every
  /// earlier command is done (unless drawn early) and its random delay has
  /// passed.
  bool go() {
    while (next_go < commands.size() && !commands[next_go].wait) ++next_go;
    if (next_go >= next_cmd) return false;
    if (!go_draw) go_draw = GoDraw{rng() % 1000 < go_early, std::uniform_int_distribution<uint64_t>(0, go_delay)(rng)};
    if (!go_draw->early && oldest < next_go) return false;
    if (go_draw->left-- > 0) return false;
    poisoned[commands[next_go++].a] = false;
    go_draw.reset();
    return true;
  }

  /// @brief Check and keep one write of the write port.
  /// @throws WriteError if the write breaks a rule of this file's header.
  void write() {
    if (oldest >= commands.size()) throw WriteError("a write after the last command's done");
    Command& c = commands[oldest];
    if (dut.w_buf_o != c.dst) throw WriteError(std::format("buffer {}, not dst {}", dut.w_buf_o, c.dst));
    if (static_cast<bool>(dut.w_bf16_o) != c.bf16) throw WriteError("bf16 is not the command's");
    for (size_t s = 0; s < kWrites; ++s) {
      if (!get_bits(dut.w_mask_o, s, 1)) continue;
      const uint64_t row = dut.w_row_o + s;
      if (!c.writes(row)) throw WriteError(std::format("row {}, not one of the command's", row));
      if (c.written[row]) throw WriteError(std::format("row {} written twice", row));
      c.written[row] = true;
      c.out[row] = static_cast<uint32_t>(get_bits(dut.w_data_o, 32 * s, 32));
    }
    ++writes;
  }

  /// @brief The oldest command's done_o.
  /// @throws WriteError if it comes before all of its rows.
  void done() {
    if (oldest >= commands.size()) throw WriteError("done_o with no command left");
    Command& c = commands[oldest];
    for (size_t row = 0; row < c.written.size(); ++row) {
      if (c.writes(row) && !c.written[row]) throw WriteError(std::format("done_o before row {}", row));
    }
    c.done = cycles;
    ++oldest;
  }

  /// @brief One clock cycle: put the memories' outputs, the RAM's register
  /// and a command on the engine, settle, take a command, a write and a done,
  /// let the memories and the RAM see the engine's outputs, then the rising
  /// edge. In reset nothing is offered and the memories do not tick.
  /// @throws AxiError, WriteError on a broken rule.
  void cycle() {
    const bool run = dut.rst_ni;
    for (size_t p = 0; p < ports.size(); ++p) {
      set_bits(dut.ar_ready_i, p, 1, ports[p].ar_ready());
      set_bits(dut.r_valid_i, p, 1, ports[p].r_valid());
      set_bits(dut.r_resp_i, 2 * p, 2, ports[p].r_resp());
      if (const auto data = ports[p].r_data(); !data.empty()) {
        std::ranges::copy(data, std::as_writable_bytes(std::span(dut.r_data_i.data() + kBeatWords * p, kBeatWords)).begin());
      }
    }
    for (size_t r = 0; r < kXReads; ++r) {
      for (size_t e = 0; e < kXValues; ++e) set_bits(dut.x_data_i, 16 * (kXValues * r + e), 16, x_q[r][e]);
    }
    if (run && !offering && next_cmd < commands.size()) offering = rng() % 1000 < offer_permille;
    dut.go_i = run && go();
    dut.cmd_valid_i = offering;
    if (offering) {
      const Command& c = commands[next_cmd];
      dut.cmd_op_i = c.op;
      dut.cmd_heads_i = c.heads;
      dut.cmd_group_i = c.group;
      dut.cmd_t_i = c.t;
      dut.cmd_cap_i = c.cap;
      dut.cmd_stride_i = c.stride;
      dut.cmd_scale_i = c.scale;
      dut.cmd_rows_i = c.rows;
      dut.cmd_cols_i = c.cols;
      dut.cmd_addr_i = c.addr;
      dut.cmd_a_i = c.a;
      dut.cmd_dst_i = c.dst;
      dut.cmd_bf16_i = c.bf16;
      dut.cmd_wait_i = c.wait;
    }
    dut.clk_i = 0;
    dut.eval();
    if (offering && dut.cmd_ready_o) {
      commands[next_cmd++].taken = cycles;
      offering = false;
    }
    if (run) {
      if (dut.w_valid_o) write();
      if (dut.done_o) done();
      for (size_t p = 0; p < ports.size(); ++p) {
        ports[p].tick({static_cast<bool>(get_bits(dut.ar_valid_o, p, 1)), get_bits(dut.ar_addr_o, kAddrBits * p, kAddrBits),
                       static_cast<unsigned>(get_bits(dut.ar_len_o, 8 * p, 8)),
                       static_cast<unsigned>(get_bits(dut.ar_size_o, 3 * p, 3)),
                       static_cast<unsigned>(get_bits(dut.ar_burst_o, 2 * p, 2))},
                      get_bits(dut.r_ready_o, p, 1));
      }
    }
    std::array<std::array<uint16_t, kXValues>, kXReads> x_next;
    for (size_t r = 0; r < kXReads; ++r) x_next[r] = read_x(dut.x_buf_o, get_bits(dut.x_addr_o, kXAddrBits * r, kXAddrBits));
    dut.clk_i = 1;
    dut.eval();
    x_q = x_next;
    if (run) ++cycles;
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
/// @return 0; 2 if the input is malformed; 3 if the engine hangs; 4 if it
///   breaks a write rule; 5 if it breaks an AXI rule (each with a message on
///   stderr).
int main(int argc, char** argv) {
  VerilatedContext context;
  context.commandArgs(argc, argv);
  auto dut = std::make_unique<Vengine>(&context);
  constexpr size_t n_ports = Params::engine__DOT__Ports;

  const std::vector<uint32_t> words = read_words(stdin);
  Reader in{words};
  std::vector<std::span<const uint32_t>> images;
  std::vector<std::vector<uint16_t>> x_bufs(kBufs);
  std::vector<Command> commands;
  try {
    const size_t n_mem = in.next();
    for (size_t p = 0; p < n_ports; ++p) images.push_back(in.take(n_mem));
    for (uint32_t b = 0, n = in.next(); b < n; ++b) {
      const uint32_t id = in.next(), n_elems = in.next();
      if (id >= kBufs) throw std::out_of_range("x buffer id past BufWidth");
      const auto packed = in.take((n_elems + 1) / 2);
      std::vector<uint16_t>& buf = x_bufs[id];
      for (uint32_t k = 0; k < n_elems; ++k) buf.push_back(static_cast<uint16_t>(packed[k / 2] >> 16 * (k % 2)));
    }
    for (uint32_t i = 0, n = in.next(); i < n; ++i) {
      const auto f = in.take(kCmdWords);
      Command c{.op = f[0], .rows = f[1], .cols = f[2], .addr = f[3] | uint64_t{f[4]} << 32, .a = f[5],
                .dst = f[6], .bf16 = f[7] != 0, .wait = f[8] != 0, .heads = f[9], .group = f[10], .t = f[11],
                .cap = f[12], .stride = f[13], .scale = f[14]};
      c.out.resize(c.n_rows());
      c.written.resize(c.n_rows());
      commands.push_back(std::move(c));
      if (f[5] >= kBufs) throw std::out_of_range("a past BufWidth");
      if ((f[0] == kOpScores || f[0] == kOpValues) && (f[12] == 0 || f[13] == 0))
        throw std::out_of_range("attention with cap or stride 0");
    }
    std::vector<bool> read(kBufs);
    for (const Command& c : commands) {
      if (c.wait && read[c.a]) throw std::out_of_range(std::format("x buffer {} read before its waiting command", c.a));
      read[c.a] = true;
    }
    if (in.at != words.size()) throw std::out_of_range("words left after the commands");
  } catch (const std::out_of_range& e) {
    std::println(stderr, "bad input: {}", e.what());
    return kExitBadInput;
  }

  const uint64_t seed = plusarg(context, "seed", 1);
  std::vector<AxiReadPort> ports;
  for (size_t p = 0; p < n_ports; ++p) {
    ports.emplace_back(std::as_bytes(images[p]), axi_settings(context, 4 * kBeatWords, p), (seed << 16) + p);
  }

  Bench bench{.dut = *dut,
              .ports = ports,
              .x_bufs = x_bufs,
              .commands = commands,
              .offer_permille = plusarg(context, "offer", 1000),
              .go_early = plusarg(context, "go_early", 0),
              .go_delay = plusarg(context, "go_delay", 0),
              .rng = std::mt19937_64(seed),
              .poisoned = std::vector<bool>(kBufs)};
  for (const Command& c : commands) {
    if (c.wait) bench.poisoned[c.a] = true;
  }
  try {
    const auto start = std::chrono::steady_clock::now();
    reset(bench);
    const auto cycles = run(
        bench, [&] { return bench.oldest == commands.size(); },
        [&] { return bench.writes + bench.oldest; });
    if (!cycles) {
      std::println(stderr, "hung: {} of {} commands taken, {} done, after {} cycles", bench.next_cmd,
                   commands.size(), bench.oldest, bench.cycles);
      return kExitHang;
    }
    for (int i = 0; i < kDrainCycles; ++i) bench.cycle();
    if (!std::ranges::all_of(ports, &AxiReadPort::idle)) throw WriteError("a memory still busy after the last done");
    const auto wall = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
    std::vector<uint32_t> out;
    for (const Command& c : commands) {
      out.insert(out.end(), c.out.begin(), c.out.end());
      out.push_back(static_cast<uint32_t>(c.taken));
      out.push_back(static_cast<uint32_t>(c.done));
    }
    uint64_t stalls = 0;
    for (const AxiReadPort& p : ports) stalls += p.r_stalls();
    out.push_back(static_cast<uint32_t>(*cycles));
    out.push_back(static_cast<uint32_t>(stalls));
    out.push_back(static_cast<uint32_t>(get_bits(dut->err_o, 0, std::min<size_t>(n_ports, 32))));
    out.push_back(static_cast<uint32_t>(wall.count()));
    std::fwrite(out.data(), sizeof(uint32_t), out.size(), stdout);
  } catch (const AxiError& e) {
    std::println(stderr, "AXI: {}", e.what());
    return kExitAxi;
  } catch (const WriteError& e) {
    std::println(stderr, "cycle {}: {}", bench.cycles, e.what());
    return kExitBadWrite;
  }
  dut->final();
  return 0;
}
