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
/// number of commands, then per command rows, cols, address (low word, high
/// word), a, dst and bf16. Commands are offered in order, each held until
/// taken.
/// Plusargs: +latency, +outstanding, +rate, +pause, +pause_max (every port's
/// AxiMemSettings, permille for rates), +slow_port and +slow_rate (one port
/// with its own rate), +bad_port and +bad (that port answers its beat number
/// bad with an error), +offer (permille of cycles a new command is offered),
/// +seed (port p draws with seed + 1 + p).
/// Output: per command its rows' results in order (FP32, or BF16 in the low
/// half), the cycle it was taken and the cycle of its done_o; then the
/// cycles, the cycles any memory waited on RREADY, and the ports with err_o.
/// Checks, each ending the run with a message: writes go to the oldest
/// command not done, to its dst, inside its rows, each row once, with its
/// bf16; done_o comes after all of its rows; nothing is written and every
/// memory is idle for kDrainCycles cycles after the last done_o.

#include <algorithm>
#include <array>
#include <map>
#include <type_traits>

#include "Vengine.h"
#include "axi_mem.h"
#include "stream.h"

namespace {

constexpr size_t kBeatWords = 8;      ///< 256 bits
constexpr size_t kCmdWords = 7;       ///< rows, cols, addr lo, addr hi, a, dst, bf16
constexpr size_t kXValues = 4;        ///< E: BF16 values per read
constexpr size_t kWrites = 8;         ///< the write port's slots
constexpr uint64_t kMaxIdleCycles = 100'000;  ///< cycles with no write before calling it a hang
constexpr int kExitBadInput = 2, kExitHang = 3, kExitBadWrite = 4, kExitAxi = 5;

/// @brief Bits [lsb, lsb + width) of a Verilated signal, width <= 64.
/// @tparam T an integer (narrow signals) or VlWide (wide ones).
template <typename T>
uint64_t get_bits(const T& sig, size_t lsb, size_t width) {
  uint64_t v = 0;
  if constexpr (std::is_integral_v<T>) {
    v = static_cast<uint64_t>(sig) >> lsb;
  } else {
    for (size_t i = 0; i < width + lsb % 32; i += 32) {
      const size_t w = lsb / 32 + i / 32;
      if (w < sizeof(T) / 4) v |= uint64_t{sig[w]} << i;
    }
    v >>= lsb % 32;
  }
  return width >= 64 ? v : v & ((uint64_t{1} << width) - 1);
}

/// @brief Set bits [lsb, lsb + width) of a Verilated signal, width <= 32.
template <typename T>
void set_bits(T& sig, size_t lsb, size_t width, uint32_t value) {
  const uint64_t mask = (uint64_t{1} << width) - 1;
  if constexpr (std::is_integral_v<T>) {
    sig = static_cast<T>((sig & ~(mask << lsb)) | ((value & mask) << lsb));
  } else {
    for (size_t i = 0; i < width; ++i) {
      const size_t b = lsb + i;
      sig[b / 32] = (sig[b / 32] & ~(1u << b % 32)) | (((value >> i) & 1u) << b % 32);
    }
  }
}

/// @brief One engine command and what it wrote.
struct Command {
  uint32_t rows, cols;
  uint64_t addr;
  uint32_t a, dst;
  bool bf16;
  std::vector<uint32_t> out;      ///< results by row
  std::vector<bool> written;      ///< rows written
  uint64_t taken = 0, done = 0;   ///< cycles
};

/// @brief A write the engine must not make, with what it did.
struct WriteError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// @brief The engine, its memories and buffers, and the test's side of its ports.
struct Bench {
  Vengine& dut;
  std::vector<AxiReadPort>& ports;
  std::map<uint32_t, std::vector<uint16_t>>& x_bufs;
  std::vector<Command>& commands;
  uint64_t offer_permille;
  std::mt19937_64 rng;            ///< draws the command offers
  size_t next_cmd = 0;            ///< the first command not yet taken
  size_t oldest = 0;              ///< the first command not yet done
  bool offering = false;          ///< a command is on the port, waiting to be taken
  std::array<uint16_t, kXValues> x_q{};  ///< the RAM's output register
  uint64_t cycles = 0, writes = 0;

  /// @brief Hold the engine in reset for kResetCycles cycles, then release it.
  void reset() {
    dut.rst_ni = 0;
    for (int i = 0; i < kResetCycles; ++i) cycle();
    dut.rst_ni = 1;
  }

  /// @brief The x RAM's read: E values of a group, zeros past the buffer.
  std::array<uint16_t, kXValues> read_x(uint32_t buf, uint64_t group) const {
    std::array<uint16_t, kXValues> v{};
    const auto it = x_bufs.find(buf);
    if (it == x_bufs.end()) return v;  // unused while no command reads it
    for (size_t e = 0; e < kXValues; ++e) {
      const size_t k = group * kXValues + e;
      if (k < it->second.size()) v[e] = it->second[k];
    }
    return v;
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
      if (row >= c.rows) throw WriteError(std::format("row {} of a {}-row command", row, c.rows));
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
    if (!std::ranges::all_of(c.written, std::identity{})) throw WriteError("done_o before every row");
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
        if (data.size() != 4 * kBeatWords) throw std::length_error("memory beat is not the engine's width");
        std::ranges::copy(data, std::as_writable_bytes(std::span(dut.r_data_i.data() + kBeatWords * p, kBeatWords)).begin());
      }
    }
    for (size_t e = 0; e < kXValues; ++e) set_bits(dut.x_data_i, 16 * e, 16, x_q[e]);
    if (run && !offering && next_cmd < commands.size()) offering = rng() % 1000 < offer_permille;
    dut.cmd_valid_i = offering;
    if (offering) {
      const Command& c = commands[next_cmd];
      dut.cmd_rows_i = c.rows;
      dut.cmd_cols_i = c.cols;
      dut.cmd_addr_i = c.addr;
      dut.cmd_a_i = c.a;
      dut.cmd_dst_i = c.dst;
      dut.cmd_bf16_i = c.bf16;
    }
    dut.clk_i = 0;
    dut.eval();
    if (offering && dut.cmd_ready_o) {
      commands[next_cmd++].taken = cycles;
      offering = false;
    }
    if (run && dut.w_valid_o) write();
    if (run && dut.done_o) done();
    if (run) {
      for (size_t p = 0; p < ports.size(); ++p) {
        ports[p].tick({static_cast<bool>(get_bits(dut.ar_valid_o, p, 1)), get_bits(dut.ar_addr_o, 34 * p, 34),
                       static_cast<unsigned>(get_bits(dut.ar_len_o, 8 * p, 8)),
                       static_cast<unsigned>(get_bits(dut.ar_size_o, 3 * p, 3)),
                       static_cast<unsigned>(get_bits(dut.ar_burst_o, 2 * p, 2))},
                      get_bits(dut.r_ready_o, p, 1));
      }
    }
    const auto x_next = read_x(dut.x_buf_o, dut.x_addr_o);
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
  const size_t n_ports = sizeof(dut->r_data_i) / (4 * kBeatWords);
  static_assert(sizeof(Vengine::x_data_i) * 8 >= 16 * kXValues, "the driver is for E = 4");

  const std::vector<uint32_t> words = read_words(stdin);
  Reader in{words};
  std::vector<std::span<const uint32_t>> images;
  std::map<uint32_t, std::vector<uint16_t>> x_bufs;
  std::vector<Command> commands;
  try {
    const size_t n_mem = in.next();
    for (size_t p = 0; p < n_ports; ++p) images.push_back(in.take(n_mem));
    for (uint32_t b = 0, n = in.next(); b < n; ++b) {
      const uint32_t id = in.next(), n_elems = in.next();
      const auto packed = in.take((n_elems + 1) / 2);
      std::vector<uint16_t>& buf = x_bufs[id];
      for (uint32_t k = 0; k < n_elems; ++k) buf.push_back(static_cast<uint16_t>(packed[k / 2] >> 16 * (k % 2)));
    }
    for (uint32_t i = 0, n = in.next(); i < n; ++i) {
      const auto f = in.take(kCmdWords);
      commands.push_back({.rows = f[0], .cols = f[1], .addr = f[2] | uint64_t{f[3]} << 32, .a = f[4],
                          .dst = f[5], .bf16 = f[6] != 0, .out = std::vector<uint32_t>(f[0]),
                          .written = std::vector<bool>(f[0])});
    }
    if (in.at != words.size()) throw std::out_of_range("words left after the commands");
  } catch (const std::out_of_range& e) {
    std::println(stderr, "bad input: {}", e.what());
    return kExitBadInput;
  }

  AxiMemSettings settings;
  settings.latency = plusarg(context, "latency", settings.latency);
  settings.outstanding = plusarg(context, "outstanding", settings.outstanding);
  settings.rate_permille = plusarg(context, "rate", settings.rate_permille);
  settings.pause_permille = plusarg(context, "pause", settings.pause_permille);
  settings.pause_max = plusarg(context, "pause_max", settings.pause_max);
  settings.beat_bytes = 4 * kBeatWords;
  const uint64_t slow_port = plusarg(context, "slow_port", UINT64_MAX);
  const uint64_t slow_rate = plusarg(context, "slow_rate", settings.rate_permille);
  const uint64_t bad_port = plusarg(context, "bad_port", UINT64_MAX);
  const uint64_t bad_beat = plusarg(context, "bad", UINT64_MAX);
  const uint64_t seed = plusarg(context, "seed", 1);
  std::vector<AxiReadPort> ports;
  for (size_t p = 0; p < n_ports; ++p) {
    AxiMemSettings s = settings;
    if (p == slow_port) s.rate_permille = slow_rate;
    if (p == bad_port) s.bad_beat = bad_beat;
    ports.emplace_back(std::as_bytes(images[p]), s, seed + 1 + p);
  }

  Bench bench{.dut = *dut,
              .ports = ports,
              .x_bufs = x_bufs,
              .commands = commands,
              .offer_permille = plusarg(context, "offer", 1000),
              .rng = std::mt19937_64(seed)};
  try {
    bench.reset();
    uint64_t idle = 0;
    while (bench.oldest < commands.size()) {
      const uint64_t before = bench.writes + bench.oldest;
      bench.cycle();
      idle = bench.writes + bench.oldest == before ? idle + 1 : 0;
      if (idle > kMaxIdleCycles) {
        std::println(stderr, "hung: {} of {} commands taken, {} done, after {} cycles", bench.next_cmd,
                     commands.size(), bench.oldest, bench.cycles);
        return kExitHang;
      }
    }
    const uint64_t cycles = bench.cycles;
    for (int i = 0; i < kDrainCycles; ++i) bench.cycle();
    if (!std::ranges::all_of(ports, &AxiReadPort::idle)) throw WriteError("a memory still busy after the last done");
    std::vector<uint32_t> out;
    for (const Command& c : commands) {
      out.insert(out.end(), c.out.begin(), c.out.end());
      out.push_back(static_cast<uint32_t>(c.taken));
      out.push_back(static_cast<uint32_t>(c.done));
    }
    uint64_t stalls = 0;
    for (const AxiReadPort& p : ports) stalls += p.r_stalls();
    out.push_back(static_cast<uint32_t>(cycles));
    out.push_back(static_cast<uint32_t>(stalls));
    out.push_back(static_cast<uint32_t>(get_bits(dut->err_o, 0, std::min<size_t>(n_ports, 32))));
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
