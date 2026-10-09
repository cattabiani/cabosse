// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief Bulk driver for rtl/port_dma.sv against the memory model of
/// axi_mem.h (verif/bulk.py builds it).
///
/// Input (little-endian uint32 on stdin): the memory's size in words, its
/// words, the number of commands, then per command its address (low word,
/// high word) and its beats. Commands are offered back to back.
/// Plusargs: +latency, +outstanding, +rate, +pause, +pause_max, +bad, +base
/// (the AxiMemSettings fields, permille for rates), +ready (permille of
/// cycles with out_ready_i high), +seed.
/// Output: the 8 data words of each beat out, in order; then cycles, bursts,
/// most bursts in flight, cycles the memory waited on RREADY, and err_o.
/// The DMA must then stay quiet for kDrainCycles cycles with the memory idle.

#include <algorithm>

#include "Vport_dma.h"
#include "axi_mem.h"
#include "stream.h"

namespace {

constexpr size_t kBeatWords = 8;  ///< 256 bits
static_assert(sizeof(Vport_dma::r_data_i) == 4 * kBeatWords, "the driver is for DataWidth = 256");
constexpr uint64_t kMaxIdleCycles = 100'000;  ///< cycles with no beat out before calling it a hang
constexpr size_t kStats = 5;  ///< words after the beats: cycles, bursts, in flight, stalls, err

/// @brief The DMA, its memory and the test's side of its ports.
struct Bench {
  Vport_dma& dut;
  AxiReadPort& port;
  std::span<const Record> commands;  ///< offered back to back
  uint64_t ready_permille;           ///< share of cycles with out_ready_i high
  std::mt19937_64 rng;               ///< draws out_ready_i
  size_t next_cmd = 0;               ///< the first command not yet taken
  std::vector<uint32_t> out;         ///< the data words of the beats out, in order

  /// @brief Hold the DMA in reset for kResetCycles cycles, then release it.
  void reset() {
    dut.rst_ni = 0;
    for (int i = 0; i < kResetCycles; ++i) cycle();
    dut.rst_ni = 1;
  }

  /// @brief One clock cycle: put the memory's outputs and our inputs on the
  /// DMA, settle, collect a command taken and a beat out, let the memory see
  /// the DMA's outputs, then the rising edge. In reset no command is offered,
  /// out_ready_i is low and the memory does not tick.
  /// @throws AxiError if the DMA breaks an AXI rule.
  void cycle() {
    const bool run = dut.rst_ni;
    dut.ar_ready_i = port.ar_ready();
    dut.r_valid_i = port.r_valid();
    dut.r_resp_i = port.r_resp();
    if (const auto data = port.r_data(); !data.empty()) {
      const std::span<std::byte, 4 * kBeatWords> beat =
          std::as_writable_bytes(std::span<uint32_t, kBeatWords>(dut.r_data_i.data(), kBeatWords));
      if (data.size() != beat.size()) throw std::length_error("memory beat is not the DMA's width");
      std::ranges::copy(data, beat.begin());
    }
    const bool offer = run && next_cmd < commands.size();
    dut.cmd_valid_i = offer;
    if (offer) {
      dut.cmd_addr_i = commands[next_cmd][0] | uint64_t{commands[next_cmd][1]} << 32;
      dut.cmd_beats_i = commands[next_cmd][2];
    }
    dut.out_ready_i = run && rng() % 1000 < ready_permille;
    dut.clk_i = 0;
    dut.eval();
    if (offer && dut.cmd_ready_o) ++next_cmd;
    if (dut.out_valid_o && dut.out_ready_i) {
      out.insert(out.end(), dut.out_data_o.data(), dut.out_data_o.data() + kBeatWords);
    }
    if (run) {
      port.tick({static_cast<bool>(dut.ar_valid_o), dut.ar_addr_o, dut.ar_len_o, dut.ar_size_o,
                 dut.ar_burst_o},
                dut.r_ready_o);
    }
    dut.clk_i = 1;
    dut.eval();
  }
};

}  // namespace

/// @brief Run the protocol of this file's header.
/// @return 0; 2 if the input is malformed; 3 if the DMA hangs; 4 if it gives
///   an extra beat or leaves the memory busy; 5 if it breaks an AXI rule
///   (each with a message on stderr).
int main(int argc, char** argv) {
  const std::vector<uint32_t> words = read_words(stdin);
  if (words.empty() || size_t{words[0]} + 2 > words.size()) {
    std::println(stderr, "input too short for its memory");
    return 2;
  }
  const size_t n_mem = words[0];
  const std::span<const uint32_t> mem_words(words.data() + 1, n_mem);
  const std::span<const uint32_t> rest(words.data() + 1 + n_mem, words.size() - 1 - n_mem);
  if (rest.size() != 1 + 3 * size_t{rest[0]}) {
    std::println(stderr, "{} words after the memory do not match {} commands", rest.size(), rest[0]);
    return 2;
  }
  const auto commands = std::views::chunk(rest.subspan(1), 3) | std::ranges::to<std::vector<Record>>();
  uint64_t total_beats = 0;
  for (Record c : commands) total_beats += c[2];

  VerilatedContext context;
  context.commandArgs(argc, argv);
  AxiMemSettings settings;
  settings.latency = plusarg(context, "latency", settings.latency);
  settings.outstanding = plusarg(context, "outstanding", settings.outstanding);
  settings.rate_permille = plusarg(context, "rate", settings.rate_permille);
  settings.pause_permille = plusarg(context, "pause", settings.pause_permille);
  settings.pause_max = plusarg(context, "pause_max", settings.pause_max);
  settings.bad_beat = plusarg(context, "bad", settings.bad_beat);
  settings.base = plusarg(context, "base", settings.base);
  settings.beat_bytes = 4 * kBeatWords;
  const uint64_t ready_permille = plusarg(context, "ready", 1000);
  const uint64_t seed = plusarg(context, "seed", 1);

  AxiReadPort port(std::as_bytes(mem_words), settings, seed + 1);
  auto dut = std::make_unique<Vport_dma>(&context);
  Bench bench{.dut = *dut,
              .port = port,
              .commands = commands,
              .ready_permille = ready_permille,
              .rng = std::mt19937_64(seed)};
  bench.out.reserve(total_beats * kBeatWords + kStats);

  try {
    bench.reset();
    uint64_t cycles = 0, idle = 0;
    while (bench.out.size() < total_beats * kBeatWords || bench.next_cmd < commands.size()) {
      const size_t before = bench.out.size();
      bench.cycle();
      ++cycles;
      idle = bench.out.size() == before ? idle + 1 : 0;
      if (idle > kMaxIdleCycles) {
        std::println(stderr, "hung: {} of {} beats, {} of {} commands after {} cycles",
                     bench.out.size() / kBeatWords, total_beats, bench.next_cmd, commands.size(), cycles);
        return 3;
      }
    }
    for (int i = 0; i < kDrainCycles; ++i) bench.cycle();
    if (bench.out.size() != total_beats * kBeatWords || !port.idle()) {
      std::println(stderr, "{} extra beats; memory {}", bench.out.size() / kBeatWords - total_beats,
                   port.idle() ? "idle" : "still busy");
      return 4;
    }
    for (uint64_t v : {cycles, port.n_bursts(), port.max_in_flight(), port.r_stalls(),
                       uint64_t{dut->err_o}}) {
      bench.out.push_back(static_cast<uint32_t>(v));
    }
  } catch (const AxiError& e) {
    std::println(stderr, "AXI: {}", e.what());
    return 5;
  }
  std::fwrite(bench.out.data(), sizeof(uint32_t), bench.out.size(), stdout);
  dut->final();
  return 0;
}
