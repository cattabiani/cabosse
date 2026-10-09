// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief Bulk driver for rtl/port_dma.sv against the memory model of
/// axi_mem.h (verif/bulk.py builds it).
///
/// Input (little-endian uint32 on stdin): the memory's size in words, its
/// words, the number of commands, then per command its address (low word,
/// high word) and its beats. Commands are offered back to back.
/// Plusargs: +latency, +outstanding, +rate, +pause, +pause_max (the
/// AxiMemSettings fields, permille for rates), +ready (permille of cycles
/// with out_ready_i high), +seed, +bad (the index of a beat the memory
/// answers with SLVERR; none by default).
/// Output: per beat out, its 8 data words and a flags word (bit 0: last);
/// then cycles, bursts, most bursts in flight, cycles the memory waited on
/// RREADY, and err_o. The DMA must then stay quiet for kDrainCycles cycles
/// with the memory idle.

#include <algorithm>
#include <cstring>

#include "Vport_dma.h"
#include "axi_mem.h"
#include "stream.h"

namespace {

constexpr int kResetCycles = 2;
constexpr int kDrainCycles = 64;
constexpr size_t kBeatWords = 8;  ///< 256 bits
static_assert(sizeof(Vport_dma::r_data_i) == 4 * kBeatWords, "the driver is for DataWidth = 256");

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
  const auto commands = std::views::chunk(rest.subspan(1), 3) | std::ranges::to<std::vector>();
  uint64_t total_beats = 0;
  for (Record c : commands) total_beats += c[2];

  VerilatedContext context;
  context.commandArgs(argc, argv);
  AxiMemSettings settings;
  settings.beat_bytes = 4 * kBeatWords;
  settings.latency = plusarg(context, "latency", settings.latency);
  settings.outstanding = plusarg(context, "outstanding", settings.outstanding);
  settings.rate_permille = plusarg(context, "rate", settings.rate_permille);
  settings.pause_permille = plusarg(context, "pause", settings.pause_permille);
  settings.pause_max = plusarg(context, "pause_max", settings.pause_max);
  const uint64_t ready_permille = plusarg(context, "ready", 1000);
  const uint64_t seed = plusarg(context, "seed", 1);
  const uint64_t bad_beat = plusarg(context, "bad", UINT64_MAX);
  std::mt19937_64 rng(seed);

  const auto mem = std::as_bytes(mem_words);
  AxiReadPort port({reinterpret_cast<const uint8_t*>(mem.data()), mem.size()}, settings, seed + 1);
  auto dut = std::make_unique<Vport_dma>(&context);

  std::vector<uint32_t> out;
  out.reserve(total_beats * (kBeatWords + 1) + 5);
  size_t next_cmd = 0;
  uint64_t n_out = 0;
  // Settle with the memory's outputs and our inputs applied, collect what
  // left the DMA, and let the memory see the edge.
  auto cycle = [&](bool run) {
    dut->ar_ready_i = port.ar_ready();
    dut->r_valid_i = port.r_valid();
    const auto data = port.r_data();
    for (size_t k = 0; k < kBeatWords; ++k) {
      uint32_t w = 0;
      if (!data.empty()) std::memcpy(&w, data.data() + 4 * k, 4);
      dut->r_data_i[k] = w;
    }
    dut->r_resp_i = port.r_valid() && port.n_beats() == bad_beat ? 2 : 0;  // SLVERR
    const bool offer = run && next_cmd < commands.size();
    dut->cmd_valid_i = offer;
    if (offer) {
      dut->cmd_addr_i = commands[next_cmd][0] | uint64_t{commands[next_cmd][1]} << 32;
      dut->cmd_beats_i = commands[next_cmd][2];
    }
    dut->out_ready_i = run && rng() % 1000 < ready_permille;
    dut->clk_i = 0;
    dut->eval();
    if (offer && dut->cmd_ready_o) ++next_cmd;
    if (dut->out_valid_o && dut->out_ready_i) {
      for (size_t k = 0; k < kBeatWords; ++k) out.push_back(dut->out_data_o[k]);
      out.push_back(dut->out_last_o);
      ++n_out;
    }
    if (dut->rst_ni) {
      port.tick({static_cast<bool>(dut->ar_valid_o), dut->ar_addr_o, dut->ar_len_o, dut->ar_size_o,
                 dut->ar_burst_o},
                dut->r_ready_o);
    }
    dut->clk_i = 1;
    dut->eval();
  };

  try {
    dut->rst_ni = 0;
    for (int i = 0; i < kResetCycles; ++i) cycle(false);
    dut->rst_ni = 1;
    const uint64_t per_beat = 64 + settings.latency + settings.pause_max;
    const uint64_t max_cycles = (total_beats + commands.size() + 16) * per_beat * 1000 /
                                std::max<uint64_t>(1, std::min(settings.rate_permille, ready_permille));
    uint64_t cycles = 0;
    while (n_out < total_beats || next_cmd < commands.size()) {
      cycle(true);
      if (++cycles > max_cycles) {
        std::println(stderr, "hung: {} of {} beats, {} of {} commands after {} cycles", n_out,
                     total_beats, next_cmd, commands.size(), cycles);
        return 3;
      }
    }
    for (int i = 0; i < kDrainCycles; ++i) cycle(true);
    if (n_out != total_beats || !port.idle()) {
      std::println(stderr, "{} extra beats; memory {}", n_out - total_beats,
                   port.idle() ? "idle" : "still busy");
      return 4;
    }
    for (uint64_t v : {cycles, port.n_bursts(), port.max_in_flight(), port.r_stalls(),
                       uint64_t{dut->err_o}}) {
      out.push_back(static_cast<uint32_t>(v));
    }
  } catch (const AxiError& e) {
    std::println(stderr, "AXI: {}", e.what());
    return 5;
  }
  std::fwrite(out.data(), sizeof(uint32_t), out.size(), stdout);
  dut->final();
  return 0;
}
