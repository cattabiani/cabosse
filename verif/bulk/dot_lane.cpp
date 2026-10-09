// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief Bulk driver for rtl/dot_lane.sv (verif/bulk.py), with its
/// valid/ready handshake, so it has its own loop instead of stream().
///
/// Record, one per beat (5 words): w0 | w1 << 16, w2 | w3 << 16, x0 | x1 << 16,
/// x2 | x3 << 16, then mask | last << 4 | gap << 8, where gap is the idle
/// cycles before the beat is offered. Plusargs: +ready=<permille> (default
/// 1000) is the share of cycles with out_ready_i high, drawn from +seed=<n>.
/// Output: one word per row (a beat with last), in order, then two words:
/// the cycles run and the stalls (cycles with in_valid_i high and in_ready_o
/// low). After the last result the lane must stay quiet for kDrainCycles
/// cycles with every beat taken; an extra result is an error.

#include <random>

#include "Vdot_lane.h"
#include "stream.h"

namespace {

constexpr size_t kFields = 5;              ///< words per beat record
constexpr uint64_t kMaxCyclesPerBeat = 64;  ///< a hung handshake fails instead of running forever
constexpr uint64_t kSlackBeats = 16;  ///< the hang limit's allowance beyond the beats

/// @brief One clock cycle with the inputs already set: settle with the clock
/// low, record the handshakes, then the rising edge.
/// @param dut the lane.
/// @param results receives out_o if out_valid_o and out_ready_i are high.
/// @param stalls counts the cycle if in_valid_i is high and in_ready_o low.
/// @return whether the input beat was taken (in_valid_i and in_ready_o).
bool cycle(Vdot_lane& dut, std::vector<uint32_t>& results, uint64_t& stalls) {
  dut.clk_i = 0;
  dut.eval();
  const bool fired = dut.in_valid_i && dut.in_ready_o;
  if (dut.in_valid_i && !dut.in_ready_o) ++stalls;
  if (dut.out_valid_o && dut.out_ready_i) results.push_back(dut.out_o);
  dut.clk_i = 1;
  dut.eval();
  return fired;
}

}  // namespace

/// @brief Run the protocol of this file's header: beats from stdin, results,
/// cycles and stalls to stdout.
/// @param argc, argv the program's arguments: Verilator's, +ready and +seed.
/// @return 0; 2 if the input is not whole beats or a plusarg is not a number;
///   3 if the lane hangs; 4 if it gives an extra result or leaves a beat
///   untaken (each with a message on stderr).
int main(int argc, char** argv) {
  const std::vector<uint32_t> words = read_words(stdin);
  if (words.size() % kFields != 0) {
    std::println(stderr, "{} words is not a whole number of {}-word beats", words.size(), kFields);
    return 2;
  }
  const auto beats = std::views::chunk(std::span(words), kFields) | std::ranges::to<std::vector>();
  const size_t n_rows = std::ranges::count_if(beats, [](Record b) { return (b[4] >> 4) & 1; });

  VerilatedContext context;
  context.commandArgs(argc, argv);
  const uint64_t ready_permille = plusarg(context, "ready", 1000);
  std::mt19937_64 rng(plusarg(context, "seed", 1));
  auto dut = std::make_unique<Vdot_lane>(&context);

  std::vector<uint32_t> results;
  results.reserve(n_rows + 2);
  uint64_t stalls = 0;

  dut->rst_ni = 0;
  dut->in_valid_i = 0;
  dut->out_ready_i = 0;
  for (int i = 0; i < kResetCycles; ++i) cycle(*dut, results, stalls);
  dut->rst_ni = 1;

  uint64_t gaps = 0;
  for (Record b : beats) gaps += b[4] >> 8;
  const uint64_t max_cycles = kMaxCyclesPerBeat * (beats.size() + kSlackBeats) + gaps;
  uint64_t cycles = 0;
  size_t next = 0;
  uint32_t idle = 0;
  while (results.size() < n_rows) {
    const bool valid = next < beats.size() && idle >= (beats[next][4] >> 8);
    if (valid) {
      const Record b = beats[next];
      dut->w_i = b[0] | uint64_t{b[1]} << 32;
      dut->x_i = b[2] | uint64_t{b[3]} << 32;
      dut->mask_i = b[4] & 0xF;
      dut->last_i = (b[4] >> 4) & 1;
    } else {
      ++idle;
    }
    dut->in_valid_i = valid;
    dut->out_ready_i = rng() % 1000 < ready_permille;
    if (cycle(*dut, results, stalls)) {
      ++next;
      idle = 0;
    }
    if (++cycles > max_cycles) {
      std::println(stderr, "hung: {} of {} results after {} cycles", results.size(), n_rows, cycles);
      return 3;
    }
  }
  const size_t n_results = results.size();
  dut->in_valid_i = 0;
  dut->out_ready_i = 1;
  for (int i = 0; i < kDrainCycles; ++i) cycle(*dut, results, stalls);
  if (results.size() != n_results || next != beats.size()) {
    std::println(stderr, "{} extra results, {} of {} beats taken", results.size() - n_results, next,
                 beats.size());
    return 4;
  }
  results.push_back(static_cast<uint32_t>(cycles));
  results.push_back(static_cast<uint32_t>(stalls));
  std::fwrite(results.data(), sizeof(uint32_t), results.size(), stdout);
  dut->final();
  return 0;
}
