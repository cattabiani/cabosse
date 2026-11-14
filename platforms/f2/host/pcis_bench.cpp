// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief `pcis_bench BYTES REPS PLAN...`: the PCIS read rate with each read
/// plan (line_reader.h, "LOAD:THREADS"), M11 step 3. For each plan, a card
/// opened with CABOSSE_PCIS_READ set to it (as core_driver would run)
/// writes a pattern of BYTES at card address 0 through PCIS, reads it back
/// REPS times, checks every byte, and prints one JSON line:
/// the plan, the bytes, each read's microseconds, the median, the rate in
/// MB/s at the median, and whether all reads matched. Run after
/// core_clock (HBM ready) and before any core_driver: it takes HBM port 0
/// and overwrites card memory. Exits 1 on an error or a mismatch.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

#include "card.h"
#include "line_reader.h"

namespace {

constexpr auto kReadyTimeout = std::chrono::seconds(5);
constexpr uint64_t kAddr = 0;
constexpr uint32_t kSeed = 20261114;

/// @brief Wait until the host owns HBM port 0 (the wrapper's reset value).
/// @throws std::runtime_error after kReadyTimeout.
void own_port0(card::Card& card) {
  if (const uint32_t id = card.peek(card::kWrapId); id != card::kWrapIdValue) {
    throw std::runtime_error(std::format("wrapper ID 0x{:08X}, not the core's CL", id));
  }
  card.poke(card::kWrapHostPort, 1);
  const auto start = std::chrono::steady_clock::now();
  constexpr uint32_t kReady = card::kWrapHbmReady | card::kWrapHostOwns;
  while ((card.peek(card::kWrapStatus) & kReady) != kReady) {
    if (std::chrono::steady_clock::now() - start > kReadyTimeout) throw std::runtime_error("HBM port 0 not ready");
  }
}

/// @brief The pattern: each 32-bit word a hash of its index.
std::vector<std::byte> pattern(uint64_t n_bytes) {
  std::vector<std::byte> out(n_bytes);
  for (uint64_t i = 0; i < n_bytes; ++i) {
    const uint32_t word = static_cast<uint32_t>(i / 4) * 2654435761u ^ kSeed;
    out[i] = static_cast<std::byte>(word >> (8 * (i % 4)));
  }
  return out;
}

/// @brief One plan's reads, printed as a JSON line.
/// @return whether every read matched.
bool bench(const std::string& plan, const std::vector<std::byte>& want, unsigned reps) {
  setenv("CABOSSE_PCIS_READ", plan.c_str(), 1);
  const auto card = card::open();
  own_port0(*card);
  card->write(kAddr, want);
  std::vector<std::byte> got(want.size());
  std::vector<double> us;
  bool ok = true;
  for (unsigned r = 0; r < reps; ++r) {
    std::ranges::fill(got, std::byte{0});
    const auto t0 = std::chrono::steady_clock::now();
    card->read(kAddr, got);
    us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    ok = ok && got == want;
  }
  std::vector<double> sorted = us;
  std::ranges::sort(sorted);
  const double median = sorted[sorted.size() / 2];
  std::string list;
  for (const double u : us) list += std::format("{}{:.1f}", list.empty() ? "" : ", ", u);
  std::println(R"({{"plan": "{}", "bytes": {}, "us": [{}], "median_us": {:.1f}, "mb_per_s": {:.1f}, "ok": {}}})", plan,
               want.size(), list, median, static_cast<double>(want.size()) / median, ok);
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::println(stderr, "usage: pcis_bench BYTES REPS PLAN...");
    return 1;
  }
  try {
    const uint64_t n_bytes = std::stoull(argv[1]);
    const unsigned reps = static_cast<unsigned>(std::stoul(argv[2]));
    if (n_bytes == 0 || reps == 0) throw std::invalid_argument("BYTES and REPS must be positive");
    const auto want = pattern(n_bytes);
    bool ok = true;
    for (int k = 3; k < argc; ++k) {
      if (!card::parse_plan(argv[k])) {
        std::println(stderr, "{}: not a read plan for this machine, skipped", argv[k]);
        ok = false;
        continue;
      }
      ok = bench(argv[k], want, reps) && ok;
    }
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::println(stderr, "pcis_bench: {}", e.what());
    return 1;
  }
}
