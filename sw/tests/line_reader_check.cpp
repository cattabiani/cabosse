// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief platforms/f2/host/line_reader.h on plain memory, for
/// sw/tests/test_line_reader.py. `line_reader_check parse TEXT` prints the
/// plan TEXT parses to, or "none". `line_reader_check copy PLAN BYTES...`
/// copies a pattern of each size (whole lines, line-aligned source) with one
/// reader, into a buffer one byte past a line boundary and with guard bytes
/// around it, and exits 1 with a message if a byte differs or a guard moved.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <print>
#include <string>
#include <vector>

#include "../../platforms/f2/host/line_reader.h"

namespace {

constexpr std::byte kGuard{0xA5};
constexpr uint64_t kGuardBytes = 64;

/// @brief Copy `n` bytes with `reader` and check them and the guards.
/// @return whether all match.
bool check(card::LineReader& reader, uint64_t n, unsigned round) {
  auto* in = static_cast<std::byte*>(::operator new(n + card::kReadLine, std::align_val_t{card::kReadLine}));
  const std::unique_ptr<std::byte, void (*)(std::byte*)> owned(in, [](std::byte* p) {
    ::operator delete(p, std::align_val_t{card::kReadLine});
  });
  for (uint64_t i = 0; i < n; ++i) in[i] = static_cast<std::byte>((i * 131 + round * 7 + i / 251) & 0xFF);
  std::vector<std::byte> out(n + 2 * kGuardBytes + 1, kGuard);
  std::byte* const at = out.data() + kGuardBytes + 1;  // off the line grid
  reader.read(at, in, n);
  for (uint64_t i = 0; i < out.size(); ++i) {
    const bool inside = i >= kGuardBytes + 1 && i < kGuardBytes + 1 + n;
    const std::byte want = inside ? in[i - kGuardBytes - 1] : kGuard;
    if (out[i] != want) {
      std::println(stderr, "{} bytes, round {}: byte {} differs", n, round, i);
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 3 && std::string(argv[1]) == "parse") {
    const auto plan = card::parse_plan(argv[2]);
    std::println("{}", plan ? card::to_string(*plan) : "none");
    return 0;
  }
  if (argc < 4 || std::string(argv[1]) != "copy") return 2;
  const auto plan = card::parse_plan(argv[2]);
  if (!plan) return 2;
  card::LineReader reader(*plan);
  bool ok = true;
  for (unsigned round = 0; round < 2; ++round) {  // the workers wait between reads
    for (int k = 3; k < argc; ++k) ok = check(reader, std::stoull(argv[k]), round) && ok;
  }
  return ok ? 0 : 1;
}
