// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief Whole 64-byte lines copied out of device memory (PCIS on BAR4,
/// mapped write-combining), with a choice of loads and threads (M11 step 3,
/// D-077). A load from that mapping is a PCIe read that waits a round trip
/// (about 1 µs, M10), so the rate depends on how many are in flight: wider
/// loads, streaming loads (VMOVNTDQA, which may fetch a whole line into a
/// buffer the line's next load hits) and several threads on disjoint slices
/// are the knobs. Which one wins on the F2 host (AMD EPYC 7R13, AVX2, no
/// AVX-512) is measured, not assumed (pcis_bench.cpp). No SDK here, so the
/// tests run it on plain memory (sw/tests/test_line_reader.py).

#pragma once

#include <algorithm>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace card {

inline constexpr uint64_t kReadLine = 64;  ///< a PCIS beat
inline constexpr unsigned kMaxReadThreads = 16;
inline constexpr uint64_t kUnrollLines = 4;  ///< lines loaded before their stores (SIMD loads)

/// @brief How each thread loads its lines.
enum class Load {
  memcpy,    ///< std::memcpy (glibc picks the instructions)
  load32,    ///< 32-byte loads (VMOVDQA), kUnrollLines lines at a time
  stream32,  ///< 32-byte streaming loads (VMOVNTDQA), kUnrollLines lines at a time
};

/// @brief A read's loads and threads.
struct ReadPlan {
  Load load = Load::memcpy;
  unsigned threads = 1;
};

/// @brief The plan's text, as parse_plan reads it ("stream32:4").
inline std::string to_string(ReadPlan p) {
  constexpr std::string_view kNames[] = {"memcpy", "load32", "stream32"};
  return std::format("{}:{}", kNames[static_cast<int>(p.load)], p.threads);
}

/// @brief Whether this machine has the plan's loads.
inline bool supported(Load load) {
#if defined(__x86_64__)
  return load == Load::memcpy || __builtin_cpu_supports("avx2");
#else
  return load == Load::memcpy;
#endif
}

/// @brief A plan from "LOAD[:THREADS]" (LOAD memcpy, load32 or stream32;
/// THREADS 1 to kMaxReadThreads, default 1).
/// @return none if the text is not one, or the loads are not supported here.
inline std::optional<ReadPlan> parse_plan(std::string_view text) {
  const auto colon = text.find(':');
  const std::string_view name = text.substr(0, colon);
  ReadPlan p;
  if (name == "memcpy") {
    p.load = Load::memcpy;
  } else if (name == "load32") {
    p.load = Load::load32;
  } else if (name == "stream32") {
    p.load = Load::stream32;
  } else {
    return std::nullopt;
  }
  if (colon != std::string_view::npos) {
    const std::string n(text.substr(colon + 1));
    if (n.empty() || n.find_first_not_of("0123456789") != std::string::npos || n.size() > 3) return std::nullopt;
    p.threads = static_cast<unsigned>(std::stoul(n));
  }
  if (p.threads < 1 || p.threads > kMaxReadThreads || !supported(p.load)) return std::nullopt;
  return p;
}

#if defined(__x86_64__)
/// @brief Lines from `in` (32-byte aligned) to `out`: kUnrollLines lines
/// of loads, then their stores, so the loads may be in flight together.
/// @tparam kStream streaming loads (VMOVNTDQA) instead of plain ones.
/// @param out the destination, any alignment.
/// @param in the source, 32-byte aligned.
/// @param n bytes, whole lines.
template <bool kStream>
__attribute__((target("avx2"))) void copy_lines_avx2(std::byte* out, const std::byte* in, uint64_t n) {
  constexpr uint64_t kHalves = kUnrollLines * 2;
  for (uint64_t at = 0; at < n;) {
    const uint64_t halves = std::min<uint64_t>(kHalves, (n - at) / (kReadLine / 2));
    __m256i v[kHalves];
    for (uint64_t h = 0; h < halves; ++h) {
      auto* p = reinterpret_cast<const __m256i*>(in + at + h * 32);
      if constexpr (kStream) {
        v[h] = _mm256_stream_load_si256(p);
      } else {
        v[h] = _mm256_load_si256(p);
      }
    }
    for (uint64_t h = 0; h < halves; ++h) _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + at + h * 32), v[h]);
    at += halves * 32;
  }
}
#endif

/// @brief Lines copied with `load`, on the calling thread.
/// @param load the loads (supported here).
/// @param out the destination.
/// @param in the source, line-aligned.
/// @param n bytes, whole lines.
inline void copy_lines(Load load, std::byte* out, const std::byte* in, uint64_t n) {
#if defined(__x86_64__)
  if (load == Load::load32) return copy_lines_avx2<false>(out, in, n);
  if (load == Load::stream32) return copy_lines_avx2<true>(out, in, n);
#endif
  std::memcpy(out, in, n);
}

/// @brief Reads whole lines with a plan: the calling thread and plan.threads
/// - 1 workers, each its slice of the lines; the workers wait on a barrier
/// between reads (made once, not a thread a read).
class LineReader {
 public:
  /// @throws std::invalid_argument if the plan's loads are not supported here.
  explicit LineReader(ReadPlan plan) : plan_(plan), start_(plan.threads), end_(plan.threads) {
    if (!supported(plan.load) || plan.threads < 1 || plan.threads > kMaxReadThreads) {
      throw std::invalid_argument(std::format("read plan {}: not on this machine", to_string(plan)));
    }
    for (unsigned k = 1; k < plan.threads; ++k) workers_.emplace_back([this, k] { work(k); });
  }
  ~LineReader() {
    if (workers_.empty()) return;
    stop_ = true;
    start_.arrive_and_wait();
  }
  LineReader(const LineReader&) = delete;
  LineReader& operator=(const LineReader&) = delete;

  ReadPlan plan() const { return plan_; }

  /// @brief Copy [in, in + n) to out.
  /// @param out the destination.
  /// @param in the source, line-aligned.
  /// @param n bytes, whole lines.
  void read(std::byte* out, const std::byte* in, uint64_t n) {
    out_ = out;
    in_ = in;
    n_ = n;
    if (workers_.empty()) return slice(0);
    start_.arrive_and_wait();
    slice(0);
    end_.arrive_and_wait();
  }

 private:
  ReadPlan plan_;
  std::barrier<> start_, end_;
  std::vector<std::jthread> workers_;  // last: they use the barriers
  std::byte* out_ = nullptr;
  const std::byte* in_ = nullptr;
  uint64_t n_ = 0;
  bool stop_ = false;

  /// @brief Thread k's lines of the current read.
  void slice(unsigned k) {
    const uint64_t lines = n_ / kReadLine, t = plan_.threads;
    const uint64_t first = lines * k / t * kReadLine, end = lines * (k + 1) / t * kReadLine;
    copy_lines(plan_.load, out_ + first, in_ + first, end - first);
  }

  /// @brief Worker k: a slice a read, until stop.
  void work(unsigned k) {
    for (;;) {
      start_.arrive_and_wait();
      if (stop_) return;
      slice(k);
      end_.arrive_and_wait();
    }
  }
};

}  // namespace card
