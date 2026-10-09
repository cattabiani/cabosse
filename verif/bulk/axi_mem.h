// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief A simulated memory behind one AXI read port (PLAN.md, M5): it
/// answers read bursts from a byte array, with a set latency, a limit on
/// bursts in flight, a bandwidth and random pauses, and checks that the
/// master keeps to the AXI subset we use. Independent of Verilator: a driver
/// copies its outputs onto the block's ports before each cycle and hands it
/// the block's outputs at each clock edge (tick()).

#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <format>
#include <random>
#include <span>
#include <stdexcept>
#include <string>

constexpr uint64_t kAxiMaxBurst = 16;  ///< beats per burst, at most (AXI3)
constexpr uint64_t kAxiPageBytes = 4096;  ///< no burst crosses a boundary of these

/// @brief How the memory behaves; every value can be set per test.
struct AxiMemSettings {
  uint64_t latency = 1;          ///< cycles from a burst's request to its first beat, at least 1
  uint64_t outstanding = 8;      ///< bursts accepted and not yet fully answered, at most
  uint64_t rate_permille = 1000; ///< chance per cycle that the next beat may start
  uint64_t pause_permille = 0;   ///< chance per cycle that a pause starts
  uint64_t pause_max = 0;        ///< a pause lasts 1 to pause_max cycles, uniformly
  uint64_t bad_beat = UINT64_MAX; ///< the index of a beat answered with SLVERR (none by default)
  uint64_t beat_bytes = 32;      ///< bytes per beat: the data width
  uint64_t base = 0;             ///< the address of the memory's first byte
};

/// @brief The read address channel as the master drives it in one cycle.
struct ArChannel {
  bool valid = false;
  uint64_t addr = 0;
  unsigned len = 0;    ///< beats - 1
  unsigned size = 0;   ///< log2(bytes per beat)
  unsigned burst = 0;  ///< 1 is INCR

  bool operator==(const ArChannel&) const = default;
};

/// @brief A broken AXI rule, with what the master did.
struct AxiError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// @brief One AXI read port's memory. Bursts are answered in order (one ID).
class AxiReadPort {
 public:
  /// @param mem the memory's bytes; must outlive the port.
  /// @param settings latency, limits, bandwidth and pauses.
  /// @param seed for the bandwidth and pause draws.
  AxiReadPort(std::span<const uint8_t> mem, const AxiMemSettings& settings, uint64_t seed)
      : mem_(mem), s_(settings), rng_(seed) {
    if (s_.latency < 1) throw std::invalid_argument("latency must be at least 1");
    if (s_.outstanding < 1) throw std::invalid_argument("outstanding must be at least 1");
  }

  /// @return ARREADY for this cycle.
  bool ar_ready() const { return bursts_.size() < s_.outstanding; }
  /// @return RVALID for this cycle.
  bool r_valid() const { return r_valid_; }
  /// @return RRESP for this cycle: SLVERR (2) on settings.bad_beat, else OKAY.
  unsigned r_resp() const { return r_valid_ && n_beats_ == s_.bad_beat ? 2 : 0; }
  /// @return RDATA for this cycle (meaningful while r_valid()).
  std::span<const uint8_t> r_data() const {
    if (!r_valid_) return {};
    const Burst& b = bursts_.front();
    return mem_.subspan(b.addr - s_.base + b.sent * s_.beat_bytes, s_.beat_bytes);
  }
  /// @brief The clock edge: take this cycle's handshakes and pick the next
  /// cycle's outputs.
  /// @param ar the master's read address channel this cycle.
  /// @param r_ready the master's RREADY this cycle.
  /// @throws AxiError if the master breaks a rule (the message says which).
  void tick(const ArChannel& ar, bool r_ready) {
    if (held_.valid && ar != held_) {
      fail("AR changed or dropped while waiting for ARREADY");
    }
    if (ar.valid && ar_ready()) {
      check(ar);
      bursts_.push_back({ar.addr, ar.len + 1u, cycle_ + s_.latency, 0});
      ++n_bursts_;
      max_in_flight_ = std::max<uint64_t>(max_in_flight_, bursts_.size());
      held_ = {};
    } else {
      held_ = ar;  // a valid request must stay the same until taken
    }
    if (r_valid_ && r_ready) {
      ++n_beats_;
      if (++bursts_.front().sent == bursts_.front().beats) bursts_.pop_front();
      r_valid_ = false;
    } else if (r_valid_) {
      ++r_stalls_;
    }
    if (pause_left_ > 0) {
      --pause_left_;
    } else if (s_.pause_max > 0 && draw(s_.pause_permille)) {
      pause_left_ = std::uniform_int_distribution<uint64_t>(1, s_.pause_max)(rng_);
    }
    ++cycle_;
    if (!r_valid_ && pause_left_ == 0 && !bursts_.empty() && bursts_.front().due <= cycle_) {
      r_valid_ = draw(s_.rate_permille);
    }
  }

  /// @return whether every accepted burst has been answered.
  bool idle() const { return bursts_.empty(); }
  uint64_t n_bursts() const { return n_bursts_; }        ///< bursts accepted
  uint64_t n_beats() const { return n_beats_; }          ///< beats taken by the master
  uint64_t r_stalls() const { return r_stalls_; }        ///< cycles with RVALID and not RREADY
  uint64_t max_in_flight() const { return max_in_flight_; }  ///< most bursts in flight at once

 private:
  /// @brief An accepted burst, answered beat by beat.
  struct Burst {
    uint64_t addr;
    uint64_t beats;
    uint64_t due;   ///< first cycle its first beat may be valid
    uint64_t sent;  ///< beats already taken
  };

  /// @brief Check a request's fields against the AXI subset: incrementing, full
  /// width, at most max_burst beats, aligned, inside one 4 KB page and the
  /// memory.
  /// @param ar the request.
  /// @throws AxiError on the first field that breaks the subset.
  void check(const ArChannel& ar) const {
    const uint64_t bytes = (ar.len + 1ull) * s_.beat_bytes;
    if (ar.burst != 1) fail(std::format("burst type {}, not INCR", ar.burst));
    if ((1ull << ar.size) != s_.beat_bytes) fail(std::format("size {} is not the data width", ar.size));
    if (ar.len + 1ull > kAxiMaxBurst) fail(std::format("{} beats, more than {}", ar.len + 1, kAxiMaxBurst));
    if (ar.addr % s_.beat_bytes != 0) fail(std::format("address 0x{:x} not beat-aligned", ar.addr));
    if (ar.addr / kAxiPageBytes != (ar.addr + bytes - 1) / kAxiPageBytes) {
      fail(std::format("burst at 0x{:x} of {} bytes crosses 4 KB", ar.addr, bytes));
    }
    if (ar.addr < s_.base || ar.addr - s_.base + bytes > mem_.size()) {
      fail(std::format("burst at 0x{:x} past the memory", ar.addr));
    }
  }

  /// @brief Stop the run on a broken rule.
  /// @param what the rule and what the master did.
  /// @throws AxiError always, with the cycle.
  [[noreturn]] void fail(const std::string& what) const {
    throw AxiError(std::format("cycle {}: {}", cycle_, what));
  }

  /// @brief A random event.
  /// @param permille its chance, in thousandths.
  /// @return whether it happens this time.
  bool draw(uint64_t permille) { return permille >= 1000 || (permille > 0 && rng_() % 1000 < permille); }

  std::span<const uint8_t> mem_;
  AxiMemSettings s_;
  std::mt19937_64 rng_;
  std::deque<Burst> bursts_;
  ArChannel held_;  ///< last cycle's request if it was valid and not taken
  bool r_valid_ = false;
  uint64_t pause_left_ = 0;
  uint64_t cycle_ = 0;
  uint64_t n_bursts_ = 0, n_beats_ = 0, r_stalls_ = 0, max_in_flight_ = 0;
};
