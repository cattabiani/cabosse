// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief A simulated memory behind one AXI read port (PLAN.md, M5) or one
/// write port (M6): it answers read bursts from a byte array, or writes
/// bursts into it and answers each once, with a set latency, a limit on
/// bursts in flight, a bandwidth and random pauses, and checks that the
/// master keeps to the AXI subset we use. Independent of Verilator: a driver
/// copies its outputs onto the block's ports before each cycle and hands it
/// the block's outputs at each clock edge (tick()).

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <format>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

constexpr uint64_t kAxiMaxBurst = 16;  ///< beats per burst, at most (AXI3)
constexpr uint64_t kAxiPageBytes = 4096;  ///< no burst crosses a boundary of these

/// @brief How the memory behaves; every value can be set per test.
struct AxiMemSettings {
  uint64_t latency = 1;          ///< cycles from a burst's request to its first beat, at least 1
  uint64_t outstanding = 8;      ///< bursts accepted and not yet fully answered, at most
  uint64_t rate_permille = 1000; ///< chance per cycle that the next beat may start
  uint64_t pause_permille = 0;   ///< chance per cycle that a pause starts
  uint64_t pause_max = 0;        ///< a pause lasts 1 to pause_max cycles, uniformly
  uint64_t bad_beat = UINT64_MAX; ///< the index of a beat (a write port's: burst) answered with SLVERR
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

/// @brief The write address channel as the master drives it in one cycle.
using AwChannel = ArChannel;

/// @brief The write data channel as the master drives it in one cycle.
struct WChannel {
  bool valid = false;
  std::span<const std::byte> data;  ///< beat_bytes
  uint64_t strb = 0;                ///< a bit per byte
  bool last = false;

  bool operator==(const WChannel& o) const {
    return valid == o.valid && strb == o.strb && last == o.last && std::ranges::equal(data, o.data);
  }
};

/// @brief A broken AXI rule, with what the master did.
struct AxiError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// @brief Check a burst's fields against the AXI subset: incrementing, full
/// width, at most kAxiMaxBurst beats, aligned, inside one 4 KB page and the
/// memory.
/// @return what is wrong, or an empty string.
inline std::string burst_error(const ArChannel& a, const AxiMemSettings& s, uint64_t mem_bytes) {
  const uint64_t bytes = (a.len + 1ull) * s.beat_bytes;
  if (a.burst != 1) return std::format("burst type {}, not INCR", a.burst);
  if ((1ull << a.size) != s.beat_bytes) return std::format("size {} is not the data width", a.size);
  if (a.len + 1ull > kAxiMaxBurst) return std::format("{} beats, more than {}", a.len + 1, kAxiMaxBurst);
  if (a.addr % s.beat_bytes != 0) return std::format("address 0x{:x} not beat-aligned", a.addr);
  if (a.addr / kAxiPageBytes != (a.addr + bytes - 1) / kAxiPageBytes) {
    return std::format("burst at 0x{:x} of {} bytes crosses 4 KB", a.addr, bytes);
  }
  if (a.addr < s.base || a.addr - s.base + bytes > mem_bytes) {
    return std::format("burst at 0x{:x} past the memory", a.addr);
  }
  return {};
}

/// @brief One AXI read port's memory. Bursts are answered in order (one ID).
class AxiReadPort {
 public:
  /// @param mem the memory's bytes; must outlive the port.
  /// @param settings latency, limits, bandwidth and pauses.
  /// @param seed for the bandwidth and pause draws.
  AxiReadPort(std::span<const std::byte> mem, const AxiMemSettings& settings, uint64_t seed)
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
  std::span<const std::byte> r_data() const {
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

  /// @brief Check a request against the AXI subset (burst_error).
  /// @throws AxiError on the first field that breaks it.
  void check(const ArChannel& ar) const {
    if (const std::string e = burst_error(ar, s_, mem_.size()); !e.empty()) fail(e);
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

  std::span<const std::byte> mem_;
  AxiMemSettings s_;
  std::mt19937_64 rng_;
  std::deque<Burst> bursts_;
  ArChannel held_;  ///< last cycle's request if it was valid and not taken
  bool r_valid_ = false;
  uint64_t pause_left_ = 0;
  uint64_t cycle_ = 0;
  uint64_t n_bursts_ = 0, n_beats_ = 0, r_stalls_ = 0, max_in_flight_ = 0;
};

/// @brief One AXI write port's memory. A burst's address is taken before its
/// data (a slave may wait for it), its beats are written as they come, and
/// it is answered once, `latency` cycles after its last beat, in order (one
/// ID). Checks, besides the subset: every channel held while valid and not
/// ready, full strobes (our subset), WLAST on each burst's last beat only,
/// and the response held until taken.
class AxiWritePort {
 public:
  /// @param mem the memory's bytes; must outlive the port.
  /// @param settings latency (last beat to response), limits, bandwidth
  ///   (W ready) and pauses (of AW and W).
  /// @param seed for the bandwidth and pause draws.
  AxiWritePort(std::span<std::byte> mem, const AxiMemSettings& settings, uint64_t seed)
      : mem_(mem), s_(settings), rng_(seed) {
    if (s_.latency < 1) throw std::invalid_argument("latency must be at least 1");
    if (s_.outstanding < 1) throw std::invalid_argument("outstanding must be at least 1");
    if (s_.beat_bytes > 64) throw std::invalid_argument("at most 64 bytes a beat (strobes in a word)");
  }

  /// @return AWREADY for this cycle.
  bool aw_ready() const { return pause_left_ == 0 && bursts_.size() < s_.outstanding; }
  /// @return WREADY for this cycle: a taken burst waits for data.
  bool w_ready() const { return w_ready_; }
  /// @return BVALID for this cycle.
  bool b_valid() const { return b_valid_; }
  /// @return BRESP for this cycle: SLVERR (2) on burst settings.bad_beat, else OKAY.
  unsigned b_resp() const { return b_valid_ && n_answered_ == s_.bad_beat ? 2 : 0; }

  /// @brief The clock edge: take this cycle's handshakes and pick the next
  /// cycle's outputs.
  /// @throws AxiError if the master breaks a rule (the message says which).
  void tick(const AwChannel& aw, const WChannel& w, bool b_ready) {
    if (held_aw_.valid && aw != held_aw_) fail("AW changed or dropped while waiting for AWREADY");
    if (held_w_.valid && !(w == held_w_)) fail("W changed or dropped while waiting for WREADY");
    if (aw.valid && aw_ready()) {
      if (const std::string e = burst_error(aw, s_, mem_.size()); !e.empty()) fail(e);
      bursts_.push_back({aw.addr, aw.len + 1u, 0, UINT64_MAX});
      ++n_bursts_;
      held_aw_ = {};
    } else {
      held_aw_ = aw;
    }
    if (w.valid && w_ready_) {
      take_beat(w);
      held_w_ = {};
    } else {
      held_w_ = w;
      held_data_.assign(w.data.begin(), w.data.end());  // the master's span may change under us
      held_w_.data = held_data_;
    }
    if (b_valid_ && b_ready) {
      bursts_.pop_front();
      ++n_answered_;
      b_valid_ = false;
    }
    if (pause_left_ > 0) {
      --pause_left_;
    } else if (s_.pause_max > 0 && draw(s_.pause_permille)) {
      pause_left_ = std::uniform_int_distribution<uint64_t>(1, s_.pause_max)(rng_);
    }
    ++cycle_;
    w_ready_ = pause_left_ == 0 && filling() != nullptr && draw(s_.rate_permille);
    if (!b_valid_ && !bursts_.empty() && bursts_.front().done <= cycle_) b_valid_ = true;
  }

  /// @return whether every taken burst has been written and answered.
  bool idle() const { return bursts_.empty(); }
  uint64_t n_bursts() const { return n_bursts_; }      ///< bursts taken
  uint64_t n_answered() const { return n_answered_; }  ///< responses taken by the master
  uint64_t n_beats() const { return n_beats_; }        ///< beats written

 private:
  /// @brief A taken burst: written beat by beat, then answered.
  struct Burst {
    uint64_t addr;
    uint64_t beats;
    uint64_t written;  ///< beats so far
    uint64_t done;     ///< first cycle its response may be valid (once written)
  };

  /// @return the first taken burst still waiting for data, or nullptr.
  Burst* filling() {
    for (Burst& b : bursts_) {
      if (b.written < b.beats) return &b;
    }
    return nullptr;
  }

  /// @brief Write one beat into the burst it belongs to.
  /// @throws AxiError on partial strobes or a misplaced WLAST.
  void take_beat(const WChannel& w) {
    Burst* b = filling();
    if (b == nullptr) fail("a W beat with no burst to fill");  // w_ready_ said otherwise
    const uint64_t full = s_.beat_bytes == 64 ? ~0ull : (1ull << s_.beat_bytes) - 1;
    if (w.strb != full) fail(std::format("strobes 0x{:x}, not a full beat", w.strb));
    if (w.data.size() != s_.beat_bytes) fail("a W beat of the wrong width");
    const bool last = b->written + 1 == b->beats;
    if (w.last != last) fail(std::format("WLAST {} on beat {} of {}", w.last, b->written + 1, b->beats));
    std::ranges::copy(w.data, mem_.subspan(b->addr - s_.base + b->written * s_.beat_bytes).begin());
    ++n_beats_;
    if (++b->written == b->beats) b->done = cycle_ + s_.latency;
  }

  /// @brief Stop the run on a broken rule.
  /// @throws AxiError always, with the cycle.
  [[noreturn]] void fail(const std::string& what) const {
    throw AxiError(std::format("cycle {}: {}", cycle_, what));
  }

  /// @return whether an event of chance permille / 1000 happens.
  bool draw(uint64_t permille) { return permille >= 1000 || (permille > 0 && rng_() % 1000 < permille); }

  std::span<std::byte> mem_;
  AxiMemSettings s_;
  std::mt19937_64 rng_;
  std::deque<Burst> bursts_;
  AwChannel held_aw_;
  WChannel held_w_;
  std::vector<std::byte> held_data_;
  bool w_ready_ = false, b_valid_ = false;
  uint64_t pause_left_ = 0;
  uint64_t cycle_ = 0;
  uint64_t n_bursts_ = 0, n_answered_ = 0, n_beats_ = 0;
};
