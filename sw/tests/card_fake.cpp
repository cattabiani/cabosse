// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief card.h without a card, for sw/tests/test_f2_platform.py: the
/// registers a map that reads back what was written, the memory a vector.
/// The wrapper's registers answer as the CL's would (ID, HBM ready,
/// HOST_PORT shown in STATUS; Ports, ChanW and Len from the environment's
/// FAKE_PORTS, FAKE_CHAN_W, FAKE_LEN), and a start (CONTROL bit 0) runs a
/// token as OCL can show it: the first STATUS read after it still the last
/// token's (the read passed the posted start), then busy, then done. It
/// checks the driver's handovers: memory touched while the core holds HBM
/// port 0, or a start while the host holds it, throws. PCIM: the host page
/// is a vector at a made-up physical address; when a token is done, the
/// armed window's card memory is copied into it (at PCIM_HOST minus that
/// address, which must land inside it) and PCIM_DONE set to its beats, or,
/// with FAKE_PCIM_ERR set, nothing copied and PCIM_ERR set. With
/// FAKE_NO_PCIM set, the PCIM registers read 0 and nothing is copied (a
/// wrapper built before PCIM).

#include <algorithm>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../platforms/f2/host/card.h"

namespace {

/// @brief An environment variable as a number.
uint32_t env(const char* name) {
  const char* v = std::getenv(name);
  if (!v) throw std::runtime_error(std::string(name) + " not set");
  return static_cast<uint32_t>(std::stoul(v));
}

/// @brief The fake card.
class FakeCard final : public card::Card {
 public:
  FakeCard() : ports_(env("FAKE_PORTS")), chan_w_(env("FAKE_CHAN_W")), len_(env("FAKE_LEN")) {
    mem_.resize(size_t{ports_} << chan_w_);
  }

  uint32_t peek(uint64_t offset) override {
    switch (offset) {
      case card::kWrapId: return card::kWrapIdValue;
      case card::kWrapStatus: return card::kWrapHbmReady | (host_owns_ ? card::kWrapHostOwns : 0);
      case card::kWrapPorts: return ports_;
      case card::kWrapChanW: return chan_w_;
      case card::kWrapLen: return len_;
      case kStatus: {
        const uint32_t s = regs_[kStatus];
        if (token_reads_ == 2) regs_[kStatus] = kBusy;
        if (token_reads_ == 1) {
          regs_[kStatus] = kDone;
          copy_window();
        }
        if (token_reads_ > 0) --token_reads_;
        return s;
      }
      default:
        if (no_pcim_ && offset >= card::kWrapPcimBaseLo && offset <= card::kWrapPcimErr) return 0;
        return regs_[offset];
    }
  }

  void poke(uint64_t offset, uint32_t value) override {
    if (offset == card::kWrapHostPort) {
      host_owns_ = value & 1;
    } else if (offset == card::kWrapPcimDone) {
      regs_[card::kWrapPcimDone] = regs_[card::kWrapPcimErr] = 0;
    } else if (offset == 0x08 && (value & 1)) {  // CONTROL's start
      if (host_owns_) throw std::runtime_error("a start while the host holds HBM port 0");
      token_reads_ = 2;
    } else {
      regs_[offset] = value;
    }
  }

  void write(uint64_t addr, std::span<const std::byte> data) override {
    std::ranges::copy(data, at(addr, data.size()).begin());
  }

  void read(uint64_t addr, std::span<std::byte> data) override {
    std::ranges::copy(at(addr, data.size()), data.begin());
  }

  card::HostPage host_page() override { return {page_, kPagePhys}; }

 private:
  uint32_t ports_, chan_w_, len_;
  std::vector<std::byte> mem_;
  std::vector<std::byte> page_ = std::vector<std::byte>(size_t{2} << 20);
  static constexpr uint64_t kPagePhys = 0x12'3460'0000;
  std::map<uint64_t, uint32_t> regs_;
  bool host_owns_ = true;  // the wrapper's reset value
  bool no_pcim_ = std::getenv("FAKE_NO_PCIM") != nullptr;
  int token_reads_ = 0;    // STATUS reads left before the token is done
  static constexpr uint64_t kStatus = 0x0C;
  static constexpr uint32_t kBusy = 1, kDone = 2;

  /// @brief The wrapper's PCIM copies of a token's window, all answered.
  void copy_window() {
    const uint64_t n = regs_[card::kWrapPcimBytes];
    if (n == 0 || no_pcim_) return;
    const uint64_t base = uint64_t{regs_[card::kWrapPcimBaseHi]} << 32 | regs_[card::kWrapPcimBaseLo];
    const uint64_t host = uint64_t{regs_[card::kWrapPcimHostHi]} << 32 | regs_[card::kWrapPcimHostLo];
    if (host < kPagePhys || host - kPagePhys + n > page_.size()) throw std::runtime_error("PCIM's window outside the page");
    if (std::getenv("FAKE_PCIM_ERR")) {
      regs_[card::kWrapPcimErr] = 1;
      return;
    }
    std::ranges::copy(std::span(mem_).subspan(base, n), page_.begin() + static_cast<ptrdiff_t>(host - kPagePhys));
    regs_[card::kWrapPcimDone] = static_cast<uint32_t>(n / 32);
  }

  std::span<std::byte> at(uint64_t addr, uint64_t n) {
    if (!host_owns_) throw std::runtime_error("memory touched while the core holds HBM port 0");
    return std::span(mem_).subspan(addr, n);
  }
};

}  // namespace

std::unique_ptr<card::Card> card::open() { return std::make_unique<FakeCard>(); }
