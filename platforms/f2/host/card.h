// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief What core_driver.cpp needs from the card: the OCL registers (the
/// core's below kWrapperBase, the wrapper's above) and card memory through
/// PCIS. card_f2.cpp implements it with AWS's SDK on the F2 instance;
/// sw/tests/card_fake.cpp with plain memory, for tests on any machine.
#pragma once

#include <cstdint>
#include <memory>
#include <span>

namespace card {

/// @brief The wrapper's registers on OCL (platforms/f2/README.md, the core's
/// CL top): above the core's 128 KiB of AXI-Lite.
inline constexpr uint64_t kWrapperBase = 0x2'0000;
inline constexpr uint64_t kWrapId = kWrapperBase + 0x00;        ///< kWrapIdValue
inline constexpr uint64_t kWrapBuild = kWrapperBase + 0x04;     ///< the build's commit, 8 hex digits
inline constexpr uint64_t kWrapHostPort = kWrapperBase + 0x08;  ///< bit 0: HBM port 0 serves PCIS (1, reset) or the core (0)
inline constexpr uint64_t kWrapStatus = kWrapperBase + 0x0C;    ///< bit 1 HBM ready, bit 3 the host owns HBM port 0
inline constexpr uint64_t kWrapPorts = kWrapperBase + 0x18;     ///< the core's Ports
inline constexpr uint64_t kWrapChanW = kWrapperBase + 0x1C;     ///< its ChanW
inline constexpr uint64_t kWrapLen = kWrapperBase + 0x20;       ///< its Len
inline constexpr uint32_t kWrapIdValue = 0xCAB0'C0E0;
inline constexpr uint32_t kWrapHbmReady = 2, kWrapHostOwns = 8;

/// @brief The card's registers and memory.
class Card {
 public:
  virtual ~Card() = default;
  /// @brief Read an OCL register.
  virtual uint32_t peek(uint64_t offset) = 0;
  /// @brief Write an OCL register.
  virtual void poke(uint64_t offset, uint32_t value) = 0;
  /// @brief Write card memory at a global HBM address, through PCIS; the
  /// writes are done (flushed) when it returns.
  virtual void write(uint64_t addr, std::span<const std::byte> data) = 0;
  /// @brief Read card memory at a global HBM address, through PCIS.
  virtual void read(uint64_t addr, std::span<std::byte> data) = 0;
};

/// @brief The card in slot 0, attached (card_f2.cpp or the fake).
std::unique_ptr<Card> open();

}  // namespace card
