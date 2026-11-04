// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief The protocol between the runtime's platforms (sw/sim_platform.py,
/// sw/f2_platform.py) and their drivers (verif/bulk/core.cpp,
/// platforms/f2/host/core_driver.cpp): little-endian uint32 words on the
/// driver's stdin and stdout. At start the driver writes kHello, Ports,
/// ChanW and Len; then each request is an opcode and its arguments
/// (verif/bulk/core.cpp's header has each one's words).
#pragma once

#include <cstdint>
#include <cstdio>
#include <span>

namespace protocol {

inline constexpr uint32_t kHello = 0x4342'5349;  ///< 'CBSI'

/// @brief A request's opcode.
enum Request : uint32_t {
  kRegWrite = 0,  ///< offset, value. No answer.
  kRegRead = 1,   ///< offset. Answer: the value.
  kMemWrite = 2,  ///< address low, high, bytes, the bytes padded to words. No answer.
  kMemRead = 3,   ///< address low, high, bytes. Answer: the bytes padded to words.
  kRun = 4,       ///< until STATUS is done or error. Answer: STATUS.
  kBufRead = 5,   ///< simulation only: buffer, first element, elements.
  kQuit = 6,
  kTrace = 7,  ///< simulation only: the controller's handshakes.
};

/// @brief Read words from stdin.
/// @param words where they go, as many as it holds.
/// @return false at the end of the input.
inline bool read_exact(std::span<uint32_t> words) {
  return std::fread(words.data(), sizeof(uint32_t), words.size(), stdin) == words.size();
}

/// @brief Write words to stdout and flush: the parent waits for them.
/// @param words the answer.
inline void answer(std::span<const uint32_t> words) {
  std::fwrite(words.data(), sizeof(uint32_t), words.size(), stdout);
  std::fflush(stdout);
}

}  // namespace protocol
