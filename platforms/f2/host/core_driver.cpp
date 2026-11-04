// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief The F2 platform's driver (sw/f2_platform.py): a long-lived child
/// process on the F2 host that serves the runtime's calls (sw/platforms.py)
/// on stdin and stdout, in the simulation driver's protocol
/// (sw/driver_protocol.h; verif/bulk/core.cpp has each request's words).
/// Requests 5 (buffer read) and 7 (trace) are the simulation's only.
///
/// HBM port 0 serves either PCIS or the core (the wrapper's HOST_PORT,
/// platforms/f2/README.md): the driver gives it to the host before a memory
/// request and to the core before a register write that starts a token
/// (CONTROL bit 0), each time waiting until the wrapper's STATUS shows it.
/// The hello's Ports, ChanW and Len are the wrapper's registers.
///
/// Exits 2 on a bad request (an unknown opcode, an address out of range,
/// input cut short), 3 if a run or a port handover takes longer than its
/// timeout, 1 on a card error, each with a message on stderr.

#include <array>
#include <chrono>
#include <cstdio>
#include <format>
#include <print>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../../../sw/driver_protocol.h"
#include "card.h"

namespace {

constexpr uint64_t kControl = 0x08, kStatus = 0x0C;  // the core's (model/commands.py, REGISTERS)
constexpr uint32_t kStart = 1, kStatusBusy = 1, kStatusDone = 2, kStatusError = 4;
constexpr auto kRunTimeout = std::chrono::seconds(60);
constexpr auto kHandoverTimeout = std::chrono::seconds(5);
constexpr int kExitCard = 1, kExitBadInput = 2, kExitHang = 3;

/// @brief A request the driver cannot serve: exit kExitBadInput.
struct BadRequest : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// @brief A run or a handover past its timeout: exit kExitHang.
struct Hang : std::runtime_error {
  using std::runtime_error::runtime_error;
};

using protocol::answer, protocol::read_exact;

/// @brief The card and who holds HBM port 0.
struct Driver {
  card::Card& card;
  uint64_t memory_bytes;  ///< Ports channels of 2^ChanW bytes
  bool host_owns = true;  ///< HOST_PORT's value, as last set
  bool started = false;   ///< a start sent, its arrival not seen yet

  /// @brief Give HBM port 0 to the host (PCIS) or to the core, and wait
  /// until the wrapper's STATUS shows it.
  /// @param host true for the host.
  /// @throws Hang after kHandoverTimeout.
  void own(bool host) {
    if (host == host_owns) return;
    card.poke(card::kWrapHostPort, host ? 1 : 0);
    const auto start = std::chrono::steady_clock::now();
    while (((card.peek(card::kWrapStatus) & card::kWrapHostOwns) != 0) != host) {
      if (std::chrono::steady_clock::now() - start > kHandoverTimeout) {
        throw Hang(std::format("HBM port 0 not handed to the {}", host ? "host" : "core"));
      }
    }
    host_owns = host;
  }

  /// @brief A memory request's address and bytes, checked against the memory.
  /// @throws BadRequest past its end or on cut input.
  std::pair<uint64_t, uint32_t> range(const char* what) {
    std::array<uint32_t, 3> a{};
    if (!read_exact(a)) throw BadRequest(what);
    const uint64_t addr = uint64_t{a[1]} << 32 | a[0];
    if (addr > memory_bytes || a[2] > memory_bytes - addr) {
      throw BadRequest(std::format("{}: 0x{:X} + {} past the memory", what, addr, a[2]));
    }
    return {addr, a[2]};
  }

  /// @brief Poll the core's STATUS until done or error. The start is a
  /// posted write, and OCL's reads may pass it: until STATUS shows the
  /// token running (busy) or failed, a done is the last token's.
  /// @return STATUS.
  /// @throws Hang after kRunTimeout.
  uint32_t run() {
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
      const uint32_t s = card.peek(kStatus);
      if (s & (kStatusBusy | kStatusError)) started = false;
      if (!started && (s & (kStatusDone | kStatusError))) return s;
      if (std::chrono::steady_clock::now() - start > kRunTimeout) throw Hang("run: no done or error");
    }
  }

  /// @brief Serve requests until quit or the end of the input.
  void serve() {
    std::array<uint32_t, 2> a{};
    for (uint32_t op = 0; read_exact(std::span(&op, 1));) {
      switch (op) {
        case protocol::kRegWrite:
          if (!read_exact(a)) throw BadRequest("register write");
          if (a[0] == kControl && (a[1] & kStart)) {
            own(false);
            started = true;
          }
          card.poke(a[0], a[1]);
          break;
        case protocol::kRegRead: {
          if (!read_exact(std::span(a).first(1))) throw BadRequest("register read");
          const uint32_t v = card.peek(a[0]);
          answer(std::span(&v, 1));
          break;
        }
        case protocol::kMemWrite: {
          const auto [addr, n] = range("memory write");
          std::vector<uint32_t> words((n + 3) / 4);
          if (!read_exact(words)) throw BadRequest("memory write's bytes");
          own(true);
          card.write(addr, std::as_bytes(std::span(words)).first(n));
          break;
        }
        case protocol::kMemRead: {
          const auto [addr, n] = range("memory read");
          std::vector<uint32_t> words((n + 3) / 4);
          own(true);
          card.read(addr, std::as_writable_bytes(std::span(words)).first(n));
          answer(words);
          break;
        }
        case protocol::kRun: {
          const uint32_t status = run();
          answer(std::span(&status, 1));
          break;
        }
        case protocol::kQuit:
          return;
        default:
          throw BadRequest(std::format("request {} (the simulation's only, or unknown)", op));
      }
    }
  }
};

}  // namespace

int main() {
  try {
    const auto card = card::open();
    if (const uint32_t id = card->peek(card::kWrapId); id != card::kWrapIdValue) {
      throw std::runtime_error(std::format("wrapper ID 0x{:08X}, not the core's CL", id));
    }
    const auto start = std::chrono::steady_clock::now();
    while (!(card->peek(card::kWrapStatus) & card::kWrapHbmReady)) {
      if (std::chrono::steady_clock::now() - start > kHandoverTimeout) throw Hang("HBM not ready");
    }
    const uint32_t ports = card->peek(card::kWrapPorts), chan_w = card->peek(card::kWrapChanW);
    const uint32_t len = card->peek(card::kWrapLen);
    // The wrapper's reset value: the host owns port 0 (README.md).
    Driver driver{.card = *card, .memory_bytes = uint64_t{ports} << chan_w};
    if (!(card->peek(card::kWrapStatus) & card::kWrapHostOwns)) driver.host_owns = false;
    const std::array<uint32_t, 4> hello{protocol::kHello, ports, chan_w, len};
    answer(hello);
    driver.serve();
    return 0;
  } catch (const BadRequest& e) {
    std::println(stderr, "bad request: {}", e.what());
    return kExitBadInput;
  } catch (const Hang& e) {
    std::println(stderr, "hung: {}", e.what());
    return kExitHang;
  } catch (const std::exception& e) {
    std::println(stderr, "card: {}", e.what());
    return kExitCard;
  }
}
