// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief `core_clock MHZ`: sets the core's CL clock (clk_hbm_axi, D-065)
/// through AWS's clock generator and waits for HBM ready, then prints one
/// JSON line: the wrapper's ID, BUILD, Ports, ChanW, Len and the
/// generator's clocks. After an AFI's load the generator holds the design
/// in reset until a clock is set (AWS_CLK_GEN_spec.md, M8), so the run
/// script calls this once, before any core_driver. A clock change resets
/// the design: never during a conversation. Exits 1 on an error (a wrong
/// wrapper ID among them), with a message on stderr.

#include <fpga_clkgen.h>
#include <fpga_mgmt.h>

#include <chrono>
#include <cstdint>
#include <exception>
#include <format>
#include <print>
#include <stdexcept>
#include <string>
#include <thread>

#include "card.h"

namespace {

constexpr int kSlot = 0;
constexpr auto kReadyTimeout = std::chrono::seconds(10);
// Groups A to C keep the frequencies AWS's defaults give their first
// clocks (unused by the design), as the bring-up's `clock` (M8).
constexpr uint32_t kClockA = 125, kClockB = 450, kClockC = 300;

/// @brief Throw on a failed SDK call.
/// @param rc the call's return code.
/// @param what the call, for the message.
void check(int rc, const char* what) {
  if (rc != 0) throw std::runtime_error(std::format("{} failed ({})", what, rc));
}

/// @brief One clock group's three clocks, in MHz, as a JSON array.
/// @param g the group.
/// @return the array's text.
std::string group_json(const fpga_clkgen_group& g) {
  return std::format("[{}, {}, {}]", g.clocks[0], g.clocks[1], g.clocks[2]);
}

/// @brief Set the clock, wait for HBM, describe the design.
/// @param mhz clk_hbm_axi's frequency.
/// @return the JSON line.
std::string set_clock(uint32_t mhz) {
  const auto card = card::open();  // attaches only: no OCL access yet
  check(aws_clkgen_set_dynamic(kSlot, kClockA, kClockB, kClockC, mhz, 0), "aws_clkgen_set_dynamic");
  const auto start = std::chrono::steady_clock::now();
  while (!(card->peek(card::kWrapStatus) & card::kWrapHbmReady)) {
    if (std::chrono::steady_clock::now() - start > kReadyTimeout) throw std::runtime_error("HBM not ready");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (const uint32_t id = card->peek(card::kWrapId); id != card::kWrapIdValue) {
    throw std::runtime_error(std::format("wrapper ID 0x{:08X}, not the core's CL", id));
  }
  fpga_clkgen_info info{};
  check(aws_clkgen_get_dynamic(kSlot, &info), "aws_clkgen_get_dynamic");
  return std::format(
      R"({{"set_mhz": {}, "wrap_id": "0x{:08X}", "build": "{:08x}", "ports": {}, "chan_w": {}, "len": {}, )"
      R"("clocks_mhz": {{"a": {}, "b": {}, "c": {}, "hbm": {}}}}})",
      mhz, card->peek(card::kWrapId), card->peek(card::kWrapBuild), card->peek(card::kWrapPorts),
      card->peek(card::kWrapChanW), card->peek(card::kWrapLen), group_json(info.clock_group_a),
      group_json(info.clock_group_b), group_json(info.clock_group_c), group_json(info.clock_group_hbm));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: core_clock MHZ");
    std::println("{}", set_clock(static_cast<uint32_t>(std::stoul(argv[1]))));
    return 0;
  } catch (const std::exception& e) {
    std::println(stderr, "core_clock: {}", e.what());
    return 1;
  }
}
