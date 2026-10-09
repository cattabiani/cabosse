// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief The memory model's checks and timing, driven by hand-made requests
/// instead of RTL (verif/tests/test_port_dma.py builds and runs it). Prints
/// one line per case: its name, then "ok" or the AxiError's message.

#include <cstdio>
#include <functional>
#include <print>
#include <utility>
#include <vector>

#include "axi_mem.h"

namespace {

constexpr ArChannel kGood{true, 0, 0, 5, 1};  ///< one 32-byte beat at 0

/// @brief Run one case against a fresh port and print its outcome.
/// @param name printed first.
/// @param settings the port's settings.
/// @param body drives the port; an AxiError it throws is printed.
void run_case(const char* name, const AxiMemSettings& settings,
              const std::function<void(AxiReadPort&)>& body) {
  static const std::vector<uint8_t> mem(64 * 1024);
  AxiReadPort port(mem, settings, 1);
  try {
    body(port);
    std::println("{}: ok", name);
  } catch (const AxiError& e) {
    std::println("{}: {}", name, e.what());
  }
}

/// @brief A request that differs from kGood in one field.
/// @param change sets the field.
/// @return the request.
ArChannel with(const std::function<void(ArChannel&)>& change) {
  ArChannel ar = kGood;
  change(ar);
  return ar;
}

}  // namespace

/// @brief Every case of the file's header.
/// @return 0.
int main() {
  const AxiMemSettings plain;
  for (auto [name, ar] : std::vector<std::pair<const char*, ArChannel>>{
           {"crosses 4 KB", with([](ArChannel& a) { a.addr = 4096 - 32, a.len = 1; })},
           {"too long", with([](ArChannel& a) { a.len = 16; })},
           {"wrong size", with([](ArChannel& a) { a.size = 4; })},
           {"not INCR", with([](ArChannel& a) { a.burst = 2; })},
           {"not beat-aligned", with([](ArChannel& a) { a.addr = 16; })},
           {"past the memory", with([](ArChannel& a) { a.addr = 64 * 1024; })},
           {"to the 4 KB boundary", with([](ArChannel& a) { a.addr = 4096 - 64, a.len = 1; })},
       }) {
    run_case(name, plain, [ar](AxiReadPort& p) { p.tick(ar, true); });
  }

  // One burst in flight, a long latency: the second request waits, and must
  // stay the same until it is taken.
  AxiMemSettings one;
  one.outstanding = 1;
  one.latency = 10;
  const ArChannel second = with([](ArChannel& a) { a.addr = 32; });
  run_case("changed while waiting", one, [&](AxiReadPort& p) {
    p.tick(kGood, true);
    p.tick(second, true);
    p.tick(with([](ArChannel& a) { a.addr = 64; }), true);
  });
  run_case("dropped while waiting", one, [&](AxiReadPort& p) {
    p.tick(kGood, true);
    p.tick(second, true);
    p.tick({}, true);
  });
  run_case("held while waiting", one, [&](AxiReadPort& p) {
    p.tick(kGood, true);
    while (!p.ar_ready()) p.tick(second, true);
    p.tick(second, true);
  });

  // Latency: the first beat is valid exactly `latency` cycles after the
  // request; with RREADY low it stays valid.
  run_case("latency", one, [&](AxiReadPort& p) {
    p.tick(kGood, false);
    uint64_t cycles = 1;
    while (!p.r_valid()) p.tick({}, false), ++cycles;
    for (int i = 0; i < 5; ++i) p.tick({}, false);
    if (cycles != one.latency || !p.r_valid() || p.r_stalls() != 5) {
      throw AxiError(std::format("first beat after {} cycles, {} stalls", cycles, p.r_stalls()));
    }
  });
  return 0;
}
