// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief The host's side of an AXI-Lite register port (s_aw_valid_i, ...,
/// as rtl/controller.sv and the bring-up design name them), one access at a
/// time, on any bench that runs a clock cycle (stream.h's CycleBench).

#pragma once

#include <format>
#include <stdexcept>

#include "stream.h"

/// @brief Cycle until `done()` holds after a settle, at most kMaxIdleCycles.
/// @throws std::runtime_error on a hang, naming `what`.
template <CycleBench B, typename Done>
void axil_until(B& bench, Done done, const char* what) {
  for (uint64_t i = 0; i < kMaxIdleCycles; ++i) {
    bench.dut.clk_i = 0;
    bench.dut.eval();
    if (done()) return;
    bench.cycle();
  }
  throw std::runtime_error(std::format("hung waiting for {}", what));
}

/// @brief Write a register: address and data together, then the response.
template <CycleBench B>
void axil_write(B& bench, uint32_t addr, uint32_t value) {
  auto& dut = bench.dut;
  dut.s_aw_valid_i = dut.s_w_valid_i = 1;
  dut.s_aw_addr_i = addr;
  dut.s_w_data_i = value;
  axil_until(bench, [&] { return dut.s_aw_ready_o != 0; }, "AWREADY");
  bench.cycle();
  dut.s_aw_valid_i = dut.s_w_valid_i = 0;
  dut.s_b_ready_i = 1;
  axil_until(bench, [&] { return dut.s_b_valid_o != 0; }, "BVALID");
  bench.cycle();
  dut.s_b_ready_i = 0;
}

/// @brief Read a register.
template <CycleBench B>
uint32_t axil_read(B& bench, uint32_t addr) {
  auto& dut = bench.dut;
  dut.s_ar_valid_i = 1;
  dut.s_ar_addr_i = addr;
  axil_until(bench, [&] { return dut.s_ar_ready_o != 0; }, "ARREADY");
  bench.cycle();
  dut.s_ar_valid_i = 0;
  dut.s_r_ready_i = 1;
  axil_until(bench, [&] { return dut.s_r_valid_o != 0; }, "RVALID");
  const uint32_t value = dut.s_r_data_o;
  bench.cycle();
  dut.s_r_ready_i = 0;
  return value;
}
