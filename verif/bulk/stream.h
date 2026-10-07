// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors
//
// The loop every bulk driver shares (the protocol is in verif/bulk.py): read
// records of n_fields little-endian uint32 from stdin, stream one per clock
// cycle through the block, collect one uint32 per record as valid_o marks
// it, and write them to stdout in order. The block has clk_i, rst_ni,
// valid_i and valid_o; set_inputs(dut, record) and output(dut) handle the
// rest of its ports.

#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "verilated.h"

template <typename Top, typename SetInputs, typename Output>
int stream(int argc, char** argv, size_t n_fields, SetInputs set_inputs, Output output) {
  constexpr int kResetCycles = 2;
  constexpr uint64_t kDrainCycles = 64;  // more than any pipeline depth we use

  auto ctx = std::make_unique<VerilatedContext>();
  ctx->commandArgs(argc, argv);
  auto dut = std::make_unique<Top>(ctx.get());

  std::vector<uint32_t> in;
  uint32_t buf[4096];
  for (size_t n; (n = std::fread(buf, sizeof(uint32_t), 4096, stdin)) > 0;) {
    in.insert(in.end(), buf, buf + n);
  }
  if (in.size() % n_fields != 0) {
    std::fprintf(stderr, "input is not a whole number of records\n");
    return 2;
  }
  const uint64_t n = in.size() / n_fields;
  std::vector<uint32_t> out;
  out.reserve(n);

  auto cycle = [&]() {  // inputs are set: settle, collect, then a rising edge
    dut->clk_i = 0;
    dut->eval();
    if (dut->valid_o) out.push_back(output(*dut));
    dut->clk_i = 1;
    dut->eval();
  };

  dut->rst_ni = 0;
  dut->valid_i = 0;
  for (int i = 0; i < kResetCycles; ++i) cycle();
  out.clear();
  dut->rst_ni = 1;

  dut->valid_i = 1;
  for (uint64_t i = 0; i < n; ++i) {
    set_inputs(*dut, &in[i * n_fields]);
    cycle();
  }
  dut->valid_i = 0;
  for (uint64_t i = 0; i < kDrainCycles && out.size() < n; ++i) cycle();

  if (out.size() != n) {
    std::fprintf(stderr, "got %zu results for %llu records\n", out.size(), (unsigned long long)n);
    return 3;
  }
  std::fwrite(out.data(), sizeof(uint32_t), out.size(), stdout);
  dut->final();
  return 0;
}
