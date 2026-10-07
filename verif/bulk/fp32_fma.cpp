// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors
//
// Bulk driver for rtl/fp32_fma.sv (verif/bulk.py): reads records of four
// little-endian uint32 (op, a, b, c) from stdin, streams one per clock cycle
// through the block, and writes one uint32 result per record to stdout, in
// order. Works for any pipeline depth: results are collected as valid_o
// marks them.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "Vfp32_fma.h"
#include "verilated.h"

namespace {

constexpr int kFields = 4;
constexpr int kResetCycles = 2;
constexpr uint64_t kDrainCycles = 64;  // more than any pipeline depth we use

std::vector<uint32_t> read_stdin() {
  std::vector<uint32_t> words;
  uint32_t buf[4096];
  size_t n;
  while ((n = std::fread(buf, sizeof(uint32_t), 4096, stdin)) > 0) words.insert(words.end(), buf, buf + n);
  return words;
}

}  // namespace

int main(int argc, char** argv) {
  auto ctx = std::make_unique<VerilatedContext>();
  ctx->commandArgs(argc, argv);
  auto dut = std::make_unique<Vfp32_fma>(ctx.get());

  const std::vector<uint32_t> in = read_stdin();
  if (in.size() % kFields != 0) {
    std::fprintf(stderr, "input is not a whole number of records\n");
    return 2;
  }
  const uint64_t n = in.size() / kFields;
  std::vector<uint32_t> out;
  out.reserve(n);

  auto cycle = [&]() {  // inputs are set; settle, collect, then a rising edge
    dut->clk_i = 0;
    dut->eval();
    if (dut->valid_o) out.push_back(dut->y_o);
    dut->clk_i = 1;
    dut->eval();
  };

  dut->rst_ni = 0;
  dut->valid_i = 0;
  for (int i = 0; i < kResetCycles; ++i) cycle();
  out.clear();
  dut->rst_ni = 1;

  for (uint64_t i = 0; i < n; ++i) {
    const uint32_t* r = &in[i * kFields];
    dut->valid_i = 1;
    dut->op_i = r[0];
    dut->a_i = r[1];
    dut->b_i = r[2];
    dut->c_i = r[3];
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
