// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors
//
// The loop every bulk driver shares (protocol: verif/bulk.py). A driver says
// how many fields a record has, how to put a record on the block's inputs,
// and which output to collect; stream() does the rest.

#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <print>
#include <ranges>
#include <span>
#include <vector>

#include "verilated.h"

using Record = std::span<const uint32_t>;

// What stream() needs from a Verilated block: our clock, reset and valid
// ports (AGENTS.md, SystemVerilog conventions).
template <typename Top>
concept Block = requires(Top top) {
  top.clk_i;
  top.rst_ni;
  top.valid_i;
  top.valid_o;
  top.eval();
};

// One clock cycle with the inputs already set: settle, collect the output
// if it is valid, then the rising edge.
template <Block Top, typename Output>
void cycle(Top& dut, Output& output, std::vector<uint32_t>& results) {
  dut.clk_i = 0;
  dut.eval();
  if (dut.valid_o) results.push_back(output(dut));
  dut.clk_i = 1;
  dut.eval();
}

inline std::vector<uint32_t> read_words(std::FILE* file) {
  std::vector<uint32_t> words;
  uint32_t buf[4096];
  while (size_t n = std::fread(buf, sizeof(uint32_t), std::size(buf), file)) {
    words.insert(words.end(), buf, buf + n);
  }
  return words;
}

template <Block Top, typename SetInputs, typename Output>
  requires std::invocable<SetInputs, Top&, Record> && std::invocable<Output, Top&>
int stream(int argc, char** argv, size_t fields, SetInputs set_inputs, Output output) {
  constexpr int kResetCycles = 2;
  constexpr int kDrainCycles = 64;  // more than any pipeline depth we use

  const std::vector<uint32_t> words = read_words(stdin);
  if (words.size() % fields != 0) {
    std::println(stderr, "{} words is not a whole number of {}-word records", words.size(), fields);
    return 2;
  }
  const size_t n = words.size() / fields;

  VerilatedContext context;
  context.commandArgs(argc, argv);
  auto dut = std::make_unique<Top>(&context);
  std::vector<uint32_t> results;
  results.reserve(n);

  dut->rst_ni = 0;
  dut->valid_i = 0;
  for (int i = 0; i < kResetCycles; ++i) cycle(*dut, output, results);
  results.clear();
  dut->rst_ni = 1;

  dut->valid_i = 1;
  for (Record record : std::views::chunk(std::span(words), fields)) {
    set_inputs(*dut, record);
    cycle(*dut, output, results);
  }
  dut->valid_i = 0;
  for (int i = 0; i < kDrainCycles && results.size() < n; ++i) cycle(*dut, output, results);

  if (results.size() != n) {
    std::println(stderr, "got {} results for {} records", results.size(), n);
    return 3;
  }
  std::fwrite(results.data(), sizeof(uint32_t), results.size(), stdout);
  dut->final();
  return 0;
}
