// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief The loop every bulk driver shares (protocol: verif/bulk.py), and
/// what drivers of handshake blocks share: fields of Verilated signals of
/// any width, the reset and a run loop that calls a hang.
///
/// A driver says how many fields a record has, how to put a record on the
/// block's inputs, and which output to collect; stream() does the rest.

#pragma once

#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <print>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "verilated.h"

constexpr int kResetCycles = 2;   ///< cycles in reset before the first input
constexpr int kDrainCycles = 64;  ///< after the last result: more than any pipeline depth we use
constexpr uint64_t kMaxIdleCycles = 100'000;  ///< handshake drivers: cycles with no progress before calling it a hang

/// @brief One record: the words of one input, in the driver's field order.
using Record = std::span<const uint32_t>;

/// @brief What stream() needs from a Verilated block: our clock, reset and
/// valid ports (AGENTS.md, SystemVerilog conventions).
template <typename Top>
concept Block = requires(Top top) {
  top.clk_i;
  top.rst_ni;
  top.valid_i;
  top.valid_o;
  top.eval();
};

/// @brief One clock cycle with the inputs already set: settle, collect the
/// output if it is valid, then the rising edge.
/// @tparam Top the Verilated block.
/// @param dut the block.
/// @param output returns the result word from the block's ports.
/// @param results receives output(dut) if valid_o is high.
template <Block Top, typename Output>
void cycle(Top& dut, Output& output, std::vector<uint32_t>& results) {
  dut.clk_i = 0;
  dut.eval();
  if (dut.valid_o) results.push_back(output(dut));
  dut.clk_i = 1;
  dut.eval();
}

/// @brief Read every little-endian uint32 in a file, in order.
/// @param file an open binary stream, read to its end.
/// @return the words; a trailing partial word is dropped.
inline std::vector<uint32_t> read_words(std::FILE* file) {
  std::vector<uint32_t> words;
  uint32_t buf[4096];
  while (size_t n = std::fread(buf, sizeof(uint32_t), std::size(buf), file)) {
    words.insert(words.end(), buf, buf + n);
  }
  return words;
}

/// @brief The value of plusarg +<name>=<n>, or `fallback` if it is absent.
/// @param context the Verilator context holding the arguments.
/// @param name the plusarg's name, without + and =.
/// @param fallback returned when the plusarg is absent.
/// @return the parsed number; a value that is not a number ends the
///   program with a message.
inline uint64_t plusarg(VerilatedContext& context, const std::string& name, uint64_t fallback) {
  const std::string match = context.commandArgsPlusMatch((name + "=").c_str());
  if (match.empty()) return fallback;
  const std::string value = match.substr(name.size() + 2);  // skip "+name="
  try {
    return std::stoull(value);
  } catch (const std::logic_error&) {
    std::println(stderr, "+{}={} is not a number", name, value);
    std::exit(2);
  }
}

/// @brief Run the bulk protocol: read records from stdin, stream one per
/// clock cycle through the block after a reset, collect a result whenever
/// valid_o is high, and write the results to stdout in order.
/// @tparam Top the Verilated block.
/// @param argc, argv the program's arguments, passed to Verilator.
/// @param fields words per record.
/// @param set_inputs puts a Record on the block's input ports.
/// @param output returns the result word from the block's ports.
/// @return 0; 2 if the input is not whole records; 3 if a result is missing
///   (each with a message on stderr).
template <Block Top, typename SetInputs, typename Output>
  requires std::invocable<SetInputs, Top&, Record> && std::invocable<Output, Top&>
int stream(int argc, char** argv, size_t fields, SetInputs set_inputs, Output output) {
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

/// @brief Bits [lsb, lsb + width) of a Verilated signal.
/// @tparam T an integer (signals up to 64 bits) or VlWide (wider ones).
/// @throws std::invalid_argument if the field spans more than 64 bits from a
///   32-bit word boundary (lsb % 32 + width > 64).
template <typename T>
uint64_t get_bits(const T& sig, size_t lsb, size_t width) {
  if (lsb % 32 + width > 64) throw std::invalid_argument("get_bits: field wider than two words");
  uint64_t v = 0;
  if constexpr (std::is_integral_v<T>) {
    v = static_cast<uint64_t>(sig) >> lsb;
  } else {
    for (size_t w = lsb / 32; w <= (lsb + width - 1) / 32 && w < sizeof(T) / 4; ++w) {
      v |= uint64_t{sig[w]} << (32 * (w - lsb / 32));
    }
    v >>= lsb % 32;
  }
  return width >= 64 ? v : v & ((uint64_t{1} << width) - 1);
}

/// @brief Set bits [lsb, lsb + width) of a Verilated signal to value, width <= 32.
/// @tparam T an integer or VlWide.
template <typename T>
void set_bits(T& sig, size_t lsb, size_t width, uint32_t value) {
  if constexpr (std::is_integral_v<T>) {
    const uint64_t mask = ((uint64_t{1} << width) - 1) << lsb;
    sig = static_cast<T>((sig & ~mask) | ((uint64_t{value} << lsb) & mask));
  } else {
    for (size_t i = 0; i < width; ++i) {
      const size_t b = lsb + i;
      sig[b / 32] = (sig[b / 32] & ~(1u << b % 32)) | (((value >> i) & 1u) << b % 32);
    }
  }
}

/// @brief What reset() and run() need from a handshake driver's bench: the
/// block (with our reset port) and one clock cycle.
template <typename B>
concept CycleBench = requires(B b) {
  b.dut.rst_ni;
  b.cycle();
};

/// @brief Hold the block in reset for kResetCycles cycles, then release it.
template <CycleBench B>
void reset(B& bench) {
  bench.dut.rst_ni = 0;
  for (int i = 0; i < kResetCycles; ++i) bench.cycle();
  bench.dut.rst_ni = 1;
}

/// @brief Run cycles until finished(), calling a hang after kMaxIdleCycles
/// cycles in a row with the same progress().
/// @return the cycles run, or nothing on a hang.
template <CycleBench B, typename Finished, typename Progress>
std::optional<uint64_t> run(B& bench, Finished finished, Progress progress) {
  uint64_t cycles = 0, idle = 0;
  while (!finished()) {
    const auto before = progress();
    bench.cycle();
    ++cycles;
    idle = progress() == before ? idle + 1 : 0;
    if (idle > kMaxIdleCycles) return std::nullopt;
  }
  return cycles;
}
