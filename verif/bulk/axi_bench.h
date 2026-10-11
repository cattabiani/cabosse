// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief What the drivers of blocks with AXI read ports share (port_dma.cpp,
/// engine.cpp): fields of Verilated signals of any width, the memory's
/// settings from plusargs, the reset, and the run loop that calls a hang.

#pragma once

#include <optional>
#include <type_traits>

#include "axi_mem.h"
#include "stream.h"

constexpr uint64_t kMaxIdleCycles = 100'000;  ///< cycles with no progress before calling it a hang

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

/// @brief The memory's settings from plusargs +latency, +outstanding, +rate,
/// +pause, +pause_max, +bad and +base (permille for rates); defaults otherwise.
/// @param context the Verilator context holding the arguments.
/// @param beat_bytes the block's data width, in bytes.
inline AxiMemSettings axi_settings(VerilatedContext& context, uint64_t beat_bytes) {
  AxiMemSettings s;
  s.latency = plusarg(context, "latency", s.latency);
  s.outstanding = plusarg(context, "outstanding", s.outstanding);
  s.rate_permille = plusarg(context, "rate", s.rate_permille);
  s.pause_permille = plusarg(context, "pause", s.pause_permille);
  s.pause_max = plusarg(context, "pause_max", s.pause_max);
  s.bad_beat = plusarg(context, "bad", s.bad_beat);
  s.base = plusarg(context, "base", s.base);
  s.beat_bytes = beat_bytes;
  return s;
}

/// @brief What reset() and run() need from a driver's bench.
template <typename B>
concept AxiBench = requires(B b) {
  b.dut.rst_ni;
  b.cycle();
};

/// @brief Hold the block in reset for kResetCycles cycles, then release it.
template <AxiBench B>
void reset(B& bench) {
  bench.dut.rst_ni = 0;
  for (int i = 0; i < kResetCycles; ++i) bench.cycle();
  bench.dut.rst_ni = 1;
}

/// @brief Run cycles until finished(), calling a hang after kMaxIdleCycles
/// cycles in a row with the same progress().
/// @return the cycles run, or nothing on a hang.
template <AxiBench B, typename Finished, typename Progress>
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
