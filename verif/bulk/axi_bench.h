// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief The memory model's settings from a driver's plusargs, for the
/// drivers of blocks with AXI ports (port_dma.cpp, engine.cpp, vec_unit.cpp).

#pragma once

#include "axi_mem.h"
#include "stream.h"

/// @brief One port's memory settings from plusargs +latency, +outstanding,
/// +rate, +pause, +pause_max, +base and +aw_after_w (every port), +bad (the beat answered
/// with an error, on port +bad_port, default 0) and +slow_rate (the rate of
/// port +slow_port, none by default); permille for rates. Each name may
/// carry a prefix (a write port's: +wlatency, ...).
/// @param context the Verilator context holding the arguments.
/// @param beat_bytes the block's data width, in bytes.
/// @param port which port's settings.
/// @param prefix put before every plusarg's name.
inline AxiMemSettings axi_settings(VerilatedContext& context, uint64_t beat_bytes, uint64_t port,
                                   const std::string& prefix = "") {
  const auto arg = [&](const char* name, uint64_t fallback) { return plusarg(context, prefix + name, fallback); };
  AxiMemSettings s;
  s.latency = arg("latency", s.latency);
  s.outstanding = arg("outstanding", s.outstanding);
  s.rate_permille = arg("rate", s.rate_permille);
  s.pause_permille = arg("pause", s.pause_permille);
  s.pause_max = arg("pause_max", s.pause_max);
  s.base = arg("base", s.base);
  s.aw_after_w = arg("aw_after_w", 0) != 0;
  s.beat_bytes = beat_bytes;
  if (port == arg("bad_port", 0)) s.bad_beat = arg("bad", s.bad_beat);
  if (port == arg("slow_port", UINT64_MAX)) s.rate_permille = arg("slow_rate", s.rate_permille);
  return s;
}
