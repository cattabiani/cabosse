// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief The memory model's settings from a driver's plusargs, for the
/// drivers of blocks with AXI read ports (port_dma.cpp, engine.cpp).

#pragma once

#include "axi_mem.h"
#include "stream.h"

/// @brief One port's memory settings from plusargs +latency, +outstanding,
/// +rate, +pause, +pause_max and +base (every port), +bad (the beat answered
/// with an error, on port +bad_port, default 0) and +slow_rate (the rate of
/// port +slow_port, none by default); permille for rates.
/// @param context the Verilator context holding the arguments.
/// @param beat_bytes the block's data width, in bytes.
/// @param port which port's settings.
inline AxiMemSettings axi_settings(VerilatedContext& context, uint64_t beat_bytes, uint64_t port) {
  AxiMemSettings s;
  s.latency = plusarg(context, "latency", s.latency);
  s.outstanding = plusarg(context, "outstanding", s.outstanding);
  s.rate_permille = plusarg(context, "rate", s.rate_permille);
  s.pause_permille = plusarg(context, "pause", s.pause_permille);
  s.pause_max = plusarg(context, "pause_max", s.pause_max);
  s.base = plusarg(context, "base", s.base);
  s.beat_bytes = beat_bytes;
  if (port == plusarg(context, "bad_port", 0)) s.bad_beat = plusarg(context, "bad", s.bad_beat);
  if (port == plusarg(context, "slow_port", UINT64_MAX)) s.rate_permille = plusarg(context, "slow_rate", s.rate_permille);
  return s;
}
