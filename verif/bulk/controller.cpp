// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief Driver for rtl/controller.sv (verif/controller.py): the host's
/// register writes over AXI-Lite, then tokens, with stub units standing in
/// for the engine and the vector unit.
///
/// Input (little-endian uint32 on stdin): the number of register writes,
/// then per write its offset and value (the command buffer, COMMANDS,
/// STRIDE, ...), made over AXI-Lite after reset; the number of tokens, then
/// per token its TOKEN and POSITION. Each token: TOKEN, POSITION, CONTROL
/// start, then STATUS read every +poll cycles until done or error.
///
/// The stubs take a command when ready (each cycle with probability +ready,
/// permille; +seed) and say done in order, one a cycle, at least +latency
/// and up to +latency_max cycles after taking it; an engine command with
/// wait no sooner than +latency cycles after its go. A go with no waiting
/// command taken and not yet given one ends the run (exit 4).
///
/// Output per token: the number of handovers, then per handover its cycle,
/// unit (0 the engine, 1 the vector unit), the engine's wait bit and group;
/// the number of gos, then each go's cycle; per unit the number of dones,
/// then their cycles; the cycle STATUS was first seen done or error;
/// STATUS, ERROR, CYCLES_LO, CYCLES_HI. Cycles count from the token's
/// CONTROL write. After the tokens: ID and VERSION. Exit 3 if a token makes
/// no progress (handovers, gos, dones) for kMaxIdleCycles.

#include <deque>
#include <random>

#include "Vcontroller.h"
#include "axil.h"

namespace {

constexpr int kExitBadGo = 4;

/// @brief A go the controller sends with no command waiting for it.
struct GoError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// @brief A stub unit: commands taken, each done in order after a delay.
struct Stub {
  struct Pending {
    uint64_t taken;
    bool wait;
    std::optional<uint64_t> go;  ///< the cycle of its go
    uint64_t latency;
  };
  std::deque<Pending> pending;
  bool ready = false;

  /// @brief Whether the oldest command is done this cycle.
  bool done(uint64_t now) const {
    if (pending.empty()) return false;
    const Pending& p = pending.front();
    if (p.wait && !p.go) return false;
    const uint64_t from = p.wait ? std::max(p.taken, *p.go) : p.taken;
    return now >= from + p.latency;
  }
};

/// @brief What a token did, cycles from its start.
struct Events {
  struct Handover {
    uint64_t cycle;
    uint32_t unit, wait, group;
  };
  std::vector<Handover> handovers;
  std::vector<uint64_t> gos, eng_dones, vec_dones;

  size_t progress() const { return handovers.size() + gos.size() + eng_dones.size() + vec_dones.size(); }
};

/// @brief The controller and its stubs.
struct Bench {
  Vcontroller& dut;
  std::mt19937_64 rng;
  uint64_t ready_permille, latency, latency_max;
  Stub eng, vec;
  uint64_t now = 0;  ///< cycles since the token's start
  Events events;

  uint64_t draw_latency() { return std::uniform_int_distribution<uint64_t>(latency, latency_max)(rng); }

  /// @brief One clock cycle: the stubs' ready and done, settle, take a
  /// command and a go, then the rising edge.
  /// @throws GoError on a go with no command for it.
  void cycle() {
    eng.ready = rng() % 1000 < ready_permille;
    vec.ready = rng() % 1000 < ready_permille;
    const bool eng_done = eng.done(now), vec_done = vec.done(now);
    dut.eng_ready_i = eng.ready;
    dut.vec_ready_i = vec.ready;
    dut.eng_done_i = eng_done;
    dut.vec_done_i = vec_done;
    dut.clk_i = 0;
    dut.eval();
    if (dut.eng_go_o) {
      const auto it = std::ranges::find_if(eng.pending, [](const Stub::Pending& p) { return p.wait && !p.go; });
      if (it == eng.pending.end()) throw GoError(std::format("cycle {}: a go with no waiting command", now));
      it->go = now;
      events.gos.push_back(now);
    }
    if (eng_done) {
      eng.pending.pop_front();
      events.eng_dones.push_back(now);
    }
    if (vec_done) {
      vec.pending.pop_front();
      events.vec_dones.push_back(now);
    }
    if (dut.eng_valid_o && eng.ready) {
      eng.pending.push_back({now, static_cast<bool>(dut.eng_wait_o), std::nullopt, draw_latency()});
      events.handovers.push_back({now, 0, dut.eng_wait_o, dut.cmd_group_o});
    }
    if (dut.vec_valid_o && vec.ready) {
      vec.pending.push_back({now, false, std::nullopt, draw_latency()});
      events.handovers.push_back({now, 1, 0, 0});
    }
    dut.clk_i = 1;
    dut.eval();
    ++now;
  }

  /// @brief Run a token: start it, poll STATUS every `poll` cycles until
  /// done or error.
  /// @return STATUS, or nothing if the token makes no progress.
  std::optional<uint32_t> token(uint32_t token, uint32_t position, uint64_t poll) {
    axil_write(*this, kRegToken, token);
    axil_write(*this, kRegPosition, position);
    events = {};
    now = 0;
    axil_write(*this, kRegControl, 1);
    return axil_until_status(*this, poll, [&] { return events.progress(); });
  }
};

/// @brief Append a count and the values.
void put_all(std::vector<uint32_t>& out, const std::vector<uint64_t>& values) {
  out.push_back(static_cast<uint32_t>(values.size()));
  for (const uint64_t v : values) out.push_back(static_cast<uint32_t>(v));
}

}  // namespace

int main(int argc, char** argv) {
  VerilatedContext context;
  context.commandArgs(argc, argv);
  auto dut = std::make_unique<Vcontroller>(&context);
  const std::vector<uint32_t> words = read_words(stdin);
  Reader in{words};
  std::vector<std::pair<uint32_t, uint32_t>> writes, tokens;
  try {
    for (auto* list : {&writes, &tokens}) {
      for (uint32_t n = in.next(); n > 0; --n) {
        const uint32_t first = in.next();  // in order: a call's arguments have none
        list->emplace_back(first, in.next());
      }
    }
  } catch (const std::out_of_range& e) {
    std::println(stderr, "bad input: {}", e.what());
    return kExitBadInput;
  }
  const uint64_t latency = plusarg(context, "latency", 1);
  Bench bench{.dut = *dut,
              .rng = std::mt19937_64(plusarg(context, "seed", 1)),
              .ready_permille = plusarg(context, "ready", 1000),
              .latency = latency,
              .latency_max = std::max(latency, plusarg(context, "latency_max", latency))};
  const uint64_t poll = std::max<uint64_t>(1, plusarg(context, "poll", 16));
  dut->eng_err_i = 0;
  dut->vec_err_i = 0;
  std::vector<uint32_t> out;
  try {
    reset(bench);
    for (const auto& [offset, value] : writes) axil_write(bench, offset, value);
    for (const auto& [token, position] : tokens) {
      const auto status = bench.token(token, position, poll);
      if (!status) {
        std::println(stderr, "hung: {} handovers after {} cycles", bench.events.handovers.size(), bench.now);
        return kExitHang;
      }
      const uint64_t seen = bench.now;
      const Events& e = bench.events;
      out.push_back(static_cast<uint32_t>(e.handovers.size()));
      for (const auto& h : e.handovers) {
        out.insert(out.end(), {static_cast<uint32_t>(h.cycle), h.unit, h.wait, h.group});
      }
      put_all(out, e.gos);
      put_all(out, e.eng_dones);
      put_all(out, e.vec_dones);
      out.insert(out.end(), {static_cast<uint32_t>(seen), *status, axil_read(bench, kRegError),
                             axil_read(bench, kRegCyclesLo), axil_read(bench, kRegCyclesHi)});
      for (uint64_t i = 0; i < kDrainCycles * bench.latency_max; ++i) bench.cycle();
      bench.eng.pending.clear();  // a failed token's, never done
      bench.vec.pending.clear();
    }
    out.push_back(axil_read(bench, kRegId));
    out.push_back(axil_read(bench, kRegVersion));
  } catch (const GoError& e) {
    std::println(stderr, "{}", e.what());
    return kExitBadGo;
  } catch (const std::runtime_error& e) {
    std::println(stderr, "{}", e.what());
    return kExitHang;
  }
  std::fwrite(out.data(), sizeof(uint32_t), out.size(), stdout);
  dut->final();
  return 0;
}
