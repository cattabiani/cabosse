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
/// Output per token: the number of handovers, then per handover its cycle
/// and unit (0 the engine, 1 the vector unit) and the engine's wait bit;
/// the number of gos, then each go's cycle; per unit the number of dones,
/// then their cycles; the cycle STATUS was first seen done or error;
/// STATUS, ERROR, CYCLES_LO, CYCLES_HI. Cycles count from the token's
/// CONTROL write. After the tokens: ID and VERSION. Exit 3 if a token hangs.

#include <deque>
#include <random>

#include "Vcontroller.h"
#include "stream.h"

namespace {

constexpr uint32_t kRegControl = 0x08, kRegStatus = 0x0C, kRegToken = 0x10, kRegPosition = 0x14;
constexpr uint32_t kRegError = 0x1C, kRegCyclesLo = 0x28, kRegCyclesHi = 0x2C, kRegId = 0x00;
constexpr uint32_t kRegVersion = 0x04;
constexpr uint32_t kStatusDone = 2, kStatusError = 4;
constexpr int kExitBadGo = 4;
constexpr uint64_t kMaxTokenCycles = 50'000'000;

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
  std::vector<uint64_t> dones;  ///< this token's

  /// @brief Whether the oldest command is done this cycle.
  bool done(uint64_t now) const {
    if (pending.empty()) return false;
    const Pending& p = pending.front();
    if (p.wait && !p.go) return false;
    const uint64_t from = p.wait ? std::max(p.taken, *p.go) : p.taken;
    return now >= from + p.latency;
  }
};

/// @brief The controller, its stubs and the host's side of AXI-Lite.
struct Bench {
  Vcontroller& dut;
  std::mt19937_64 rng;
  uint64_t ready_permille, latency, latency_max;
  Stub eng, vec;
  uint64_t cycle = 0;  ///< since the token's start
  struct Handover {
    uint64_t cycle;
    uint32_t unit, wait;
  };
  std::vector<Handover> handovers;
  std::vector<uint64_t> gos;

  uint64_t draw_latency() { return std::uniform_int_distribution<uint64_t>(latency, latency_max)(rng); }

  /// @brief One clock cycle: the stubs' ready and done, settle, take a
  /// command and a go, then the rising edge.
  /// @throws GoError on a go with no command for it.
  void tick() {
    eng.ready = rng() % 1000 < ready_permille;
    vec.ready = rng() % 1000 < ready_permille;
    const bool eng_done = eng.done(cycle), vec_done = vec.done(cycle);
    dut.eng_ready_i = eng.ready;
    dut.vec_ready_i = vec.ready;
    dut.eng_done_i = eng_done;
    dut.vec_done_i = vec_done;
    dut.eng_err_i = 0;
    dut.vec_err_i = 0;
    dut.clk_i = 0;
    dut.eval();
    if (dut.eng_go_o) {
      const auto it = std::ranges::find_if(eng.pending, [](const Stub::Pending& p) { return p.wait && !p.go; });
      if (it == eng.pending.end()) throw GoError(std::format("cycle {}: a go with no waiting command", cycle));
      it->go = cycle;
      gos.push_back(cycle);
    }
    if (eng_done) {
      eng.pending.pop_front();
      eng.dones.push_back(cycle);
    }
    if (vec_done) {
      vec.pending.pop_front();
      vec.dones.push_back(cycle);
    }
    if (dut.eng_valid_o && eng.ready) {
      eng.pending.push_back({cycle, static_cast<bool>(dut.eng_wait_o), std::nullopt, draw_latency()});
      handovers.push_back({cycle, 0, dut.eng_wait_o});
    }
    if (dut.vec_valid_o && vec.ready) {
      vec.pending.push_back({cycle, false, std::nullopt, draw_latency()});
      handovers.push_back({cycle, 1, 0});
    }
    dut.clk_i = 1;
    dut.eval();
    ++cycle;
  }

  /// @brief One AXI-Lite write: address and data together, then the response.
  void write(uint32_t offset, uint32_t value) {
    dut.s_awvalid_i = 1;
    dut.s_wvalid_i = 1;
    dut.s_awaddr_i = offset;
    dut.s_wdata_i = value;
    dut.s_bready_i = 1;
    dut.clk_i = 0;
    dut.eval();
    while (!dut.s_awready_o) {
      tick();
      dut.clk_i = 0;
      dut.eval();
    }
    tick();
    dut.s_awvalid_i = 0;
    dut.s_wvalid_i = 0;
    while (true) {
      dut.clk_i = 0;
      dut.eval();
      const bool b = dut.s_bvalid_o;
      tick();
      if (b) break;
    }
  }

  /// @brief One AXI-Lite read.
  uint32_t read(uint32_t offset) {
    dut.s_arvalid_i = 1;
    dut.s_araddr_i = offset;
    dut.s_rready_i = 1;
    dut.clk_i = 0;
    dut.eval();
    while (!dut.s_arready_o) {
      tick();
      dut.clk_i = 0;
      dut.eval();
    }
    tick();
    dut.s_arvalid_i = 0;
    while (true) {
      dut.clk_i = 0;
      dut.eval();
      if (dut.s_rvalid_o) {
        const uint32_t data = dut.s_rdata_o;
        tick();
        return data;
      }
      tick();
    }
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
  std::vector<std::pair<uint32_t, uint32_t>> writes;
  std::vector<std::pair<uint32_t, uint32_t>> tokens;
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
  std::vector<uint32_t> out;
  try {
    dut->rst_ni = 0;
    for (int i = 0; i < kResetCycles; ++i) bench.tick();
    dut->rst_ni = 1;
    for (const auto& [offset, value] : writes) bench.write(offset, value);
    for (const auto& [token, position] : tokens) {
      bench.write(kRegToken, token);
      bench.write(kRegPosition, position);
      bench.handovers.clear();
      bench.gos.clear();
      bench.eng.dones.clear();
      bench.vec.dones.clear();
      bench.cycle = 0;
      bench.write(kRegControl, 1);
      uint32_t status = 0;
      while (!(status & (kStatusDone | kStatusError))) {
        if (bench.cycle > kMaxTokenCycles) {
          std::println(stderr, "hung: {} handovers after {} cycles", bench.handovers.size(), bench.cycle);
          return kExitHang;
        }
        for (uint64_t i = 0; i < poll; ++i) bench.tick();
        status = bench.read(kRegStatus);
      }
      const uint64_t seen = bench.cycle;
      out.push_back(static_cast<uint32_t>(bench.handovers.size()));
      for (const auto& h : bench.handovers) out.insert(out.end(), {static_cast<uint32_t>(h.cycle), h.unit, h.wait});
      put_all(out, bench.gos);
      put_all(out, bench.eng.dones);
      put_all(out, bench.vec.dones);
      out.insert(out.end(), {static_cast<uint32_t>(seen), status, bench.read(kRegError),
                             bench.read(kRegCyclesLo), bench.read(kRegCyclesHi)});
      for (uint64_t i = 0; i < kDrainCycles * bench.latency_max; ++i) bench.tick();
      bench.eng.pending.clear();  // a failed token's, never done
      bench.vec.pending.clear();
    }
    out.push_back(bench.read(kRegId));
    out.push_back(bench.read(kRegVersion));
  } catch (const GoError& e) {
    std::println(stderr, "{}", e.what());
    return kExitBadGo;
  }
  std::fwrite(out.data(), sizeof(uint32_t), out.size(), stdout);
  dut->final();
  return 0;
}
