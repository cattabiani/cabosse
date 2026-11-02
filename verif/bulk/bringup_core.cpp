// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief Bulk driver for platforms/f2/bringup/bringup_core.sv: register
/// accesses on its AXI-Lite port, and its HBM ports on the memory model of
/// axi_mem.h, every port reaching all of one shared memory (as HBM's switch
/// does) (verif/bulk.py builds it).
///
/// Input (little-endian uint32 on stdin): the memory's size in words, its
/// words, the number of operations, then per operation three words: kind,
/// register offset, value. Kinds: 0 write the value, 1 read (its value goes
/// out), 2 read STATUS until busy is clear, 3 wait the value's cycles,
/// 4 the host writes one beat at the offset (a byte address), its words the
/// value plus their index, 5 the host reads one beat at the offset (its 8
/// words go out), 6 the host posts that read and does not wait, 7 the host
/// takes the posted read's beat (its 8 words go out). The host's accesses
/// are one-beat bursts on the host port (PCIS's place), one at a time.
/// Plusargs: the memory's (axi_settings; +w... for the write side), +seed.
/// Output: the values read, in order; then the memory's words.

#include <algorithm>

#include "Vbringup_core.h"
#include "Vbringup_core___024root.h"
#include "axi_bench.h"

namespace {

using Params = Vbringup_core___024root;  // the build's parameters (verif/bulk.py: --public-params)
constexpr size_t kPorts = Params::bringup_core__DOT__Ports;
constexpr size_t kBeatWords = 8;  ///< 256 bits
constexpr size_t kBeatBytes = 4 * kBeatWords;
constexpr size_t kAddrBits = 34;
constexpr uint32_t kStatus = 0x00C;
constexpr uint64_t kMaxPolls = 1'000'000;  ///< STATUS reads before calling a run hung

/// @brief The core, its memories and an AXI-Lite master doing one access at a time.
struct Bench {
  Vbringup_core& dut;
  std::vector<AxiReadPort>& rd;
  std::vector<AxiWritePort>& wr;

  /// @brief One clock cycle: the memories' outputs onto the core, settle, the
  /// memories see the core's outputs, then the rising edge. In reset the
  /// memories do not tick.
  /// @throws AxiError if a port breaks an AXI rule.
  void cycle() {
    for (size_t p = 0; p < kPorts; ++p) {
      set_bits(dut.ar_ready_i, p, 1, rd[p].ar_ready());
      set_bits(dut.r_valid_i, p, 1, rd[p].r_valid());
      set_bits(dut.r_resp_i, 2 * p, 2, rd[p].r_resp());
      set_bits(dut.r_last_i, p, 1, rd[p].r_last());
      if (const auto data = rd[p].r_data(); !data.empty()) {
        std::ranges::copy(data, std::as_writable_bytes(std::span(dut.r_data_i.data() + kBeatWords * p, kBeatWords)).begin());
      }
      set_bits(dut.aw_ready_i, p, 1, wr[p].aw_ready());
      set_bits(dut.w_ready_i, p, 1, wr[p].w_ready());
      set_bits(dut.b_valid_i, p, 1, wr[p].b_valid());
      set_bits(dut.b_resp_i, 2 * p, 2, wr[p].b_resp());
    }
    dut.clk_i = 0;
    dut.eval();
    if (dut.rst_ni) {
      for (size_t p = 0; p < kPorts; ++p) tick_port(p);
    }
    dut.clk_i = 1;
    dut.eval();
  }

  /// @brief Port p's memories see its outputs this cycle.
  void tick_port(size_t p) {
    const auto bit = [&](const auto& sig) { return get_bits(sig, p, 1) != 0; };
    rd[p].tick({bit(dut.ar_valid_o), get_bits(dut.ar_addr_o, kAddrBits * p, kAddrBits),
                static_cast<unsigned>(get_bits(dut.ar_len_o, 8 * p, 8)),
                static_cast<unsigned>(get_bits(dut.ar_size_o, 3 * p, 3)),
                static_cast<unsigned>(get_bits(dut.ar_burst_o, 2 * p, 2))},
               bit(dut.r_ready_o));
    const bool w_valid = bit(dut.w_valid_o);
    const std::span<const std::byte> w_beat = std::as_bytes(std::span(dut.w_data_o.data() + kBeatWords * p, kBeatWords));
    wr[p].tick({bit(dut.aw_valid_o), get_bits(dut.aw_addr_o, kAddrBits * p, kAddrBits),
                static_cast<unsigned>(get_bits(dut.aw_len_o, 8 * p, 8)),
                static_cast<unsigned>(get_bits(dut.aw_size_o, 3 * p, 3)),
                static_cast<unsigned>(get_bits(dut.aw_burst_o, 2 * p, 2))},
               {w_valid, w_beat, get_bits(dut.w_strb_o, kBeatBytes * p, kBeatBytes), bit(dut.w_last_o)},
               bit(dut.b_ready_o));
  }

  /// @brief Cycle until `done()` holds after a settle, at most kMaxIdleCycles.
  /// @throws std::runtime_error on a hang.
  template <typename Done>
  void until(Done done, const char* what) {
    for (uint64_t i = 0; i < kMaxIdleCycles; ++i) {
      dut.clk_i = 0;
      dut.eval();
      if (done()) return;
      cycle();
    }
    throw std::runtime_error(std::format("hung waiting for {}", what));
  }

  /// @brief Write a register: address and data together, then the response.
  void write(uint32_t addr, uint32_t value) {
    dut.s_aw_valid_i = dut.s_w_valid_i = 1;
    dut.s_aw_addr_i = addr;
    dut.s_w_data_i = value;
    until([&] { return dut.s_aw_ready_o != 0; }, "AWREADY");
    cycle();
    dut.s_aw_valid_i = dut.s_w_valid_i = 0;
    dut.s_b_ready_i = 1;
    until([&] { return dut.s_b_valid_o != 0; }, "BVALID");
    cycle();
    dut.s_b_ready_i = 0;
  }

  /// @brief Read a register.
  uint32_t read(uint32_t addr) {
    dut.s_ar_valid_i = 1;
    dut.s_ar_addr_i = addr;
    until([&] { return dut.s_ar_ready_o != 0; }, "ARREADY");
    cycle();
    dut.s_ar_valid_i = 0;
    dut.s_r_ready_i = 1;
    until([&] { return dut.s_r_valid_o != 0; }, "RVALID");
    const uint32_t value = dut.s_r_data_o;
    cycle();
    dut.s_r_ready_i = 0;
    return value;
  }

  /// @brief The host writes one beat.
  /// @param addr its byte address, beat-aligned.
  /// @param value word i of the beat is value + i.
  /// @throws std::runtime_error on a hang.
  void host_write(uint64_t addr, uint32_t value) {
    dut.h_aw_valid_i = dut.h_w_valid_i = 1;
    dut.h_aw_addr_i = addr;
    dut.h_aw_len_i = 0;
    dut.h_aw_burst_i = 1;
    for (size_t i = 0; i < kBeatWords; ++i) set_bits(dut.h_w_data_i, 32 * i, 32, value + static_cast<uint32_t>(i));
    dut.h_w_strb_i = 0xFFFF'FFFF;
    dut.h_w_last_i = 1;
    // AW and W may be taken in different cycles; each drops after its handshake.
    for (uint64_t i = 0; dut.h_aw_valid_i || dut.h_w_valid_i; ++i) {
      if (i == kMaxIdleCycles) throw std::runtime_error("hung waiting for the host's AWREADY and WREADY");
      dut.clk_i = 0;
      dut.eval();
      const bool aw_taken = dut.h_aw_valid_i && dut.h_aw_ready_o, w_taken = dut.h_w_valid_i && dut.h_w_ready_o;
      cycle();
      if (aw_taken) dut.h_aw_valid_i = 0;
      if (w_taken) dut.h_w_valid_i = 0;
    }
    dut.h_b_ready_i = 1;
    until([&] { return dut.h_b_valid_o != 0; }, "the host's BVALID");
    cycle();
    dut.h_b_ready_i = 0;
  }

  /// @brief The host posts a one-beat read; host_take() collects its beat.
  /// @param addr its byte address, beat-aligned.
  /// @throws std::runtime_error on a hang.
  void host_post(uint64_t addr) {
    dut.h_ar_valid_i = 1;
    dut.h_ar_addr_i = addr;
    dut.h_ar_len_i = 0;
    dut.h_ar_burst_i = 1;
    until([&] { return dut.h_ar_ready_o != 0; }, "the host's ARREADY");
    cycle();
    dut.h_ar_valid_i = 0;
  }

  /// @brief The beat of the read host_post() posted.
  /// @return its 8 words.
  /// @throws std::runtime_error on a hang.
  std::array<uint32_t, kBeatWords> host_take() {
    dut.h_r_ready_i = 1;
    until([&] { return dut.h_r_valid_o != 0; }, "the host's RVALID");
    std::array<uint32_t, kBeatWords> beat;
    for (size_t i = 0; i < kBeatWords; ++i) beat[i] = static_cast<uint32_t>(get_bits(dut.h_r_data_o, 32 * i, 32));
    cycle();
    dut.h_r_ready_i = 0;
    return beat;
  }
};

}  // namespace

/// @brief Run the protocol of this file's header.
/// @return 0; 2 if the input is malformed; 3 on a hang or memory left busy;
///   5 if a port breaks an AXI rule (each with a message on stderr).
int main(int argc, char** argv) {
  const std::vector<uint32_t> words = read_words(stdin);
  Reader in{words};
  std::vector<uint32_t> mem;
  std::vector<Record> ops;
  try {
    mem = in.take(in.next()) | std::ranges::to<std::vector>();
    const size_t n_ops = in.next();
    ops = std::views::chunk(in.take(3 * n_ops), 3) | std::ranges::to<std::vector<Record>>();
  } catch (const std::out_of_range& e) {
    std::println(stderr, "{}", e.what());
    return kExitBadInput;
  }

  VerilatedContext context;
  context.commandArgs(argc, argv);
  const uint64_t seed = plusarg(context, "seed", 1);
  auto dut = std::make_unique<Vbringup_core>(&context);
  const std::span<std::byte> bytes = std::as_writable_bytes(std::span(mem));
  std::vector<AxiReadPort> rd;
  std::vector<AxiWritePort> wr;
  for (size_t p = 0; p < kPorts; ++p) {
    rd.emplace_back(bytes, axi_settings(context, kBeatBytes, p), seed + 2 * p);
    wr.emplace_back(bytes, axi_settings(context, kBeatBytes, p, "w"), seed + 2 * p + 1);
  }
  Bench bench{.dut = *dut, .rd = rd, .wr = wr};
  dut->hbm_ready_i = 1;

  std::vector<uint32_t> out;
  try {
    reset(bench);
    for (Record op : ops) {
      switch (op[0]) {
        case 0: bench.write(op[1], op[2]); break;
        case 1: out.push_back(bench.read(op[1])); break;
        case 2:
          for (uint64_t polls = 1; bench.read(kStatus) & 1; ++polls) {
            if (polls == kMaxPolls) throw std::runtime_error("run hung: STATUS.busy stays set");
          }
          break;
        case 3:
          for (uint32_t i = 0; i < op[2]; ++i) bench.cycle();
          break;
        case 4: bench.host_write(op[1], op[2]); break;
        case 5:
          bench.host_post(op[1]);
          std::ranges::copy(bench.host_take(), std::back_inserter(out));
          break;
        case 6: bench.host_post(op[1]); break;
        case 7: std::ranges::copy(bench.host_take(), std::back_inserter(out)); break;
        default: std::println(stderr, "unknown operation {}", op[0]); return kExitBadInput;
      }
    }
    for (int i = 0; i < kDrainCycles; ++i) bench.cycle();
    for (size_t p = 0; p < kPorts; ++p) {
      if (!rd[p].idle() || !wr[p].idle()) {
        std::println(stderr, "port {}: memory still busy", p);
        return kExitHang;
      }
    }
  } catch (const AxiError& e) {
    std::println(stderr, "AXI: {}", e.what());
    return 5;
  } catch (const std::runtime_error& e) {
    std::println(stderr, "{}", e.what());
    return kExitHang;
  }
  out.insert(out.end(), mem.begin(), mem.end());
  std::fwrite(out.data(), sizeof(uint32_t), out.size(), stdout);
  dut->final();
  return 0;
}
