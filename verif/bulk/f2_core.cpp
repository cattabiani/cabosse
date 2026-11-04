// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief platforms/f2/core/f2_core.sv, the core as F2 sees it, behind the
/// simulation platform's protocol (sw/driver_protocol.h; the requests as
/// verif/bulk/core.cpp's header, but 5, buffer read, which F2 does not
/// have). The memory is one array at global addresses, as the HBM switch
/// gives every port all of it: each port reads its bursts where their
/// address points, so a port reading the wrong channel reads the wrong
/// data. Port 0 also writes (the vector unit, through f2_port0); a write on
/// any other port is an AXI error.
///
/// Memory requests go into the array directly, the host owning HBM port 0
/// (HOST_PORT set, as platforms/f2/host/core_driver.cpp does before them);
/// a register write that starts a token first hands port 0 to the core.
/// Each handover waits for the wrapper's STATUS. Plusargs: the memory's
/// (axi_settings; +w... for port 0's writes), +seed, +poll. Exits as
/// core.cpp.

#include <cstdlib>

#include "../../sw/driver_protocol.h"
#include "Vf2_core.h"
#include "Vf2_core___024root.h"
#include "axi_bench.h"
#include "axil.h"

namespace {

using Params = Vf2_core___024root;  // the build's parameters (--public-params)
constexpr size_t kPorts = Params::f2_core__DOT__Ports;
constexpr size_t kChanW = Params::f2_core__DOT__ChanW;
constexpr size_t kLen = Params::f2_core__DOT__Len;
constexpr size_t kAddrBits = 34;
constexpr size_t kBeatBytes = 32;
constexpr size_t kBeatWords = kBeatBytes / 4;
constexpr uint32_t kControl = 0x08, kStart = 1;
constexpr uint32_t kWrapHostPort = 0x2'0008, kWrapStatus = 0x2'000C, kWrapHostOwns = 8;
constexpr int kExitAxi = 5;

/// @brief The controller's handshakes with the units (request 7's order).
enum Event : size_t { kEngTake, kEngGo, kEngDone, kVecTake, kVecDone, kEvents };

/// @brief The wrapper, its memory, and which side holds HBM port 0.
struct Bench {
  Vf2_core& dut;
  std::vector<AxiReadPort> rd;
  AxiWritePort wr;  ///< port 0's
  uint64_t progress = 0;
  uint64_t now = 0;
  bool host_owns = true;  ///< HOST_PORT's reset value
  std::array<std::vector<uint64_t>, kEvents> trace;

  /// @brief Note this cycle's handshakes (the core settled, before the edge).
  void record() {
    const auto& r = *dut.rootp;
    const std::array<bool, kEvents> seen{
        r.f2_core__DOT__u_core__DOT__eng_valid && r.f2_core__DOT__u_core__DOT__eng_ready,
        r.f2_core__DOT__u_core__DOT__eng_go != 0, r.f2_core__DOT__u_core__DOT__eng_done != 0,
        r.f2_core__DOT__u_core__DOT__vec_valid && r.f2_core__DOT__u_core__DOT__vec_ready,
        r.f2_core__DOT__u_core__DOT__vec_done != 0};
    for (size_t e = 0; e < kEvents; ++e) {
      if (seen[e]) trace[e].push_back(now);
    }
  }

  /// @brief One clock cycle: the memories' outputs onto the wrapper, settle,
  /// the memories see its outputs, the rising edge.
  /// @throws AxiError on a broken AXI rule, or a write on a port but 0.
  void cycle() {
    for (size_t p = 0; p < kPorts; ++p) {
      set_bits(dut.ar_ready_i, p, 1, rd[p].ar_ready());
      set_bits(dut.r_valid_i, p, 1, rd[p].r_valid());
      set_bits(dut.r_resp_i, 2 * p, 2, rd[p].r_resp());
      set_bits(dut.r_last_i, p, 1, rd[p].r_last());
      if (const auto data = rd[p].r_data(); !data.empty()) {
        std::ranges::copy(data, std::as_writable_bytes(std::span(dut.r_data_i.data() + kBeatWords * p, kBeatWords)).begin());
      }
    }
    set_bits(dut.aw_ready_i, 0, 1, wr.aw_ready());
    set_bits(dut.w_ready_i, 0, 1, wr.w_ready());
    set_bits(dut.b_valid_i, 0, 1, wr.b_valid());
    set_bits(dut.b_resp_i, 0, 2, wr.b_resp());
    dut.clk_i = 0;
    dut.eval();
    if (dut.rst_ni) tick();
    if (dut.rst_ni) record();
    dut.clk_i = 1;
    dut.eval();
    ++now;
  }

  /// @brief The memories see the wrapper's outputs this cycle.
  void tick() {
    const auto bit = [&](const auto& sig, size_t p) { return get_bits(sig, p, 1) != 0; };
    for (size_t p = 0; p < kPorts; ++p) {
      progress += bit(dut.r_valid_i, p) && bit(dut.r_ready_o, p);
      rd[p].tick({bit(dut.ar_valid_o, p), get_bits(dut.ar_addr_o, kAddrBits * p, kAddrBits),
                  static_cast<unsigned>(get_bits(dut.ar_len_o, 8 * p, 8)),
                  static_cast<unsigned>(get_bits(dut.ar_size_o, 3 * p, 3)),
                  static_cast<unsigned>(get_bits(dut.ar_burst_o, 2 * p, 2))},
                 bit(dut.r_ready_o, p));
      if (p > 0 && (bit(dut.aw_valid_o, p) || bit(dut.w_valid_o, p))) {
        throw AxiError(std::format("a write on port {}: only port 0 writes", p));
      }
    }
    const bool w_valid = bit(dut.w_valid_o, 0);
    progress += w_valid && bit(dut.w_ready_i, 0);
    wr.tick({bit(dut.aw_valid_o, 0), get_bits(dut.aw_addr_o, 0, kAddrBits),
             static_cast<unsigned>(get_bits(dut.aw_len_o, 0, 8)), static_cast<unsigned>(get_bits(dut.aw_size_o, 0, 3)),
             static_cast<unsigned>(get_bits(dut.aw_burst_o, 0, 2))},
            {w_valid, std::as_bytes(std::span(dut.w_data_o.data(), kBeatWords)), get_bits(dut.w_strb_o, 0, kBeatBytes),
             bit(dut.w_last_o, 0)},
            bit(dut.b_ready_o, 0));
  }

  /// @brief Give HBM port 0 to the host or to the core (HOST_PORT), and
  /// wait until the wrapper's STATUS shows it.
  /// @throws std::runtime_error if it never does.
  void own(bool host) {
    if (host == host_owns) return;
    axil_write(*this, kWrapHostPort, host ? 1 : 0);
    for (uint64_t i = 0; ((axil_read(*this, kWrapStatus) & kWrapHostOwns) != 0) != host; ++i) {
      if (i == kMaxIdleCycles) throw std::runtime_error("HBM port 0 not handed over");
    }
    host_owns = host;
  }
};

/// @brief Read words from stdin.
/// @return false at the end of the input.
bool read_exact(std::span<uint32_t> words) {
  return std::fread(words.data(), sizeof(uint32_t), words.size(), stdin) == words.size();
}

/// @brief Write words to stdout and flush: the parent waits for them.
void answer(std::span<const uint32_t> words) {
  std::fwrite(words.data(), sizeof(uint32_t), words.size(), stdout);
  std::fflush(stdout);
}

/// @brief The bytes [addr, addr + n) of the memory.
/// @throws std::out_of_range past its end.
std::span<std::byte> bytes_at(std::span<std::byte> mem, uint64_t addr, uint64_t n) {
  if (addr > mem.size() || n > mem.size() - addr) throw std::out_of_range(std::format("0x{:X} + {} past the memory", addr, n));
  return mem.subspan(addr, n);
}

}  // namespace

int main(int argc, char** argv) {
  VerilatedContext context;
  context.commandArgs(argc, argv);
  auto dut = std::make_unique<Vf2_core>(&context);
  const uint64_t total = (uint64_t{1} << kChanW) * kPorts;
  std::unique_ptr<std::byte, decltype(&std::free)> raw(static_cast<std::byte*>(std::calloc(total, 1)), &std::free);
  if (!raw) {
    std::println(stderr, "cannot allocate {} bytes of memory", total);
    return 1;
  }
  const std::span<std::byte> mem(raw.get(), total);
  const uint64_t seed = plusarg(context, "seed", 1);
  std::vector<AxiReadPort> rd;
  for (size_t p = 0; p < kPorts; ++p) rd.emplace_back(mem, axi_settings(context, kBeatBytes, p), (seed << 16) + p);
  Bench bench{.dut = *dut,
              .rd = std::move(rd),
              .wr = AxiWritePort(mem, axi_settings(context, kBeatBytes, 0, "w"), (seed << 16) + kPorts)};
  const uint64_t poll = std::max<uint64_t>(1, plusarg(context, "poll", 64));
  dut->hbm_ready_i = 1;
  dut->h_ar_valid_i = dut->h_aw_valid_i = dut->h_w_valid_i = 0;
  dut->h_r_ready_i = dut->h_b_ready_i = 1;
  try {
    reset(bench);
    const std::array<uint32_t, 4> hello{protocol::kHello, kPorts, kChanW, kLen};
    answer(hello);
    std::array<uint32_t, 3> a{};
    for (uint32_t op = 0; read_exact(std::span(&op, 1));) {
      switch (op) {
        case protocol::kRegWrite:
          if (!read_exact(std::span(a).first(2))) throw std::out_of_range("register write");
          if (a[0] == kControl && (a[1] & kStart)) bench.own(false);
          axil_write(bench, a[0], a[1]);
          break;
        case protocol::kRegRead: {
          if (!read_exact(std::span(a).first(1))) throw std::out_of_range("register read");
          const uint32_t v = axil_read(bench, a[0]);
          answer(std::span(&v, 1));
          break;
        }
        case protocol::kMemWrite: {
          if (!read_exact(a)) throw std::out_of_range("memory write");
          std::vector<uint32_t> words((a[2] + 3) / 4);
          if (!read_exact(words)) throw std::out_of_range("memory write's bytes");
          bench.own(true);
          const auto dst = bytes_at(mem, uint64_t{a[1]} << 32 | a[0], a[2]);
          std::ranges::copy(std::as_bytes(std::span(words)).first(dst.size()), dst.begin());
          break;
        }
        case protocol::kMemRead: {
          if (!read_exact(a)) throw std::out_of_range("memory read");
          bench.own(true);
          const auto src = bytes_at(mem, uint64_t{a[1]} << 32 | a[0], a[2]);
          std::vector<uint32_t> words((a[2] + 3) / 4);
          std::ranges::copy(src, std::as_writable_bytes(std::span(words)).begin());
          answer(words);
          break;
        }
        case protocol::kRun: {
          const auto status = axil_until_status(bench, poll, [&] { return bench.progress; });
          if (!status) {
            std::println(stderr, "hung: no memory traffic for {} cycles", kMaxIdleCycles);
            return kExitHang;
          }
          answer(std::span(&*status, 1));
          break;
        }
        case protocol::kQuit:
          dut->final();
          return 0;
        case protocol::kTrace: {
          std::vector<uint32_t> out;
          for (auto& cycles : bench.trace) {
            out.push_back(static_cast<uint32_t>(cycles.size()));
            for (const uint64_t c : cycles) out.insert(out.end(), {static_cast<uint32_t>(c), static_cast<uint32_t>(c >> 32)});
            cycles.clear();
          }
          answer(out);
          break;
        }
        default:
          throw std::out_of_range(std::format("request {} (not F2's, or unknown)", op));
      }
    }
  } catch (const AxiError& e) {
    std::println(stderr, "AXI: {}", e.what());
    return kExitAxi;
  } catch (const std::out_of_range& e) {
    std::println(stderr, "bad request: {}", e.what());
    return kExitBadInput;
  } catch (const std::runtime_error& e) {
    std::println(stderr, "{}", e.what());
    return kExitHang;
  }
  dut->final();
  return 0;
}
