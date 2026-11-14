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
/// Each handover waits for the wrapper's STATUS.
///
/// PCIM (M11): the wrapper's copies of OUTPUT's logits land in a host page
/// (kHostBytes at the made-up physical address kHostBase, a memory model of
/// 64-byte beats), as core_driver.cpp's hugepage on F2. Request 8, output
/// read, arms the window on its first call for an address and size (a
/// memory read that time); after it, a start clears PCIM_DONE and fills the
/// window's bytes in the page with 0xFF, and an output read waits until
/// PCIM_DONE counts the window's beats, then answers from the page. A copy
/// outside the window, strobes that break PCIe's contiguous byte enables,
/// or an error response stop the driver (AXI error). Plusargs: the
/// memories' (axi_settings; +w... for port 0's writes, +p... for PCIM),
/// +seed, +poll. Exits as core.cpp.

#include <algorithm>
#include <bit>
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
constexpr uint32_t kWrapPcimBaseLo = 0x2'0024, kWrapPcimBaseHi = 0x2'0028, kWrapPcimBytes = 0x2'002C;
constexpr uint32_t kWrapPcimHostLo = 0x2'0030, kWrapPcimHostHi = 0x2'0034, kWrapPcimDone = 0x2'0038;
constexpr uint32_t kWrapPcimErr = 0x2'003C;
constexpr uint64_t kHostBase = 0x12'3460'0000;  ///< the host page's physical address (2 MiB-aligned, above 4 GiB)
constexpr uint64_t kHostBytes = uint64_t{2} << 20;
constexpr size_t kPcimBytes = 64;  ///< PCIM's beat
constexpr int kExitAxi = 5;

/// @brief The controller's handshakes with the units (request 7's order).
enum Event : size_t { kEngTake, kEngGo, kEngDone, kVecTake, kVecDone, kEvents };

/// @brief The wrapper, its memory, and which side holds HBM port 0.
struct Bench {
  Vf2_core& dut;
  std::vector<AxiReadPort> rd;
  AxiWritePort wr;  ///< port 0's
  AxiWritePort pcim;  ///< the host page, behind PCIM
  std::span<std::byte> host;  ///< its bytes
  uint64_t win_addr = 0, win_bytes = 0;  ///< PCIM's window (0 bytes: not armed)
  bool pending = false;  ///< a token started since the window's copies were last read
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
    dut.pcim_aw_ready_i = pcim.aw_ready();
    dut.pcim_w_ready_i = pcim.w_ready();
    dut.pcim_b_valid_i = pcim.b_valid();
    dut.pcim_b_resp_i = pcim.b_resp();
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
    const uint64_t strb = dut.pcim_w_strb_o;
    if (dut.pcim_w_valid_o && (strb == 0 || std::popcount(strb) != 64 - std::countl_zero(strb) - std::countr_zero(strb))) {
      throw AxiError(std::format("PCIM strobes 0x{:016X} not contiguous", strb));
    }
    pcim.tick({dut.pcim_aw_valid_o != 0, dut.pcim_aw_addr_o, 0, std::countr_zero(kPcimBytes), 1},
              {dut.pcim_w_valid_o != 0, std::as_bytes(std::span(dut.pcim_w_data_o.data(), kPcimBytes / 4)), strb, true},
              true);
  }

  /// @brief Arm PCIM's window: card memory [addr, addr + n) copied to the
  /// host page's first n bytes.
  void arm(uint64_t addr, uint64_t n) {
    axil_write(*this, kWrapPcimBaseLo, static_cast<uint32_t>(addr));
    axil_write(*this, kWrapPcimBaseHi, static_cast<uint32_t>(addr >> 32));
    axil_write(*this, kWrapPcimBytes, static_cast<uint32_t>(n));
    axil_write(*this, kWrapPcimHostLo, static_cast<uint32_t>(kHostBase));
    axil_write(*this, kWrapPcimHostHi, static_cast<uint32_t>(kHostBase >> 32));
    axil_write(*this, kWrapPcimDone, 0);
    win_addr = addr;
    win_bytes = n;
    pending = false;
  }

  /// @brief Before a start: the last token's copies all answered, then the
  /// count cleared and the window's bytes in the page set to 0xFF.
  void start_copies() {
    if (win_bytes == 0) return;
    if (pending) wait_copies();
    std::ranges::fill(host.first(win_bytes), std::byte{0xFF});
    axil_write(*this, kWrapPcimDone, 0);
    pending = true;
  }

  /// @brief Wait until PCIM_DONE counts the window's beats.
  /// @throws AxiError on an error response or more copies than beats.
  /// @throws std::runtime_error if they never all come.
  void wait_copies() {
    const uint64_t beats = win_bytes / kBeatBytes;
    for (uint64_t i = 0;; ++i) {
      if (const uint32_t e = axil_read(*this, kWrapPcimErr); e != 0) throw AxiError(std::format("PCIM: {} error responses", e));
      const uint32_t done = axil_read(*this, kWrapPcimDone);
      if (done == beats) break;
      if (done > beats) throw AxiError(std::format("PCIM: {} copies answered, the window has {} beats", done, beats));
      if (i == kMaxIdleCycles) throw std::runtime_error(std::format("PCIM: {} of {} copies answered", done, beats));
    }
    pending = false;
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

using protocol::answer, protocol::read_exact;

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
  std::vector<std::byte> host(kHostBytes);
  AxiMemSettings pcim = axi_settings(context, kPcimBytes, 0, "p");
  pcim.base = kHostBase;
  std::vector<AxiReadPort> rd;
  for (size_t p = 0; p < kPorts; ++p) rd.emplace_back(mem, axi_settings(context, kBeatBytes, p), (seed << 16) + p);
  Bench bench{.dut = *dut,
              .rd = std::move(rd),
              .wr = AxiWritePort(mem, axi_settings(context, kBeatBytes, 0, "w"), (seed << 16) + kPorts),
              .pcim = AxiWritePort(host, pcim, (seed << 16) + kPorts + 1),
              .host = host};
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
          if (a[0] == kControl && (a[1] & kStart)) {
            bench.start_copies();
            bench.own(false);
          }
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
        case protocol::kOutputRead: {
          if (!read_exact(a)) throw std::out_of_range("output read");
          const uint64_t addr = uint64_t{a[1]} << 32 | a[0], n = a[2];
          std::vector<uint32_t> words((n + 3) / 4);
          const auto out = std::as_writable_bytes(std::span(words)).first(n);
          if (bench.win_bytes != 0 && addr == bench.win_addr && n == bench.win_bytes) {
            if (bench.pending) bench.wait_copies();
            if (std::ranges::any_of(bench.host.subspan(n), [](std::byte b) { return b != std::byte{0}; })) {
              throw AxiError("PCIM: a copy outside the window");
            }
            std::ranges::copy(bench.host.first(n), out.begin());
          } else {
            bench.own(true);
            std::ranges::copy(bytes_at(mem, addr, n), out.begin());
            if (n > 0 && n <= kHostBytes && addr % kBeatBytes == 0 && n % kBeatBytes == 0) bench.arm(addr, n);
          }
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
