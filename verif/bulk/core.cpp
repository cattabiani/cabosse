// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief The simulation platform's driver for rtl/core.sv (sw/sim_platform.py):
/// a long-lived child process that serves the runtime's six calls
/// (sw/platforms.py) on stdin and stdout, with the HBM behind the core's AXI
/// ports in a memory model (axi_mem.h).
///
/// At start it writes kHello, Ports, ChanW, Len and Lanes. Then requests,
/// little-endian uint32 words, each an opcode and its arguments:
/// - 0 register write: offset, value (AXI-Lite). No answer.
/// - 1 register read: offset. Answer: the value.
/// - 2 memory write: address low, high, bytes, then the bytes padded to
///   words (straight into the memory, as the host's loads). No answer.
/// - 3 memory read: address low, high, bytes. Answer: the bytes padded to words.
/// - 4 run: cycles until STATUS is done or error, polled over AXI-Lite every
///   +poll cycles. Answer: the cycles run, low and high word.
/// - 5 buffer read: buffer, first element, elements (host_r port, Lanes a
///   cycle). Answer: the elements' raw bits, a word each.
/// - 6 quit.
/// The memory is Ports channels of 2^ChanW bytes (calloc: the pages no one
/// writes cost nothing); engine port p reads channel p, the vector unit's
/// port the whole of it at global addresses. Plusargs: the memory's
/// (axi_settings; +w... for the vector unit's write side), +seed, +poll.
/// Exits 3 if a run makes no progress for kMaxIdleCycles (STATUS polls aside),
/// 5 on a broken AXI rule, each with a message on stderr.

#include <cstdlib>
#include <cstring>

#include "Vcore.h"
#include "Vcore___024root.h"
#include "axi_bench.h"
#include "axil.h"

namespace {

using Params = Vcore___024root;  // the build's parameters (verif/bulk.py: --public-params)
constexpr size_t kPorts = Params::core__DOT__Ports;
constexpr size_t kChanW = Params::core__DOT__ChanW;
constexpr size_t kLen = Params::core__DOT__Len;
constexpr size_t kLanes = Params::core__DOT__Lanes;
constexpr size_t kAddrBits = 34;
constexpr size_t kBeatBytes = 32;
constexpr size_t kBeatWords = kBeatBytes / 4;
constexpr uint32_t kHello = 0x4342'5349;  // 'CBSI'
constexpr uint32_t kRegStatus = 0x0C, kStatusDone = 2, kStatusError = 4;
constexpr int kExitAxi = 5;

/// @brief Word i of a beat's bytes.
uint32_t word_at(std::span<const std::byte> bytes, size_t i) {
  uint32_t w;
  std::memcpy(&w, bytes.data() + 4 * i, 4);
  return w;
}

/// @brief The core, its memory and the host's side of its ports.
struct Bench {
  Vcore& dut;
  std::span<std::byte> mem;  ///< every channel, channel c at c << kChanW
  std::vector<AxiReadPort> ports;
  AxiReadPort vec_rd;
  AxiWritePort vec_wr;
  std::array<std::byte, kBeatBytes> w_beat{};
  uint64_t progress = 0;  ///< memory beats moved: a run's sign of life

  /// @brief One clock cycle: the memories' outputs onto the core, settle,
  /// the memories see the core's outputs, the rising edge.
  /// @throws AxiError on a broken AXI rule.
  void cycle() {
    for (size_t p = 0; p < kPorts; ++p) {
      set_bits(dut.e_ar_ready_i, p, 1, ports[p].ar_ready());
      set_bits(dut.e_r_valid_i, p, 1, ports[p].r_valid());
      set_bits(dut.e_r_resp_i, 2 * p, 2, ports[p].r_resp());
      if (const auto data = ports[p].r_data(); !data.empty()) {
        std::ranges::copy(data, std::as_writable_bytes(std::span(dut.e_r_data_i.data() + kBeatWords * p, kBeatWords)).begin());
      }
    }
    dut.v_ar_ready_i = vec_rd.ar_ready();
    dut.v_r_valid_i = vec_rd.r_valid();
    dut.v_r_resp_i = vec_rd.r_resp();
    if (const auto data = vec_rd.r_data(); !data.empty()) {
      std::ranges::copy(data, std::as_writable_bytes(std::span(dut.v_r_data_i.data(), kBeatWords)).begin());
    }
    dut.v_aw_ready_i = vec_wr.aw_ready();
    dut.v_w_ready_i = vec_wr.w_ready();
    dut.v_b_valid_i = vec_wr.b_valid();
    dut.v_b_resp_i = vec_wr.b_resp();
    dut.clk_i = 0;
    dut.eval();
    if (dut.rst_ni) {
      for (size_t p = 0; p < kPorts; ++p) {
        progress += get_bits(dut.e_r_valid_i, p, 1) && get_bits(dut.e_r_ready_o, p, 1);
        ports[p].tick({static_cast<bool>(get_bits(dut.e_ar_valid_o, p, 1)), get_bits(dut.e_ar_addr_o, kAddrBits * p, kAddrBits),
                       static_cast<unsigned>(get_bits(dut.e_ar_len_o, 8 * p, 8)),
                       static_cast<unsigned>(get_bits(dut.e_ar_size_o, 3 * p, 3)),
                       static_cast<unsigned>(get_bits(dut.e_ar_burst_o, 2 * p, 2))},
                      get_bits(dut.e_r_ready_o, p, 1));
      }
      progress += (dut.v_r_valid_i && dut.v_r_ready_o) + (dut.v_w_valid_o && dut.v_w_ready_i);
      vec_rd.tick({static_cast<bool>(dut.v_ar_valid_o), dut.v_ar_addr_o, dut.v_ar_len_o, dut.v_ar_size_o, dut.v_ar_burst_o},
                  dut.v_r_ready_o);
      std::ranges::copy(std::as_bytes(std::span(dut.v_w_data_o.data(), kBeatWords)), w_beat.begin());
      vec_wr.tick({static_cast<bool>(dut.v_aw_valid_o), dut.v_aw_addr_o, dut.v_aw_len_o, dut.v_aw_size_o, dut.v_aw_burst_o},
                  {static_cast<bool>(dut.v_w_valid_o), w_beat, get_bits(dut.v_w_strb_o, 0, kBeatBytes),
                   static_cast<bool>(dut.v_w_last_o)},
                  dut.v_b_ready_o);
    }
    dut.clk_i = 1;
    dut.eval();
  }

  /// @brief Run until STATUS is done or error.
  /// @return the cycles run, or nothing on a hang.
  std::optional<uint64_t> run(uint64_t poll) {
    uint64_t cycles = 0, idle = 0;
    while (true) {
      const uint64_t before = progress;
      for (uint64_t i = 0; i < poll; ++i) cycle();
      cycles += poll;
      if (axil_read(*this, kRegStatus) & (kStatusDone | kStatusError)) return cycles;
      idle = progress == before ? idle + poll : 0;
      if (idle > kMaxIdleCycles) return std::nullopt;
    }
  }

  /// @brief n elements of a buffer from `first`, Lanes a cycle.
  std::vector<uint32_t> read_buffer(uint32_t buf, uint32_t first, uint32_t n) {
    std::vector<uint32_t> out;
    dut.host_r_valid_i = 1;
    dut.host_r_buf_i = buf;
    for (uint32_t at = first; at < first + n; at += kLanes) {
      dut.host_r_addr_i = at;
      cycle();
      dut.clk_i = 0;
      dut.eval();
      for (size_t l = 0; l < kLanes && at + l < first + n; ++l) {
        out.push_back(static_cast<uint32_t>(get_bits(dut.host_r_data_o, 32 * l, 32)));
      }
    }
    dut.host_r_valid_i = 0;
    return out;
  }
};

/// @brief Read n words from stdin.
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
  auto dut = std::make_unique<Vcore>(&context);
  const uint64_t channel_bytes = uint64_t{1} << kChanW;
  const uint64_t total = channel_bytes * kPorts;
  std::unique_ptr<std::byte, decltype(&std::free)> raw(static_cast<std::byte*>(std::calloc(total, 1)), &std::free);
  if (!raw) {
    std::println(stderr, "cannot allocate {} bytes of memory", total);
    return 1;
  }
  const std::span<std::byte> mem(raw.get(), total);
  const uint64_t seed = plusarg(context, "seed", 1);
  std::vector<AxiReadPort> ports;
  for (size_t p = 0; p < kPorts; ++p) {
    ports.emplace_back(mem.subspan(p * channel_bytes, channel_bytes), axi_settings(context, kBeatBytes, p), (seed << 16) + p);
  }
  Bench bench{.dut = *dut,
              .mem = mem,
              .ports = std::move(ports),
              .vec_rd = AxiReadPort(mem, axi_settings(context, kBeatBytes, kPorts), (seed << 16) + kPorts),
              .vec_wr = AxiWritePort(mem, axi_settings(context, kBeatBytes, 0, "w"), (seed << 16) + kPorts + 1)};
  const uint64_t poll = std::max<uint64_t>(1, plusarg(context, "poll", 64));
  dut->host_r_valid_i = 0;
  try {
    reset(bench);
    const std::array<uint32_t, 5> hello{kHello, kPorts, kChanW, kLen, kLanes};
    answer(hello);
    std::array<uint32_t, 3> a{};
    for (uint32_t op = 0; read_exact(std::span(&op, 1));) {
      switch (op) {
        case 0:
          if (!read_exact(std::span(a).first(2))) throw std::out_of_range("register write");
          axil_write(bench, a[0], a[1]);
          break;
        case 1: {
          if (!read_exact(std::span(a).first(1))) throw std::out_of_range("register read");
          const uint32_t v = axil_read(bench, a[0]);
          answer(std::span(&v, 1));
          break;
        }
        case 2: {
          if (!read_exact(a)) throw std::out_of_range("memory write");
          std::vector<uint32_t> words((a[2] + 3) / 4);
          if (!read_exact(words)) throw std::out_of_range("memory write's bytes");
          const auto dst = bytes_at(mem, uint64_t{a[1]} << 32 | a[0], a[2]);
          std::memcpy(dst.data(), words.data(), dst.size());
          break;
        }
        case 3: {
          if (!read_exact(a)) throw std::out_of_range("memory read");
          const auto src = bytes_at(mem, uint64_t{a[1]} << 32 | a[0], a[2]);
          std::vector<uint32_t> words((a[2] + 3) / 4);
          std::memcpy(words.data(), src.data(), src.size());
          answer(words);
          break;
        }
        case 4: {
          const auto cycles = bench.run(poll);
          if (!cycles) {
            std::println(stderr, "hung: no memory traffic for {} cycles", kMaxIdleCycles);
            return kExitHang;
          }
          const std::array<uint32_t, 2> c{static_cast<uint32_t>(*cycles), static_cast<uint32_t>(*cycles >> 32)};
          answer(c);
          break;
        }
        case 5: {
          if (!read_exact(a)) throw std::out_of_range("buffer read");
          if (a[1] > kLen || a[2] > kLen - a[1]) throw std::out_of_range("buffer read past the buffer");
          answer(bench.read_buffer(a[0], a[1], a[2]));
          break;
        }
        case 6:
          dut->final();
          return 0;
        default:
          throw std::out_of_range(std::format("unknown request {}", op));
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
