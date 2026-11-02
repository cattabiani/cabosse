// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief The bring-up design's host tool (platforms/f2/README.md, PLAN.md
/// M8), run on the F2 instance with the AFI loaded: one measurement per call,
/// its result as one line of JSON on stdout (scripts/f2/bringup_plan.py
/// drives it and keeps the lines). Built there against AWS's SDK
/// (libfpga_mgmt, which has fpga_pci and fpga_clkgen), see the Makefile.
///
/// Commands:
/// - `info`: the design's ID, BUILD, STATUS and the clock generator's clocks.
/// - `clock MHZ`: clk_hbm_axi (the core's clock, D-065) to MHZ, through
///   aws_clkgen_set_dynamic (which resets the design); waits for HBM, then
///   measures the clock: CYCLES against the host's clock over 0.2 s.
/// - `run NAME=VALUE...`: a run of the test engines (README.md's registers,
///   by lower-case name: mode, port_en, beats, ...); every port's counters.
/// - `ocl N`: N reads of SCRATCH and N writes, each timed (OCL latency).
/// - `pcis-write BYTES OFFSET SEED` and `pcis-read BYTES OFFSET SEED`: the
///   host writes or reads card memory through PCIS (BAR4, write-combining),
///   the pattern of README.md; throughput, and for reads the words unlike it.
///
/// Exit status 0, or 1 with {"error": ...} on stdout.

#include <fpga_clkgen.h>
#include <fpga_mgmt.h>
#include <fpga_pci.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <format>
#include <map>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kSlot = 0;
constexpr int kPorts = 32;
constexpr uint32_t kId = 0xCAB0'B001;
constexpr int kStatusBusy = 1, kStatusHbmReady = 2, kStatusRefused = 4, kStatusHostOwns = 8;
constexpr uint64_t kPortCounters = 0x100, kPortStride = 0x40;
constexpr auto kPollTimeout = std::chrono::seconds(60);

/// @brief README.md's registers, by lower-case name.
const std::map<std::string, uint64_t> kRegs = {
    {"id", 0x000},       {"build", 0x004},    {"scratch", 0x008},  {"status", 0x00C},
    {"cycles_lo", 0x010}, {"cycles_hi", 0x014}, {"ctrl", 0x020},     {"port_en", 0x024},
    {"mode", 0x028},     {"base_lo", 0x02C},  {"base_hi", 0x030},  {"stride", 0x034},
    {"beats", 0x038},    {"rd_burst", 0x03C}, {"wr_burst", 0x040}, {"rd_limit", 0x044},
    {"wr_limit", 0x048}, {"strb", 0x04C},     {"seed", 0x050},     {"host_port", 0x054},
};

/// @brief A port's counters, in their order (README.md).
constexpr std::array<const char*, 15> kCounters = {
    "rd_beats", "rd_cycles", "rd_first", "rd_stalls", "rd_maxgap", "rd_bad", "rd_occ_lo", "rd_occ_hi",
    "wr_beats", "wr_cycles", "wr_first", "wr_maxout", "wr_occ_lo", "wr_occ_hi", "flags"};

using Clock = std::chrono::steady_clock;

/// @brief Seconds between two instants.
double seconds(Clock::time_point from, Clock::time_point to) {
  return std::chrono::duration<double>(to - from).count();
}

/// @brief A failed SDK call.
void check(int rc, const char* what) {
  if (rc != 0) throw std::runtime_error(std::format("{} failed ({})", what, rc));
}

/// @brief The design's registers on OCL (BAR0).
struct Regs {
  pci_bar_handle_t bar = PCI_BAR_HANDLE_INIT;

  Regs() { check(fpga_pci_attach(kSlot, FPGA_APP_PF, APP_PF_BAR0, 0, &bar), "attach BAR0"); }
  ~Regs() { fpga_pci_detach(bar); }
  Regs(const Regs&) = delete;
  Regs& operator=(const Regs&) = delete;

  /// @brief Read a register at its offset.
  uint32_t peek(uint64_t offset) const {
    uint32_t v = 0;
    check(fpga_pci_peek(bar, offset, &v), "peek");
    return v;
  }
  /// @brief Write a register at its offset.
  void poke(uint64_t offset, uint32_t value) const { check(fpga_pci_poke(bar, offset, value), "poke"); }
  /// @brief Read a register by name.
  uint32_t get(const std::string& name) const { return peek(kRegs.at(name)); }
  /// @brief Write a register by name.
  void set(const std::string& name, uint32_t value) const { poke(kRegs.at(name), value); }

  /// @brief The free-running cycle counter (CYCLES_LO latches CYCLES_HI).
  uint64_t cycles() const {
    const uint64_t lo = get("cycles_lo");
    return uint64_t{get("cycles_hi")} << 32 | lo;
  }

  /// @brief Poll STATUS until every bit of `mask` reads `want`.
  /// @return STATUS then.
  /// @throws std::runtime_error after kPollTimeout.
  uint32_t wait_status(uint32_t mask, uint32_t want, const char* what) const {
    const auto start = Clock::now();
    for (;;) {
      const uint32_t s = get("status");
      if ((s & mask) == want) return s;
      if (Clock::now() - start > kPollTimeout) throw std::runtime_error(std::format("timeout: {}", what));
    }
  }
};

/// @brief The pattern's word at byte address addr (README.md).
uint32_t pattern(uint64_t addr, uint32_t seed) { return static_cast<uint32_t>(addr >> 2) ^ seed; }

/// @brief A JSON object, built field by field, printed on one line.
struct Json {
  std::string body;

  /// @brief Add a field whose value is already JSON.
  Json& raw(const std::string& key, const std::string& value) {
    body += std::format("{}\"{}\": {}", body.empty() ? "" : ", ", key, value);
    return *this;
  }
  /// @brief Add a number.
  template <typename T>
  Json& num(const std::string& key, T value) { return raw(key, std::format("{}", value)); }
  /// @brief Add a string.
  Json& str(const std::string& key, const std::string& value) { return raw(key, std::format("\"{}\"", value)); }
  /// @brief The object's text.
  std::string text() const { return "{" + body + "}"; }
};

/// @brief The clock generator's clocks, in MHz, as JSON.
std::string clocks_json() {
  fpga_clkgen_info info{};
  check(aws_clkgen_get_dynamic(kSlot, &info), "aws_clkgen_get_dynamic");
  const auto group = [](const fpga_clkgen_group& g) {
    return std::format("[{}, {}, {}]", g.clocks[0], g.clocks[1], g.clocks[2]);
  };
  return Json{}
      .raw("a", group(info.clock_group_a))
      .raw("b", group(info.clock_group_b))
      .raw("c", group(info.clock_group_c))
      .raw("hbm", group(info.clock_group_hbm))
      .text();
}

/// @brief `info`.
Json info(const Regs& regs) {
  const uint32_t id = regs.get("id");
  if (id != kId) throw std::runtime_error(std::format("ID {:#x}, not the bring-up design's", id));
  return Json{}
      .str("id", std::format("{:#010x}", id))
      .str("build", std::format("{:08x}", regs.get("build")))
      .num("status", regs.get("status"))
      .raw("clocks_mhz", clocks_json());
}

/// @brief The core's clock measured: CYCLES over 0.2 s of the host's clock.
double measure_mhz(const Regs& regs) {
  const auto t0 = Clock::now();
  const uint64_t c0 = regs.cycles();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  const uint64_t c1 = regs.cycles();
  return static_cast<double>(c1 - c0) / seconds(t0, Clock::now()) / 1e6;
}

/// @brief `clock MHZ`. Groups A to C keep the frequencies AWS's defaults
/// give their first clocks (unused by the design).
Json set_clock(uint32_t mhz) {
  check(aws_clkgen_set_dynamic(kSlot, 125, 450, 300, mhz, 0), "aws_clkgen_set_dynamic");
  const Regs regs;
  regs.wait_status(kStatusHbmReady, kStatusHbmReady, "HBM ready after the clock change");
  return Json{}.num("set_mhz", mhz).num("measured_mhz", measure_mhz(regs)).raw("clocks_mhz", clocks_json());
}

/// @brief `run NAME=VALUE...`.
Json run(const Regs& regs, std::span<char*> args) {
  regs.wait_status(kStatusHbmReady | kStatusBusy, kStatusHbmReady, "HBM ready and idle before a run");
  Json settings;
  uint32_t port_en = 0;
  for (const std::string arg : args) {
    const auto eq = arg.find('=');
    if (eq == std::string::npos || !kRegs.contains(arg.substr(0, eq))) {
      throw std::runtime_error(std::format("not NAME=VALUE with a register's name: {}", arg));
    }
    const uint32_t value = static_cast<uint32_t>(std::stoul(arg.substr(eq + 1), nullptr, 0));
    regs.set(arg.substr(0, eq), value);
    settings.num(arg.substr(0, eq), value);
    if (arg.substr(0, eq) == "port_en") port_en = value;
  }
  const auto t0 = Clock::now();
  regs.set("ctrl", 1);
  const uint32_t status = regs.wait_status(kStatusBusy, 0, "the run's end");
  const double elapsed = seconds(t0, Clock::now());
  if (status & kStatusRefused) throw std::runtime_error("start refused: the settings break README.md's rules");
  std::string ports = "[";
  for (int p = 0; p < kPorts; ++p) {
    if (!(port_en >> p & 1)) continue;
    Json c;
    c.num("port", p);
    for (size_t i = 0; i < kCounters.size(); ++i) {
      c.num(kCounters[i], regs.peek(kPortCounters + kPortStride * p + 4 * i));
    }
    ports += (ports.size() > 1 ? ", " : "") + c.text();
  }
  return Json{}.raw("settings", settings.text()).num("elapsed_s", elapsed).raw("ports", ports + "]");
}

/// @brief The latency percentiles of a set of timings, in ns.
std::string percentiles(std::vector<double> ns) {
  std::ranges::sort(ns);
  const auto at = [&](double q) { return ns[static_cast<size_t>(q * static_cast<double>(ns.size() - 1))]; };
  return Json{}.num("min", ns.front()).num("p50", at(0.5)).num("p99", at(0.99)).num("max", ns.back()).text();
}

/// @brief `ocl N`.
Json ocl(const Regs& regs, size_t n) {
  std::vector<double> reads, writes;
  for (size_t i = 0; i < n; ++i) {
    const auto t0 = Clock::now();
    regs.poke(kRegs.at("scratch"), static_cast<uint32_t>(i));
    const auto t1 = Clock::now();
    if (regs.get("scratch") != static_cast<uint32_t>(i)) throw std::runtime_error("SCRATCH read back wrong");
    const auto t2 = Clock::now();
    writes.push_back(seconds(t0, t1) * 1e9);
    reads.push_back(seconds(t1, t2) * 1e9);
  }
  return Json{}.num("n", n).raw("read_ns", percentiles(reads)).raw("write_ns", percentiles(writes));
}

/// @brief Card memory through PCIS (BAR4), write-combining.
struct Card {
  pci_bar_handle_t bar = PCI_BAR_HANDLE_INIT;

  Card() { check(fpga_pci_attach(kSlot, FPGA_APP_PF, APP_PF_BAR4, BURST_CAPABLE, &bar), "attach BAR4"); }
  ~Card() { fpga_pci_detach(bar); }
  Card(const Card&) = delete;
  Card& operator=(const Card&) = delete;

  /// @brief The mapped words from byte offset `offset`, `bytes` long.
  std::span<volatile uint32_t> words(uint64_t offset, uint64_t bytes) const {
    void* p = nullptr;
    check(fpga_pci_get_address(bar, offset, bytes / 4, &p), "fpga_pci_get_address");
    return {static_cast<volatile uint32_t*>(p), bytes / 4};
  }
};

/// @brief Make sure the host owns HBM port 0 and HBM is ready (README.md).
void own_port_0(const Regs& regs) {
  regs.set("host_port", 1);
  regs.wait_status(kStatusHostOwns | kStatusHbmReady, kStatusHostOwns | kStatusHbmReady,
                   "the host owning HBM port 0, HBM ready");
}

/// @brief `pcis-write BYTES OFFSET SEED`: the pattern from a host buffer,
/// 64-byte copies (write-combining makes whole beats of them).
Json pcis_write(const Regs& regs, uint64_t bytes, uint64_t offset, uint32_t seed) {
  own_port_0(regs);
  std::vector<uint32_t> host(bytes / 4);
  for (size_t i = 0; i < host.size(); ++i) host[i] = pattern(offset + 4 * i, seed);
  const Card card;
  const auto dst = card.words(offset, bytes);
  const auto t0 = Clock::now();
  std::memcpy(const_cast<uint32_t*>(dst.data()), host.data(), bytes);  // NOLINT: device memory
  __builtin_ia32_sfence();  // the write-combining buffers out
  const double s = seconds(t0, Clock::now());
  // A read flushes the posted writes before the next step.
  const uint32_t last = dst[dst.size() - 1];
  return Json{}.num("bytes", bytes).num("offset", offset).num("seconds", s)
      .num("gb_per_s", static_cast<double>(bytes) / s / 1e9).num("last_ok", last == host.back() ? 1 : 0);
}

/// @brief `pcis-read BYTES OFFSET SEED`: card memory into a host buffer,
/// then the words unlike the pattern counted.
Json pcis_read(const Regs& regs, uint64_t bytes, uint64_t offset, uint32_t seed) {
  own_port_0(regs);
  std::vector<uint32_t> host(bytes / 4);
  const Card card;
  const auto src = card.words(offset, bytes);
  const auto t0 = Clock::now();
  std::memcpy(host.data(), const_cast<const uint32_t*>(src.data()), bytes);  // NOLINT: device memory
  const double s = seconds(t0, Clock::now());
  uint64_t bad = 0;
  for (size_t i = 0; i < host.size(); ++i) bad += host[i] != pattern(offset + 4 * i, seed);
  return Json{}.num("bytes", bytes).num("offset", offset).num("seconds", s)
      .num("gb_per_s", static_cast<double>(bytes) / s / 1e9).num("bad_words", bad);
}

/// @brief Argument i of the command line as a number (decimal or 0x hex).
uint64_t number_arg(std::span<char*> args, size_t i) {
  if (i >= args.size()) throw std::runtime_error(std::format("{} needs {} arguments", args[0], i));
  return std::stoull(args[i], nullptr, 0);
}

/// @brief The command line's command.
Json dispatch(std::span<char*> args) {
  if (args.empty()) throw std::runtime_error("no command (see bringup.cpp's header)");
  const std::string cmd = args[0];
  const auto u64 = [&](size_t i) { return number_arg(args, i); };
  if (cmd == "clock") return set_clock(static_cast<uint32_t>(u64(1)));
  const Regs regs;
  if (cmd == "info") return info(regs);
  if (cmd == "run") return run(regs, args.subspan(1));
  if (cmd == "ocl") return ocl(regs, u64(1));
  if (cmd == "pcis-write") return pcis_write(regs, u64(1), u64(2), static_cast<uint32_t>(u64(3)));
  if (cmd == "pcis-read") return pcis_read(regs, u64(1), u64(2), static_cast<uint32_t>(u64(3)));
  throw std::runtime_error(std::format("unknown command {}", cmd));
}

}  // namespace

/// @brief Run one command (this file's header).
int main(int argc, char** argv) {
  try {
    check(fpga_mgmt_init(), "fpga_mgmt_init");
    const Json result = dispatch(std::span(argv + 1, static_cast<size_t>(argc - 1)));
    std::println("{}", result.text());
    return 0;
  } catch (const std::exception& e) {
    std::println("{}", Json{}.str("error", e.what()).text());
    return 1;
  }
}
