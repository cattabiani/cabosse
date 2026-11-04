// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief card.h on F2, with AWS's SDK (libfpga_mgmt's fpga_pci): OCL is
/// BAR0, card memory is PCIS on BAR4 (write-combining), mapped a range at a
/// time. Whole 64-byte lines are written as two 32-byte non-temporal stores
/// each (AVX2: the F2 host has no AVX-512, M8); a range's partial first and
/// last lines byte by byte, which PCIS carries as byte strobes (M8). After
/// the stores a fence and a read of the last line's word: the posted writes
/// before it are done. Reads go through a line-aligned copy.

#include <immintrin.h>

#include <fpga_mgmt.h>
#include <fpga_pci.h>

#include <algorithm>
#include <cstring>
#include <format>
#include <stdexcept>
#include <vector>

#include "card.h"

namespace {

constexpr int kSlot = 0;
constexpr uint64_t kLine = 64;  ///< a PCIS beat, and a write-combining line

/// @brief Throw on a failed SDK call.
/// @param rc the call's return code.
/// @param what the call, for the message.
void check(int rc, const char* what) {
  if (rc != 0) throw std::runtime_error(std::format("{} failed ({})", what, rc));
}

/// @brief Whole lines from `in` to write-combining `out`, two 32-byte
/// non-temporal stores a line.
/// @param out the mapped card memory, line-aligned.
/// @param in the bytes, as many as `n`.
/// @param n bytes, whole lines.
__attribute__((target("avx2"))) void store_lines(std::byte* out, const std::byte* in, uint64_t n) {
  for (uint64_t at = 0; at < n; at += kLine) {
    const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + at));
    const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + at + kLine / 2));
    _mm256_stream_si256(reinterpret_cast<__m256i*>(out + at), lo);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(out + at + kLine / 2), hi);
  }
}

/// @brief The F2 card: BAR0 and BAR4 attached.
class F2Card final : public card::Card {
 public:
  F2Card() {
    check(fpga_mgmt_init(), "fpga_mgmt_init");
    check(fpga_pci_attach(kSlot, FPGA_APP_PF, APP_PF_BAR0, 0, &ocl_), "attach BAR0");
    check(fpga_pci_attach(kSlot, FPGA_APP_PF, APP_PF_BAR4, BURST_CAPABLE, &pcis_), "attach BAR4");
    if (!__builtin_cpu_supports("avx2")) throw std::runtime_error("PCIS writes need AVX2");
  }
  ~F2Card() override {
    fpga_pci_detach(pcis_);
    fpga_pci_detach(ocl_);
  }
  F2Card(const F2Card&) = delete;
  F2Card& operator=(const F2Card&) = delete;

  uint32_t peek(uint64_t offset) override {
    uint32_t v = 0;
    check(fpga_pci_peek(ocl_, offset, &v), "peek");
    return v;
  }

  void poke(uint64_t offset, uint32_t value) override { check(fpga_pci_poke(ocl_, offset, value), "poke"); }

  void write(uint64_t addr, std::span<const std::byte> data) override {
    if (data.empty()) return;
    const uint64_t first = addr / kLine * kLine;
    const uint64_t end = (addr + data.size() + kLine - 1) / kLine * kLine;
    std::byte* const card = map(first, end - first);
    // Partial lines at the ends, byte by byte (each a strobed write).
    const uint64_t head = std::min<uint64_t>(data.size(), (kLine - addr % kLine) % kLine);
    const uint64_t body = (data.size() - head) / kLine * kLine;
    auto* const out = reinterpret_cast<volatile std::byte*>(card + (addr - first));
    for (uint64_t i = 0; i < head; ++i) out[i] = data[i];
    store_lines(card + (addr - first) + head, data.data() + head, body);
    for (uint64_t i = head + body; i < data.size(); ++i) out[i] = data[i];
    _mm_sfence();
    // A read: the posted writes before it are done.
    (void)*reinterpret_cast<volatile uint32_t*>(card + (end - first) - 4);
  }

  void read(uint64_t addr, std::span<std::byte> data) override {
    if (data.empty()) return;
    const uint64_t first = addr / kLine * kLine;
    const uint64_t end = (addr + data.size() + kLine - 1) / kLine * kLine;
    const std::byte* const card = map(first, end - first);
    std::vector<std::byte> lines(end - first);
    std::memcpy(lines.data(), card, lines.size());  // device memory: whole lines
    std::ranges::copy(std::span(lines).subspan(addr - first, data.size()), data.begin());
  }

 private:
  pci_bar_handle_t ocl_ = PCI_BAR_HANDLE_INIT;
  pci_bar_handle_t pcis_ = PCI_BAR_HANDLE_INIT;

  /// @brief Card memory [offset, offset + n) mapped.
  std::byte* map(uint64_t offset, uint64_t n) {
    void* p = nullptr;
    check(fpga_pci_get_address(pcis_, offset, n, &p), "fpga_pci_get_address");
    return static_cast<std::byte*>(p);
  }
};

}  // namespace

std::unique_ptr<card::Card> card::open() { return std::make_unique<F2Card>(); }
