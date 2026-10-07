// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief Bulk driver for rtl/bf16_mac.sv (verif/bulk.py). Record: w, x, acc
/// (w and x in the low 16 bits).

#include "Vbf16_mac.h"
#include "stream.h"

int main(int argc, char** argv) {
  auto set_inputs = [](Vbf16_mac& dut, Record r) {
    dut.w_i = r[0];
    dut.x_i = r[1];
    dut.acc_i = r[2];
  };
  auto output = [](Vbf16_mac& dut) { return dut.y_o; };
  return stream<Vbf16_mac>(argc, argv, 3, set_inputs, output);
}
