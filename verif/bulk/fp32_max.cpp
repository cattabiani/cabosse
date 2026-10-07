// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief Bulk driver for rtl/fp32_max.sv (verif/bulk.py). Record: a, b.

#include "Vfp32_max.h"
#include "stream.h"

int main(int argc, char** argv) {
  auto set_inputs = [](Vfp32_max& dut, Record r) {
    dut.a_i = r[0];
    dut.b_i = r[1];
  };
  auto output = [](Vfp32_max& dut) { return dut.y_o; };
  return stream<Vfp32_max>(argc, argv, 2, set_inputs, output);
}
