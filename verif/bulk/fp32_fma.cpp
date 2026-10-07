// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief Bulk driver for rtl/fp32_fma.sv (verif/bulk.py). Record: op, a, b, c.

#include "Vfp32_fma.h"
#include "stream.h"

int main(int argc, char** argv) {
  auto set_inputs = [](Vfp32_fma& dut, Record r) {
    dut.op_i = r[0];
    dut.a_i = r[1];
    dut.b_i = r[2];
    dut.c_i = r[3];
  };
  auto output = [](Vfp32_fma& dut) { return dut.y_o; };
  return stream<Vfp32_fma>(argc, argv, 4, set_inputs, output);
}
