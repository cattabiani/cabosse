// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors

/// @file
/// @brief Bulk driver for rtl/fp32_chain.sv (verif/bulk.py). Record: op, a, b, c.

#include "Vfp32_chain.h"
#include "stream.h"

int main(int argc, char** argv) {
  auto set_inputs = [](Vfp32_chain& dut, Record r) {
    dut.op_i = r[0];
    dut.a_i = r[1];
    dut.b_i = r[2];
    dut.c_i = r[3];
  };
  auto output = [](Vfp32_chain& dut) { return dut.y_o; };
  return stream<Vfp32_chain>(argc, argv, 4, set_inputs, output);
}
