// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cabosse Authors
//
// Bulk driver for rtl/fp32_fma.sv (verif/bulk.py). Record: op, a, b, c.

#include "Vfp32_fma.h"
#include "stream.h"

int main(int argc, char** argv) {
  return stream<Vfp32_fma>(
      argc, argv, 4,
      [](Vfp32_fma& dut, const uint32_t* r) {
        dut.op_i = r[0];
        dut.a_i = r[1];
        dut.b_i = r[2];
        dut.c_i = r[3];
      },
      [](Vfp32_fma& dut) { return dut.y_o; });
}
