// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Types of the lanes' multiply-add (D-038): bf16_mul's product, which
// mac_add adds to the FP32 accumulator.

package bf16_mac_pkg;

  // A BF16 x BF16 product: (-1)^sign * sig * 2^(exp - 153), so exp is biased
  // like FP32's (127) and sig has 3 bits below FP32's 24: guard, round,
  // sticky. exp >= 1, and sig[26] = 1 whenever exp > 1, as for an FP32
  // value; the exponent may exceed 254. The product is exact down to
  // exponent 1; below it, sig is shifted right to exponent 1 and the bits
  // shifted out collect into the sticky bit (sig[0]). A zero has sig = 0.
  typedef struct packed {
    logic               sign;
    logic signed [10:0] exp;
    logic [26:0]        sig;
    logic               is_inf;
    logic               is_nan;
  } product_t;

endpackage
