# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""FP32 special values as bit patterns, shared by the golden-model tests
(test_arith) and the RTL tests (verif/tests): zeros, the subnormal and normal
boundaries, one and its neighbour, the largest finite value, infinities, the
canonical NaN and a signalling NaN; both signs where it applies."""

from golden import arith

F32_SPECIAL_BITS = [
    sign | bits
    for bits in (
        0x0000_0000,  # zero
        0x0000_0001,  # smallest subnormal
        0x007F_FFFF,  # largest subnormal
        0x0080_0000,  # smallest normal
        0x3F80_0000,  # 1
        0x3F80_0001,  # 1 + 2^-23
        0x7F7F_FFFF,  # largest finite
        0x7F80_0000,  # infinity
    )
    for sign in (0, 0x8000_0000)
] + [arith.NAN_F32_BITS, 0x7F80_0001]  # canonical NaN, a signalling NaN
