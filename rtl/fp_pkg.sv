// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Operands of our FP units (D-038, D-039): fp_add adds two of them with one
// rounding to FP32; fp_product makes them from products.

package fp_pkg;

  localparam int unsigned SigW = 27;  // FP32's 24 significand bits, guard, round, sticky

  // An operand: (-1)^sign * sig * 2^(exp - 126 - W) for a significand of W
  // bits (sig[W-1] has weight 2^(exp - 127)); exp is biased like FP32's.
  // exp >= 1, and sig[W-1] = 1 whenever exp > 1 (a subnormal or zero has exp
  // 1, as in FP32), so (exp, sig) orders magnitudes. A product's exp may
  // exceed 254 (up to 382); one below exponent 1 is shifted to it, the bits
  // shifted out collected into the sticky bit, sig[0]. fp_add takes the same
  // layout for any W, as a vector of W + 12 bits; this is the one for W = SigW.
  typedef struct packed {
    logic            sign;
    logic [8:0]      exp;
    logic [SigW-1:0] sig;
    logic            is_inf;
    logic            is_nan;
  } operand_t;

  // An FP32 value as an operand (a BF16 value b is decode_f32({b, 16'h0})).
  function automatic operand_t decode_f32(logic [31:0] f);
    operand_t o;
    o.sign   = f[31];
    o.exp    = (f[30:23] == 0) ? 9'd1 : {1'b0, f[30:23]};
    o.sig    = {f[30:23] != 0, f[22:0], 3'b000};
    o.is_inf = (f[30:23] == 8'hFF) && (f[22:0] == 0);
    o.is_nan = (f[30:23] == 8'hFF) && (f[22:0] != 0);
    return o;
  endfunction

endpackage
