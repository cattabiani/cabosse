// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
// Copyright 2026 The Cabosse Authors
//
// Types and helpers of the lanes' multiply-add (D-038), shared by bf16_mul
// and mac_add.

package bf16_mac_pkg;

  localparam int unsigned W = 27;  // significand bits: FP32's 24, guard, round, sticky

  // An operand of mac_add: (-1)^sign * sig * 2^(exp - 153), exp biased like
  // FP32's (127). exp >= 1, and sig[W-1] = 1 whenever exp > 1 (a subnormal
  // or zero has exp 1, as in FP32), so (exp, sig) orders magnitudes. A
  // product's exp may exceed 254 (up to 382). bf16_mul's product is exact
  // down to exponent 1; below it, sig is shifted to exponent 1 and the bits
  // shifted out collect into the sticky bit (sig[0]).
  typedef struct packed {
    logic         sign;
    logic [8:0]   exp;
    logic [W-1:0] sig;
    logic         is_inf;
    logic         is_nan;
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

  // sig >> n, with the bits shifted out ORed into bit 0 (sticky).
  function automatic logic [W-1:0] shift_right_sticky(logic [W-1:0] sig, logic [8:0] n);
    logic [W-1:0] out_mask;  // the bits shifted out
    if (n >= 9'(W)) return W'(|sig);
    out_mask = ~({W{1'b1}} << n[4:0]);
    return (sig >> n[4:0]) | W'(|(sig & out_mask));
  endfunction

endpackage
