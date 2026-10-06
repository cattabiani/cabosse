# Cabosse numerics spec (v0, draft)

This is the contract for what the accelerator computes, bit for bit. The
golden model (`model/`) and the RTL (`rtl/`) both implement it. Tests compare
them bit-exactly. When they disagree, this document decides which one is
wrong. Change this document first, then both implementations (AGENTS.md
rule 8).

Status: draft for M1. Values marked **[param]** are fixed by measurement in M1
or by the architecture in M2 (section 7).

## 1. Formats

| Name | Bits | Layout (sign/exp/mantissa) | Used for |
|------|------|----------------------------|----------|
| BF16 | 16 | 1/8/7 | weights, multiplier inputs, KV cache |
| FP32 | 32 | 1/8/23 (IEEE 754 binary32) | accumulation, vector unit, residual stream, logits |

## 2. Primitive operations

Every result below is defined as the exact mathematical result, then rounded
once. Notation: `f32(·)` rounds to FP32, and `bf16(·)` rounds to BF16. Both
round to nearest, ties to even.

| Op | Definition |
|----|------------|
| `bf16(x)` | FP32 → BF16, round to nearest even |
| `up(b)` | BF16 → FP32, exact (append 16 zero bits) |
| `add(a, b)` | `f32(a + b)` |
| `mul(a, b)` | `f32(a · b)` |
| `fma(a, b, c)` | `f32(a · b + c)`: one rounding (D-019) |
| `mac(w, x, acc)` | `fma(up(w), up(x), acc)` for BF16 `w`, `x` and FP32 `acc` |
| `max(a, b)` | the larger of a and b, with `max(-0, +0) = +0`; NaN if a or b is NaN |
| `int bit tricks` | integer add/subtract/shift on the 32-bit pattern of an FP32 value (section 5) |

Special values (D-012, D-016):
- **Subnormals** are supported on input and output (IEEE 754, as in
  PyTorch). The golden model has a flush-to-zero switch for experiments only.
- **Infinities** propagate per IEEE 754.
- **NaN:** any NaN result is the canonical NaN: BF16 `0x7FC0`, FP32
  `0x7FC00000`. When comparing against torch, NaNs match by class, not by
  bits, because torch's CPU conversion produces `0xFFFF`.
- **Zeros:** signed zeros follow IEEE 754 (`x + (-x) = +0` in round to
  nearest).

## 3. Dot product (the lanes)

A dot product `y = Σₖ w[k]·x[k]` over `k = 0…K-1` with BF16 `w`, `x`
is computed with `A` = 16 interleaved accumulators (D-027):

```
acc[j] = +0.0                                  for j = 0…A-1
for k = 0 … K-1:                               # in increasing k
    j = k mod A
    acc[j] = mac(w[k], x[k], acc[j])
y = tree_sum(acc[0 … A-1])
```

`tree_sum` is a fixed balanced binary tree with `A` a power of two, pairing
neighbours level by level:
`tree_sum(a0,a1,a2,a3) = add(add(a0,a1), add(a2,a3))`.

`K ≥ 1`: an empty dot product is not defined. In a partial last group, the
accumulators of missing elements are not updated: they keep their value (in
hardware, a write enable). Feeding zeros instead is not equivalent: a tiny
negative product underflows to -0 (`mac(-2^-80, 2^-80, +0) = -0`), and then
`mac(0, 0, -0) = +0` changes the sign bit.

Splitting one dot product across several lanes (split-K) is not allowed in
v0. If M2 needs it, this section changes.

## 4. Vector unit operations

All FP32 unless noted. `n ≥ 1` is the vector length.

- **Reductions.** `sum(x)` and `sum_squares(x)` use the loop of section 3 with
  `S` = 128 interleaved partial sums (D-030) in place of `A` (the vector
  unit's adder has a latency too), and a step in place of
  `mac(w[k], x[k], acc)`: `add(xᵢ, acc)` for `sum`, `fma(xᵢ, xᵢ, acc)` for
  `sum_squares`. Unlike section 3, padding the last group with +0 gives the
  same bits: a partial sum that starts at +0 can never become -0 (an exact zero
  sum is +0, an add never underflows to zero because subnormals are kept, and
  `xᵢ·xᵢ ≥ 0`). A sum of only -0 values is +0. `max(x)` is the `max` of
  section 2 over all elements. `max` is exact, commutative and associative, so
  its order does not matter.
- **Embedding lookup:** `h = up(E[token])`. E is the BF16 table (tied with the
  classifier).
- **RMSNorm** (weight `g`, BF16; `eps` = `f32(rms_norm_eps)` from the model
  config, 1e-5 for SmolLM2):
  ```
  ss  = sum_squares(x)
  var = mul(ss, inv_n)                         # inv_n = f32(1/n), constant
  r   = rsqrt(add(var, eps))                   # section 5
  yᵢ  = mul(up(gᵢ), mul(xᵢ, r))
  ```
  This uses a multiply by `f32(1/n)` instead of torch's divide by `n`, so it
  is not bit-identical to torch here (tolerance comparison only).
- **Softmax** of FP32 scores `s`:
  ```
  m  = max(s)
  eᵢ = exp(add(sᵢ, -m))                        # unary minus is an exact sign flip
  z  = sum(e)
  pᵢ = mul(eᵢ, recip(z))
  ```
  For finite scores the largest gives `exp(+0) = 1` exactly, so `z ≥ 1` and
  `recip(z)` needs no scaling. A score of -Inf gets `p = 0`. A score of +Inf
  or NaN, or all scores -Inf, gives NaN everywhere (IEEE propagation). Decode
  has no masked scores, so only finite scores occur there.
  `e` need not be stored: recomputing `exp(add(sᵢ, -m))` gives the same bits
  (D-022).
  `p` is not exactly normalized: even for `n = 1`, `p₀ = recip(1) = 0.9999935`.
- **RoPE** (head dim `d` even, `half = d/2`, tables `C[pos]`, `S[pos]` of
  length `d`, FP32): the tables are inputs, computed on the host by
  `transformers`' own rotary embedding (D-020), in FP32 before its cast to
  the model dtype: `inv_freq = 1/θ^(2i/d)`, `emb = [pos·inv_freq,
  pos·inv_freq]`, `C = cos(emb)`, `S = sin(emb)`. They come from the host's
  float32 math library, which is not correctly rounded (about 5% of
  SmolLM2's values are 1 ulp off, measured on x86 Linux), so they may differ
  between platforms. The accelerator and the golden model use whatever
  tables the host gives them. Pinned values (table hashes, fixtures) come
  from the reference platform, x86-64 Linux (D-023).
  ```
  rᵢ = -x[i+half]  for i < half;   rᵢ = x[i-half]  for i ≥ half
  outᵢ = fma(xᵢ, Cᵢ, mul(rᵢ, Sᵢ))
  ```
  This is `transformers`' `x·cos + rotate_half(x)·sin` with one rounding
  fewer. At position 0 (`C = 1`, `S = 0`) the output equals `x`, except that
  -0 can become +0 and an infinite partner gives NaN (`Inf·0`).
- **Attention**, per query head (GQA: query head `h` uses KV head
  `h // (n_heads / n_kv_heads)`):
  ```
  q̂ = bf16(RoPE(q))      K[t] = bf16(RoPE(k_t))      V[t] = bf16(v_t)
  sₜ = mul(dot(q̂, K[t]), scale)                # scale = f32(d^-0.5): 0.125 for d = 64
  p  = bf16(softmax(s))                        # over the positions t
  oᵢ = dot over t of (pₜ, V[t][i])             # section 3, positions as k
  ```
  The KV cache stores `K[t]`, `V[t]` in BF16, written once per token.
  `scale` is a host constant, computed as `transformers` does: the float64
  value `d**-0.5`, rounded to FP32. That is two roundings, but it equals the
  correctly rounded `1/√d` for every `d` up to 100 000 (checked; the test
  covers `d ≤ 4096`).
- **SwiGLU:** `a = gate·x`, `b = up·x` (dot products), then
  `sᵢ = mul(aᵢ, recip(add(1.0, exp(-aᵢ))))` (SiLU), `hᵢ = mul(sᵢ, bᵢ)`.
  For `aᵢ < -88.7`, `exp(-aᵢ)` overflows and `sᵢ = -0`, as in torch. For
  large positive `aᵢ`, `sᵢ = aᵢ·recip(1) = aᵢ·0.9999935`. `silu(-Inf)` is NaN
  (`-Inf·0`), as in torch.
- **Residual add:** `x = add(x, delta)`.
- **Matvec input rounding** (D-011): every vector entering a matrix-vector
  product is `bf16(·)` first. That applies to the RMSNorm outputs, the
  attention output `o`, and the SwiGLU output `h`.
- **Logits:** `dot(E[v], bf16(final_norm(x)))` for every vocabulary entry `v`,
  in FP32. Sampling or argmax is not part of this spec (Q-21).
- **Decode step**, one token at a time at position `pos = 0, 1, …` (a prompt
  is fed the same way, Q-20). Weights are the checkpoint's BF16 tensors;
  `dot(W, x)` is one dot product per row of `W` (section 3):
  ```
  h = up(E[token])
  for each layer:
      x = bf16(rmsnorm(h, g_attn))
      q, k, v = dot(Wq, x), dot(Wk, x), dot(Wv, x)          # FP32, split into heads
      K[pos], V[pos] = bf16(RoPE(k)), bf16(v)                # per KV head, before attention
      o = attention over t = 0 … pos, per query head          # as above
      h = add(h, dot(Wo, bf16(o)))
      x = bf16(rmsnorm(h, g_mlp))
      h = add(h, dot(Wdown, bf16(swiglu(dot(Wgate, x), dot(Wup, x)))))
  logits = dot(E, bf16(rmsnorm(h, g_final)))                # E: tied embedding, or lm_head
  ```
  This is the order of `transformers`' Llama decoder layer (pre-norm,
  residual adds). Biases are not supported in v0 (SmolLM2 has none).

## 5. Function approximations (D-020)

`bits(x)` reinterprets an FP32 value as a 32-bit integer, and `float(i)` does
the reverse. All arithmetic uses the primitives from section 2.

- **rsqrt(x) ≈ 1/√x** (`R_RSQRT = 0x5F3759DF`, `N_RSQRT = 2`):
  ```
  s = (x < 2^-125)                             # subnormal x, and the lowest binade
  if s: x = mul(x, 2^24)                       # exact; keeps h = x/2 normal (also under FTZ)
  y = float(R_RSQRT - (bits(x) >> 1))          # first guess, max error 2^-4.9
  h = mul(0.5, x)
  repeat N_RSQRT times:                        # each step roughly doubles the bits
      y = mul(y, fma(-mul(h, y), y, 1.5))      # unary minus is an exact sign flip
  if s: y = mul(y, 2^12)
  ```
- **recip(x) ≈ 1/x** (`R_RECIP = 0x7EF311C3`, `N_RECIP = 2`). Works on `|x|`,
  and the sign is restored at the end:
  ```
  a = |x|;  k = 2^24 if a < 2^-126,  2^-24 if a ≥ 2^125,  else 1
  if k ≠ 1: a = mul(a, k)                      # bit trick only sees [2^-126, 2^125)
  y = float(R_RECIP - bits(a))                 # first guess, max error 2^-4.3
  repeat N_RECIP times:
      y = fma(y, fma(-a, y, 1.0), y)
  if k ≠ 1: y = mul(y, k)                      # may overflow to inf or round to subnormal
  result = y with the sign of x
  ```
- **exp(x) = 2^(x·log₂e)** (`LOG2E = 0x3FB8AA3B`, degree 4):
  ```
  t = mul(x, LOG2E);  t = clamp(t, -150, 150)  # saturate; NaN handled separately
  i = round_to_nearest_even(t)                 # as an integer
  f = add(t, -i)                               # f in [-0.5, 0.5], exact (Sterbenz)
  p = (((c4·f + c3)·f + c2)·f + c1)·f + c0     # Horner, each step one fma
  E = exponent_field(p) + i
  if E ≥ 255: result = +Inf
  elif E ≥ 1: result = float(bits(p) + (i << 23))           # exact, normal
  else:       result = mul(float(bits(p) + ((i + 64) << 23)), 2^-64)
              # p·2^(i+64) is exact and normal; the multiply rounds p·2^i once
              # into the subnormal range (or to +0)
  ```
  Coefficients of 2ᶠ on [-0.5, 0.5] (Chebyshev interpolation, rounded to
  FP32 and pinned): `c0..c4 = 0x3F800000, 0x3F317061, 0x3E75FD26, 0x3D650E71,
  0x3C1E5FB0`. The clamp at ±150 is exact: |t| > 150 means a result above
  the FP32 range (+Inf) or below 2⁻¹⁵⁰ (rounds to +0). With E ≥ -24 after
  the clamp, `i + 64` always gives a normal intermediate.
- **Special inputs** (checked before the bit tricks): `rsqrt(±0) = ±Inf`,
  `rsqrt(+Inf) = +0`, `rsqrt(x < 0) = NaN`, `recip(±0) = ±Inf`,
  `recip(±Inf) = ±0`, `exp(-Inf) = +0`, `exp(+Inf) = +Inf`, NaN in → NaN out.

**Accuracy** (measured, `model/tests/test_funcs.py`): max relative error
2⁻¹⁷·⁷ for rsqrt and 2⁻¹⁷·² for recip (exhaustive over one period of the
bit-trick error), and 2⁻¹⁷·⁰⁸ for exp (exhaustive over every x with a normal
result). For subnormal exp results (x < -87.3), |t| ≥ 128 is rounded to a
coarser grid and the error is 2⁻¹⁶·⁶³ (exhaustive), plus half a subnormal ulp
from the final rounding. exp is limited by the
rounding of `x·log₂e`, so a higher degree does not help. All three are about
256× below the BF16 rounding (2⁻⁹) that follows every use. Note that the
approximations are not exact even at simple points: `recip(1) = 0.9999935`.
The end-to-end check against `transformers` (M1) confirms these choices.

## 6. What is not bit-identical to PyTorch, by design

Summation order (sections 3 and 4), FMA instead of a separate multiply and
add, the approximated rsqrt/recip/exp, `1/n` as a multiply, and BF16
rounding at the D-011 points. These are compared against `transformers`
with tolerances (M1 exit criteria), never bit-exactly.

## 7. Parameters

| Param | Meaning | Fixed by |
|-------|---------|----------|
| `A` | interleaved accumulators per lane | M2: 16 (D-027; 4 multiply-adds per lane × a 4-cycle accumulate loop) |
| `S` | vector-unit reduction width | M2: 128 (D-030; up to 32 elements per cycle × a 4-cycle add loop) |
| `R_RSQRT`, `N_RSQRT` | rsqrt first-guess constant, Newton steps | M1: `0x5F3759DF`, 2 |
| `R_RECIP`, `N_RECIP` | recip first-guess constant, Newton steps | M1: `0x7EF311C3`, 2 |
| exp degree, coefficients | 2ᶠ polynomial | M1: degree 4, section 5 |
