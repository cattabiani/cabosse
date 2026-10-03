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
| `max(a, b)` | the larger of a and b, with `max(-0, +0) = +0` |
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
is computed with `A` interleaved accumulators **[param, from M2]**:

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

An accumulator can never become -0: it starts at +0, `+0 + -0 = +0`, and an
exact cancellation also gives +0 (round to nearest). So the hardware may pad a
partial last group with zeros: `mac(0, 0, acc) = acc` bit for bit.

Splitting one dot product across several lanes (split-K) is not allowed in
v0. If M2 needs it, this section changes.

## 4. Vector unit operations

All FP32 unless noted. `n` is the vector length. `S` is the vector-unit
reduction width **[param, from M2]**. Reductions use the same scheme as
section 3: `S` interleaved partials in increasing index order, then
`tree_sum`.

- **Embedding lookup:** `h = up(E[token])`. E is the BF16 table (tied with the
  classifier).
- **RMSNorm** (weight `g`, BF16; `eps` = `f32(1e-5)`):
  ```
  ss  = Σ fma(xᵢ, xᵢ, ·)                       # reduction, as above
  var = mul(ss, inv_n)                         # inv_n = f32(1/n), constant
  r   = rsqrt(add(var, eps))                   # section 5
  yᵢ  = mul(up(gᵢ), mul(xᵢ, r))
  ```
  This uses a multiply by `f32(1/n)` instead of torch's divide by `n`, so it
  is not bit-identical to torch here (tolerance comparison only).
- **RoPE** (head dim `d`, `half = d/2`, tables `C[pos]`, `S[pos]` of length
  `d`, FP32): the tables are computed on the host with the same code as
  `transformers`:
  `inv_freq = 1/θ^(2i/d)`, `emb = [pos·inv_freq, pos·inv_freq]`,
  `C = cos(emb)`, `S = sin(emb)`.
  ```
  rᵢ = -x[i+half]  for i < half;   rᵢ = x[i-half]  for i ≥ half
  outᵢ = fma(xᵢ, Cᵢ, mul(rᵢ, Sᵢ))
  ```
- **Attention**, per query head (GQA: query head `h` uses KV head
  `h // (n_heads / n_kv_heads)`):
  ```
  q̂ = bf16(RoPE(q))      K[t] = bf16(RoPE(k_t))      V[t] = bf16(v_t)
  sₜ = mul(dot(q̂, K[t]), scale)                # scale = 1/√d, exact for d = 64
  m  = max over t of sₜ                        # order irrelevant: max is exact
  eₜ = exp(add(sₜ, -m))
  z  = Σ eₜ                                    # reduction
  pₜ = bf16(mul(eₜ, recip(z)))
  oᵢ = dot over t of (pₜ, V[t][i])             # section 3, positions as k
  ```
  The KV cache stores `K[t]`, `V[t]` in BF16, written once per token.
- **SwiGLU:** `a = gate·x`, `b = up·x` (dot products), then
  `sᵢ = mul(aᵢ, recip(add(1.0, exp(-aᵢ))))`, `hᵢ = mul(sᵢ, bᵢ)`.
- **Residual add:** `x = add(x, delta)`.
- **Matvec input rounding** (D-011): every vector entering a matrix-vector
  product is `bf16(·)` first. That applies to the RMSNorm outputs, the
  attention output `o`, and the SwiGLU output `h`.
- **Logits:** `dot(E[v], bf16(final_norm(x)))` for every vocabulary entry `v`,
  in FP32. Sampling or argmax is not part of this spec (Q-21).

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
| `A` | interleaved accumulators per lane | M2 (adder pipeline depth); provisional 8 |
| `S` | vector-unit reduction width | M2 |
| `R_RSQRT`, `N_RSQRT` | rsqrt first-guess constant, Newton steps | M1: `0x5F3759DF`, 2 |
| `R_RECIP`, `N_RECIP` | recip first-guess constant, Newton steps | M1: `0x7EF311C3`, 2 |
| exp degree, coefficients | 2ᶠ polynomial | M1: degree 4, section 5 |
