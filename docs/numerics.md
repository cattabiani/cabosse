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

- **rsqrt(x) ≈ 1/√x:**
  ```
  y = float(R_RSQRT - (bits(x) >> 1))          # R_RSQRT [param], e.g. 0x5F375A86
  repeat N_RSQRT times:                        # [param]
      h = mul(0.5, x)
      t = fma(-mul(h, y), y, 1.5)
      y = mul(y, t)
  ```
- **recip(x) ≈ 1/x:**
  ```
  y = float(R_RECIP - bits(x))                 # R_RECIP [param], e.g. 0x7EF311C3
  repeat N_RECIP times:                        # [param]
      e = fma(-x, y, 1.0)
      y = fma(y, e, y)
  ```
- **exp(x) = 2^(x·log₂e):**
  ```
  t = mul(x, LOG2E)                            # LOG2E = f32(log₂ e)
  i = round_to_nearest_even(t) as integer
  f = add(t, -i)                               # f in [-0.5, 0.5], exact (Sterbenz)
  p = polynomial of degree D_EXP in f, Horner with fma   # coefficients [param]
  result = float(bits(p) + (i << 23))         # multiply by 2^i via the exponent field
  ```
  Results that would overflow become +Inf. Results below the normal range
  become +0. This is a property of this approximation, not a flush-to-zero
  rule: such values only appear in softmax, where they are negligible.
- **Special inputs:** `rsqrt(+0) = +Inf`, `rsqrt(x<0) = NaN`,
  `recip(±0) = ±Inf`, `exp(-Inf) = +0`, `exp(+Inf) = +Inf`, NaN in → NaN out.
  The bit tricks alone do not produce these, so they are handled explicitly.

Accuracy target: about BF16 level (relative error ≲ 2⁻⁹). Every result
passes through `bf16(·)` before it reaches a multiplier. M1 picks the smallest
`N_RSQRT`, `N_RECIP`, `D_EXP` that keep end-to-end accuracy no worse than
torch's BF16 run.

## 6. What is not bit-identical to PyTorch, by design

Summation order (sections 3 and 4), FMA instead of a separate multiply and
add, the approximated rsqrt/recip/exp, `1/n` as a multiply, and BF16
rounding at the D-011 points. These are compared against `transformers`
with tolerances (M1 exit criteria), never bit-exactly.

## 7. Parameters

| Param | Meaning | Fixed by |
|-------|---------|----------|
| `A` | interleaved accumulators per lane | M2 (adder pipeline depth) |
| `S` | vector-unit reduction width | M2 |
| `R_RSQRT`, `N_RSQRT` | rsqrt magic constant, Newton steps | M1 |
| `R_RECIP`, `N_RECIP` | recip magic constant, Newton steps | M1 |
| `D_EXP`, coefficients | exp polynomial | M1 |
