# M3 interim report: the cost of subnormal support

The last open M3 exit criterion: the cost of subnormal support in Yosys
cells (D-016). This report goes into the M3 checkpoint report with
[M3-interim.md](M3-interim.md). Every number below is generated from
[data/M3-subnormal-yosys.txt](data/M3-subnormal-yosys.txt) by
`scripts/report_m3_subnormal.py`.

<!-- begin: provenance -->
Measured on 2026-10-08, commit `2e92602`, Yosys 0.69+190 with yosys-slang, generic `synth -flatten`.
<!-- end: provenance -->

## Result

Subnormal support costs a few percent of each unit's cells, the vector
unit's FP32 fma more than the lanes' multiply-add. D-016 (keep subnormals,
as PyTorch) stays unless the owner decides otherwise. The F2 resource cost
comes with the first build (exit criterion).

<!-- begin: cost -->
| unit | with subnormals | flush to zero | cost (cells) | share of the unit |
|---|---|---|---|---|
| `bf16_mac` (lanes) | 2,568 | 2,492 | +76 | +3.0 % |
| `fp32_fma` (vector unit) | 7,953 | 7,554 | +399 | +5.0 % |
| `fp_product`, M = 8 | 870 | 750 | +120 | +13.8 % |
| `fp_product`, M = 24 | 4,939 | 4,406 | +533 | +10.8 % |
| `fp_add`, W = 27 | 1,531 | 1,556 | -25 | -1.6 % |
| `fp_add`, W = 51 | 2,564 | 2,570 | -6 | -0.2 % |
<!-- end: cost -->

## What was measured

`Ftz` = 1 (default 0) builds the units' flush-to-zero variant. It matches
the golden model's `settings.override(ftz=True)` bit for bit: subnormal
inputs and results become zeros of their sign; the product stays exact and
the result is rounded once, at subnormal precision, before the flush. The
cost is the difference in cells between the two builds.

- The saving is in `fp_product`: with normal inputs the product of the
  significands is in [1, 4), so its leading-zero count and normalize shift
  become one bit.
- `fp_add` loses nothing and gains the flushes: it still needs subnormal
  rounding, because a tiny exact product changes the fma's rounding.
- `fp32_max` and `bf16_to_fp32` only move bits: subnormals cost them
  nothing, so they have no variant.

The blocks FTZ (as the golden model defines it) must keep, and the one it
removes, for scale:

<!-- begin: parts -->
| block | cells |
|---|---|
| `sticky_shift`, 27 bits (shift below exponent 1, M = 8) | 331 |
| `sticky_shift`, 51 bits (shift below exponent 1, M = 24) | 666 |
| `leading_zeros`, 16 bits (product, M = 8) | 41 |
| `leading_zeros`, 48 bits (product, M = 24) | 172 |
<!-- end: parts -->

A non-fused FTZ that flushes the product too could also drop the shift
below exponent 1 and the subnormal rounding, but would no longer match the
golden switch.

## How far to trust the numbers

Yosys gives the same count on every run of the same source, but the count
also moves with how the source is written. The units with `Ftz` = 0 have
the same logic as before the parameter existed:

<!-- begin: baseline -->
| unit | `94a894f` (no `Ftz`) | this commit, `Ftz` = 0 | difference |
|---|---|---|---|
| `bf16_mac` | 2,552 | 2,568 | +16 |
| `fp32_fma` | 7,964 | 7,953 | -11 |
<!-- end: baseline -->

So differences of that size are not significant; both costs above are
several times larger.

## Verification

The `Ftz` = 1 builds are tested against the golden ftz switch on the
special values and the random batch, at every pipeline depth (`bf16_mac`;
`fp32_fma`'s fma, add, mul). Each flush is reached: the rows it changes,
of which a test requires a minimum per flush:

<!-- begin: hits -->
| unit, operation | product input | addend | result |
|---|---|---|---|
| `bf16_mac mac` | 2,618 | 28,523 | 31,110 |
| `fp32_fma fma` | 1,361 | 20,661 | 19,570 |
| `fp32_fma add` | 552 | 544 | 545 |
| `fp32_fma mul` | 2,233 | – | 22,174 |
<!-- end: hits -->

Removing the flush of the product's inputs, of the addend or of the result
makes the tests fail. Keeping the full leading-zero count under FTZ passes,
as it should: the same value from more logic. Lint and Yosys synthesis
cover the `Ftz` = 1 builds.
