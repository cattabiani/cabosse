# M3 interim report: our own FP units, timing on F2's part

Status on 2026-10-08, branch `m3/own-fp` (PR #36). M3 is not finished: the
cost of subnormal support (D-016) is still to measure. Timing numbers are
Vivado's, measured; nothing here ran on the FPGA itself.

## Result

| exit criterion | status |
|---|---|
| Each FP operation bit-exact against the golden model on ≥ 10⁸ inputs and every special-value class | met: fma, add, mul, max on 10⁸ each; the lanes' multiply-add on all 2³² BF16 pairs |
| Every mismatch reported and settled by the owner | met: one (`max` with a NaN), settled by D-037 |
| The accumulate loop closes in 4 cycles at 250 MHz, or `A` changes | met: +0.877 ns slack, `A` = 16 unchanged |
| Lint clean, units synthesize in Yosys | met |
| Cost of subnormal support in Yosys cells (D-016) | to do |

## The question the AWS runs answered

The lanes accumulate into FP32 partial sums. An adder that takes several
cycles needs `A` rotating partial sums to accept one input per cycle; with
`A` = 16 (D-027) and 4 multiply-adds per lane, each partial sum must go
round its loop in 4 cycles. So the question was: does product + accumulator
+ one rounding fit in 4 cycles of 4.0 ns (250 MHz) on F2's part?

All runs: Vivado 2025.2, `xcvu47p-fsvh2892-2-e`, out of context, 4.0 ns
clock, on an AWS build instance (16 vCPUs, 32 GB), about 10 minutes each.
Slack is the margin left in the 4.0 ns: positive means timing is met.
Scripts: `scripts/f2/time_loops.sh`, `platforms/f2/timing/`. Logs:
[data/f2/](data/f2/).

| log | what was tried | result |
|---|---|---|
| 1 `acc-loop` | CVFPU's FMA in a loop of 2, 3, 4, 5 cycles | fails every length: -1.115, -0.777, -0.694, -0.527 ns |
| 2 `acc-loop-retime` | the same with Vivado's retiming | 4 cycles -0.639 ns; 5, 6, 8 cycles pass (+0.300, +0.183, +0.340) |
| 3 `mac-loop` | our own BF16 multiply-add (D-038) | 4 cycles +0.146 ns, 3 cycles +0.057 ns (retimed); worst path the product, outside the loop |
| 4 `fma-path` | our own `fp32_fma`, product in one cycle | fails: -2.700 ns, -2.023 ns retimed |
| 5 `fma-pipelined` | `fp32_fma` with 1 or 2 registers inside the product | 1: +0.155 ns; 2: +0.588 ns |
| 6 `mac-loop-pipelined` | the lanes' unit with 1 register inside the product | 4 cycles +0.657 ns; 3 cycles +0.813 ns (retimed) |
| 7 `review-recheck` | runs of log 6 again, after two review cleanups | 4 cycles +0.877 ns (+0.953 retimed); 3 cycles +0.841 ns (retimed); `fp32_fma` +0.514 ns |

## What we learned

1. **CVFPU cannot run the loop in 4 cycles at 250 MHz** (logs 1, 2). Its
   fixed stages are long (up to 27 logic levels), so extra registers barely
   help; even retimed it needs about 4.5 cycles, so 8 with `A` a power of
   two. It also spends about 1,100 LUTs and 2 DSPs on a 24 × 24 multiply
   that BF16 inputs do not need.
2. **A BF16-specific unit fits** (log 3). The exact 8 × 8-bit product goes
   outside the loop; inside it stays only product + accumulator with one
   rounding. Same bits as the spec, about 625 LUTs, no DSPs. This settled
   the loop criterion and led to D-038, then D-039.
3. **The FP32 fma's multiply must be split** (logs 4, 5). In one cycle the
   path is op select, 24 × 24 multiply on DSPs, normalize and shift: 6.7 ns.
   Retiming cannot move registers into a DSP multiply, so `fp_product` now
   has explicit stages (multiply | normalize | shift below exponent 1).
   Two registers give +0.588 ns: a 6-cycle fma. The vector unit's fma is
   not in a loop, so the extra latency does not change throughput.
4. **The lanes' product was right at the 4 ns edge, and a register fixed
   it** (logs 3 to 6). Three rewrites of the same function gave +0.146,
   +0.016 and -0.053 ns. Vivado gives the same result for the same RTL (the
   fma's +0.588 ns twice), so the spread came from the rewrites, not from
   noise. One register inside the product (`MulRegs` = 1, a 5-cycle unit)
   gives +0.657 ns, and the worst path is now the loop's own (3.3 ns). The
   register sits outside the loop, so `A` and the bits do not change.
5. **The review cleanups kept the margin** (log 7). The same four runs on
   the final RTL all pass: the loop +0.877 ns, its worst path now inside
   `fp_add`'s round stage (3.1 ns); `fp32_fma` +0.514 ns (log 6: +0.588),
   its worst path still the product's normalize stage (3.4 ns). The
   differences are of the size seen between rewrites in log 3 to 5.

## Resources (Vivado, per unit)

| unit | LUTs | registers | DSPs |
|---|---|---|---|
| CVFPU's FMA (for comparison) | about 1,100 | | 2 |
| `bf16_mac`, `MulRegs` = 1 (in the loop harness) | 633 | 275 | 0 |
| `fp32_fma`, `MulRegs` = 2 | 1,262 | 535 | 2 |

From log 7, on the final RTL (log 6: 596 LUTs and 1,183 LUTs). Yosys
(generic `synth`, flattened): `bf16_mac` 2,552 cells, `fp32_fma` 7,964.

## Decisions taken in this step

- **D-037:** `max` is IEEE 754-2019 `maximumNumber` (the one mismatch with
  CVFPU, settled in favour of the standard; no model output changes).
- **D-038:** the lanes' multiply-add is our own BF16 unit. Fallback, not
  needed: CVFPU in 256 lanes × 2 with an 8-cycle loop, 5-9% fewer tokens/s
  (estimate).
- **D-039:** all FP units are our own; CVFPU and `rtl/vendor/` are removed.
  One set of units instead of two that must be kept equal.
- Defaults, recorded in PLAN.md: `fp32_fma` `MulRegs` = 2 (6 cycles),
  `bf16_mac` `MulRegs` = 1 (5 cycles).

## Verification

- Fast suite 306 passed, slow RTL suite 7 passed: 10⁸ inputs per
  operation, all 2³² BF16 pairs, every special-value combination, each
  pipeline depth (`bf16_mac` with 0, 1 and 2 product registers).
- Mutation check (before the review cleanups): 19 hand-made changes to the
  units, 15 caught by the fast tests. The other 4 give the same bits; two
  were redundant logic, now removed.

## Obstacles

- CVFPU's timing (above) changed the plan from wrapping a mature unit to
  writing our own; the bit-exactness tests written for CVFPU carried over,
  changed only for CVFPU's pipeline parameter.
- Log 5's last run was cut from its replay by the instance shutdown; it was
  read from the live console. Log 6 was rebuilt from console snapshots
  saved during the run, after Claude Code's permission check failed and
  stopped all shell commands at the end of the session. Both contain every result line.
- No AWS instance is left running (checked 2026-10-08, after log 7).

## Left for M3

- The cost of subnormal support in Yosys cells (D-016), against a
  flush-to-zero variant.
- Then the M3 checkpoint report.
