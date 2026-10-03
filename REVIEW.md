# Review checklist

For agents reviewing changes in this repo: a PR, a branch, a diff, or the
final check before declaring a step done (AGENTS.md, working style).

Review only; do not change files unless the owner asks. The owner picks which
findings to fix.

## Scope

Read the full diff against `main`, plus the parts of `docs/numerics.md`,
`model/golden/` and `model/tests/` it touches. Run `ruff format --check .`,
`ruff check .` and `pytest`, and report their real output.

## Checklist

1. **Spec ↔ code ↔ tests.** Every rule in the changed spec text is implemented
   exactly (order of operations, which primitive rounds where, constants), and
   the code does nothing the spec does not say. Spec changes come with the
   golden model and tests in the same PR (AGENTS.md rule 8).
2. **Every claim is backed.** Each factual sentence in the spec, docstrings and
   comments ("cannot be -0", "exact", "order does not matter", "z ≥ 1", an
   error bound) is either proven in a sentence or tested. Try to construct a
   counterexample for each; past wrong claims were found this way.
3. **Edge cases**, for every new operation:
   - signed zeros (-0 results, `x + (-x) = +0`, -0 vs +0 in max and padding);
   - NaN (canonical `0x7FC00000` / `0x7FC0` on output) and ±Inf, including
     Inf - Inf;
   - subnormal inputs and results; overflow to Inf;
   - flush-to-zero on (`settings.override(ftz=True)`): only an experiment
     switch, but it must not break the algorithm;
   - shapes: length 1, length < the interleave width, not a multiple of it,
     empty input (must be rejected).
4. **Tests can fail.** Tests compare bit-exactly against an independent
   restatement of the spec (constants copied, not imported; exact oracle in
   `model/tests/oracle.py`). For each important rule, name a plausible
   mutation (drop a rounding, change the order, zero-pad, swap fma for
   mul+add) and check that some test fails. If a mutation survives, either
   the test has a gap or the mutation is harmless; say which and why.
5. **Tolerances are derived, not tuned.** Each accuracy bound has a written
   derivation; report the measured value next to it. A bound loosened to
   make a test pass is a finding (AGENTS.md rule 9).
6. **Hardware-model hygiene.** Explicit dtypes; no float64 inside modelled
   computations (only in emulation like `arith.fma`, and in tests); no
   private helpers used across modules; functions named after the hardware
   operation. Keep it simple: flag over-engineering too.
7. **Repo rules.** SPDX headers and copyright line, Python 3.14 typing, no
   new dependencies, no weights or build outputs, PLAN.md / decision log
   updated if a decision was made, milestone scope respected.

## Report

A numbered list, most severe first. For each finding: file and line, what is
wrong, a concrete failing input or scenario, and a suggested fix. Mark
uncertain findings as such. End with the mutations tried and their result,
and anything checked that was fine (briefly).
