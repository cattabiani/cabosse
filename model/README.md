# model/

Python reference code. Nothing here runs on the accelerator.

Planned contents:

- **Golden model** (M1): a PyTorch implementation of Llama-style decode that uses
  the hardware's exact number formats and summation order. It is the bit-exact
  reference for RTL tests. It is checked against PyTorch, with tolerances.
- **Performance model** (M2): a first-order model of tokens/s from memory
  bandwidth, lane count, and clock frequency. We use it to size the design and
  compare against measurements.

Model weights are downloaded into a git-ignored directory and never committed.

## Layout and tests

- `golden/arith.py`: the arithmetic primitives of `docs/numerics.md`
  section 2 (`bf16`, `up`, `add`, `mul`, `fma`, `mac`).
- `golden/funcs.py`: the function approximations of section 5 (`rsqrt`,
  `recip`, `exp`).
- `golden/settings.py`: experiment switches that are not part of the spec
  (e.g. flush-to-zero).
- `tests/`: pytest tests. `tests/oracle.py` is an independent, exact
  (rational-arithmetic) reference used to check rounding.

From the repository root: `pytest` runs the fast tests (seconds), and
`pytest -m slow` runs the exhaustive ones (minutes).
