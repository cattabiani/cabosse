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
