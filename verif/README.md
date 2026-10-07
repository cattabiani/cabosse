# verif/

cocotb (Python) testbenches that run the RTL under Verilator.

Each RTL block gets a testbench that drives it with inputs and compares its
outputs with the golden model in `../model/`. The comparison is bit-exact.
Shared test utilities (AXI memory models, BF16/FP32 helpers that call into the
golden model) also live here.

- `rtl.py`: how the tests run the tools (see its docstring).
- `bulk.py` and `bulk/`: high-volume tests, a C++ driver per block that
  streams inputs through Verilator with no return to Python per input.
- `tests/`: `test_<block>.py` per RTL block (the pytest launcher and its
  cocotb tests together), and `test_rtl_checks.py` (lint and synthesis).
