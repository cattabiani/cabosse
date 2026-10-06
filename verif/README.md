# verif/

cocotb (Python) testbenches that run the RTL under Verilator.

Each RTL block gets a testbench that drives it with inputs and compares its
outputs with the golden model in `../model/`. The comparison is bit-exact.
Shared test utilities (AXI memory models, BF16/FP32 helpers that call into the
golden model) also live here.

- `rtl.py`: runs a cocotb testbench under Verilator from pytest, and the
  lint and Yosys synthesis checks; finds the OSS CAD Suite in `../.tools/`.
- `tests/`: one test file per RTL block (the pytest launcher and its cocotb
  tests together), plus `test_rtl_checks.py` (every file lints clean and
  synthesizes).

`pytest` from the repo root runs these with the golden-model tests. Without
Verilator or Yosys on `PATH` (as in CI for now) the RTL tests skip.
