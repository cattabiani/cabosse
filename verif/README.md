# verif/

cocotb (Python) testbenches that run the RTL under Verilator.

Each RTL block gets a testbench that drives it with inputs and compares its
outputs with the golden model in `../model/`. The comparison is bit-exact.
Shared test utilities (AXI memory models, BF16/FP32 helpers that call into the
golden model) also live here.

First tests arrive in M3.
