// Every RTL source in compile order (packages first), for every tool:
// Verilator and yosys-slang read it with -F, which resolves the paths below
// against this file's folder. platforms/f2/timing/timing.tcl reads it for
// Vivado too: keep to paths and // comments.
fp_pkg.sv
leading_zeros.sv
sticky_shift.sv
fp_add.sv
fp_product.sv
bf16_mac.sv
bf16_to_fp32.sv
fp32_fma.sv
fp32_max.sv
