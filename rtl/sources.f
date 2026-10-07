// Every RTL source in compile order (packages first), for every tool:
// Verilator and yosys-slang read it with -F, which resolves the paths below
// against this file's folder. Vendored files: rtl/vendor/README.md.
+incdir+vendor/common_cells/include
vendor/common_cells/src/cf_math_pkg.sv
vendor/cvfpu/src/fpnew_pkg.sv
vendor/common_cells/src/lzc.sv
vendor/cvfpu/src/fpnew_classifier.sv
vendor/cvfpu/src/fpnew_rounding.sv
vendor/cvfpu/src/fpnew_fma.sv
vendor/cvfpu/src/fpnew_noncomp.sv
bf16_to_fp32.sv
fp32_fma.sv
fp32_max.sv
