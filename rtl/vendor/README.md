# rtl/vendor/

Third-party RTL, copied as is at pinned commits by
[`scripts/vendor_rtl.sh`](../../scripts/vendor_rtl.sh): only the files the FP
units use (D-033). Rerun the script after changing a pin, and review the diff.
These files keep their own license headers; [`lint.vlt`](lint.vlt) exempts
them from our Verilator `-Wall` rule.

| Project | Pinned at | License | Files |
|---|---|---|---|
| [CVFPU](https://github.com/openhwgroup/cvfpu) (FPnew) | `develop`, commit `77811635cb7e` (2026-09-14) | Solderpad 0.51 | `fpnew_pkg`, `fpnew_classifier`, `fpnew_rounding`, `fpnew_fma`, `fpnew_noncomp` |
| [common_cells](https://github.com/pulp-platform/common_cells) | `v1.40.0`, commit `1281545696eb` | Solderpad 0.51; `assertions.svh` Apache-2.0 | `cf_math_pkg`, `lzc`, `registers.svh`, `assertions.svh` |

Why these pins:
- CVFPU's last release, v0.8.1, is from 2023; `develop` has fixes since.
- CVFPU asks for common_cells 1.21.0, but `lzc.sv` in that release carries
  an "unreleased, redistribution forbidden" header. From v1.39.0 on it is
  Solderpad 0.51; the files we use have the same interface.

No local changes.
