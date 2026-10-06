# rtl/vendor/

Third-party RTL, copied as is at pinned commits by
[`scripts/vendor_rtl.sh`](../../scripts/vendor_rtl.sh) (the pins and the file
list): only the files the FP units use (D-033). Their checksums are in
`SHA256SUMS`, and a test fails if a file changes. They keep their own license
headers; [`lint.vlt`](lint.vlt) exempts them from our Verilator `-Wall` rule.
Compile order: [`../sources.f`](../sources.f).

| Project | Pinned at | License |
|---|---|---|
| [CVFPU](https://github.com/openhwgroup/cvfpu) (FPnew) | `develop`, 2026-09-14 | Solderpad 0.51 |
| [common_cells](https://github.com/pulp-platform/common_cells) | `v1.40.0` | Solderpad 0.51; `assertions.svh` Apache-2.0 |

Why these pins:
- CVFPU's last release, v0.8.1, is from 2023; `develop` has fixes since.
- CVFPU asks for common_cells 1.21.0, but `lzc.sv` in that release carries
  an "unreleased, redistribution forbidden" header. From v1.39.0 on it is
  Solderpad 0.51; the files we use have the same interface.

No local changes.
