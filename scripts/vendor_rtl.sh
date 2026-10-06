#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# Fetch the vendored RTL (rtl/vendor/README.md) at its pinned commits, as is:
# the files the FP units need, and each project's license. Rerun after
# changing a pin; review the diff.
set -euo pipefail
cd "$(dirname "$0")/../rtl/vendor"

CVFPU=77811635cb7e8649f1d9484733ee6a0bdb44a0e6         # openhwgroup/cvfpu, develop, 2026-09-14
COMMON_CELLS=1281545696eb3fcba50ec5b4275993476a3c710e  # pulp-platform/common_cells, v1.40.0

fetch() {  # fetch REPO SHA PATH DEST
  mkdir -p "$(dirname "$4")"
  curl -fsSL "https://raw.githubusercontent.com/$1/$2/$3" -o "$4"
}

for f in fpnew_pkg fpnew_classifier fpnew_rounding fpnew_fma fpnew_noncomp; do
  fetch openhwgroup/cvfpu $CVFPU src/$f.sv cvfpu/src/$f.sv
done
fetch openhwgroup/cvfpu $CVFPU LICENSE.solderpad cvfpu/LICENSE.solderpad

for f in src/cf_math_pkg.sv src/lzc.sv include/common_cells/registers.svh include/common_cells/assertions.svh LICENSE; do
  fetch pulp-platform/common_cells $COMMON_CELLS $f common_cells/$f
done
