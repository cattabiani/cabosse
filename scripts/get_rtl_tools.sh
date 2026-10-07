#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# Fetch the OSS CAD Suite (Verilator, Yosys with slang) at its pinned release
# into .tools/oss-cad-suite, where the tests find it (README, development
# setup). Does nothing if it is already there; to change the pin, edit RELEASE
# and remove .tools/oss-cad-suite.
set -euo pipefail
cd "$(dirname "$0")/.."

RELEASE=2026-10-06
TGZ="oss-cad-suite-linux-x64-${RELEASE//-/}.tgz"

if [ -x .tools/oss-cad-suite/bin/verilator ]; then
  exit 0
fi
mkdir -p .tools
curl -fsSL "https://github.com/YosysHQ/oss-cad-suite-build/releases/download/$RELEASE/$TGZ" \
  | tar xz -C .tools
