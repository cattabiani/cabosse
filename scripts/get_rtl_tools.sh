#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# Fetch the OSS CAD Suite (Verilator, Yosys with slang) at its pinned release
# into .tools/oss-cad-suite, where the tests find it (README, development
# setup). Does nothing if that release is already there; replaces any other.
set -euo pipefail
cd "$(dirname "$0")/.."

RELEASE=2026-10-06
SUITE=.tools/oss-cad-suite

[ "$(cat "$SUITE/.release" 2>/dev/null)" = "$RELEASE" ] && exit 0
rm -rf "$SUITE"
mkdir -p .tools
curl -fsSL "https://github.com/YosysHQ/oss-cad-suite-build/releases/download/$RELEASE/oss-cad-suite-linux-x64-${RELEASE//-/}.tgz" \
  | tar xz -C .tools
echo "$RELEASE" > "$SUITE/.release"
