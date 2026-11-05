#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# Runs pytest under a memory cap, so a test run cannot take the machine
# (and the editor) down with it: the kernel kills the run alone when it
# passes the cap. Arguments go to pytest as they are.
#
#   scripts/test.sh                  # the default suite
#   scripts/test.sh -m slow          # the slow suite
#   MEM=16G scripts/test.sh ...      # another cap (default 12G; less if the
#                                    # machine has less free, minus RESERVE_GB=4)
#
# Without a user systemd (CI, containers) it runs pytest uncapped.
set -euo pipefail

cd "$(dirname "$0")/.."
MEM="${MEM:-12G}"
RESERVE_GB="${RESERVE_GB:-4}"  # left to the rest of the machine (desktop, editor)
export PATH="$PWD/.tools/bin:$PATH"

cmd=(.venv/bin/python -m pytest "$@")
if ! systemd-run --user --scope --quiet true >/dev/null 2>&1; then
  exec "${cmd[@]}"  # no user systemd: no cap to set
fi

# The cap: MEM, or less when the machine has less free now (MemAvailable
# minus RESERVE_GB), so the run hits its own cap before the system's OOM
# killer picks another process.
if [[ ! "$MEM" =~ ^[0-9]+[Gg]$ ]]; then
  echo "test.sh: MEM=$MEM: give whole gigabytes, such as 12G" >&2
  exit 2
fi
mem_gb=${MEM%[Gg]}
avail_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
cap_gb=$(( avail_gb - RESERVE_GB < mem_gb ? avail_gb - RESERVE_GB : mem_gb ))
if (( cap_gb < 2 )); then
  echo "test.sh: only ${avail_gb} GB free; close something first" >&2
  exit 1
fi
if (( cap_gb < mem_gb )); then
  echo "test.sh: capping at ${cap_gb}G (${avail_gb} GB free)" >&2
fi
exec systemd-run --user --scope --quiet -p MemoryMax="${cap_gb}G" -p MemorySwapMax=0 "${cmd[@]}"
