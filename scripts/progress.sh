#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# A progress bar for a pytest run writing to a log (pytest -q's one
# character per test: . passed, F failed, E error, s skipped, x xfail).
# Counts the run's tests first, then redraws every 2 s until the log has
# pytest's summary line. Read-only: it never touches the run.
#
#   scripts/progress.sh LOG [pytest selection, as given to scripts/test.sh]
#   scripts/progress.sh .tools/runs-1a/3-slow.log -m slow
set -euo pipefail

cd "$(dirname "$0")/.."
log="$1"
shift
total=$(.venv/bin/python -m pytest -q -p no:cacheprovider --collect-only "$@" 2>/dev/null \
  | awk '/tests? collected|tests? selected/ {split($1, a, "/"); print a[1]; exit}')
total=${total:-0}
start=$(date +%s)

while true; do
  # The result characters: pytest -q's lines before the summary, without the [ nn%] markers.
  results=$(sed -n '/^[.FEsxX]*\( *\[ *[0-9]*%\]\)\?$/p' "$log" 2>/dev/null | sed 's/ *\[ *[0-9]*%\]//' | tr -d '\n')
  done_n=${#results}
  failed=$(tr -cd 'FE' <<<"$results" | wc -c)
  width=40
  filled=$(( total > 0 ? done_n * width / total : 0 ))
  bar=$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' $(( width - filled )) '' | tr ' ' '-')
  mins=$(( ($(date +%s) - start) / 60 ))
  printf '\r[%s] %d/%d  failed %d  %d min watched ' "$bar" "$done_n" "$total" "$failed" "$mins"
  if grep -qE '^(=+ )?[0-9]+ (passed|failed)|(passed|failed).* in [0-9.]+s' "$log" 2>/dev/null; then
    echo
    grep -E '^FAILED|^ERROR|passed|failed' "$log" | tail -10
    exit 0
  fi
  sleep 2
done
