#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# M9's CPU baselines on the F2 host (PLAN.md, D-056, D-067), run by
# scripts/f2/run_core.sh after the core's plan, in the same session:
#
# - llama.cpp at LLAMA_TAG (MIT), built CPU-only on the instance; the
#   official checkpoint converted to GGUF BF16 by llama.cpp's own script
#   (the format only, in a venv of its own pinned requirements);
#   llama-bench: N_GEN tokens generated at each of DEPTHS, JSON;
# - transformers on the same CPU: scripts/measure_m7.py cpu (D-067).
#
# Usage: cpu_baseline.sh OUT_DIR   (with /root/venv, /root/.local/bin/uv and
# $CABOSSE_WEIGHTS as run_core.sh leaves them)

LLAMA_TAG=b11672               # llama.cpp release of 2026-10-16
CONVERT_PYTHON=3.12            # llama.cpp's pinned converter requirements need it
DEPTHS=0,46,1023               # positions: the start, the long fixture's prompt, near its end
N_GEN=128                      # tokens generated per llama-bench test
REPS=5                         # llama-bench repetitions

out=$1
repo=$(cd "$(dirname "$0")/../.." && pwd)
model=$CABOSSE_WEIGHTS/SmolLM2-135M-Instruct
mkdir -p "$out"
fail() { echo "CABOSSE-BASELINE fail $1"; exit 1; }

lscpu > "$out/lscpu.txt"
git clone -q --depth 1 -b $LLAMA_TAG https://github.com/ggml-org/llama.cpp.git /root/llama.cpp || fail "clone"
command -v cmake >/dev/null || apt-get install -y -q cmake || fail "cmake install"
cmake -S /root/llama.cpp -B /root/llama.cpp/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-14 -DGGML_NATIVE=ON -DLLAMA_CURL=OFF > "$out/cmake.log" 2>&1 \
  && cmake --build /root/llama.cpp/build -j --target llama-bench >> "$out/cmake.log" 2>&1 || fail "build"

/root/.local/bin/uv venv -q -p $CONVERT_PYTHON /root/convert \
  && /root/.local/bin/uv pip install -q -p /root/convert/bin/python --index-strategy unsafe-best-match \
       -r /root/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt > "$out/convert.log" 2>&1 \
  && /root/convert/bin/python /root/llama.cpp/convert_hf_to_gguf.py "$model" --outtype bf16 \
       --outfile /root/smollm2-bf16.gguf >> "$out/convert.log" 2>&1 || fail "convert"

/root/llama.cpp/build/bin/llama-bench -m /root/smollm2-bf16.gguf -p 0 -n $N_GEN -d $DEPTHS \
  -r $REPS -o json > "$out/llama_bench.json" 2> "$out/llama_bench.log" || fail "llama-bench"

/root/venv/bin/python "$repo/scripts/measure_m7.py" cpu > "$out/transformers.txt" 2>&1 || fail "transformers"
/root/venv/bin/python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["cpu"], indent=1))' \
  "$repo/reports/data/M7.json" > "$out/transformers.json" || fail "transformers json"
echo "CABOSSE-BASELINE pass"
