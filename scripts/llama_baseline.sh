#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# llama.cpp on this PC's CPU and GPU, at the depths M9 used on the F2 host
# (scripts/f2/cpu_baseline.sh), for the M9 report. Nothing is installed:
# everything lives in the git-ignored .tools/ and weights/.
#
# - CPU: llama.cpp LLAMA_TAG built from source with GGML_NATIVE, as on F2;
# - GPU: the same release's prebuilt Linux CUDA binary and its cudart
#   libraries (needs only the NVIDIA driver), all layers on the GPU;
# - the official checkpoint converted to GGUF BF16 by llama.cpp's own
#   script, in a Python 3.12 venv of its pinned requirements.
#
# One-time setup (about 600 MB of downloads):
#   git clone --depth 1 --branch b11672 https://github.com/ggml-org/llama.cpp .tools/llama.cpp-b11672
#   cmake -S .tools/llama.cpp-b11672 -B .tools/llama.cpp-b11672/build -DCMAKE_BUILD_TYPE=Release \
#     -DGGML_NATIVE=ON -DLLAMA_CURL=OFF && cmake --build .tools/llama.cpp-b11672/build -j8 --target llama-bench
#   gh release download b11672 -R ggml-org/llama.cpp -D .tools/llama.cpp-b11672-cuda \
#     -p 'llama-b11672-bin-ubuntu-cuda-13.4-x64.tar.gz' -p 'cudart-llama-b11672-bin-ubuntu-cuda-13.4-x64.tar.gz'
#     (then untar both there)
#   uv venv -p 3.12 .tools/llama-convert && uv pip install -p .tools/llama-convert/bin/python \
#     --index-strategy unsafe-best-match -r .tools/llama.cpp-b11672/requirements/requirements-convert_hf_to_gguf.txt
#
# Usage: scripts/llama_baseline.sh OUT_DIR
set -euo pipefail

LLAMA_TAG=b11672      # the release M9 ran on the F2 host
DEPTHS=0,46,1023      # M9's positions (scripts/f2/cpu_baseline.sh)
N_GEN=128             # tokens generated per llama-bench test
REPS=5                # llama-bench repetitions

repo=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?usage: llama_baseline.sh OUT_DIR}
src=$repo/.tools/llama.cpp-$LLAMA_TAG
cuda=$repo/.tools/llama.cpp-$LLAMA_TAG-cuda
weights=${CABOSSE_WEIGHTS:-$repo/weights}
gguf=$weights/smollm2-135m-instruct-bf16.gguf
threads=$(lscpu -p=core | grep -v '^#' | sort -u | wc -l)  # physical cores, as on F2
mkdir -p "$out"

if [ ! -f "$gguf" ]; then
  "$repo/.tools/llama-convert/bin/python" "$src/convert_hf_to_gguf.py" \
    "$weights/SmolLM2-135M-Instruct" --outtype bf16 --outfile "$gguf" > "$out/convert.log" 2>&1
fi

{
  echo "commit $(git -C "$repo" rev-parse HEAD)"
  echo "llama.cpp $(git -C "$src" rev-parse --short HEAD) ($LLAMA_TAG)"
  echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "$out/describe.txt"
lscpu > "$out/lscpu.txt"

"$src/build/bin/llama-bench" -m "$gguf" -p 0 -n $N_GEN -d $DEPTHS -r $REPS -t "$threads" \
  -ngl 0 -o json > "$out/llama_bench_cpu.json" 2> "$out/llama_bench_cpu.log"
echo "load after cpu $(cut -d' ' -f1-3 /proc/loadavg)" >> "$out/describe.txt"
LD_LIBRARY_PATH=$cuda/llama-$LLAMA_TAG:$cuda/cudart-llama-$LLAMA_TAG-bin-ubuntu-cuda-13.4-x64 \
  "$cuda/llama-$LLAMA_TAG/llama-bench" -m "$gguf" -p 0 -n $N_GEN -d $DEPTHS -r $REPS \
  -ngl 99 -o json > "$out/llama_bench_cuda.json" 2> "$out/llama_bench_cuda.log"
echo "done: $out"
