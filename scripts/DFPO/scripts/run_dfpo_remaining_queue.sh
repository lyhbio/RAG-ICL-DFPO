#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-/mnt/nfs/wanghongyin/anaconda3/envs/llama_factory/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON_BIN" ./run_dfpo_queue.py --repeats 1-10 --devices cuda:0 cuda:1 cuda:2 cuda:3 --largest-first
