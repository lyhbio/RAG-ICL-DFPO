#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/mnt/nfs/wanghongyin/anaconda3/envs/llama_factory/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON_BIN" ./run_dfpo_repeats.py --datasets NCBI_Disease BC5CDR_Chemical --repeats 2-10 --device cuda:0 --batch-size 1 &
pid0=$!
"$PYTHON_BIN" ./run_dfpo_repeats.py --datasets NLM_Gene BC5CDR_Disease --repeats 2-10 --device cuda:1 --batch-size 1 &
pid1=$!
"$PYTHON_BIN" ./run_dfpo_repeats.py --datasets BC5CDR_RE Chemdner --repeats 2-10 --device cuda:2 --batch-size 1 &
pid2=$!
"$PYTHON_BIN" ./run_dfpo_repeats.py --datasets Biorelex DDI --repeats 2-10 --device cuda:3 --batch-size 1 &
pid3=$!

wait "$pid0"
wait "$pid1"
wait "$pid2"
wait "$pid3"

"$PYTHON_BIN" ./summarize_dfpo_repeats.py --repeats 1-10
