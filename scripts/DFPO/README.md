# DFPO Five-Run Reproduction

This folder contains the scripts needed to reproduce repeated DFPO inference
and summarize variance across repeated runs.

## What This Folder Does

- Runs DFPO model inference repeatedly on the eight evaluation datasets.
- Writes prediction JSON files such as `DDI/DDI_predict_1.json`.
- Supports resumable `.partial` files for interrupted long runs.
- Supports multi-GPU task scheduling for missing dataset/repeat jobs.
- Computes Precision/Recall/F1 for each repeat.
- Summarizes mean, variance, standard deviation, min/max, and malformed outputs.

## Files

- `scripts/dfpo_common.py`
  - Dataset configuration and paths.
- `scripts/run_dfpo_repeats.py`
  - Runs one or more repeats for selected datasets.
- `scripts/run_dfpo_queue.py`
  - Distributes missing dataset/repeat tasks across multiple GPUs.
- `scripts/summarize_dfpo_repeats.py`
  - Scores prediction JSON files and writes summary tables.
- `scripts/run_dfpo_10_repeats.sh`
  - Example four-GPU launcher for 10 repeats.
- `scripts/run_dfpo_remaining_queue.sh`
  - Example queue launcher for missing repeats.
- `tables/`
  - Current repeat metric/summary tables copied from the working analysis.

## Required Data Layout

The scripts expect this structure under the `DFPO/` folder:

```text
DFPO/
  EvaluationDataset/collate/ner/bc5cdr_chemical/bc5cdr_chemical_test_processed.json
  EvaluationDataset/collate/ner/bc5cdr_disease/bc5cdr_disease_test_processed.json
  EvaluationDataset/collate/ner/ncbi_disease/ncbi_disease_test_processed.json
  EvaluationDataset/collate/ner/chemdner/chemdner_test_processed.json
  EvaluationDataset/collate/ner/nlm_gene/nlm_gene_test_processed.json
  EvaluationDataset/collate/re/bc5cdr/bc5cdr_test_processed.json
  EvaluationDataset/collate/re/ddi_corpus/ddi_corpus_test_processed.json
  EvaluationDataset/collate/re/biorelex/biorelex_test_processed.json
```

Prediction files are written under:

```text
DFPO/BC5CDR_Chemical/BC5CDR_Chemical_predict_<repeat>.json
DFPO/BC5CDR_Disease/BC5CDR_Disease_predict_<repeat>.json
DFPO/NCBI_Disease/NCBI_Disease_predict_<repeat>.json
DFPO/Chemdner/Chemdner_predict_<repeat>.json
DFPO/NLM_Gene/NLM_Gene_predict_<repeat>.json
DFPO/BC5CDR_RE/BC5CDR_RE_predict_<repeat>.json
DFPO/DDI/DDI_predict_<repeat>.json
DFPO/Biorelex/Biorelex_predict_<repeat>.json
```

## Model Requirement

The default model path in `scripts/dfpo_common.py` is:

```text
/mnt/nfs/wangyu/Biomarker/DFPO-Gemma2
```

For GitHub reproduction, either:

- edit `DEFAULT_MODEL_PATH`, or
- pass `--model-path /path/to/DFPO-Gemma2`.

## Example: Run Five Repeats

Single GPU:

```bash
python scripts/run_dfpo_repeats.py \
  --datasets NCBI_Disease \
  --repeats 1-5 \
  --model-path /path/to/DFPO-Gemma2 \
  --device cuda:0 \
  --batch-size 1
```

Four-GPU queue for all missing files:

```bash
python scripts/run_dfpo_queue.py \
  --datasets BC5CDR_Chemical BC5CDR_Disease NCBI_Disease Chemdner NLM_Gene BC5CDR_RE DDI Biorelex \
  --repeats 1-5 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --largest-first \
  --batch-size 1
```

Summarize five repeats:

```bash
python scripts/summarize_dfpo_repeats.py --repeats 1-5 --out-dir tables
```

Outputs:

```text
tables/dfpo_repeat_metrics.csv
tables/dfpo_repeat_summary.csv
tables/dfpo_missing_repeats.csv
tables/dfpo_repeat_summary.md
```

## Why There Is a Four-GPU Queue

Each DFPO repeat is saved as a separate prediction JSON file:

```text
<dataset>/<dataset>_predict_<repeat>.json
```

For five repeats on eight datasets, the expected output is:

```text
8 datasets x 5 repeats = 40 prediction JSON files
```

`run_dfpo_queue.py` checks which of these expected prediction files are missing
and only runs the missing `dataset x repeat` tasks. Existing prediction files
are skipped and are not overwritten.

This is useful because the datasets have very different sizes:

```text
Chemdner: 3000 documents
BC5CDR_Chemical / BC5CDR_Disease / BC5CDR_RE: 500 documents each
DDI: 279 documents
Biorelex: 198 documents
NCBI_Disease / NLM_Gene: 100 documents each
```

Running everything on one GPU would be slow and can leave other GPUs idle.
The queue lets multiple GPUs take tasks in parallel:

```text
cuda:0 -> one missing dataset x repeat task
cuda:1 -> one missing dataset x repeat task
cuda:2 -> one missing dataset x repeat task
cuda:3 -> one missing dataset x repeat task
```

When a GPU finishes, it automatically takes the next missing task. After all
missing tasks finish, the queue calls `summarize_dfpo_repeats.py` to regenerate
the metric and summary tables.

In short:

```text
run_dfpo_repeats.py      = runs model inference and writes prediction JSON
run_dfpo_queue.py        = fills missing prediction JSON files across GPUs
summarize_dfpo_repeats.py = computes F1, mean, variance, and std tables
```

`dfpo_missing_repeats.csv` records any expected prediction JSON files that were
not found during summarization. It means the prediction output for that repeat
has not been generated yet; it does not mean that the evaluation dataset or
model files are missing.

## Notes

- `batch-size 1` is the safest setting for 24GB GPUs because some Chemdner
  examples produce long outputs.
- Interrupted runs can be resumed from `*.partial` files.
- Use `--seed-mode repeat` only if you want the repeat number to set the random
  seed. The working analysis used stochastic generation with `seed-mode none`.
