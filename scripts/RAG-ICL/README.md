# RAG-ICL / Random ICL Raw Reproduction

This folder contains the no-DMCR reproduction script for the randomized ICL
baselines used in the paper.

## What This Folder Does

- Generates raw DERS/FERS prediction JSON files.
- Computes raw Precision/Recall/F1 from those raw JSON files.
- Does not apply DMCR correction.

## Files

- `scripts/reproduce_icl_raw.py`
  - Runs DERS/FERS inference.
  - Writes raw JSON files.
  - Writes a raw metrics TSV.
- `requirements.txt`
  - Minimal Python dependencies for this script.

## Data Layout

The script expects:

```text
<data-root>/
  <dataset>/
    Data/train_data/*.json
    Data/test_data/*.json
```

Datasets:

```text
BC5CDR_Chemical
BC5CDR_Disease
NCBI_Disease
Chemdner
NLM_Gene
BC5CDR_RE
DDI
Biorelex
```

Task types:

```text
BC5CDR_Chemical: chemical
BC5CDR_Disease: disease
NCBI_Disease: disease
Chemdner: chemical
NLM_Gene: gene
BC5CDR_RE: chemical-induced disease
DDI: relation
Biorelex: binding interaction
```

## Example

```bash
python scripts/reproduce_icl_raw.py \
  --data-root /path/to/Paper_Evaluate_Pipeline/code \
  --dataset NCBI_Disease \
  --task-type disease \
  --model qwen2:7b \
  --model-slug qwen2 \
  --methods DERS FERS \
  --sample 1 \
  --repetition 1 \
  --seed 911011 \
  --output-dir outputs/icl_raw \
  --metrics-tsv outputs/icl_raw/NCBI_Disease/qwen2/sample_1_rep_1_raw_metrics.tsv \
  --ollama-host http://127.0.0.1:11435
```

## Complete 10-Seed Grid

Run:

```text
8 datasets x 5 models x 2 methods x 5 sample counts x 10 repetitions = 4000 runs
```

Use the original seed table to reproduce the reported settings exactly:

```text
seed_provenance_4000.tsv
```

DERS samples context examples independently for each test instance. FERS samples
one fixed context set per run and reuses it for all test instances.
