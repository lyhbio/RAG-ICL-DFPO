# Data Manifest

For DMCR correction/evaluation, raw JSON files are produced by:

```text
RAG-ICL/scripts/reproduce_icl_raw.py
```

Metric calculation requires the same test data files:

```text
BC5CDR_Chemical/Data/test_data/bc5cdr_chemical_test_new.json
BC5CDR_Disease/Data/test_data/bc5cdr_disease_test_processed.json
NCBI_Disease/Data/test_data/ncbi_disease_test_processed.json
Chemdner/Data/test_data/chemdner_test_processed.json
NLM_Gene/Data/test_data/nlm_gene_test_processed.json
BC5CDR_RE/Data/test_data/test.json
DDI/Data/test_data/ddi_test_processed.json
Biorelex/Data/test_data/biorelex_test_processed.json
```

The DMCR figure script also needs RAG-ICL raw/fix source files for all models,
datasets, and sample counts.
