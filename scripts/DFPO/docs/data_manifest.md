# DFPO Data Manifest

Required evaluation files:

```text
EvaluationDataset/collate/ner/bc5cdr_chemical/bc5cdr_chemical_test_processed.json
EvaluationDataset/collate/ner/bc5cdr_disease/bc5cdr_disease_test_processed.json
EvaluationDataset/collate/ner/ncbi_disease/ncbi_disease_test_processed.json
EvaluationDataset/collate/ner/chemdner/chemdner_test_processed.json
EvaluationDataset/collate/ner/nlm_gene/nlm_gene_test_processed.json
EvaluationDataset/collate/re/bc5cdr/bc5cdr_test_processed.json
EvaluationDataset/collate/re/ddi_corpus/ddi_corpus_test_processed.json
EvaluationDataset/collate/re/biorelex/biorelex_test_processed.json
```

Optional existing prediction files for resumable/recomputed summaries:

```text
BC5CDR_Chemical/BC5CDR_Chemical_predict_1.json ... BC5CDR_Chemical_predict_5.json
BC5CDR_Disease/BC5CDR_Disease_predict_1.json ... BC5CDR_Disease_predict_5.json
NCBI_Disease/NCBI_Disease_predict_1.json ... NCBI_Disease_predict_5.json
Chemdner/Chemdner_predict_1.json ... Chemdner_predict_5.json
NLM_Gene/NLM_Gene_predict_1.json ... NLM_Gene_predict_5.json
BC5CDR_RE/BC5CDR_RE_predict_1.json ... BC5CDR_RE_predict_5.json
DDI/DDI_predict_1.json ... DDI_predict_5.json
Biorelex/Biorelex_predict_1.json ... Biorelex_predict_5.json
```

Current copied summary outputs:

```text
tables/dfpo_repeat_metrics.csv
tables/dfpo_repeat_summary.csv
tables/dfpo_missing_repeats.csv
tables/dfpo_repeat_summary.md
```
