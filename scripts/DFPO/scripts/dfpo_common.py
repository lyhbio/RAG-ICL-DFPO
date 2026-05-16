from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "/mnt/nfs/wangyu/Biomarker/DFPO-Gemma2"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    task: str
    directory: str
    test_path: str
    prompt_label: str
    default_device: str

    @property
    def output_prefix(self) -> str:
        return self.name

    def output_path(self, repeat: int) -> Path:
        return ROOT / self.directory / f"{self.output_prefix}_predict_{repeat}.json"


DATASETS: dict[str, DatasetConfig] = {
    "BC5CDR_Chemical": DatasetConfig(
        name="BC5CDR_Chemical",
        task="ner",
        directory="BC5CDR_Chemical",
        test_path="EvaluationDataset/collate/ner/bc5cdr_chemical/bc5cdr_chemical_test_processed.json",
        prompt_label="Chemical",
        default_device="cuda:0",
    ),
    "BC5CDR_Disease": DatasetConfig(
        name="BC5CDR_Disease",
        task="ner",
        directory="BC5CDR_Disease",
        test_path="EvaluationDataset/collate/ner/bc5cdr_disease/bc5cdr_disease_test_processed.json",
        prompt_label="Disease",
        default_device="cuda:1",
    ),
    "BC5CDR_RE": DatasetConfig(
        name="BC5CDR_RE",
        task="re",
        directory="BC5CDR_RE",
        test_path="EvaluationDataset/collate/re/bc5cdr/bc5cdr_test_processed.json",
        prompt_label="chemical-induced disease",
        default_device="cuda:2",
    ),
    "Biorelex": DatasetConfig(
        name="Biorelex",
        task="re",
        directory="Biorelex",
        test_path="EvaluationDataset/collate/re/biorelex/biorelex_test_processed.json",
        prompt_label="binding interaction",
        default_device="cuda:3",
    ),
    "Chemdner": DatasetConfig(
        name="Chemdner",
        task="ner",
        directory="Chemdner",
        test_path="EvaluationDataset/collate/ner/chemdner/chemdner_test_processed.json",
        prompt_label="Chemical",
        default_device="cuda:2",
    ),
    "DDI": DatasetConfig(
        name="DDI",
        task="re",
        directory="DDI",
        test_path="EvaluationDataset/collate/re/ddi_corpus/ddi_corpus_test_processed.json",
        prompt_label="drug-drug interaction",
        default_device="cuda:3",
    ),
    "NCBI_Disease": DatasetConfig(
        name="NCBI_Disease",
        task="ner",
        directory="NCBI_Disease",
        test_path="EvaluationDataset/collate/ner/ncbi_disease/ncbi_disease_test_processed.json",
        prompt_label="Disease",
        default_device="cuda:0",
    ),
    "NLM_Gene": DatasetConfig(
        name="NLM_Gene",
        task="ner",
        directory="NLM_Gene",
        test_path="EvaluationDataset/collate/ner/nlm_gene/nlm_gene_test_processed.json",
        prompt_label="Gene",
        default_device="cuda:1",
    ),
}
