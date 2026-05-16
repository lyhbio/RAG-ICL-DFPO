from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dfpo-dmcr")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAG_SOURCE_DIR = Path(
    "/mnt/disk1/wanghongyin/work/Paper_Evaluate_Pipeline/"
    "RAG-ICL_OriginResults_Logs/RAG-ICL"
)
PIPELINE_ROOT = Path("/mnt/disk1/wanghongyin/work/Paper_Evaluate_Pipeline")
OUTPUT_DIR = Path("/mnt/disk1/wanghongyin/work/DFPO_Evaluation/figures/dmcr_raw_vs_fix")

DATASETS = [
    "BC5CDR_Chemical",
    "BC5CDR_Disease",
    "NCBI_Disease",
    "Chemdner",
    "NLM_Gene",
    "BC5CDR_RE",
    "DDI",
    "Biorelex",
]
SOURCE_DATASET_NAMES = {
    "BC5CDR_Chemical": "BC5CDR-Chemical",
}
RE_DATASETS = {"BC5CDR_RE", "DDI", "Biorelex"}

MODEL_ORDER = [
    "phi3_14b-medium-4k-instruct-q3_K_S",
    "gemma2",
    "llama3.1",
    "mistral",
    "qwen2",
    "llama3.1_70b",
]
MODEL_DISPLAY = {
    "phi3_14b-medium-4k-instruct-q3_K_S": "Phi3-14B",
    "gemma2": "Gemma2",
    "llama3.1": "Llama3.1",
    "mistral": "Mistral",
    "qwen2": "Qwen2",
    "llama3.1_70b": "Llama3.1-70B",
}
MODEL_SLUG = {
    "phi3_14b-medium-4k-instruct-q3_K_S": "phi3_14b",
    "gemma2": "gemma2",
    "llama3.1": "llama3_1",
    "mistral": "mistral",
    "qwen2": "qwen2",
    "llama3.1_70b": "llama3_1_70b",
}
FIGURE_NAMES = {
    "phi3_14b-medium-4k-instruct-q3_K_S": "figure6_phi3_14b_dmcr_raw_vs_fix",
    "gemma2": "figureS1_gemma2_dmcr_raw_vs_fix",
    "llama3.1": "figureS2_llama3_1_dmcr_raw_vs_fix",
    "mistral": "figureS3_mistral_dmcr_raw_vs_fix",
    "qwen2": "figureS4_qwen2_dmcr_raw_vs_fix",
    "llama3.1_70b": "figureS5_llama3_1_70b_dmcr_raw_vs_fix",
}


@dataclass(frozen=True)
class Metric:
    precision: float
    recall: float
    f1: float


def source_dataset_name(dataset: str) -> str:
    return SOURCE_DATASET_NAMES.get(dataset, dataset)


def data_dataset_name(dataset: str) -> str:
    return dataset


def canonical_dataset_name(source_name: str) -> str:
    return "BC5CDR_Chemical" if source_name == "BC5CDR-Chemical" else source_name


def load_test_data(dataset: str) -> dict:
    data_dir = PIPELINE_ROOT / data_dataset_name(dataset) / "Data" / "test_data"
    candidates = sorted(data_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No test data JSON found in {data_dir}")
    return json.loads(candidates[0].read_text())


def score_ner(gold: dict, pred: dict) -> Metric:
    tp = fp = fn = 0
    for doc_id, truth_item in gold.items():
        true_entities = extract_truth_entities(truth_item)
        pred_entities = pred.get(doc_id, [])
        if not (
            isinstance(pred_entities, list)
            and all(isinstance(entity, str) for entity in pred_entities)
        ):
            pred_entities = []
        true_set = {entity.lower() for entity in true_entities}
        pred_set = {entity.lower() for entity in pred_entities}
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
    return prf(tp, fp, fn)


def score_re(gold: dict, pred: dict) -> Metric:
    tp = fp = fn = 0
    for doc_id, truth_item in gold.items():
        truth_pairs = extract_truth_pairs(truth_item)
        ground_truth_set = {(pair[0].lower(), pair[1].lower()) for pair in truth_pairs}
        predictions = pred.get(doc_id, [])
        try:
            pred_set = {(pair[0].lower(), pair[1].lower()) for pair in predictions}
        except Exception:
            pred_set = {()}
        tp += len(ground_truth_set & pred_set)
        fp += len(pred_set - ground_truth_set)
        fn += len(ground_truth_set - pred_set)
    return prf(tp, fp, fn)


def extract_truth_entities(truth_item) -> list[str]:
    if isinstance(truth_item, list) and len(truth_item) >= 2:
        entities = truth_item[1]
    elif isinstance(truth_item, dict):
        entities = truth_item.get("entities", truth_item.get("labels", []))
    else:
        entities = []
    normalized: list[str] = []
    for entity in entities:
        if isinstance(entity, str):
            normalized.append(entity)
        elif isinstance(entity, dict) and entity:
            value = next(iter(entity.values()))
            if isinstance(value, str):
                normalized.append(value)
    return normalized


def extract_truth_pairs(truth_item) -> list[tuple[str, str]]:
    if isinstance(truth_item, list) and len(truth_item) >= 2:
        pairs = truth_item[1]
    elif isinstance(truth_item, dict):
        pairs = truth_item.get("triples", truth_item.get("relations", []))
    else:
        pairs = []
    normalized: list[tuple[str, str]] = []
    for pair in pairs:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            normalized.append((str(pair[0]), str(pair[1])))
        elif isinstance(pair, dict):
            left = pair.get("drug", pair.get("head", pair.get("entity1")))
            right = pair.get("target", pair.get("tail", pair.get("entity2")))
            if left is not None and right is not None:
                normalized.append((str(left), str(right)))
    return normalized


def prf(tp: int, fp: int, fn: int) -> Metric:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metric(precision=precision, recall=recall, f1=f1)


def discover_models() -> list[str]:
    models = sorted({path.parts[-2] for path in RAG_SOURCE_DIR.glob("*/*/RAG-ICL")})
    ordered = [model for model in MODEL_ORDER if model in models]
    ordered += [model for model in models if model not in ordered]
    return ordered


def infer_sample_num(path: Path) -> int:
    match = re.search(r"(?:fix_)?(\d+)_sample\.json$", path.name)
    if not match:
        raise ValueError(f"Cannot infer sample number from {path}")
    return int(match.group(1))


def collect_long_metrics(models: list[str]) -> pd.DataFrame:
    rows = []
    test_cache = {dataset: load_test_data(dataset) for dataset in DATASETS}
    for dataset in DATASETS:
        ds_source = source_dataset_name(dataset)
        for model in models:
            run_dir = RAG_SOURCE_DIR / ds_source / model / "RAG-ICL"
            for stage in ["raw", "fix"]:
                pattern = "[0-9]_sample.json" if stage == "raw" else "fix_[0-9]_sample.json"
                for path in sorted(run_dir.glob(pattern), key=infer_sample_num):
                    sample_num = infer_sample_num(path)
                    pred = json.loads(path.read_text())
                    if dataset in RE_DATASETS:
                        metric = score_re(test_cache[dataset], pred)
                    else:
                        metric = score_ner(test_cache[dataset], pred)
                    rows.append(
                        {
                            "dataset": dataset,
                            "model_source": model,
                            "model": MODEL_DISPLAY.get(model, model),
                            "stage": stage,
                            "observation_type": "sample_count_not_seed_repeat",
                            "sample_num": sample_num,
                            "repetition": 1,
                            "source_file": str(path),
                            "precision": metric.precision,
                            "recall": metric.recall,
                            "f1": metric.f1,
                            "f1_percent": metric.f1 * 100.0,
                        }
                    )
    return pd.DataFrame(rows)


def build_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        long_df.groupby(["model_source", "model", "dataset", "sample_num", "stage"], as_index=False)
        .agg(
            mean_f1=("f1_percent", "mean"),
            std_f1=("f1_percent", "std"),
            n_repeats=("f1_percent", "count"),
            observed_repetitions=("repetition", lambda values: ",".join(map(str, sorted(set(values))))),
        )
        .rename(columns={"sample_num": "sample"})
    )
    summary["observation_type"] = "rag_sample_setting_single_run"
    return summary[
        [
            "model",
            "model_source",
            "dataset",
            "sample",
            "stage",
            "mean_f1",
            "std_f1",
            "n_repeats",
            "observed_repetitions",
            "observation_type",
        ]
    ]


def build_completeness(long_df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for model_source in models:
        model = MODEL_DISPLAY.get(model_source, model_source)
        for dataset in DATASETS:
            for sample in range(1, 6):
                for stage in ["raw", "fix"]:
                    sub = long_df[
                        (long_df["model_source"] == model_source)
                        & (long_df["dataset"] == dataset)
                        & (long_df["sample_num"] == sample)
                        & (long_df["stage"] == stage)
                    ]
                    observed = sorted(set(sub["repetition"].astype(int).tolist()))
                    rows.append(
                        {
                            "model": model,
                            "model_source": model_source,
                            "dataset": dataset,
                            "sample": sample,
                            "stage": stage,
                            "expected_observations": 1,
                            "actual_n": len(observed),
                            "observed_observations": ",".join(map(str, observed)),
                            "complete_single_run": len(observed) == 1,
                            "available_observation_type": "rag_single_run_per_sample",
                            "note": "RAG-ICL source has one raw/fix file per sample setting; seed repetitions are not expected.",
                        }
                    )
    return pd.DataFrame(rows)


def value_or_nan(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return math.nan
    return float(frame.iloc[0][column])


def value_or_zero(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return 0.0
    return float(frame.iloc[0][column])


def text_or_empty(frame: pd.DataFrame, column: str) -> str:
    if frame.empty:
        return ""
    return str(frame.iloc[0][column])


def pretty_dataset(dataset: str) -> str:
    return {
        "BC5CDR_Chemical": "BC5CDR Chemical",
        "BC5CDR_Disease": "BC5CDR Disease",
        "NCBI_Disease": "NCBI Disease",
        "Chemdner": "Chemdner",
        "NLM_Gene": "NLM Gene",
        "BC5CDR_RE": "BC5CDR RE",
        "DDI": "DDI",
        "Biorelex": "BioRelEx",
    }.get(dataset, dataset)


def plot_model(summary: pd.DataFrame, model_source: str, draft: bool = False) -> list[Path]:
    model_name = MODEL_DISPLAY.get(model_source, model_source)
    model_df = summary[summary["model_source"] == model_source].copy()
    if model_df.empty:
        print(f"[warn] No summary rows for {model_source}")
        return []

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 19,
            "axes.titlesize": 19,
            "axes.labelsize": 19,
            "xtick.labelsize": 18,
            "ytick.labelsize": 16,
            "legend.fontsize": 21,
            "figure.titlesize": 22,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    raw_color = "#E99A86"
    fix_color = "#B76E86"
    fig, axes = plt.subplots(2, 4, figsize=(18.2, 10.4), subplot_kw={"projection": "polar"})
    axes = axes.ravel()
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    closed_angles = np.concatenate([angles, [angles[0]]])
    axis_labels = [str(idx) for idx in range(1, 6)]

    for ax, dataset in zip(axes, DATASETS):
        ds_df = model_df[model_df["dataset"] == dataset]
        if ds_df.empty:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
        max_value = 0.0
        stage_values: dict[str, np.ndarray] = {}
        for stage in ["raw", "fix"]:
            stage_df = ds_df[ds_df["stage"] == stage].sort_values("sample")
            values_by_sample = dict(zip(stage_df["sample"].astype(int), stage_df["mean_f1"] / 100.0))
            values = np.array([values_by_sample.get(sample, np.nan) for sample in range(1, 6)], dtype=float)
            stage_values[stage] = values
            if np.isfinite(values).any():
                max_value = max(max_value, float(np.nanmax(values)))

        radial_top = max(0.1, math.ceil((max_value + 0.04) * 20) / 20)
        radial_top = min(1.0, radial_top)
        ticks = [radial_top / 4, radial_top]
        ax.set_ylim(0, radial_top)
        ax.set_xticks(angles)
        ax.set_xticklabels(axis_labels, fontsize=18)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.2f}".rstrip("0").rstrip(".") for tick in ticks], fontsize=16, color="#555555")
        ax.set_rlabel_position(18)
        ax.tick_params(axis="x", pad=0)
        ax.tick_params(axis="y", pad=0)
        ax.grid(color="#B8B8B8", linewidth=0.75, alpha=0.9)
        ax.spines["polar"].set_color("#222222")
        ax.spines["polar"].set_linewidth(0.9)

        for stage, color, label, alpha in [
            ("raw", raw_color, "Origin", 0.20),
            ("fix", fix_color, "DMCR", 0.18),
        ]:
            values = stage_values.get(stage)
            if values is None or not np.isfinite(values).any():
                continue
            closed_values = np.concatenate([values, [values[0]]])
            ax.plot(closed_angles, closed_values, color=color, linewidth=2.3, label=label)
            ax.fill(closed_angles, closed_values, color=color, alpha=alpha)

        ax.set_xlabel(
            pretty_dataset(dataset),
            labelpad=18,
            fontsize=21,
            fontweight="bold",
            color="#3F7FB5",
        )
        ax.xaxis.set_label_coords(0.5, -0.12)

    handles = [
        plt.Line2D([0], [0], color=raw_color, linewidth=4.0),
        plt.Line2D([0], [0], color=fix_color, linewidth=4.0),
    ]
    fig.legend(
        handles,
        ["Origin", "DMCR"],
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.072),
        borderaxespad=0.0,
    )
    fig.text(
        0.5,
        0.03,
        "1-5 = Number of examples",
        ha="center",
        va="center",
        fontsize=20,
        color="#333333",
    )
    suffix = "draft_" if draft else ""
    fig.subplots_adjust(left=0.032, right=0.988, bottom=0.20, top=0.985, wspace=0.28, hspace=0.46)

    base = OUTPUT_DIR / (suffix + FIGURE_NAMES.get(model_source, f"{MODEL_SLUG.get(model_source, model_source)}_dmcr_raw_vs_fix"))
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RAG-ICL DMCR raw vs fix F1 figures.")
    parser.add_argument("--draft-model", default=None, help="Generate only one draft model.")
    parser.add_argument("--all", action="store_true", help="Generate all model figures.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[data-source] {RAG_SOURCE_DIR}")
    models = discover_models()
    print("[model-mapping]")
    for model in models:
        print(f"  {model} -> {MODEL_DISPLAY.get(model, model)}")

    long_df = collect_long_metrics(models)
    summary = build_summary(long_df)
    completeness = build_completeness(long_df, models)

    long_path = OUTPUT_DIR / "dmcr_raw_fix_long_f1_by_sample.tsv"
    summary_path = OUTPUT_DIR / "dmcr_raw_fix_f1_by_sample_summary.tsv"
    completeness_path = OUTPUT_DIR / "dmcr_raw_fix_completeness_check.tsv"
    long_df.to_csv(long_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    completeness.to_csv(completeness_path, sep="\t", index=False)
    print(f"[wrote] {long_path}")
    print(f"[wrote] {summary_path}")
    print(f"[wrote] {completeness_path}")

    incomplete = completeness[~completeness["complete_single_run"]]
    print(
        f"[completeness] {len(incomplete)}/{len(completeness)} model-dataset-sample-stage rows "
        "are missing their expected single RAG-ICL observation."
    )
    print("[completeness-note] RAG-ICL files provide sample_count 1-5 raw/fix pairs; no seed repetitions are expected.")

    if args.draft_model:
        target = args.draft_model
        if target not in models:
            matching = [model for model in models if target.lower() in model.lower() or target.lower() in MODEL_DISPLAY.get(model, "").lower()]
            if len(matching) == 1:
                target = matching[0]
            else:
                raise SystemExit(f"Cannot resolve draft model {args.draft_model!r}; candidates={models}")
        paths = plot_model(summary, target, draft=True)
        for path in paths:
            print(f"[wrote] {path}")
    elif args.all:
        for model in MODEL_ORDER:
            if model in models:
                for path in plot_model(summary, model):
                    print(f"[wrote] {path}")
            else:
                print(f"[missing-model] {model} ({MODEL_DISPLAY.get(model, model)}) not found in source")
    else:
        print("[info] No figures requested. Use --draft-model phi3 or --all.")


if __name__ == "__main__":
    main()
