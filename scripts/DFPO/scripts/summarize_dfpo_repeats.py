from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev, variance

import pandas as pd

from dfpo_common import DATASETS, ROOT, DatasetConfig
from run_dfpo_repeats import parse_repeats


def normalize_entity(value) -> str:
    return str(value).lower()


def metric_ner(test_data: dict, model_output: dict) -> tuple[int, int, int, int]:
    total_tp = total_fp = total_fn = malformed = 0
    for doc_id, predictions in model_output.items():
        true_entities = test_data[doc_id][1]
        if not isinstance(predictions, list):
            malformed += 1
            predictions = []
        true_set = {normalize_entity(entity) for entity in true_entities}
        pred_set = {normalize_entity(entity) for entity in predictions}
        total_tp += len(true_set & pred_set)
        total_fp += len(pred_set - true_set)
        total_fn += len(true_set - pred_set)
    return total_tp, total_fp, total_fn, malformed


def metric_re(test_data: dict, model_output: dict) -> tuple[int, int, int, int]:
    total_tp = total_fp = total_fn = malformed = 0
    for doc_id, predictions in model_output.items():
        ground_truth_set = {(pred[0].lower(), pred[1].lower()) for pred in test_data[doc_id][1]}
        try:
            model_output_set = {(pred[0].lower(), pred[1].lower()) for pred in predictions}
        except Exception:
            malformed += 1
            model_output_set = set()
        total_tp += len(ground_truth_set & model_output_set)
        total_fp += len(model_output_set - ground_truth_set)
        total_fn += len(ground_truth_set - model_output_set)
    return total_tp, total_fp, total_fn, malformed


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def score_repeat(config: DatasetConfig, repeat: int) -> dict:
    output_path = config.output_path(repeat)
    test_data = json.loads((ROOT / config.test_path).read_text())
    model_output = json.loads(output_path.read_text())
    if config.task == "ner":
        tp, fp, fn, malformed = metric_ner(test_data, model_output)
    else:
        tp, fp, fn, malformed = metric_re(test_data, model_output)
    precision, recall, f1 = prf(tp, fp, fn)
    return {
        "dataset": config.name,
        "task": config.task,
        "repeat": repeat,
        "prediction_file": str(output_path.relative_to(ROOT)),
        "records": len(model_output),
        "malformed_outputs": malformed,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarize_group(rows: list[dict]) -> dict:
    out = {
        "dataset": rows[0]["dataset"],
        "task": rows[0]["task"],
        "n_repeats": len(rows),
        "repeats": ",".join(str(row["repeat"]) for row in rows),
        "missing_repeats": "",
    }
    for metric in ["precision", "recall", "f1"]:
        values = [row[metric] for row in rows]
        out[f"{metric}_mean"] = mean(values)
        out[f"{metric}_variance"] = variance(values) if len(values) >= 2 else 0.0
        out[f"{metric}_std"] = stdev(values) if len(values) >= 2 else 0.0
        out[f"{metric}_min"] = min(values)
        out[f"{metric}_max"] = max(values)
    out["total_malformed_outputs"] = sum(row["malformed_outputs"] for row in rows)
    return out


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DFPO repeat prediction metrics.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=sorted(DATASETS))
    parser.add_argument("--repeats", default="1-10")
    parser.add_argument("--out-dir", default="tables")
    args = parser.parse_args()

    repeats = parse_repeats(args.repeats)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    missing_rows: list[dict] = []
    for dataset_name in args.datasets:
        config = DATASETS[dataset_name]
        rows = []
        missing = []
        for repeat in repeats:
            if config.output_path(repeat).exists():
                row = score_repeat(config, repeat)
                detail_rows.append(row)
                rows.append(row)
            else:
                missing.append(repeat)
                missing_rows.append(
                    {
                        "dataset": config.name,
                        "repeat": repeat,
                        "missing_file": str(config.output_path(repeat).relative_to(ROOT)),
                    }
                )
        if rows:
            summary = summarize_group(rows)
            summary["missing_repeats"] = ",".join(map(str, missing))
            summary_rows.append(summary)

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    missing_df = pd.DataFrame(missing_rows)

    detail_path = out_dir / "dfpo_repeat_metrics.csv"
    summary_path = out_dir / "dfpo_repeat_summary.csv"
    missing_path = out_dir / "dfpo_missing_repeats.csv"
    markdown_path = out_dir / "dfpo_repeat_summary.md"

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    missing_df.to_csv(missing_path, index=False)
    markdown_path.write_text(dataframe_to_markdown(summary_df) + "\n")

    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {missing_path}")
    print(f"Wrote {markdown_path}")
    if missing_rows:
        print(f"Missing prediction files: {len(missing_rows)}")


if __name__ == "__main__":
    main()
