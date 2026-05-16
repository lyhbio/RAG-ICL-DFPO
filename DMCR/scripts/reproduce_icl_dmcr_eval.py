#!/usr/bin/env python3
"""Apply DMCR-style correction to ICL raw JSON and evaluate raw/fixed outputs."""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


DEFAULT_DATA_ROOT = Path("/mnt/disk1/wanghongyin/work/Paper_Evaluate_Pipeline/code")
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11435"
DEFAULT_DMCR_MODEL = "gemma2:9b"
TIMEOUT_FLAG = "<TIMEOUT>"
RELATION_TASKS = {"relation", "chemical-induced disease", "binding interaction"}


def make_json_safe(obj: Any) -> Any:
    if obj is Ellipsis:
        return None
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, set):
        cleaned = [make_json_safe(x) for x in obj]
        cleaned = [x for x in cleaned if x is not None]
        try:
            return sorted(cleaned, key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True, default=str))
        except Exception:
            return sorted(str(x) for x in cleaned)
    if isinstance(obj, tuple):
        return [x for x in (make_json_safe(v) for v in obj) if x is not None]
    if isinstance(obj, list):
        return [x for x in (make_json_safe(v) for v in obj) if x is not None]
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            safe_key = make_json_safe(key)
            safe_value = make_json_safe(value)
            if safe_key is None or safe_value is None:
                continue
            out[str(safe_key)] = safe_value
        return out
    return str(obj)


def normalize_ollama_host(host: str | None) -> str:
    host = (host or DEFAULT_OLLAMA_HOST).strip() or DEFAULT_OLLAMA_HOST
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host


def make_ollama(model_name: str, ollama_host: str, temperature: float = 0.0):
    from langchain_community.llms import Ollama

    kwargs: dict[str, Any] = {
        "model": model_name,
        "base_url": normalize_ollama_host(ollama_host),
        "temperature": temperature,
    }
    if os.environ.get("OLLAMA_NUM_PREDICT"):
        kwargs["num_predict"] = int(os.environ["OLLAMA_NUM_PREDICT"])
    return Ollama(**kwargs)


def load_first_json(directory: Path) -> dict[str, Any]:
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {directory}")
    with files[0].open() as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def task_family(task_type: str) -> str:
    return "relation" if task_type.strip().lower() in RELATION_TASKS else "entity"


def dmcr_prompt(task_type: str, value: str) -> str:
    if task_family(task_type) == "relation":
        return (
            "Convert the following extracted relations into a Python list format containing "
            'the entity pairs as strings. Each entity pair should be in the format: '
            '[["entity1", "entity2"], ["entity3", "entity4"]]. Only output the list '
            "without any additional information or explanation.\n\nInput relations:"
            + value
        )
    return (
        "Given the following extracted entities, convert them into a Python list format "
        'containing the entities as strings. Example format: ["entity1", "entity2", "entity3"]. '
        "Do not output anything except for the extracted information. Do not add any clarifying "
        "information. Input entities:\n"
        + value
    )


def invoke_with_timeout(llm: Any, prompt: str, doc_id: str, timeout: int) -> str:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(llm.invoke, prompt)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            print(f"Time error {doc_id}")
            return TIMEOUT_FLAG


def apply_dmcr(raw: dict[str, Any], task_type: str, llm: Any, timeout: int) -> tuple[dict[str, Any], int]:
    fixed = dict(raw)
    malformed_count = 0
    for doc_id, value in tqdm(raw.items(), desc="DMCR correction"):
        if not isinstance(value, str):
            continue
        malformed_count += 1
        response = invoke_with_timeout(llm, dmcr_prompt(task_type, value), doc_id, timeout)
        if response == TIMEOUT_FLAG:
            fixed[doc_id] = []
            continue
        try:
            fixed[doc_id] = ast.literal_eval(response)
        except Exception:
            print(f"Format error {doc_id}")
            fixed[doc_id] = response
    return fixed, malformed_count


def normalize_entity_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).lower() for item in value if isinstance(item, str)]


def normalize_relation_list(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append((str(item[0]).lower(), str(item[1]).lower()))
        except Exception:
            continue
    return out


def score_predictions(test_data: dict[str, Any], predictions: dict[str, Any], task_type: str) -> dict[str, float | int]:
    total_tp = total_fp = total_fn = 0
    relation = task_family(task_type) == "relation"
    for doc_id, item in test_data.items():
        gold = item[1]
        pred = predictions.get(doc_id, [])
        if relation:
            gold_set = set(normalize_relation_list(gold))
            pred_set = set(normalize_relation_list(pred))
        else:
            gold_set = set(normalize_entity_list(gold))
            pred_set = set(normalize_entity_list(pred))
        total_tp += len(gold_set & pred_set)
        total_fp += len(pred_set - gold_set)
        total_fn += len(gold_set - pred_set)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply DMCR correction and evaluate ICL raw/fixed JSON.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root containing <dataset>/Data/test_data.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--fixed-json", type=Path, default=None, help="Defaults to fix_<raw-json name> beside raw JSON.")
    parser.add_argument("--metrics-tsv", type=Path, default=None, help="Optional TSV with raw/fixed metrics.")
    parser.add_argument("--dmcr-model", default=DEFAULT_DMCR_MODEL)
    parser.add_argument("--ollama-host", default=os.environ.get("EVALUATOR_OLLAMA_HOST") or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--skip-dmcr", action="store_true", help="Only evaluate existing raw/fixed files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = json.load(args.raw_json.open(), object_pairs_hook=OrderedDict)
    fixed_path = args.fixed_json or (args.raw_json.parent / f"fix_{args.raw_json.name}")

    if args.skip_dmcr and fixed_path.exists():
        fixed = json.load(fixed_path.open(), object_pairs_hook=OrderedDict)
        malformed_count = sum(1 for value in raw.values() if isinstance(value, str))
    else:
        llm = make_ollama(args.dmcr_model, args.ollama_host, args.temperature)
        fixed, malformed_count = apply_dmcr(raw, args.task_type, llm, args.timeout)
        fixed_path.parent.mkdir(parents=True, exist_ok=True)
        with fixed_path.open("w") as f:
            json.dump(make_json_safe(fixed), f, indent=4, ensure_ascii=False)
        print(f"[wrote] {fixed_path}")

    test_data = load_first_json(args.data_root / args.dataset / "Data" / "test_data")
    rows = []
    for stage, preds in [("raw", raw), ("fixed", fixed)]:
        metric = score_predictions(test_data, preds, args.task_type)
        metric.update(
            {
                "dataset": args.dataset,
                "stage": stage,
                "json_path": str(args.raw_json if stage == "raw" else fixed_path),
                "malformed_raw_outputs": malformed_count if stage == "fixed" else "",
            }
        )
        rows.append(metric)
        print(
            f"[{stage}] TP={metric['total_tp']} FP={metric['total_fp']} FN={metric['total_fn']} "
            f"Precision={metric['precision']:.4f} Recall={metric['recall']:.4f} F1={metric['f1']:.4f}"
        )

    metrics_tsv = args.metrics_tsv or (fixed_path.parent / f"{args.raw_json.stem}_raw_fixed_metrics.tsv")
    pd.DataFrame(rows).to_csv(metrics_tsv, sep="\t", index=False)
    print(f"[wrote] {metrics_tsv}")


if __name__ == "__main__":
    main()
