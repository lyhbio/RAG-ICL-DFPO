#!/usr/bin/env python3
"""Generate raw DERS/FERS ICL prediction JSON files.

This script is a compact, reproducible replacement for the raw-prediction part
of the per-dataset `DERS/main.py` and `FERS/main.py` files used in the original
experiments. It runs inference, writes raw prediction JSON, and reports raw
Precision/Recall/F1. DMCR correction is handled by
`reproduce_icl_dmcr_eval.py`.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


DEFAULT_DATA_ROOT = Path("/mnt/disk1/wanghongyin/work/Paper_Evaluate_Pipeline/code")
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11435"
TIMEOUT_FLAG = "<TIMEOUT>"
ENTITY_TASKS = {"chemical", "disease", "gene"}
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


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


def limit_mapping(data: dict[str, Any], limit: int | None) -> dict[str, Any]:
    if not limit or limit <= 0:
        return data
    return dict(list(data.items())[:limit])


def task_family(task_type: str) -> str:
    normalized = task_type.strip().lower()
    if normalized in RELATION_TASKS:
        return "relation"
    if normalized in ENTITY_TASKS:
        return "entity"
    raise ValueError(
        f"Unknown task type {task_type!r}. Use an entity type such as disease/chemical/gene "
        "or a relation type such as relation/chemical-induced disease/binding interaction."
    )


def build_prompt(task_type: str, query: str, examples: list[Any]) -> str:
    family = task_family(task_type)
    if family == "relation":
        instruction = (
            f"Extract the {task_type} relations from the given text. Below are some examples "
            "demonstrating the input text and the expected output format.\n\n"
        )
        final = (
            f"Now, extract the {task_type} relations from the following text. Do not output "
            "anything except for the extracted information. Do not add any clarifying information.\n\n"
        )
    else:
        task_lower = task_type.lower()
        instruction = (
            f"Your task is to identify and extract {task_lower} entities from the given text. "
            "Below are some examples demonstrating the input text and the expected output format. "
            f"Provide the output as a single Python list containing the {task_lower} as strings.\n\n"
        )
        final = (
            f"Now, extract the {task_lower} entities from the following text. Do not output "
            "anything except for the extracted information. Do not add any clarifying information.\n\n"
        )

    for idx, example in enumerate(examples, start=1):
        instruction += (
            f"### Example {idx}\n"
            f"Input: {example[0]}\n"
            f"Output: {json.dumps(example[1], ensure_ascii=False)}\n\n"
        )
    return instruction + final + f"Input: {query}\nOutput:"


def invoke_with_timeout(llm: Any, prompt: str, doc_id: str, timeout: int) -> str:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(llm.invoke, prompt)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            print(f"Time error {doc_id}")
            return TIMEOUT_FLAG


def parse_response(response: str, doc_id: str) -> Any:
    try:
        return ast.literal_eval(response)
    except Exception:
        print(f"Format error {doc_id}")
        print(response)
        return response


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


def selected_train_keys(method: str, train_keys: list[str], sample: int) -> list[str] | None:
    if method == "FERS":
        return random.sample(train_keys, sample)
    if method == "DERS":
        return None
    raise ValueError(f"Unsupported method {method!r}; expected DERS or FERS.")


def run_method(args: argparse.Namespace, method: str) -> dict[str, Any]:
    dataset_root = args.data_root / args.dataset
    train_data = load_first_json(dataset_root / "Data" / "train_data")
    test_data = limit_mapping(load_first_json(dataset_root / "Data" / "test_data"), args.smoke_limit)
    train_keys = list(train_data.keys())
    if args.sample > len(train_keys):
        raise ValueError(f"sample={args.sample} is larger than train set size {len(train_keys)}")

    # Match the original scripts: each method process starts from the supplied seed.
    seed_everything(args.seed)
    llm = make_ollama(args.model, args.ollama_host, args.temperature)
    fixed_keys = selected_train_keys(method, train_keys, args.sample)
    result: dict[str, Any] = {}

    for doc_id, item in tqdm(test_data.items(), desc=f"{args.dataset} {method} sample={args.sample} rep={args.repetition}"):
        example_keys = fixed_keys if fixed_keys is not None else random.sample(train_keys, args.sample)
        examples = [train_data[key] for key in example_keys]
        prompt = build_prompt(args.task_type, item[0], examples)
        response = invoke_with_timeout(llm, prompt, doc_id, args.timeout)
        result[doc_id] = [] if response == TIMEOUT_FLAG else parse_response(response, doc_id)

    out_dir = args.output_dir / args.dataset / args.model_slug / method / f"{args.sample}_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.repetition}.json"
    with out_path.open("w") as f:
        json.dump(make_json_safe(result), f, indent=4, ensure_ascii=False)
    print(f"[wrote] {out_path}")
    metrics = score_predictions(test_data, result, args.task_type)
    metrics.update(
        {
            "dataset": args.dataset,
            "model": args.model,
            "model_slug": args.model_slug,
            "method": method,
            "sample": args.sample,
            "repetition": args.repetition,
            "seed": args.seed,
            "stage": "raw",
            "json_path": str(out_path),
            "malformed_raw_outputs": sum(1 for value in result.values() if isinstance(value, str)),
        }
    )
    print(
        f"[raw metrics] {method} sample={args.sample} rep={args.repetition} "
        f"TP={metrics['total_tp']} FP={metrics['total_fp']} FN={metrics['total_fn']} "
        f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw DERS/FERS ICL prediction JSON files.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root containing <dataset>/Data/{train_data,test_data}.")
    parser.add_argument("--dataset", required=True, help="Dataset directory name, e.g. NCBI_Disease or DDI.")
    parser.add_argument("--task-type", required=True, help="Entity/relation type used in prompts, e.g. disease, chemical, gene, relation.")
    parser.add_argument("--model", required=True, help="Ollama model name, e.g. qwen2:7b.")
    parser.add_argument("--model-slug", default=None, help="Output directory model name. Defaults to sanitized --model.")
    parser.add_argument("--methods", nargs="+", default=["DERS", "FERS"], choices=["DERS", "FERS"])
    parser.add_argument("--sample", type=int, required=True, choices=range(1, 6), help="Number of in-context examples.")
    parser.add_argument("--repetition", type=int, required=True, help="Repetition index used in the output JSON filename.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for example selection.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/icl_raw"))
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--smoke-limit", type=int, default=None, help="Optional limit on test documents for smoke tests.")
    parser.add_argument("--metrics-tsv", type=Path, default=None, help="Optional TSV path for raw metrics.")
    args = parser.parse_args()
    args.methods = [method.upper() for method in args.methods]
    args.model_slug = args.model_slug or args.model.replace(":", "_").replace("/", "_")
    args.ollama_host = normalize_ollama_host(args.ollama_host)
    return args


def main() -> None:
    args = parse_args()
    rows = []
    for method in args.methods:
        rows.append(run_method(args, method))
    metrics_tsv = args.metrics_tsv or (
        args.output_dir / args.dataset / args.model_slug / f"sample_{args.sample}_rep_{args.repetition}_raw_metrics.tsv"
    )
    metrics_tsv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_tsv, sep="\t", index=False)
    print(f"[wrote] {metrics_tsv}")


if __name__ == "__main__":
    main()
