from __future__ import annotations

import argparse
import ast
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dfpo_common import DATASETS, DEFAULT_MODEL_PATH, ROOT, DatasetConfig


def parse_repeats(value: str) -> list[int]:
    repeats: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            repeats.extend(range(int(start), int(end) + 1))
        else:
            repeats.append(int(part))
    return sorted(dict.fromkeys(repeats))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(config: DatasetConfig, text: str) -> str:
    if config.task == "ner":
        return (
            f"Extract and list the names of all {config.prompt_label}s mentioned in the following text. "
            f"Provide the output as a single Python list containing the {config.prompt_label} names as strings. \n"
            "Do not output anything except for the extracted information. Do not add any clarifying information.\n\n"
            f"Input: {text}\nOutput:"
        )
    return (
        f"Extract the {config.prompt_label} relations from the following text. Output should be a nested python list.\n"
        "Do not output anything except for the extracted information. Do not add any clarifying information.\n\n"
        f"Input: {text}\nOutput:"
    )


def predict_batch(model, tokenizer, user_inputs: list[str], max_new_tokens: int) -> list[str]:
    texts = []
    for user_input in user_inputs:
        messages = [{"role": "user", "content": user_input}]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        )
    encoded = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            eos_token_id=[tokenizer.eos_token_id],
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    responses = []
    prompt_width = encoded["input_ids"].shape[1]
    for output in outputs:
        responses.append(tokenizer.decode(output[prompt_width:], skip_special_tokens=True))
    return responses


def run_repeat(
    config: DatasetConfig,
    model,
    tokenizer,
    repeat: int,
    *,
    overwrite: bool,
    seed_mode: str,
    max_new_tokens: int,
    limit_docs: int | None,
    batch_size: int,
) -> None:
    output_path = config.output_path(repeat)
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path} already exists")
        return
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")

    if seed_mode == "repeat":
        set_seed(repeat)

    test_data = json.loads((ROOT / config.test_path).read_text())
    items = list(test_data.items())
    if limit_docs is not None:
        items = items[:limit_docs]

    if partial_path.exists() and not overwrite:
        result = json.loads(partial_path.read_text())
        print(f"[resume] {partial_path} with {len(result)} completed records")
    else:
        result = {}
    format_errors = 0
    items = [(key, value) for key, value in items if key not in result]
    progress = tqdm(total=len(items), desc=f"{config.name} repeat {repeat}")
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        prompts = [build_prompt(config, value[0]) for _, value in batch]
        try:
            outputs = predict_batch(model, tokenizer, prompts, max_new_tokens)
        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            torch.cuda.empty_cache()
            outputs = []
            for prompt in prompts:
                outputs.extend(predict_batch(model, tokenizer, [prompt], max_new_tokens))
        for (key, _), data in zip(batch, outputs):
            try:
                result[key] = ast.literal_eval(data)
            except Exception:
                format_errors += 1
                print(f"Format error {key}: {data}")
                result[key] = data
        progress.update(len(batch))
        partial_path.write_text(json.dumps(result, indent=4, ensure_ascii=False))
    progress.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=4, ensure_ascii=False))
    if partial_path.exists():
        partial_path.unlink()
    print(f"[done] wrote {output_path} ({len(result)} records, {format_errors} format errors)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DFPO notebook-style prediction repeats.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=sorted(DATASETS))
    parser.add_argument("--repeats", default="1-10", help="Comma/range syntax, e.g. 2-10 or 1,3,5.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default=None, help="Override dataset default device, e.g. cuda:0.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed-mode", choices=["none", "repeat"], default="none")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-docs", type=int, default=None, help="Smoke-test on the first N documents.")
    args = parser.parse_args()

    repeats = parse_repeats(args.repeats)
    for dataset_name in args.datasets:
        config = DATASETS[dataset_name]
        device = args.device or config.default_device
        print(f"[load] {dataset_name}: {args.model_path} -> {device}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        )
        model.to(device)
        model.eval()
        model.generation_config.disable_compile = True

        for repeat in repeats:
            run_repeat(
                config,
                model,
                tokenizer,
                repeat,
                overwrite=args.overwrite,
                seed_mode=args.seed_mode,
                max_new_tokens=args.max_new_tokens,
                limit_docs=args.limit_docs,
                batch_size=args.batch_size,
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
