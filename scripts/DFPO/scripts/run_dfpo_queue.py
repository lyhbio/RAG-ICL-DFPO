from __future__ import annotations

import argparse
import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from dfpo_common import DATASETS
from run_dfpo_repeats import parse_repeats


DEFAULT_PYTHON = "/mnt/nfs/wanghongyin/anaconda3/envs/llama_factory/bin/python"
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Task:
    dataset: str
    repeat: int


def build_tasks(datasets: list[str], repeats: list[int], largest_first: bool) -> list[Task]:
    tasks = [
        Task(dataset=dataset, repeat=repeat)
        for dataset in datasets
        for repeat in repeats
        if not DATASETS[dataset].output_path(repeat).exists()
    ]
    if largest_first:
        size_order = {
            "Chemdner": 3000,
            "BC5CDR_Chemical": 500,
            "BC5CDR_Disease": 500,
            "BC5CDR_RE": 500,
            "DDI": 279,
            "Biorelex": 198,
            "NCBI_Disease": 100,
            "NLM_Gene": 100,
        }
        tasks.sort(key=lambda task: (-size_order.get(task.dataset, 0), task.dataset, task.repeat))
    return tasks


def worker(device: str, task_queue: queue.Queue[Task], args: argparse.Namespace) -> None:
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            return
        print(f"[start] {task.dataset} repeat {task.repeat} on {device}", flush=True)
        cmd = [
            args.python_bin,
            str(SCRIPT_DIR / "run_dfpo_repeats.py"),
            "--datasets",
            task.dataset,
            "--repeats",
            str(task.repeat),
            "--device",
            device,
            "--batch-size",
            str(args.batch_size),
        ]
        if args.seed_mode:
            cmd += ["--seed-mode", args.seed_mode]
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("TORCH_COMPILE_DISABLE", "1")
        subprocess.run(cmd, check=True, env=env)
        print(f"[finish] {task.dataset} repeat {task.repeat} on {device}", flush=True)
        task_queue.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue missing DFPO repeat jobs across GPUs.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=sorted(DATASETS))
    parser.add_argument("--repeats", default="1-10")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed-mode", choices=["none", "repeat"], default="none")
    parser.add_argument("--largest-first", action="store_true")
    args = parser.parse_args()

    tasks = build_tasks(args.datasets, parse_repeats(args.repeats), args.largest_first)
    print(f"[queue] {len(tasks)} missing tasks")
    task_queue: queue.Queue[Task] = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    threads = []
    for device in args.devices:
        thread = threading.Thread(target=worker, args=(device, task_queue, args), daemon=False)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    subprocess.run([args.python_bin, str(SCRIPT_DIR / "summarize_dfpo_repeats.py"), "--repeats", args.repeats], check=True)


if __name__ == "__main__":
    main()
