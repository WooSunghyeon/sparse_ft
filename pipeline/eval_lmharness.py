#!/usr/bin/env python
"""
MMLU / CSR evaluation using vLLM backend via lm-eval-harness.
8 GPUs via tensor parallel. Results appended to results.md.
"""
import argparse
import json
import logging
import os
import subprocess
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# CSR subtasks (commonsense reasoning)
CSR_TASKS = "hellaswag,winogrande,arc_easy,arc_challenge,piqa,boolq,openbookqa"
MMLU_TASK = "mmlu"


def run_eval(model_path, task, num_fewshot, num_gpus=8):
    """Run lm_eval with vLLM backend."""
    if task == "csr":
        task_str = CSR_TASKS
    elif task == "mmlu":
        task_str = "mmlu"
    else:
        task_str = task

    output_path = os.path.join(model_path, f"eval_{task}.json")

    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", f"pretrained={model_path},tensor_parallel_size={num_gpus},dtype=bfloat16,max_model_len=2048",
        "--tasks", task_str,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", "auto",
        "--output_path", output_path,
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - start

    if result.returncode != 0:
        logger.error(f"lm_eval failed:\n{result.stderr[-2000:]}")
        return None, elapsed

    # Parse results
    logger.info(f"lm_eval completed in {elapsed:.0f}s")
    logger.info(result.stdout[-1000:])

    # Try to find results json
    scores = {}
    for root, dirs, files in os.walk(output_path):
        for f in files:
            if f.endswith(".json") and "results" in f:
                with open(os.path.join(root, f)) as fh:
                    data = json.load(fh)
                if "results" in data:
                    for task_name, metrics in data["results"].items():
                        acc = metrics.get("acc,none", metrics.get("acc_norm,none", metrics.get("acc", 0)))
                        scores[task_name] = acc * 100 if acc < 1 else acc

    if not scores:
        # Try parsing from stdout
        for line in result.stdout.split("\n"):
            if "|" in line and "acc" in line.lower():
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    try:
                        name = parts[1]
                        acc = float(parts[3]) * 100 if float(parts[3]) < 1 else float(parts[3])
                        scores[name] = acc
                    except (ValueError, IndexError):
                        pass

    return scores, elapsed


def append_results(results_file, method, task, scores, elapsed):
    """Append to results.md."""
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, "a") as f:
        if task == "mmlu":
            avg = sum(scores.values()) / len(scores) if scores else 0
            f.write(f"\n## MMLU Results (5-shot)\n\n")
            f.write(f"| Method | Avg | Details |\n|--------|-----|--------|\n")
            detail = ", ".join(f"{k}:{v:.1f}" for k, v in sorted(scores.items())[:5])
            f.write(f"| {method} | {avg:.1f} | {detail} |\n")
        elif task == "csr":
            avg = sum(scores.values()) / len(scores) if scores else 0
            f.write(f"\n## CSR Results (0-shot)\n\n")
            f.write(f"| Method | Avg | Details |\n|--------|-----|--------|\n")
            detail = ", ".join(f"{k}:{v:.1f}" for k, v in sorted(scores.items()))
            f.write(f"| {method} | {avg:.1f} | {detail} |\n")

    # Copy to workspace
    import shutil
    shutil.copy2(results_file, "/workspace/rapa/results.md")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--task", required=True, choices=["mmlu", "csr"])
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--results_file", default="/home1/irteam/rapa/results/results.md")
    p.add_argument("--num_gpus", type=int, default=8)
    args = p.parse_args()

    scores, elapsed = run_eval(args.model_path, args.task, args.num_fewshot, args.num_gpus)
    if scores:
        append_results(args.results_file, args.method, args.task, scores, elapsed)
        logger.info(f"[{args.method}] {args.task} scores: {scores}")
    else:
        logger.error(f"[{args.method}] {args.task} evaluation failed")


if __name__ == "__main__":
    main()
