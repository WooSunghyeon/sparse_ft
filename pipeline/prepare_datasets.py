#!/usr/bin/env python
"""
Prepare all datasets in LMFlow format.
- oasst1 (timdettmers/openassistant-guanaco) for MT-Bench methods
- MMLU from /home1/irteam/datasets/mmlu/mmlu.json (already in LMFlow format)
- CSR from /home1/irteam/datasets/merge/merge.json (already in LMFlow format)
"""
import json
import os
import sys
from datasets import load_dataset

RAPA_BASE = "/home1/irteam/rapa"
DATA_DIR = os.path.join(RAPA_BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def prepare_oasst1():
    """Convert timdettmers/openassistant-guanaco to LMFlow text_only format."""
    out_path = os.path.join(DATA_DIR, "oasst1_lmflow.json")
    if os.path.exists(out_path):
        print(f"[oasst1] Already exists: {out_path}")
        return out_path

    print("[oasst1] Downloading timdettmers/openassistant-guanaco ...")
    ds = load_dataset("timdettmers/openassistant-guanaco", split="train")

    instances = []
    for item in ds:
        text = item.get("text", "")
        if text:
            instances.append({"text": text})

    lmflow_data = {"type": "text_only", "instances": instances}
    with open(out_path, "w") as f:
        json.dump(lmflow_data, f, ensure_ascii=False, indent=2)
    print(f"[oasst1] Saved {len(instances)} instances to {out_path}")
    return out_path


def check_existing(path, name):
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        n = len(d.get("instances", []))
        print(f"[{name}] {path}: {n} instances, type={d.get('type','?')}")
    else:
        print(f"[{name}] NOT FOUND: {path}")


if __name__ == "__main__":
    oasst1_path = prepare_oasst1()
    print(f"\n[oasst1] Path: {oasst1_path}")

    mmlu_path = "/home1/irteam/datasets/mmlu/mmlu.json"
    csr_path = "/home1/irteam/datasets/merge/merge.json"
    check_existing(mmlu_path, "MMLU")
    check_existing(csr_path, "CSR")

    # Write paths config
    paths = {
        "oasst1": oasst1_path,
        "mmlu": mmlu_path,
        "csr": csr_path,
    }
    with open(os.path.join(DATA_DIR, "dataset_paths.json"), "w") as f:
        json.dump(paths, f, indent=2)
    print(f"\n[paths] Saved to {DATA_DIR}/dataset_paths.json")
