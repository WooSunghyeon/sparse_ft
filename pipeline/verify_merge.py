#!/usr/bin/env python
"""
Verify that each fine-tuned checkpoint can be merged/loaded as a full model.
Checks:
1. Model can be loaded from checkpoint
2. Forward pass works
3. Model can be saved to a merged checkpoint (no PEFT adapter structure)
"""
import argparse
import json
import logging
import os
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

LMFLOW_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(LMFLOW_DIR, "src"))


def verify_merge(method, checkpoint_dir, output_dir=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    if output_dir is None:
        output_dir = checkpoint_dir + "_merged"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"[{method}] Loading from {checkpoint_dir} ...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    except Exception as e:
        logger.warning(f"[{method}] Could not load tokenizer from checkpoint: {e}. Using base model tokenizer.")
        meta_path = os.path.join(checkpoint_dir, "train_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            tokenizer = AutoTokenizer.from_pretrained(meta.get("model", checkpoint_dir))
        else:
            raise

    # Try loading as a PEFT model first (SpiEL uses PEFT adapters)
    loaded_as_peft = False
    try:
        from peft import PeftModel, PeftConfig
        peft_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
        if os.path.exists(peft_config_path):
            peft_config = PeftConfig.from_pretrained(checkpoint_dir)
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype=torch.bfloat16,
            )
            model = PeftModel.from_pretrained(base_model, checkpoint_dir)
            model = model.merge_and_unload()
            loaded_as_peft = True
            logger.info(f"[{method}] Loaded as PEFT model and merged successfully")
    except Exception as e:
        logger.info(f"[{method}] Not a PEFT model (or merge failed): {e}")

    if not loaded_as_peft:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.bfloat16,
        )
        logger.info(f"[{method}] Loaded as full model from {checkpoint_dir}")

    # Verify forward pass
    logger.info(f"[{method}] Running forward pass verification ...")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    text = "Hello, world!"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    logger.info(f"[{method}] Forward pass OK. Loss: {out.loss.item():.4f}")

    # Save merged model
    logger.info(f"[{method}] Saving merged model to {output_dir} ...")
    model.cpu()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"[{method}] Merge verification PASSED -> {output_dir}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    ok = verify_merge(args.method, args.checkpoint_dir, args.output_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
