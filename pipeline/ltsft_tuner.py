"""LT-SFT (Lottery Ticket Sparse Fine-Tuning) — DeepSpeed 8-GPU compatible.

Random sparse element selection (lottery ticket style).
Uses buffer-based frozen weights + Parameter delta, same pattern as SIFT/SMT/S2FT.
Target trainable params: ~170M.
"""
import logging
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

from lmflow.pipeline.rapa.sift_tuner import SparseLinear  # noqa: E402


class TextDataset(TorchDataset):
    def __init__(self, instances, tokenizer, max_len):
        self.data = []
        for item in instances:
            text = item.get("text", "")
            enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].squeeze()
            self.data.append({"input_ids": ids, "labels": ids.clone()})
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def train_ltsft(
    model_name_or_path, dataset_path, output_dir, num_train_epochs=1, max_steps=-1,
    per_device_train_batch_size=1, gradient_accumulation_steps=1, learning_rate=5e-5,
    lr_scheduler_type="linear", max_seq_length=512, target_params=170_000_000,
    bf16=True, hf_token=None, seed=42, report_to="none", **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    tok_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch.bfloat16 if bf16 else torch.float32, **tok_kwargs)

    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and any(t in n for t in target_modules)]
    total_target = sum(m.weight.numel() for _, m in layers)
    sparse_rate = min(target_params / total_target, 1.0) if total_target > 0 else 0.025
    logger.info(f"[LT-SFT] {len(layers)} layers, sparse_rate={sparse_rate:.4f}")

    for i, (name, module) in enumerate(layers):
        train_num = max(1, int(module.weight.numel() * sparse_rate))
        flat_idx = torch.randint(0, module.weight.numel(), (train_num,), dtype=torch.long)
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], SparseLinear(module, flat_idx))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[LT-SFT] trainable={trainable:,}")

    with open(dataset_path) as f:
        raw = json.load(f)
    train_dataset = TextDataset(raw.get("instances", []), tokenizer, max_seq_length)

    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=num_train_epochs, max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size, gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate, lr_scheduler_type=lr_scheduler_type, bf16=bf16,
        save_strategy="no" if 0 < max_steps < 100 else "epoch", logging_steps=5,
        report_to=report_to, seed=seed, dataloader_num_workers=4, remove_unused_columns=False,
        deepspeed="/home1/irteam/rapa/LMFlow/configs/rapa/ds_zero1_sift.json",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=tokenizer)
    trainer.train(resume_from_checkpoint=get_last_checkpoint(output_dir))

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, SparseLinear):
                df = torch.zeros(m.weight.numel(), dtype=m.weight.dtype, device=m.weight.device)
                df.scatter_(0, m.flat_idx, m.sparse_delta.data.to(m.weight.dtype))
                m.weight.data += df.view(m.weight.shape)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"[LT-SFT] Model saved to {output_dir}")
    return output_dir
