#!/usr/bin/env python3
"""
finetune_unsloth_oss20b.py

Unsloth QLoRA finetune script (single-file, hardcoded params).
- Uses Unsloth FastLanguageModel for 4-bit loading + fast training
- Trains LoRA adapters (QLoRA) on your ChatML JSONL dataset
- Designed so you can just:  python3 finetune_unsloth_oss20b.py

Dataset format (JSONL):
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

Before running (once):
  pip install -U unsloth trl transformers datasets accelerate bitsandbytes

Run:
  python3 finetune_unsloth_oss20b.py
"""

import os
import json
import builtins
import psutil

builtins.psutil = psutil  # <-- MUST be before importing unsloth/trl

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch
from datasets import load_dataset

from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer

builtins.psutil = psutil

# =========================
# HARD-CODED CONFIG
# =========================

# Pick ONE of these:
# - "unsloth/gpt-oss-20b-unsloth-bnb-4bit"  -> best for QLoRA training (bnb 4bit)
# - "unsloth/gpt-oss-20b"                   -> MXFP4-based variant (often not trainable the same way)
MODEL_NAME = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"

DATASET_PATH = "/home/jovyan/work/kevin_fine_tune-code-oss20b/data/processed/evol_code_chatml.jsonl"

OUTPUT_DIR = "runs/oss20b_unsloth_qlora"
RUN_NAME = "oss20b_unsloth_qlora"

SEED = 42

# Sequence length
MAX_SEQ_LENGTH = 768  # match your screenshot; if OOM, drop to 768 or 512

# QLoRA load
LOAD_IN_4BIT = True
DTYPE = None  # None = autodetect (bf16 if supported, else fp16)

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_BIAS = "none"

# Target modules (as per screenshot)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Unsloth extras
USE_GRADIENT_CHECKPOINTING = "unsloth"  # True or "unsloth"
RANDOM_STATE = 3407
USE_RSLORA = False
LOFTQ_CONFIG = None

# Train hyperparams
NUM_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 18
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
LR_SCHEDULER = "cosine"
LOGGING_STEPS = 10
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3

# Eval split
EVAL_RATIO = 0.02  # set 0.0 to disable
EVAL_STEPS = 400

# Optional: limit rows for a quick smoke test (None = full)
MAX_ROWS: Optional[int] = 17500


# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("finetune_unsloth")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_run_meta(path: str, meta: Dict[str, Any]) -> None:
    with open(os.path.join(path, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def format_chatml_with_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    # For training: do NOT add_generation_prompt; we want full supervised text
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main() -> None:
    torch.manual_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_dir = os.path.join(OUTPUT_DIR, RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)

    LOGGER.info(f"Run dir: {run_dir}")
    LOGGER.info(f"Loading model/tokenizer via Unsloth: {MODEL_NAME}")

    # 1) Load model with Unsloth 4-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        dtype=DTYPE,                       # None for auto
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        full_finetuning=False,             # LoRA / QLoRA path
        # token="hf_...",                  # if gated
    )

    # 2) Add LoRA adapters (Unsloth optimized)
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        random_state=RANDOM_STATE,
        use_rslora=USE_RSLORA,
        loftq_config=LOFTQ_CONFIG,
    )

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    # 3) Load your ChatML JSONL dataset
    LOGGER.info(f"Loading dataset: {DATASET_PATH}")
    ds = load_dataset("json", data_files=DATASET_PATH, split="train")

    if MAX_ROWS is not None:
        ds = ds.select(range(min(MAX_ROWS, len(ds))))
        LOGGER.info(f"Subsetting to MAX_ROWS={len(ds)}")

    if EVAL_RATIO and EVAL_RATIO > 0:
        split = ds.train_test_split(test_size=EVAL_RATIO, seed=SEED)
        train_ds, eval_ds = split["train"], split["test"]
        LOGGER.info(f"Split sizes: train={len(train_ds)} eval={len(eval_ds)}")
    else:
        train_ds, eval_ds = ds, None
        LOGGER.info(f"Split sizes: train={len(train_ds)} eval=0")

    # 4) Convert messages -> single training text using the model chat template
    def to_text(example: Dict[str, Any]) -> Dict[str, Any]:
        return {"text": format_chatml_with_template(tokenizer, example["messages"])}

    LOGGER.info("Formatting dataset with tokenizer chat template...")
    train_ds = train_ds.map(to_text, remove_columns=train_ds.column_names)
    if eval_ds is not None:
        eval_ds = eval_ds.map(to_text, remove_columns=eval_ds.column_names)

    # 5) Trainer args
    bf16 = is_bfloat16_supported()
    fp16 = not bf16

    args = TrainingArguments(
        output_dir=run_dir,
        run_name=RUN_NAME,

        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,

        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,

        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
    
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=EVAL_STEPS if eval_ds is not None else None,


        bf16=bf16,
        fp16=fp16,

        # Big VRAM win for QLoRA:
        optim="paged_adamw_8bit",

        report_to="none",
        remove_unused_columns=False,
    )

    # 6) SFTTrainer (TRL) — trains on dataset_text_field="text"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=args,
        packing=False,  # safer for ChatML; set True only if you know you want packing
    )

    LOGGER.info("Starting training...")
    trainer.train()

    # 7) Save adapters
    adapter_dir = os.path.join(run_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)

    LOGGER.info(f"Saving LoRA adapter to: {adapter_dir}")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Save metadata
    meta = {
        "model_name": MODEL_NAME,
        "dataset_path": DATASET_PATH,
        "max_seq_length": MAX_SEQ_LENGTH,
        "train": {
            "epochs": NUM_EPOCHS,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "grad_accum": GRAD_ACCUM_STEPS,
            "lr": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "scheduler": LR_SCHEDULER,
            "optim": "paged_adamw_8bit",
            "bf16": bf16,
            "fp16": fp16,
        },
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "bias": LORA_BIAS,
            "target_modules": LORA_TARGET_MODULES,
        },
        "eval_ratio": EVAL_RATIO,
        "timestamp": now_stamp(),
    }
    save_run_meta(run_dir, meta)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
