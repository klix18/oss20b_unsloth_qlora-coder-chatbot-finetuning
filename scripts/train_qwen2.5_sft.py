#!/usr/bin/env python3
"""
train_sft.py

Portfolio-quality SFT script for ChatML JSONL -> Qwen2.5-Coder-14B-Instruct (4-bit NF4) + QLoRA (PEFT)
using plain Transformers Trainer (no TRL notebook dependencies).

Expected dataset format (one JSON per line):
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

Examples:
  # Pilot (5k) quick run
  python train_sft.py train \
    --dataset_path data/processed/evol_code_chatml_5k.jsonl \
    --model_name_or_path Qwen/Qwen2.5-Coder-14B-Instruct \
    --seq_len 2048 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --num_train_epochs 1 --learning_rate 2e-4 \
    --output_dir runs

  # Full run
  python train_sft.py train \
    --dataset_path data/processed/evol_code_chatml.jsonl \
    --output_dir runs --run_name evol80k_qwen25coder14b_qlora

  # Resume from latest checkpoint inside the run folder
  python train_sft.py train \
    --dataset_path data/processed/evol_code_chatml.jsonl \
    --resume latest \
    --output_dir runs --run_name evol80k_qwen25coder14b_qlora

  # Quick compare: base vs adapter
  python train_sft.py infer \
    --base_model Qwen/Qwen2.5-Coder-14B-Instruct \
    --adapter_dir runs/<your_run>/adapter \
    --prompt "Write a Python function to deduplicate a list while preserving order."
"""

from __future__ import annotations

import os
import re
import json
import math
import time
import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)


# ----------------------------
# Logging
# ----------------------------

def setup_logging(log_level: str = "INFO") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


LOGGER = logging.getLogger("train_sft")


# ----------------------------
# Utilities
# ----------------------------

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_run_name(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", name)
    return name[:120] if len(name) > 120 else name


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_latest_checkpoint(run_dir: str) -> Optional[str]:
    """
    Find latest Trainer checkpoint in run_dir (checkpoint-XXXX).
    Returns path or None.
    """
    if not os.path.isdir(run_dir):
        return None
    candidates = []
    for d in os.listdir(run_dir):
        if d.startswith("checkpoint-"):
            m = re.match(r"checkpoint-(\d+)$", d)
            if m:
                step = int(m.group(1))
                candidates.append((step, os.path.join(run_dir, d)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


# ----------------------------
# Data pipeline
# ----------------------------

def _format_chatml_example(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Turn [{"role":..., "content":...}, ...] into a single chat-formatted text using the model's chat template.
    We include BOTH user + assistant turns in the training text.
    """
    # Some tokenizers support apply_chat_template; Qwen2.5 does.
    # We DO NOT add a generation prompt during training; we want full supervised text.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_dataset(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    seq_len: int,
    eval_ratio: float,
    seed: int,
    num_proc: int = 4,
    max_rows: Optional[int] = None,
) -> DatasetDict:
    """
    Loads JSONL, formats with chat template, tokenizes with truncation to seq_len,
    and creates labels = input_ids (later padded to -100 by collator).
    """
    LOGGER.info(f"Loading dataset: {dataset_path}")
    ds = load_dataset("json", data_files=dataset_path, split="train")

    if max_rows is not None and max_rows > 0:
        max_rows = min(max_rows, len(ds))
        LOGGER.info(f"Subsetting dataset to first {max_rows} rows (for debugging)")
        ds = ds.select(range(max_rows))

    # Split
    if eval_ratio <= 0.0:
        dsd = DatasetDict(train=ds)
    else:
        dsd = ds.train_test_split(test_size=eval_ratio, seed=seed)
        # rename for Trainer expectations
        dsd = DatasetDict(train=dsd["train"], eval=dsd["test"])

    LOGGER.info(f"Split sizes: train={len(dsd['train'])} eval={len(dsd.get('eval', [])) if 'eval' in dsd else 0}")

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        # batch["messages"] is a list of lists (when batched) or list (single)
        texts = []
        for msgs in batch["messages"]:
            texts.append(_format_chatml_example(tokenizer, msgs))

        toks = tokenizer(
            texts,
            truncation=True,
            max_length=seq_len,
            padding=False,
            return_attention_mask=True,
        )

        # labels = input_ids copy; padding handled later
        toks["labels"] = [ids.copy() for ids in toks["input_ids"]]
        return toks

    LOGGER.info("Tokenizing (chat template -> tokens -> labels)...")
    dsd = dsd.map(
        preprocess,
        batched=True,
        num_proc=num_proc,
        remove_columns=[c for c in dsd["train"].column_names if c not in ("messages",)],
        desc="Preprocess",
    )

    return dsd


# ----------------------------
# Collator
# ----------------------------

@dataclass
class CausalLMCollator:
    """
    Pads input_ids/attention_mask/labels to longest in batch.
    Labels padding becomes -100 so it doesn't contribute to loss.
    """
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        # Determine max len (+ round up)
        max_len = max(x.size(0) for x in input_ids)
        if self.pad_to_multiple_of is not None:
            m = self.pad_to_multiple_of
            max_len = int(math.ceil(max_len / m) * m)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # Qwen tokenizers usually have pad; if not, fall back to eos
            pad_id = self.tokenizer.eos_token_id

        def pad_1d(x: torch.Tensor, pad_value: int) -> torch.Tensor:
            if x.size(0) == max_len:
                return x
            return torch.cat([x, torch.full((max_len - x.size(0),), pad_value, dtype=x.dtype)], dim=0)

        batch_input_ids = torch.stack([pad_1d(x, pad_id) for x in input_ids], dim=0)
        batch_attention = torch.stack([pad_1d(x, 0) for x in attention_mask], dim=0)
        batch_labels = torch.stack([pad_1d(x, -100) for x in labels], dim=0)

        # Also mask out label positions where input is padding (extra safety)
        batch_labels = torch.where(batch_attention.bool(), batch_labels, torch.full_like(batch_labels, -100))

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": batch_labels,
        }


# ----------------------------
# Model (QLoRA)
# ----------------------------

def load_model_and_tokenizer(
    model_name_or_path: str,
    use_4bit: bool,
    fp16: bool,
    gradient_checkpointing: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Optional[List[str]] = None,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    LOGGER.info(f"Loading tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        # Qwen usually has pad; if not, set to eos.
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if use_4bit:
        compute_dtype = torch.float16 if fp16 else torch.bfloat16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    LOGGER.info(f"Loading base model: {model_name_or_path} (4bit={use_4bit})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=(torch.float16 if fp16 else torch.bfloat16) if not use_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )

    # QLoRA prep
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if target_modules is None:
        # Good default for Qwen2.5 decoder blocks
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    # Print trainable params
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    return model, tokenizer


# ----------------------------
# Trainer args compatibility
# ----------------------------

def make_training_args(**kwargs) -> TrainingArguments:
    """
    Some envs use eval_strategy instead of evaluation_strategy.
    This helper tries both.
    """
    try:
        return TrainingArguments(**kwargs)
    except TypeError as e:
        msg = str(e)
        if "evaluation_strategy" in msg and "eval_strategy" not in kwargs:
            kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
            return TrainingArguments(**kwargs)
        raise


# ----------------------------
# Train
# ----------------------------

def train_cmd(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    set_seed(args.seed)

    # Resolve run dir
    run_name = args.run_name
    if not run_name:
        base = sanitize_run_name(args.model_name_or_path.split("/")[-1])
        data_tag = sanitize_run_name(os.path.splitext(os.path.basename(args.dataset_path))[0])
        run_name = f"{data_tag}_{base}_qlora_{now_stamp()}"
    else:
        run_name = sanitize_run_name(run_name)

    run_dir = os.path.join(args.output_dir, run_name)
    ensure_dir(run_dir)

    LOGGER.info(f"Run dir: {run_dir}")

    # Load model/tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        use_4bit=not args.no_4bit,
        fp16=args.fp16,
        gradient_checkpointing=not args.no_grad_ckpt,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )

    # Data
    dsd = build_dataset(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        num_proc=args.num_proc,
        max_rows=args.max_rows,
    )

    collator = CausalLMCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    # Eval strategy
    do_eval = "eval" in dsd and len(dsd["eval"]) > 0
    evaluation_strategy = "steps" if (do_eval and args.eval_steps > 0) else "no"

    # Steps math for logging
    steps_per_epoch = math.ceil(len(dsd["train"]) / (args.per_device_train_batch_size * max(1, args.world_size)) / args.gradient_accumulation_steps)
    LOGGER.info(
        f"Approx steps/epoch (world_size={args.world_size}): {steps_per_epoch} | "
        f"Total steps ~ {int(steps_per_epoch * args.num_train_epochs)}"
    )

    # TrainingArguments
    ta_kwargs = dict(
        output_dir=run_dir,
        run_name=run_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=False,  # keep off in this environment per your note
        report_to=args.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
    )

    if do_eval:
        ta_kwargs.update(
            dict(
                evaluation_strategy=evaluation_strategy,
                eval_steps=args.eval_steps,
                load_best_model_at_end=False,
            )
        )
    else:
        ta_kwargs.update(dict(evaluation_strategy="no"))

    training_args = make_training_args(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["eval"] if do_eval else None,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # Resume handling
    resume_path = None
    if args.resume:
        if args.resume.lower() == "latest":
            resume_path = find_latest_checkpoint(run_dir)
            if resume_path is None:
                LOGGER.warning("No checkpoint found to resume from; starting fresh.")
        else:
            resume_path = args.resume
            if not os.path.isdir(resume_path):
                raise FileNotFoundError(f"--resume path not found: {resume_path}")

    LOGGER.info(f"Starting training (resume_from_checkpoint={resume_path})")
    train_result = trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_state()

    # Save adapter + tokenizer (portfolio-friendly layout)
    adapter_dir = os.path.join(run_dir, "adapter")
    ensure_dir(adapter_dir)

    LOGGER.info(f"Saving adapter to: {adapter_dir}")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Save a tiny metadata json for reproducibility
    meta = {
        "run_name": run_name,
        "run_dir": run_dir,
        "model_name_or_path": args.model_name_or_path,
        "dataset_path": args.dataset_path,
        "seq_len": args.seq_len,
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "use_4bit": (not args.no_4bit),
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        },
        "training": {
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
        },
        "final": {
            "global_step": int(train_result.global_step) if hasattr(train_result, "global_step") else None,
            "train_runtime_sec": float(train_result.metrics.get("train_runtime", 0.0)) if hasattr(train_result, "metrics") else None,
            "train_loss": float(train_result.metrics.get("train_loss", float("nan"))) if hasattr(train_result, "metrics") else None,
        },
        "timestamp": now_stamp(),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    LOGGER.info("Done.")


# ----------------------------
# Inference / compare base vs adapter
# ----------------------------

@torch.no_grad()
def generate_one(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        use_cache=True,
    )
    decoded = tokenizer.decode(out[0], skip_special_tokens=False)

    # Best-effort: strip everything before assistant start if template includes it
    # (Different chat templates differ; keep simple + robust.)
    return decoded


def infer_cmd(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)

    # Base tokenizer/model
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    LOGGER.info(f"Loading base model (4-bit): {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    ).eval()

    LOGGER.info("Generating with base...")
    base_out = generate_one(
        model=base,
        tokenizer=tok,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Adapter model
    LOGGER.info(f"Loading adapter: {args.adapter_dir}")
    adapted = PeftModel.from_pretrained(base, args.adapter_dir).eval()

    LOGGER.info("Generating with adapter...")
    adapt_out = generate_one(
        model=adapted,
        tokenizer=tok,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n" + "=" * 80)
    print("PROMPT:\n" + args.prompt)
    print("=" * 80)
    print("\n[BASE OUTPUT]\n")
    print(base_out)
    print("\n" + "-" * 80)
    print("\n[ADAPTER OUTPUT]\n")
    print(adapt_out)
    print("\n" + "=" * 80 + "\n")


# ----------------------------
# CLI
# ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT QLoRA Trainer for ChatML JSONL -> Qwen2.5-Coder-14B-Instruct")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Train
    t = sub.add_parser("train", help="Train QLoRA adapter with Transformers Trainer")
    t.add_argument("--dataset_path", type=str, required=True, help="Path to ChatML JSONL (full or pilot).")
    t.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-Coder-14B-Instruct")
    t.add_argument("--output_dir", type=str, default="runs", help="Base folder for runs.")
    t.add_argument("--run_name", type=str, default="", help="Optional fixed run name (no timestamp unless you include it).")
    t.add_argument("--resume", type=str, default="", help="Path to checkpoint dir, or 'latest' to auto-resume inside run_dir.")

    # Data / tokenize
    t.add_argument("--seq_len", type=int, default=2048)
    t.add_argument("--eval_ratio", type=float, default=0.02)
    t.add_argument("--num_proc", type=int, default=4)
    t.add_argument("--max_rows", type=int, default=0, help="Debug only: limit rows (0 = no limit).")

    # Training
    t.add_argument("--per_device_train_batch_size", type=int, default=1)
    t.add_argument("--per_device_eval_batch_size", type=int, default=1)
    t.add_argument("--gradient_accumulation_steps", type=int, default=16)
    t.add_argument("--num_train_epochs", type=float, default=1.0)
    t.add_argument("--learning_rate", type=float, default=2e-4)
    t.add_argument("--warmup_ratio", type=float, default=0.03)
    t.add_argument("--lr_scheduler_type", type=str, default="cosine")
    t.add_argument("--weight_decay", type=float, default=0.0)
    t.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging / checkpointing / eval
    t.add_argument("--logging_steps", type=int, default=10)
    t.add_argument("--save_steps", type=int, default=50)
    t.add_argument("--save_total_limit", type=int, default=3)
    t.add_argument("--eval_steps", type=int, default=50)

    # QLoRA / performance
    t.add_argument("--no_4bit", action="store_true", help="Disable 4-bit loading (not recommended for 14B on 48GB).")
    t.add_argument("--fp16", action="store_true", help="Use FP16 compute (recommended).")
    t.add_argument("--no_grad_ckpt", action="store_true", help="Disable gradient checkpointing.")
    t.add_argument("--lora_r", type=int, default=16)
    t.add_argument("--lora_alpha", type=int, default=32)
    t.add_argument("--lora_dropout", type=float, default=0.0)
    t.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="*",
        default=None,
        help="Override LoRA target modules (space-separated). Default is good for Qwen2.5.",
    )

    # System
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--log_level", type=str, default="INFO")
    t.add_argument("--report_to", type=str, default="none", help="Set to 'wandb' if you want; default none.")
    t.add_argument("--dataloader_num_workers", type=int, default=2)
    t.add_argument("--optim", type=str, default="adamw_torch")

    # If running DDP yourself, pass this for accurate step estimates (Trainer still works without it)
    t.add_argument("--world_size", type=int, default=1)

    # Infer
    inf = sub.add_parser("infer", help="Quick generation compare: base vs adapter")
    inf.add_argument("--base_model", type=str, required=True)
    inf.add_argument("--adapter_dir", type=str, required=True)
    inf.add_argument("--prompt", type=str, required=True)
    inf.add_argument("--max_new_tokens", type=int, default=256)
    inf.add_argument("--temperature", type=float, default=0.2)
    inf.add_argument("--top_p", type=float, default=0.95)
    inf.add_argument("--log_level", type=str, default="INFO")

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    # Normalize max_rows
    if hasattr(args, "max_rows") and args.max_rows and args.max_rows > 0:
        pass
    elif hasattr(args, "max_rows"):
        args.max_rows = None

    # Normalize resume flag
    if hasattr(args, "resume") and args.resume:
        args.resume = args.resume.strip()

    if args.cmd == "train":
        # Default fp16 ON (per your environment note)
        if not args.fp16:
            # If user didn't pass --fp16, still default to True for safety.
            args.fp16 = True
        train_cmd(args)
    elif args.cmd == "infer":
        infer_cmd(args)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
