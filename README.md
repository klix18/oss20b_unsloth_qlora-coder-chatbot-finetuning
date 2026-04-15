# GPT-OSS-20B & Qwen 2.5 Coder — SFT Fine-Tuning for Code Generation

Supervised fine-tuning experiments on **GPT-OSS-20B** (via Unsloth QLoRA) and **Qwen 2.5 Coder 14B** for code instruction-following. This project documents the full iteration cycle — including a deliberate pivot from OSS-20B to Qwen 2.5 when evaluation showed weak signal — emphasizing engineering rigor, honest evaluation, and practical trade-offs under real hardware constraints.

## Motivation

Large code-generation models are expensive to fine-tune and difficult to evaluate meaningfully. This project explores whether QLoRA SFT on instruction/code data can produce measurable improvements on a single 48 GB GPU, and what happens when the answer is "barely." The OSS-20B experiments validated the pipeline; the Qwen 2.5 pivot delivered cleaner signal.

## Timeline & Approach

**Phase 1 — GPT-OSS-20B (QLoRA SFT)**
- Dataset: Evol-Instruct-Code-80k, formatted to ChatML
- Training: 1-epoch QLoRA under 48 GB VRAM constraint (batch=2, grad_accum=18, 477 steps)
- LoRA config: r=16, alpha=16, targeting q/k/v/o/gate/up/down_proj (~8M trainable params, 0.04% of total)
- Result: Narrow win over base (48 vs 46 wins, 6 ties) — signal is real but weak

**Phase 2 — Alignment Data (Abandoned)**
- Attempted synthetic preference data generation for reward modeling
- Abandoned due to noisy, unstable signal — deliberate decision that more data isn't better when the signal is weak

**Phase 3 — Pivot to Qwen 2.5 Coder 14B**
- Stronger code prior and better instruction-following out of the box
- Re-ran SFT with improved setup and naming conventions
- Cleaner evaluation signal and more reproducible results

## Evaluation Design

The evaluation pipeline was designed for reliability over convenience:

- **Judge:** GPT-4o-mini, temperature=0, strict JSON schema
- **Bias mitigation:** Deterministic A/B position flipping to reduce order effects
- **Crash safety:** Dual output format — `.jsonl` (streaming, append-only, resumable) + `.json` (aggregated summary)
- **Memory efficiency:** Base model loaded once; fine-tuned model uses PEFT wrapper (no duplicate weights)
- **Batched generation + batched judging** with streaming checkpoints every 10 questions

## OSS-20B Eval Results

| Metric | Value |
|---|---|
| Base wins | 46 |
| Fine-tuned wins | 48 |
| Ties | 6 |
| Verdict | Fine-tuned (narrow margin) |

Improvements showed up in safer list mutation, clearer explanations, and more defensive coding patterns — consistent with what you'd expect from 1-epoch QLoRA on a very large base model.

## Repository Structure

```
├── scripts/
│   ├── train_oss_sft.py                              # OSS-20B SFT training
│   ├── train_qwen2.5_sft.py                          # Qwen 2.5 Coder SFT training
│   └── alignment_preference_data_generation_oss.py   # Preference data gen (abandoned)
├── data/
│   └── processed/                                     # ChatML-formatted training data
├── eval/
│   ├── sft_oss_eval.json                              # OSS eval (initial)
│   ├── sft_oss_eval-2.json                            # OSS eval (streaming + checkpointed)
│   └── sft_oss_eval-2.jsonl                           # Per-question streaming output
├── runs/                                              # Training run artifacts
│   ├── oss20b_unsloth_qlora/
│   └── qwen25coder-14b-sft-pilot/
├── eval_oss_sft.py          # Eval script v1
├── eval_oss_sft-2.py        # Eval script v2 (streaming + checkpointed)
├── sft_eval_oss.ipynb       # OSS evaluation notebook
├── sft_eval_qwen.ipynb      # Qwen evaluation notebook
├── setup_debug.ipynb        # Data inspection + pipeline validation
├── requirements.txt
└── README.md
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train OSS-20B
python scripts/train_oss_sft.py

# Train Qwen 2.5 Coder
python scripts/train_qwen2.5_sft.py

# Evaluate
python eval_oss_sft-2.py
```

Requires a GPU with ≥48 GB VRAM for OSS-20B, or ≥24 GB for Qwen 2.5 Coder 14B.

## Lessons Learned

- **1-epoch QLoRA on a 20B model under VRAM constraints produces marginal gains.** The signal was real but not compelling. This is a useful negative result — it sets a lower bound on what's achievable with minimal compute.
- **Know when to pivot.** OSS-20B was expensive to iterate on with high eval variance. Qwen 2.5 Coder had a stronger code prior, making SFT more efficient and evaluation cleaner.
- **Abandoning noisy alignment data was the right call.** Preference data generation isn't free — if the base model's outputs are too similar, chosen/rejected pairs degrade into noise. Recognizing this early saved compute and avoided polluting downstream training.
- **Evaluation infrastructure matters.** Streaming checkpoints, crash-safe outputs, and deterministic bias mitigation aren't glamorous, but they're what make results trustworthy.

## Author

Kevin Li
