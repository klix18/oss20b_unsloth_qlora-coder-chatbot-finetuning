# OSS 20B & Qwen 2.5 SFT Experiments — Training, Evaluation, and Design Notes

This repository documents a series of supervised fine-tuning (SFT) and evaluation experiments conducted on **GPT-OSS-20B** (via Unsloth) and later **Qwen 2.5 Coder** models. The work focuses on practical constraints (VRAM, time), evaluation rigor, and engineering trade-offs rather than chasing idealized benchmarks.

The goal of this README is to give an **honest, engineer-to-engineer account** of:
- what was attempted,
- what worked and didn’t,
- why certain pivots were made,
- and how to interpret the results.

---

## High-Level Timeline

1. **Data preparation & debugging**
   - Dataset inspection, formatting, ChatML alignment
   - Small pilots to validate pipelines

2. **Initial SFT attempt on GPT-OSS-20B**
   - QLoRA fine-tuning under strict VRAM limits (48 GB)
   - One-epoch training on curated instruction/code data

3. **Evaluation pipeline development**
   - Base vs finetuned comparisons
   - Streaming + checkpointed evals with GPT-4o-mini as judge

4. **Realization of OSS limitations**
   - Pilot evals showed weak signal and high variance
   - Alignment / preference data generation deemed unnecessary & noisy

5. **Pivot to Qwen 2.5 Coder**
   - Cleaner behavior, better inductive bias for code
   - Re-run SFT with improved setup and naming

6. **Final evaluations & conclusions**
   - Quantitative + qualitative assessment
   - Honest interpretation of results

---

## Repository Structure

kevin_fine_tune-code-oss20b/
├── data/
│ ├── raw/
│ │ └── evol_instruct_code_80k/
│ └── processed/
│ ├── evol_code_chatml.jsonl
│ ├── evol_code_chatml_5k.jsonl
│ └── alignment_qa.jsonl
│
├── eval/
│ ├── sft_oss_eval.json # original OSS eval
│ ├── sft_oss_eval-2.json # streaming + checkpointed eval
│ └── sft_oss_eval-2.jsonl # per-question streaming output
│
├── runs/
│ ├── oss20b_unsloth_qlora/
│ ├── qwen25coder-14b-sft-pilot/
│ ├── evol_code_chatml_5k_Qwen2.5-Coder-14B-Instruct-*
│ └── misc eval artifacts
│
├── scripts/
│ ├── train_oss_sft.py
│ ├── train_oss_sft_2.py
│ ├── train_qwen2.5_sft.py
│ └── alignment_preference_data_generation_oss.py
│
├── eval_oss_sft.py
├── eval_oss_sft-2.py
├── sft_eval_oss.ipynb
├── sft_eval_qwen.ipynb
├── setup_debug.ipynb
└── README.md

yaml
Copy code

---

## Data Preparation & Debugging

### `setup_debug.ipynb`
This notebook was used to:
- Inspect raw Evol-Instruct / Evol-Code datasets
- Normalize ChatML formatting
- Validate token lengths and truncation behavior
- Ensure compatibility with Unsloth + PEFT

At this stage, **no training decisions were made**—the goal was simply to ensure the pipeline would not fail mid-run.

---

## GPT-OSS-20B Fine-Tuning (QLoRA)

### Why OSS-20B?
- Strong open-weight baseline
- Supported by Unsloth for aggressive memory optimizations
- Feasible under **48 GB VRAM** using 4-bit QLoRA

### Key Constraints
- Single GPU
- VRAM-bound → small batch size + heavy gradient accumulation
- Time-bounded → **1 epoch only**

### Training Configuration (Summary)
- Model: `unsloth/gpt-oss-20b-unsloth-bnb-4bit`
- LoRA:
  - r = 16
  - alpha = 16
  - target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Batch:
  - per-device = 2
  - grad accumulation = 18
- Total steps: 477
- Trainable params: ~8M (~0.04%)

This was **not** intended to produce a fully aligned assistant—only to test whether *any measurable improvement* could be detected under realistic constraints.

---

## Alignment & Preference Data (Abandoned)

An attempt was made to:
- Generate QA and preference data automatically
- Use it for alignment or reward-style fine-tuning

This was **intentionally abandoned** for two reasons:
1. **Generated preference data was unstable and noisy**
2. For SFT comparison, it did not materially improve signal quality

This decision was deliberate: *more data is not better if the signal is weak.*

---

## Evaluation Design

### Why Two Outputs (`.json` + `.jsonl`)?

- **`.jsonl`**  
  - Streaming, append-only
  - One result per question
  - Crash-safe, resumable
- **`.json`**  
  - Aggregated summary
  - Final counts, winner, metadata

This is intentional and standard practice for long-running evals.

### Judge Model
- `gpt-4o-mini`
- Temperature = 0
- Strict JSON schema enforcement

### Key Design Choices
- **Base model loaded once**
- Finetuned model uses PEFT wrapper (no duplicate base weights)
- Deterministic A/B flipping to reduce position bias
- Batched generation + batched judging
- Streaming checkpoints every 10 questions

---

## Evaluation Results (OSS-20B)

Example (from `sft_oss_eval-2.json`):

- Base wins: 46
- Finetuned wins: 48
- Ties: 6
- Final winner: **finetuned (narrow margin)**

### Interpretation (Important)
- The signal is **weak but real**
- Many ties indicate **behavioral similarity**
- Improvements show up mostly in:
  - safer list mutation
  - clearer explanations
  - more defensive coding patterns

This is **exactly what you’d expect** from:
- 1-epoch QLoRA
- limited data
- very large base model

---

## Pivot to Qwen 2.5 Coder

### Why Pivot?
- OSS-20B is expensive to iterate on
- Eval variance was high
- Qwen 2.5 Coder has:
  - stronger code prior
  - cleaner instruction following
  - better SFT efficiency

### Two Qwen Runs
- **Short-named run**  
  - Done via notebook (`sft_eval_qwen.ipynb`)
  - Pilot only
- **Long-named run**  
  - Done via `train_qwen2.5_sft.py`
  - Cleaner, reproducible, better tracked
  - This is the *real* Qwen result

Evaluations were also run for Qwen, following the same base vs SFT comparison logic.

---

## What This Project Demonstrates

- Practical SFT under **real hardware constraints**
- Honest evaluation without cherry-picking
- Willingness to abandon noisy alignment paths
- Engineering-first thinking over benchmark chasing
- Clear separation between:
  - pilots
  - experiments
  - final runs

---

## What This Project Does *Not* Claim

- That OSS-20B is “fully aligned”
- That SFT magically transforms large models
- That preference data generation is trivial
- That win counts alone prove model superiority

---

## Final Takeaway

This repo reflects **how real model iteration actually looks**:
- You test.
- You measure.
- You pivot.
- You simplify.
- You keep what works.

The OSS-20B run validated the pipeline.  
The Qwen 2.5 run delivered cleaner signal.  
The evaluations are honest, reproducible, and transparent.

That was the goal.

— Kevin