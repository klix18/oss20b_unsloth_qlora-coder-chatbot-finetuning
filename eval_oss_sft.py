#!/usr/bin/env python3
"""
eval_oss_sft.py

Fast(ish) eval for SFT adapter vs base WITHOUT loading base twice.

Key design choices:
- Load BASE ONCE with Unsloth 4-bit fast path.
- Build FT model via PEFT wrapper (adds only adapter weights, reuses base).
- Batched generation for base + ft.
- Batched judging (1 API call per batch) to avoid 100 sequential judge calls.
- Prints per-question verdicts + final winner; saves full details JSON.

Output JSON:
  /home/jovyan/work/kevin_fine_tune-code-oss20b/eval/sft_oss_eval.json

Env:
  OPENAI_API_KEY required
  Optional: JUDGE_MODEL (default: gpt-4o-mini)

Run:
  python /home/jovyan/work/kevin_fine_tune-code-oss20b/eval_oss_sft.py

Optional flags:
  --n 100
  --batch-size 8
  --verbose 0|1   (verbose prints full answers; default 0 prints short previews)
"""

import os
import json
import time
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch
from unsloth import FastLanguageModel

# Optional .env loading
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# OpenAI judge client
try:
    from openai import OpenAI  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: openai. Install with: pip install openai") from e

# PEFT adapter attach WITHOUT reloading base weights
try:
    from peft import PeftModel  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: peft. Install with: pip install peft") from e


# ----------------------------
# Paths / Config
# ----------------------------
BASE_MODEL = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
ADAPTER_DIR = "/home/jovyan/work/kevin_fine_tune-code-oss20b/runs/oss20b_unsloth_qlora/oss20b_unsloth_qlora/adapter"
OUT_PATH = "/home/jovyan/work/kevin_fine_tune-code-oss20b/eval/sft_oss_eval.json"

MAX_SEQ_LENGTH = 1024

# Generation params
GEN_MAX_NEW_TOKENS = 256
GEN_TEMPERATURE = 0.2
GEN_TOP_P = 0.95
GEN_TOP_K = 0

# Judge model
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

# Deterministic
SEED = 42


# ----------------------------
# Robust JSON extraction
# ----------------------------
def parse_first_valid_json(text: str) -> Optional[Any]:
    """
    Find the first valid JSON object/array within a possibly noisy string.
    Tries to parse balanced {...} OR [...]
    """
    s = text.strip()
    if not s:
        return None

    # Fast path: direct JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    def try_parse_balanced(start_ch: str, end_ch: str) -> Optional[Any]:
        n = len(s)
        for i in range(n):
            if s[i] != start_ch:
                continue
            depth = 0
            in_str = False
            esc = False
            for j in range(i, n):
                ch = s[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == start_ch:
                        depth += 1
                    elif ch == end_ch:
                        depth -= 1
                        if depth == 0:
                            cand = s[i : j + 1]
                            try:
                                return json.loads(cand)
                            except Exception:
                                break
        return None

    # Try object, then array
    obj = try_parse_balanced("{", "}")
    if obj is not None:
        return obj
    arr = try_parse_balanced("[", "]")
    if arr is not None:
        return arr
    return None


def strip_model_artifacts(text: str) -> str:
    # Keep this conservative; prefer decode(skip_special_tokens=True) anyway.
    toks = ["<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|return|>"]
    for t in toks:
        text = text.replace(t, "")
    return text.strip()


def preview(text: str, n: int = 300) -> str:
    t = " ".join(text.strip().split())
    return t if len(t) <= n else t[:n] + "…"


# ----------------------------
# Prompt set (deterministic, expands to N without errors)
# ----------------------------
BASE_QUESTIONS: List[str] = [
    # Python / Core
    "Write a Python function to deduplicate a list while preserving order.",
    "Explain concurrency vs parallelism in Python. When would you use asyncio vs multiprocessing?",
    "Implement a context manager that times a code block and logs milliseconds.",
    "Show how to use dataclasses with type hints for a simple config object.",
    "Write a robust function to parse an int from a string with clear error handling.",
    "Explain the difference between list, tuple, and set and when to use each.",
    "Implement an LRU cache in Python (custom or functools-based).",
    "Write a generator that streams lines from a large file and skips comments.",
    "Explain Python’s GIL and its practical implications.",
    "Demonstrate how to safely handle mutable default arguments.",

    # Async / Concurrency
    "Explain how asyncio scheduling works at a high level.",
    "When should you avoid asyncio?",
    "Compare threading vs multiprocessing in Python.",
    "Explain backpressure in async systems.",
    "How would you cancel and clean up async tasks safely?",

    # HTTP / Backend
    "Write a Python function to retry an HTTP request with exponential backoff and jitter.",
    "Explain idempotency in HTTP APIs and why it matters.",
    "Design a minimal FastAPI endpoint for file upload with validation and size limits.",
    "Explain 401 vs 403 vs 404 with real examples.",
    "What is a circuit breaker and when should you use it?",
    "How would you implement rate limiting for an API?",
    "Explain REST pagination: offset vs cursor.",
    "How do you safely handle user-provided filenames?",
    "What is request tracing and why are correlation IDs useful?",
    "Explain graceful shutdown for a web service.",

    # Data / ETL
    "Design a clean interface for a data-cleaning pipeline using dataclasses and type hints.",
    "Explain how to avoid data leakage in an ML pipeline. Give 3 concrete examples and mitigations.",
    "Implement a streaming JSONL reader that validates required keys and yields typed objects.",
    "Propose a deduplication strategy using MinHash/LSH. Explain parameters and failure modes.",
    "Explain batch vs streaming ETL trade-offs.",
    "How would you design an incremental ETL job (watermarks, idempotency)?",
    "Explain schema evolution and backward compatibility.",
    "How do you validate data at pipeline boundaries?",
    "What is exactly-once vs at-least-once delivery (and why does it matter)?",
    "How would you handle late-arriving data in analytics pipelines?",

    # ML / Modeling
    "Explain look-ahead bias in backtesting and how to avoid it.",
    "What is survivorship bias and why does it matter in equity datasets?",
    "Compare VaR and Expected Shortfall conceptually. When is ES preferred?",
    "Why might a model have high AUC but poor calibration?",
    "What does overfitting look like in time-series forecasting models?",
    "Explain train/val/test splits for time series and why random splitting can be wrong.",
    "Give 3 examples of data leakage and how to prevent each.",
    "What is concept drift and how can you monitor it in production?",
    "Explain the bias-variance tradeoff in practical terms.",
    "When would you retrain vs fine-tune a model?",

    # Systems
    "Explain the difference between a process and a thread.",
    "What is a race condition? Give a concrete example.",
    "Explain optimistic vs pessimistic locking in databases.",
    "How do database transactions work (ACID at a high level)?",
    "Explain deadlocks and how to prevent them.",
    "What is load shedding and when is it appropriate?",
    "Explain message queues vs pub/sub; when would you choose each?",
    "What is a thundering herd problem and how do you mitigate it?",
    "Explain horizontal vs vertical scaling with trade-offs.",
    "What is eventual consistency and where is it used?",

    # SQL / Storage
    "Write SQL to compute daily active users from an events table (user_id, ts, event_type).",
    "Explain INNER JOIN vs LEFT JOIN with examples.",
    "What does database normalization aim to achieve?",
    "How do indexes speed up queries and when can they hurt?",
    "Explain composite indexes and leftmost-prefix rules.",
    "What is an execution plan and how do you use it to debug slow queries?",
    "Explain window functions with one example (ROW_NUMBER or SUM OVER).",
    "What is a primary key vs unique constraint?",
    "How would you model soft deletes and their query implications?",
    "Explain isolation levels at a high level (read committed vs serializable).",

    # Testing / Quality
    "Compare unit tests vs integration tests.",
    "Explain property-based testing with Hypothesis; show a tiny example.",
    "How do you test code that calls external HTTP APIs?",
    "What is a golden test and what are its trade-offs?",
    "Explain flaky tests and how to reduce them.",
    "What should you mock and what should you avoid mocking?",
    "Explain test isolation and why it matters.",
    "How would you test async code in Python (pytest-asyncio etc.)?",
    "What is snapshot testing and when is it useful?",
    "How do you structure tests for a CLI tool (argparse/typer)?",
]

TEMPLATES = [
    "Explain {concept} in simple terms, then give a concise Python example.",
    "Give a production checklist for {concept}.",
    "What are 3 common pitfalls for {concept}, and how do you avoid them?",
    "Show a minimal, production-ready example of {concept} with type hints.",
    "Compare {a} vs {b}. When would you choose each?",
    "Write a short production-minded explanation of {concept} and include a tiny code snippet.",
]

CONCEPTS = [
    "rate limiting",
    "structured logging",
    "graceful shutdown",
    "retry policies",
    "pagination (cursor vs offset)",
    "feature flags",
    "caching (TTL vs LRU)",
    "database transactions",
    "schema migrations",
    "message queues",
    "idempotent consumers",
    "dead-letter queues",
    "data validation",
    "PII redaction",
    "deduplication",
    "backpressure",
    "async I/O",
    "multiprocessing",
    "thread safety",
    "testing flaky code",
    "configuration management",
    "dependency injection",
    "observability (metrics vs logs vs traces)",
    "load shedding",
    "streaming vs batch ETL",
    "exactly-once vs at-least-once delivery",
    "JWT verification and rotation",
    "secrets management",
    "blue/green deployments",
    "canary releases",
]

COMPARISONS = [
    ("threads", "asyncio"),
    ("multiprocessing", "multithreading"),
    ("REST", "GraphQL"),
    ("SQL", "NoSQL"),
    ("unit tests", "property-based tests"),
    ("batch processing", "stream processing"),
    ("push", "pull"),
    ("JWT", "opaque session tokens"),
    ("ORM", "raw SQL"),
    ("monolith", "microservices"),
]


def build_questions(n: int, seed: int = SEED) -> List[str]:
    random.seed(seed)
    pool = list(dict.fromkeys(BASE_QUESTIONS))  # de-dupe, preserve order

    i = 0
    while len(pool) < n:
        c = CONCEPTS[i % len(CONCEPTS)]
        t = TEMPLATES[i % len(TEMPLATES)]
        if "{concept}" in t:
            pool.append(t.format(concept=c))
        else:
            a, b = COMPARISONS[i % len(COMPARISONS)]
            pool.append(t.format(a=a, b=b))
        if i % 3 == 0:
            a, b = COMPARISONS[(i // 3) % len(COMPARISONS)]
            pool.append(TEMPLATES[4].format(a=a, b=b))
        pool = list(dict.fromkeys(pool))
        i += 1

    return pool[:n]


# ----------------------------
# Fast batched generation (Unsloth)
# ----------------------------
@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_seq_length: int,
) -> List[str]:
    msgs_list = [[{"role": "user", "content": p}] for p in prompts]
    chats = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in msgs_list
    ]

    enc = tokenizer(
        chats,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only continuation
    results: List[str] = []
    for i in range(out.size(0)):
        in_len = int(attention_mask[i].sum().item())
        cont_ids = out[i, in_len:]
        txt = tokenizer.decode(cont_ids, skip_special_tokens=True)
        results.append(txt.strip())
    return results


# ----------------------------
# Judge (batched)
# ----------------------------
JUDGE_SYSTEM = (
    "You are an expert software engineer evaluating two model answers.\n"
    "Prefer correctness, safety, completeness, clarity, and Python best practices.\n"
    "Be strict.\n"
    "Return ONLY valid JSON.\n"
)

def build_judge_payload(items: List[Dict[str, str]]) -> str:
    """
    items: [{id, prompt, A, B}, ...]
    """
    return json.dumps(
        {
            "instructions": (
                "Evaluate each item independently. Prefer correctness, safety, completeness, clarity.\n"
                "Return JSON exactly as:\n"
                "{\n"
                '  "items": [\n'
                '    {"id": "...", "winner": "A"|"B"|"tie", "scores": {"A":0-10,"B":0-10}, "justification":"1-3 sentences"}\n'
                "  ]\n"
                "}\n"
                "No extra keys. No markdown."
            ),
            "items": items,
        },
        ensure_ascii=False,
        indent=0,
    )

def judge_batch(
    client: OpenAI,
    items: List[Dict[str, str]],
    model: str,
) -> List[Dict[str, Any]]:
    """
    Returns list aligned to `items` order: [{"id", "winner", "scores", "justification"}, ...]
    """
    content = build_judge_payload(items)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": content},
        ],
    )

    raw = (resp.choices[0].message.content or "").strip()
    parsed = parse_first_valid_json(raw)
    if not isinstance(parsed, dict) or "items" not in parsed or not isinstance(parsed["items"], list):
        raise ValueError(f"Judge did not return expected JSON.\nRAW:\n{raw}")

    out_items = parsed["items"]
    # Index by id for robustness
    by_id: Dict[str, Dict[str, Any]] = {}
    for o in out_items:
        if not isinstance(o, dict):
            continue
        _id = o.get("id")
        if isinstance(_id, str):
            by_id[_id] = o

    results: List[Dict[str, Any]] = []
    for it in items:
        _id = it["id"]
        o = by_id.get(_id)
        if not o:
            raise ValueError(f"Judge missing item id={_id}. RAW:\n{raw}")
        winner = o.get("winner")
        scores = o.get("scores")
        if winner not in ("A", "B", "tie"):
            raise ValueError(f"Judge invalid winner for id={_id}: {winner}. RAW:\n{raw}")
        if not isinstance(scores, dict) or "A" not in scores or "B" not in scores:
            raise ValueError(f"Judge invalid scores for id={_id}: {scores}. RAW:\n{raw}")
        results.append(o)

    return results


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="Number of prompts to evaluate (default 100).")
    ap.add_argument("--batch-size", type=int, default=8, help="Generation batch size (default 8).")
    ap.add_argument("--verbose", type=int, default=0, help="Print full answers (1) or short previews (0).")
    args = ap.parse_args()

    n_prompts = int(args.n)
    gen_batch_size = int(args.batch_size)
    verbose = bool(args.verbose)

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Speed knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")
    judge_client = OpenAI(api_key=api_key)

    questions = build_questions(n_prompts, seed=SEED)

    print("\n" + "=" * 120)
    print("EVAL: Base vs LoRA (NO double base load) | batched generation + batched judging")
    print(f"BASE_MODEL  : {BASE_MODEL}")
    print(f"ADAPTER_DIR : {ADAPTER_DIR}")
    print(f"JUDGE_MODEL : {JUDGE_MODEL}")
    print(f"N           : {len(questions)}")
    print(f"BATCH_SIZE  : {gen_batch_size}")
    print("=" * 120 + "\n")

    # Load BASE ONCE
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        load_in_4bit=True,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
    )

    # Tokenizer tweaks for decoder-only batching (usually helps)
    try:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception:
        pass

    # Prepare base inference
    base_model = FastLanguageModel.for_inference(base_model)

    # Attach adapter WITHOUT reloading base weights
    ft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    ft_model = FastLanguageModel.for_inference(ft_model)

    results: List[Dict[str, Any]] = []
    win_counts = {"base": 0, "ft": 0, "tie": 0}

    t0 = time.time()

    # Evaluate in batches
    for start in range(0, len(questions), gen_batch_size):
        batch_q = questions[start : start + gen_batch_size]

        base_outs = generate_batch(
            base_model, tokenizer, batch_q,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            top_k=GEN_TOP_K,
            max_seq_length=MAX_SEQ_LENGTH,
        )
        ft_outs = generate_batch(
            ft_model, tokenizer, batch_q,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            top_k=GEN_TOP_K,
            max_seq_length=MAX_SEQ_LENGTH,
        )

        # Build judge items (randomize A/B per item, deterministically)
        judge_items: List[Dict[str, str]] = []
        presentations: List[Dict[str, str]] = []

        for i, q in enumerate(batch_q):
            base_txt = strip_model_artifacts(base_outs[i])
            ft_txt = strip_model_artifacts(ft_outs[i])

            # Deterministic flip
            # (use a seeded RNG stream: stable across runs)
            flip = random.random() < 0.5
            if not flip:
                A_is, A_txt = "base", base_txt
                B_is, B_txt = "ft", ft_txt
            else:
                A_is, A_txt = "ft", ft_txt
                B_is, B_txt = "base", base_txt

            q_idx = start + i + 1
            item_id = f"{q_idx:04d}"

            judge_items.append({
                "id": item_id,
                "prompt": q,
                "A": A_txt,
                "B": B_txt,
            })
            presentations.append({
                "id": item_id,
                "A_is": A_is,
                "B_is": B_is,
                "base": base_txt,
                "ft": ft_txt,
                "prompt": q,
                "idx": q_idx,
            })

        # Batched judge call
        verdicts = judge_batch(judge_client, judge_items, model=JUDGE_MODEL)

        # Consume verdicts (aligned by id)
        for pres, verdict in zip(presentations, verdicts):
            winner = verdict["winner"]
            if winner == "tie":
                winner_model = "tie"
            elif winner == "A":
                winner_model = pres["A_is"]
            else:
                winner_model = pres["B_is"]

            if winner_model == "base":
                win_counts["base"] += 1
            elif winner_model == "ft":
                win_counts["ft"] += 1
            else:
                win_counts["tie"] += 1

            # Print (fast mode by default)
            print("\n" + "-" * 120)
            print(f"Q{pres['idx']:03d}/{len(questions)} | winner={winner_model.upper()} | scores={verdict.get('scores')}")
            print(f"PROMPT: {pres['prompt']}")

            if verbose:
                print("\n[BASE]\n" + pres["base"])
                print("\n[FINETUNED]\n" + pres["ft"])
            else:
                print("\nBASE(preview): " + preview(pres["base"]))
                print("\nFT  (preview): " + preview(pres["ft"]))
            print("\nJUDGE: " + str(verdict))

            results.append({
                "idx": pres["idx"],
                "id": pres["id"],
                "prompt": pres["prompt"],
                "base": pres["base"],
                "finetuned": pres["ft"],
                "judge": verdict,
                "presentation": {"A_is": pres["A_is"], "B_is": pres["B_is"]},
                "winner_model": winner_model,
            })

    dt = time.time() - t0

    # Final winner
    if win_counts["ft"] > win_counts["base"]:
        final_winner = "finetuned"
    elif win_counts["base"] > win_counts["ft"]:
        final_winner = "base"
    else:
        final_winner = "tie"

    print("\n" + "#" * 120)
    print(f"FINAL WINNER: {final_winner.upper()}")
    print(f"COUNTS: base={win_counts['base']} | finetuned={win_counts['ft']} | tie={win_counts['tie']}")
    print(f"ELAPSED: {dt:.2f}s")
    print("#" * 120 + "\n")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    payload = {
        "summary": {
            "final_winner": final_winner,
            "counts": win_counts,
            "n_questions": len(questions),
            "base_model": BASE_MODEL,
            "adapter_dir": ADAPTER_DIR,
            "judge_model": JUDGE_MODEL,
            "gen": {
                "max_seq_length": MAX_SEQ_LENGTH,
                "max_new_tokens": GEN_MAX_NEW_TOKENS,
                "temperature": GEN_TEMPERATURE,
                "top_p": GEN_TOP_P,
                "top_k": GEN_TOP_K,
                "batch_size": gen_batch_size,
                "seed": SEED,
            },
            "elapsed_seconds": dt,
            "notes": [
                "Base loaded once; FT is PEFT wrapper around same base weights.",
                "Judging is batched to reduce API calls.",
                "Use --verbose 1 to print full answers (slower).",
            ],
        },
        "results": results,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved eval JSON to: {OUT_PATH}")


if __name__ == "__main__":
    main()
