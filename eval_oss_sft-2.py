#!/usr/bin/env python3
"""
eval_oss_sft-2.py

Eval SFT adapter vs base WITHOUT loading base twice.

Changes vs previous:
- NO requirement for 100 questions (no assert).
- Streaming:
  - Append per-question results to JSONL
  - Write full checkpoint JSON every 10 completed questions (atomic write)

Outputs:
- OUT_JSON  : /home/jovyan/work/kevin_fine_tune-code-oss20b/eval/sft_oss_eval-2.json
- OUT_JSONL : /home/jovyan/work/kevin_fine_tune-code-oss20b/eval/sft_oss_eval-2.jsonl

Env:
  OPENAI_API_KEY required
  Optional: JUDGE_MODEL (default: gpt-4o-mini)

Run:
  python /home/jovyan/work/kevin_fine_tune-code-oss20b/eval_oss_sft-2.py --n 95
"""

import os
import json
import time
import argparse
import hashlib
from typing import Any, Dict, List, Optional

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

OUT_JSON = "/home/jovyan/work/kevin_fine_tune-code-oss20b/eval/sft_oss_eval-2.json"
OUT_JSONL = "/home/jovyan/work/kevin_fine_tune-code-oss20b/eval/sft_oss_eval-2.jsonl"

MAX_SEQ_LENGTH = 1536

# Generation params
GEN_MAX_NEW_TOKENS = 700
GEN_TEMPERATURE = 0.25
GEN_TOP_P = 0.90
GEN_TOP_K = 0

# Judge model
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

# Deterministic
SEED = 42

# Completion terminator
END_TOKEN = "<<END>>"

# Streaming checkpoint cadence
CHECKPOINT_EVERY = 10


# ----------------------------
# Questions (no fixed length required)
# ----------------------------
QUESTIONS_2: List[str] = [
    # Python correctness + robustness
       # Python correctness + robustness (1–20)
    "Write a Python function `chunked(iterable, n)` that yields lists of size n. Handle n<=0 safely. End with <<END>>.",
    "Explain `*args` vs `**kwargs` with a concrete example and one common pitfall. End with <<END>>.",
    "Show how to implement a dataclass with validation in `__post_init__` (include type hints). End with <<END>>.",
    "Write a safe `parse_bool(s: str) -> bool` that handles common user inputs and raises ValueError otherwise. End with <<END>>.",
    "Explain why mutable default arguments are dangerous, and show the correct pattern. End with <<END>>.",
    "Implement `safe_get(d, path, default)` where path is like 'a.b.c'. Include tests (tiny). End with <<END>>.",
    "Explain `is` vs `==` with one example where `is` is wrong. End with <<END>>.",
    "Write a context manager that temporarily sets an env var and restores it. End with <<END>>.",
    "Explain Python’s exception chaining (`raise ... from ...`) and why it helps debugging. End with <<END>>.",
    "Implement a small `Result[T]` style pattern (Ok/Err) using typing. End with <<END>>.",
    "Explain how `__repr__` should differ from `__str__` and show a good `__repr__`. End with <<END>>.",
    "Write `dedupe_by_key(items, key_fn)` preserving order and explain complexity. End with <<END>>.",
    "Explain `@staticmethod` vs `@classmethod` with a real use case. End with <<END>>.",
    "Implement a tiny CLI using argparse that supports `--input`, `--output`, and `--dry-run`. End with <<END>>.",
    "Explain how to design custom exceptions for a library (hierarchy + message). End with <<END>>.",
    "Write `retry(fn, tries, base_delay, max_delay)` with exponential backoff + jitter. End with <<END>>.",
    "Explain the difference between logging and printing in production code. End with <<END>>.",
    "Show a clean pattern for resource cleanup without a context manager (try/finally). End with <<END>>.",
    "Write `atomic_write_text(path, content)` that avoids partial writes. End with <<END>>.",
    "Explain how Python hashing impacts dict/set performance and one security reason hashes are salted. End with <<END>>.",

    # Async + concurrency (21–35)
    "Explain the difference between I/O-bound and CPU-bound tasks and what to use in Python for each. End with <<END>>.",
    "Write a minimal asyncio example that runs 20 tasks with a concurrency limit of 5. End with <<END>>.",
    "Explain why calling `requests.get()` inside asyncio code is a bug and show the correct alternative at a high level. End with <<END>>.",
    "Implement a thread-safe counter with `threading.Lock`, and explain what race it prevents. End with <<END>>.",
    "Explain deadlocks in simple terms and list 3 practical prevention strategies. End with <<END>>.",
    "Show a producer/consumer queue example using `queue.Queue` with graceful shutdown. End with <<END>>.",
    "Explain why multiprocessing can be slower for small tasks and what overhead dominates. End with <<END>>.",
    "Write a small example using `concurrent.futures` that times out slow tasks safely. End with <<END>>.",
    "Explain cancellation in asyncio and why you must handle `asyncio.CancelledError`. End with <<END>>.",
    "Explain backpressure with a concrete pipeline example (not just definition). End with <<END>>.",
    "Show how to rate-limit async calls (token bucket or semaphore) with code. End with <<END>>.",
    "Explain the “thundering herd” problem and 3 mitigations. End with <<END>>.",
    "Explain what “structured concurrency” means and how it improves correctness. End with <<END>>.",
    "Write an async context manager (async with) that opens/closes a resource. End with <<END>>.",
    "Explain starvation vs fairness in scheduling with one real example. End with <<END>>.",

    # HTTP / APIs / reliability (36–55)
    "Explain idempotency keys with a payments example, and what to store server-side. End with <<END>>.",
    "Compare PUT vs PATCH with an example payload for each. End with <<END>>.",
    "Design an API error format (JSON) that’s developer-friendly and consistent. End with <<END>>.",
    "Explain why retries can cause duplicate side effects and how to design around it. End with <<END>>.",
    "Explain timeouts: connect vs read vs total timeout, and a sane default strategy. End with <<END>>.",
    "Write a minimal FastAPI endpoint that validates JSON input with Pydantic and returns typed errors. End with <<END>>.",
    "Explain 401 vs 403 vs 404 vs 409 with one example each. End with <<END>>.",
    "Explain how to do safe logging of requests without leaking secrets/PII. End with <<END>>.",
    "Explain circuit breakers and when to use them vs just retries. End with <<END>>.",
    "Explain pagination: offset vs cursor and why cursor is better at scale. End with <<END>>.",
    "Explain caching: TTL vs LRU (simple), and give a concise Python example for each. End with <<END>>.",
    "Explain webhooks vs polling and how to verify webhook authenticity. End with <<END>>.",
    "Explain why “at-least-once” delivery is common and how consumers must behave. End with <<END>>.",
    "Write a tiny example of an idempotent message handler (dedupe by message_id). End with <<END>>.",
    "Explain how to roll out risky API changes safely (versioning, compat, canary). End with <<END>>.",
    "Explain observability: logs vs metrics vs traces, and what each is best for. End with <<END>>.",
    "Explain correlation IDs and how they flow through services. End with <<END>>.",
    "Explain graceful shutdown in a web service and what should happen on SIGTERM. End with <<END>>.",
    "Explain rate limiting strategies (fixed window vs sliding vs token bucket) and tradeoffs. End with <<END>>.",
    "Explain what makes a good health check vs a bad one in Kubernetes. End with <<END>>.",

    # Data pipelines / dedupe / PII (56–75)
    "Design a JSONL schema for cleaned text documents with dedupe + PII flags. End with <<END>>.",
    "Explain 3 pitfalls of deduplication (false pos/neg, normalization, scaling) and mitigations. End with <<END>>.",
    "Explain what MinHash is used for (intuition), and what parameters control recall/precision. End with <<END>>.",
    "Write a streaming JSONL reader that validates required keys and yields dicts; don’t load whole file. End with <<END>>.",
    "Explain schema migrations: 3 common pitfalls and safe rollout patterns. End with <<END>>.",
    "Give a production checklist for PII redaction (detection, review, audit, regression tests). End with <<END>>.",
    "Explain why regex-only PII detection fails and how to combine approaches safely. End with <<END>>.",
    "Explain “exactly-once” vs “effectively-once” processing and how idempotency makes it practical. End with <<END>>.",
    "Explain late-arriving data and watermarking with one example. End with <<END>>.",
    "Compare batch vs stream processing and when each is the right tool. End with <<END>>.",
    "Explain data validation at boundaries and how to fail fast without losing debuggability. End with <<END>>.",
    "Write a minimal typed “record validator” function that returns a list of errors. End with <<END>>.",
    "Explain how to design a rollback-safe ETL deployment (shadow runs, canary, checks). End with <<END>>.",
    "Explain lineage and why it matters for debugging ML/data issues. End with <<END>>.",
    "Explain how to handle partial failures in a pipeline (DLQ, replay, checkpoints). End with <<END>>.",
    "Explain why “append-only” logs are helpful for auditing. End with <<END>>.",
    "Show a minimal example of atomic checkpointing for a batch job (write tmp then rename). End with <<END>>.",
    "Explain what makes a good dedupe key vs a bad one with examples. End with <<END>>.",
    "Explain how to measure dedup quality without labels (spot checks, sampling, heuristics). End with <<END>>.",
    "Explain config management pitfalls (drift, secrets, env mismatch) and avoidances. End with <<END>>.",

    # Systems / DB / queues (76–100)
    "Explain ACID transactions in simple terms with one real failure scenario they prevent. End with <<END>>.",
    "Explain isolation levels (read committed vs repeatable read vs serializable) in practical terms. End with <<END>>.",
    "Explain optimistic vs pessimistic locking, and when each breaks down. End with <<END>>.",
    "Explain indexes and when adding an index can make things worse. End with <<END>>.",
    "Write SQL to compute a 7-day rolling average of daily signups. End with <<END>>.",
    "Explain window functions with one concrete example query. End with <<END>>.",
    "Compare SQL vs NoSQL and choose one for: (a) payments ledger, (b) clickstream. Explain why. End with <<END>>.",
    "Explain dead-letter queues and what metadata you should attach to DLQ messages. End with <<END>>.",
    "Explain message ordering guarantees and why you often can’t rely on ordering. End with <<END>>.",
    "Explain how to design consumers to be idempotent and re-entrant. End with <<END>>.",
    "Explain what backpressure looks like in Kafka-like systems and how to respond. End with <<END>>.",
    "Explain how load balancers distribute traffic and what sticky sessions do. End with <<END>>.",
    "Explain retries vs hedged requests and when hedging is dangerous. End with <<END>>.",
    "Explain cache stampede prevention (locking, jitter, stale-while-revalidate). End with <<END>>.",
    "Explain why distributed locks are hard and when you should avoid them. End with <<END>>.",
    "Explain eventual consistency with a user-visible example (e.g., social feed). End with <<END>>.",
    "Explain blue/green vs canary deployments and how to pick between them. End with <<END>>.",
    "Give a production checklist for testing flaky code (determinism, time control, isolation). End with <<END>>.",
    "Explain the difference between mocks, fakes, and stubs with one example each. End with <<END>>.",
    "Explain why “coverage” isn’t quality, and what to measure instead. End with <<END>>.",
]


# ----------------------------
# Small utilities
# ----------------------------
def parse_first_valid_json(text: str) -> Optional[Any]:
    s = (text or "").strip()
    if not s:
        return None
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

    obj = try_parse_balanced("{", "}")
    if obj is not None:
        return obj
    arr = try_parse_balanced("[", "]")
    if arr is not None:
        return arr
    return None


def cut_at_end_token(text: str, end_token: str = END_TOKEN) -> str:
    t = (text or "").strip()
    if not t:
        return t
    idx = t.find(end_token)
    if idx >= 0:
        t = t[:idx].rstrip()
    return t


def preview(text: str, n: int = 320) -> str:
    t = " ".join((text or "").strip().split())
    return t if len(t) <= n else t[:n] + "…"


def stable_flip(seed: int, item_id: str) -> bool:
    h = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).digest()
    return (h[0] % 2) == 1


def atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_jsonl(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ----------------------------
# Generation (Unsloth)
# ----------------------------
ANSWER_SYSTEM = (
    "You are a senior software engineer teaching a CS student.\n"
    "Rules:\n"
    "1) Be correct and production-minded.\n"
    "2) Explain briefly, then show a clear example.\n"
    "3) If code is requested, include it.\n"
    f"4) ALWAYS finish with the exact token {END_TOKEN} on its own line.\n"
)

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
    msgs_list = [
        [{"role": "system", "content": ANSWER_SYSTEM},
         {"role": "user", "content": p}]
        for p in prompts
    ]
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
        eos_token_id=tokenizer.eos_token_id,
    )

    results: List[str] = []
    for i in range(out.size(0)):
        in_len = int(attention_mask[i].sum().item())
        cont_ids = out[i, in_len:]
        txt = tokenizer.decode(cont_ids, skip_special_tokens=True)
        txt = cut_at_end_token(txt, END_TOKEN)
        results.append(txt.strip())
    return results


# ----------------------------
# Judge (batched)
# ----------------------------
JUDGE_SYSTEM = (
    "You are an expert software engineer evaluating two answers.\n"
    "Goal: pick the answer that is BEST for a CS student AND most production-ready.\n"
    "Be strict and decisive.\n"
    "Only output valid JSON.\n"
)

def build_judge_payload(items: List[Dict[str, str]]) -> str:
    return json.dumps(
        {
            "instructions": (
                "Evaluate each item independently.\n"
                "Rubric:\n"
                "- Correctness & completeness (45%)\n"
                "- Production-minded quality (25%): safe defaults, edge cases, idiomatic, avoids footguns\n"
                "- Teaching quality (25%): clear mental model, short example, explains 'why'\n"
                "- Organization/clarity (5%)\n"
                "Ties are allowed ONLY if answers are materially indistinguishable.\n"
                "Return JSON exactly:\n"
                "{\n"
                '  "items": [\n'
                '    {"id":"...","winner":"A"|"B"|"tie","scores":{"A":0-10,"B":0-10},"justification":"1-3 sentences"}\n'
                "  ]\n"
                "}\n"
                "No extra keys. No markdown."
            ),
            "items": items,
        },
        ensure_ascii=False,
        indent=0,
    )

def judge_batch(client: OpenAI, items: List[Dict[str, str]], model: str) -> List[Dict[str, Any]]:
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

    by_id: Dict[str, Dict[str, Any]] = {}
    for o in parsed["items"]:
        if isinstance(o, dict) and isinstance(o.get("id"), str):
            by_id[o["id"]] = o

    out: List[Dict[str, Any]] = []
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
        out.append(o)
    return out


def maybe_double_judge(
    client: OpenAI,
    items: List[Dict[str, str]],
    verdicts: List[Dict[str, Any]],
    model: str,
    enabled: bool,
) -> List[Dict[str, Any]]:
    if not enabled:
        return verdicts

    swapped = [{"id": it["id"], "prompt": it["prompt"], "A": it["B"], "B": it["A"]} for it in items]
    verdicts2 = judge_batch(client, swapped, model=model)
    by_id2 = {v["id"]: v for v in verdicts2}

    fixed: List[Dict[str, Any]] = []
    for v in verdicts:
        v2 = by_id2.get(v["id"])
        if not v2:
            fixed.append(v)
            continue

        orig, swap = v["winner"], v2["winner"]
        consistent = True
        if orig == "A" and swap not in ("B", "tie"):
            consistent = False
        if orig == "B" and swap not in ("A", "tie"):
            consistent = False

        if consistent:
            fixed.append(v)
        else:
            fixed.append({
                "id": v["id"],
                "winner": "tie",
                "scores": {"A": v["scores"]["A"], "B": v["scores"]["B"]},
                "justification": (v.get("justification", "") + " [unstable on A/B swap -> forced tie]").strip(),
            })
    return fixed


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="How many prompts to run (0 = all available).")
    ap.add_argument("--gen-batch", type=int, default=6, help="Generation batch size.")
    ap.add_argument("--judge-batch", type=int, default=4, help="Judge batch size per API call.")
    ap.add_argument("--verbose", type=int, default=0, help="Print full answers (1) or previews (0).")
    ap.add_argument("--double-judge", type=int, default=0, help="Stability check by swapping A/B (cost ~2x).")
    args = ap.parse_args()

    gen_batch_size = max(1, int(args.gen_batch))
    judge_batch_size = max(1, int(args.judge_batch))
    verbose = bool(args.verbose)
    double_judge = bool(args.double_judge)

    # Determine N
    available = len(QUESTIONS_2)
    if args.n and args.n > 0:
        N = min(int(args.n), available)
    else:
        N = available

    questions = QUESTIONS_2[:N]

    # Determinism
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

    print("\n" + "=" * 120)
    print("EVAL: Base vs LoRA (NO double base load) | STREAM every question + checkpoint/10")
    print(f"BASE_MODEL     : {BASE_MODEL}")
    print(f"ADAPTER_DIR    : {ADAPTER_DIR}")
    print(f"JUDGE_MODEL    : {JUDGE_MODEL}")
    print(f"N              : {N} (available: {available})")
    print(f"GEN_BATCH      : {gen_batch_size}")
    print(f"JUDGE_BATCH    : {judge_batch_size}")
    print(f"MAX_NEW_TOKENS : {GEN_MAX_NEW_TOKENS}")
    print(f"OUT_JSON       : {OUT_JSON}")
    print(f"OUT_JSONL      : {OUT_JSONL}")
    print("=" * 120 + "\n")

    # Start fresh JSONL (optional: comment out if you want append-resume behavior)
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        f.write("")

    # Load BASE ONCE
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        load_in_4bit=True,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
    )

    # Tokenizer tweaks
    try:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception:
        pass

    base_model = FastLanguageModel.for_inference(base_model)

    # Attach adapter WITHOUT reloading base weights
    ft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    ft_model = FastLanguageModel.for_inference(ft_model)

    results: List[Dict[str, Any]] = []
    win_counts = {"base": 0, "ft": 0, "tie": 0}

    t0 = time.time()
    completed = 0

    def build_payload(is_final: bool) -> Dict[str, Any]:
        if win_counts["ft"] > win_counts["base"]:
            final_winner = "finetuned"
        elif win_counts["base"] > win_counts["ft"]:
            final_winner = "base"
        else:
            final_winner = "tie"

        return {
            "summary": {
                "final_winner": final_winner,
                "counts": win_counts,
                "n_questions": N,
                "base_model": BASE_MODEL,
                "adapter_dir": ADAPTER_DIR,
                "judge_model": JUDGE_MODEL,
                "gen": {
                    "max_seq_length": MAX_SEQ_LENGTH,
                    "max_new_tokens": GEN_MAX_NEW_TOKENS,
                    "temperature": GEN_TEMPERATURE,
                    "top_p": GEN_TOP_P,
                    "top_k": GEN_TOP_K,
                    "gen_batch_size": gen_batch_size,
                    "judge_batch_size": judge_batch_size,
                    "seed": SEED,
                    "end_token": END_TOKEN,
                },
                "elapsed_seconds": time.time() - t0,
                "is_final": is_final,
                "streaming": {
                    "jsonl_path": OUT_JSONL,
                    "checkpoint_every": CHECKPOINT_EVERY,
                },
            },
            "results": results,
            "questions": questions,
        }

    # Evaluate in generation batches
    for start in range(0, N, gen_batch_size):
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

        # Build judge items (deterministic flip per id)
        judge_items_all: List[Dict[str, str]] = []
        pres_all: List[Dict[str, Any]] = []

        for i, q in enumerate(batch_q):
            q_idx = start + i + 1
            item_id = f"{q_idx:04d}"

            base_txt = base_outs[i].strip()
            ft_txt = ft_outs[i].strip()

            flip = stable_flip(SEED, item_id)
            if not flip:
                A_is, A_txt = "base", base_txt
                B_is, B_txt = "ft", ft_txt
            else:
                A_is, A_txt = "ft", ft_txt
                B_is, B_txt = "base", base_txt

            pres_all.append({
                "idx": q_idx,
                "id": item_id,
                "prompt": q,
                "base": base_txt,
                "finetuned": ft_txt,
                "A_is": A_is,
                "B_is": B_is,
                "flip": flip,
            })
            judge_items_all.append({
                "id": item_id,
                "prompt": q,
                "A": A_txt,
                "B": B_txt,
            })

        # Judge in smaller batches for reliability
        verdict_by_id: Dict[str, Dict[str, Any]] = {}
        for jstart in range(0, len(judge_items_all), judge_batch_size):
            jbatch = judge_items_all[jstart : jstart + judge_batch_size]
            verdicts = judge_batch(judge_client, jbatch, model=JUDGE_MODEL)
            verdicts = maybe_double_judge(judge_client, jbatch, verdicts, model=JUDGE_MODEL, enabled=double_judge)
            for v in verdicts:
                verdict_by_id[v["id"]] = v

        # Consume results one-by-one so we can stream per question
        for pres in pres_all:
            verdict = verdict_by_id[pres["id"]]
            w = verdict["winner"]
            if w == "tie":
                winner_model = "tie"
            elif w == "A":
                winner_model = pres["A_is"]
            else:
                winner_model = pres["B_is"]

            if winner_model == "base":
                win_counts["base"] += 1
            elif winner_model == "ft":
                win_counts["ft"] += 1
            else:
                win_counts["tie"] += 1

            row = {
                "idx": pres["idx"],
                "id": pres["id"],
                "prompt": pres["prompt"],
                "base": pres["base"],
                "finetuned": pres["finetuned"],
                "judge": verdict,
                "presentation": {"A_is": pres["A_is"], "B_is": pres["B_is"], "flip": pres["flip"]},
                "winner_model": winner_model,
            }

            results.append(row)
            append_jsonl(OUT_JSONL, row)  # stream per-question

            completed += 1

            print("\n" + "-" * 120)
            print(f"Q{pres['idx']:03d}/{N} | winner={winner_model.upper()} | scores={verdict.get('scores')}")
            print(f"PROMPT: {pres['prompt']}")
            if verbose:
                print("\n[BASE]\n" + pres["base"])
                print("\n[FINETUNED]\n" + pres["finetuned"])
            else:
                print("\nBASE(preview): " + preview(pres["base"]))
                print("\nFT  (preview): " + preview(pres["finetuned"]))
            print("\nJUDGE: " + str(verdict))

            # checkpoint every 10
            if completed % CHECKPOINT_EVERY == 0:
                atomic_write_json(OUT_JSON, build_payload(is_final=False))
                print(f"[checkpoint] wrote {completed}/{N} to {OUT_JSON}")

    # Final write
    atomic_write_json(OUT_JSON, build_payload(is_final=True))

    # Final winner print
    payload = build_payload(is_final=True)
    print("\n" + "#" * 120)
    print(f"FINAL WINNER: {payload['summary']['final_winner'].upper()}")
    print(f"COUNTS: base={win_counts['base']} | finetuned={win_counts['ft']} | tie={win_counts['tie']}")
    print(f"ELAPSED: {payload['summary']['elapsed_seconds']:.2f}s")
    print(f"✅ Saved final JSON to:  {OUT_JSON}")
    print(f"✅ Streamed JSONL to:    {OUT_JSONL}")
    print("#" * 120 + "\n")


if __name__ == "__main__":
    main()
