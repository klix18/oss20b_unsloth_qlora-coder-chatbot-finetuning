#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
align_oss.py (Unsloth single-load + adapter attach + GPT-5 mini judge)

- Load base once (4-bit) with Unsloth
- Attach QLoRA adapter to the same model
- Generate 2 answers per question (2 system prompts)
- Judge with GPT-5 mini (NO temperature param; model doesn't support 0.0)
- Save DPO-style JSONL to:
  /home/jovyan/work/kevin_fine_tune-code-oss20b/data/raw/alignment_qa.jsonl
- Flush every 10
- Pilot = 10
"""

# Must be set BEFORE importing torch/transformers
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRANSFORMERS_NO_CACHING_ALLOCATOR_WARMUP", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from dotenv import load_dotenv
from openai import OpenAI

from unsloth import FastLanguageModel
from peft import PeftModel


# -----------------------------
# HARD-CODED SETTINGS
# -----------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

BASE_MODEL = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
ADAPTER_DIR = "/home/jovyan/work/kevin_fine_tune-code-oss20b/runs/oss20b_unsloth_qlora/oss20b_unsloth_qlora/adapter"

OUT_PATH = "/home/jovyan/work/kevin_fine_tune-code-oss20b/data/raw/alignment_qa.jsonl"
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

NUM_QUESTIONS = 500
FLUSH_EVERY = 10

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.95
TOP_K = 0

JUDGE_MODEL = "gpt-5-mini"

SYSTEM_PROMPT_A = (
    "You are a senior software engineer and computer science researcher. "
    "Respond in an academic, developer, professional tone. "
    "Be technically correct, explicit about assumptions, and include concise code where helpful. "
    "Prefer robust best practices and clear reasoning."
)

SYSTEM_PROMPT_B = (
    "You are a staff-level Python engineer writing production-quality guidance. "
    "Respond professionally and pragmatically. "
    "Structure: (1) short explanation, (2) clean reference implementation, (3) pitfalls/tests. "
    "Keep it tight and avoid fluff."
)


QUESTIONS: List[str] = [
    # =========================
    # Python Core & Semantics
    # =========================
    "Explain the difference between concurrency and parallelism in Python. When would you use asyncio vs multiprocessing?",
    "How does Python’s Global Interpreter Lock (GIL) work, and what problems does it solve?",
    "Explain Python’s memory management model, including reference counting and cyclic garbage collection.",
    "What is late binding in Python closures, and how can it cause bugs?",
    "Explain the difference between __getattr__ and __getattribute__.",
    "How does Python’s method resolution order (MRO) work in multiple inheritance?",
    "What are descriptors in Python, and what problems do they solve?",
    "Explain the difference between shallow and deep copies.",
    "How does Python handle integer overflow internally?",
    "What are Python metaclasses, and when should they be used?",
    "Explain the lifecycle of a Python object.",
    "What is the difference between __new__ and __init__?",
    "How does Python handle truthiness and falsiness?",
    "What are Python weak references and when are they useful?",
    "Explain how Python handles exceptions internally.",
    "How does Python’s import system work?",
    "What is name mangling in Python, and why does it exist?",
    "How does Python handle Unicode strings internally?",
    "Explain the difference between staticmethod and classmethod.",
    "What are slots in Python classes, and what are their trade-offs?",
    "Explain Python’s garbage collection of reference cycles.",
    "What is the difference between is and ==?",
    "How does Python optimize bytecode execution?",
    "What are Python frame objects, and when are they relevant?",
    "Explain the difference between generators and iterators.",
    "What does yield from do, and when should it be used?",
    "How does Python handle recursion limits?",
    "What are context managers, and how does the with protocol work?",
    "Explain Python’s hashing strategy and hash randomization.",
    "What is monkey patching, and why is it risky?",
    "How does Python resolve local vs global variables?",
    "What are frozen dataclasses, and when are they appropriate?",
    "Explain Python’s memory arenas and allocators.",
    "What is the difference between copy.copy and copy.deepcopy?",
    "How does Python handle floating-point precision?",
    "What are sentinel objects, and why use them?",
    "How does super() work under the hood?",
    "What is the difference between __slots__ and dataclasses?",
    "Explain Python’s small integer caching.",
    "What are namespace packages?",
    "How does Python handle signal interrupts?",
    "Explain Python exception chaining.",
    "How does Python optimize function calls?",
    "What are interned strings in Python?",
    "Explain Python slicing semantics.",
    "What is the difference between yield and return?",
    "How does Python’s bytecode caching work?",
    "What are Python annotations used for at runtime?",
    "How does Python handle mutable default arguments?",
    "What are Python’s truth-testing rules?",
    "Explain the difference between list, tuple, and deque.",
    "What is the role of __all__ in a module?",
    "How does Python evaluate boolean expressions?",
    "What are Python’s data model dunder methods?",
    "Explain how Python handles attribute lookup.",
    "What is the difference between globals(), locals(), and vars()?",
    "How does Python handle circular imports?",
    "Explain the cost model of Python function calls.",
    "What are Python’s reference cycles and how are they collected?",
    "What is the difference between bytes and bytearray?",
    "Explain Python’s argument passing model.",
    "How does Python manage stack frames?",
    "What are Python’s truthy and falsy values?",
    "What is the difference between exec and eval?",
    "How does Python handle memory fragmentation?",
    "Explain the behavior of finally blocks.",
    "What is the difference between repr and str?",
    "How does Python handle attribute assignment?",
    "Explain Python’s exception hierarchy.",
    "What are Python’s internal caching mechanisms?",
    "How does Python implement dictionaries internally?",
    "What is the difference between callable objects and functions?",
    "Explain Python’s import hooks.",
    "What is the role of __enter__ and __exit__?",
    "How does Python manage thread-local storage?",
    "What is the difference between compile-time and runtime errors?",
    "Explain Python’s object mutability rules.",

    # =========================
    # Async, Concurrency, Systems
    # =========================
    "How does asyncio’s event loop work internally?",
    "What is cooperative multitasking in asyncio?",
    "How does async/await differ from threading?",
    "Explain backpressure in async systems.",
    "What are common pitfalls when mixing threads and asyncio?",
    "How does Python multiprocessing serialize objects?",
    "What is the difference between spawn, fork, and forkserver?",
    "How does asyncio handle task scheduling?",
    "Explain cancellation semantics in asyncio.",
    "How do you debug deadlocks in Python?",
    "What are race conditions, and how do they occur in Python?",
    "Explain thread safety in Python programs.",
    "How does Python’s queue module ensure thread safety?",
    "What are asyncio Futures and Tasks?",
    "How does Python handle blocking I/O in async code?",
    "Explain event-driven architecture in Python.",
    "What are common causes of memory leaks in Python?",
    "How do you profile CPU-bound vs I/O-bound workloads?",
    "Explain the difference between concurrency primitives: Lock, RLock, Semaphore.",
    "How does Python handle process crashes in multiprocessing?",
    "What are common async performance bottlenecks?",
    "How do you implement rate limiting in async services?",
    "Explain starvation and fairness in schedulers.",
    "How does Python’s signal handling interact with threads?",
    "What are async context managers?",
    "How do you safely shut down async services?",
    "Explain structured concurrency.",
    "What is the difference between greenlets and asyncio?",
    "How do you implement retries safely in distributed systems?",
    "Explain idempotency in API design.",
    "How do you detect and prevent thundering herd problems?",
    "What are the trade-offs between threads and processes?",
    "How does Python handle file descriptors?",
    "Explain cooperative cancellation patterns.",
    "How do you handle graceful shutdown in CLI tools?",
    "What are common sources of nondeterminism in concurrent code?",
    "How does asyncio handle blocking libraries?",
    "What is the difference between async generators and sync generators?",
    "Explain timeout handling in async systems.",
    "How does Python interact with epoll/kqueue under the hood?",
    "What are best practices for async error propagation?",
    "Explain bounded vs unbounded queues.",
    "How do you avoid priority inversion?",
    "What are async race conditions?",
    "Explain how futures differ from promises.",
    "How does Python handle inter-process communication?",
    "What is the difference between cooperative and preemptive multitasking?",
    "How do you implement async retries with jitter?",
    "Explain async context propagation.",
    "How do you test concurrent code reliably?",

    # =========================
    # Data Structures & Algorithms
    # =========================
    "Explain the time complexity of Python list operations.",
    "How does Python’s dict achieve average O(1) lookups?",
    "What are collision resolution strategies in hash tables?",
    "Explain the difference between BFS and DFS.",
    "How do you detect cycles in a graph?",
    "What are stable vs unstable sorting algorithms?",
    "Explain the space-time trade-offs in caching.",
    "What is amortized analysis?",
    "How do you implement an LRU cache?",
    "Explain the difference between heaps and balanced trees.",
    "What are bloom filters and when should they be used?",
    "How does MinHash approximate Jaccard similarity?",
    "Explain locality-sensitive hashing (LSH).",
    "What are failure modes of probabilistic data structures?",
    "How do you deduplicate large datasets efficiently?",
    "Explain trie-based data structures.",
    "What is the difference between adjacency lists and matrices?",
    "How does Python’s heapq work internally?",
    "Explain the difference between recursion and iteration.",
    "How do you implement top-k efficiently?",
    "What are the trade-offs between arrays and linked lists?",
    "Explain union-find and its optimizations.",
    "How do you handle large-scale text indexing?",
    "What is the difference between exact and approximate algorithms?",
    "Explain how hashing impacts performance.",
    "What are skip lists?",
    "How do you design a memory-efficient streaming algorithm?",
    "Explain consistent hashing.",
    "What are common pitfalls in algorithm benchmarking?",
    "Explain sliding window techniques.",
    "How do you implement deduplication with rolling hashes?",
    "What is the difference between eager and lazy evaluation?",
    "Explain cache eviction policies.",
    "What is a radix tree?",
    "How do you evaluate algorithmic scalability?",
    "Explain how Python compares strings.",
    "What are vectorized operations?",
    "Explain the difference between functional and imperative styles.",
    "How do you avoid quadratic behavior in Python code?",
    "What is the trade-off between precomputation and on-demand computation?",

    # =========================
    # Packaging, Tooling, CLI
    # =========================
    "Explain how to structure a Python package for a CLI tool using argparse or typer.",
    "What is the difference between setup.py, pyproject.toml, and setup.cfg?",
    "How do Python entry points work?",
    "Explain version pinning vs version ranges.",
    "How do you design a maintainable CLI command hierarchy?",
    "What are best practices for CLI error handling?",
    "How do you handle configuration in CLI tools?",
    "Explain semantic versioning.",
    "How do you test CLI tools effectively?",
    "What is the difference between editable installs and standard installs?",
    "How does pip resolve dependency conflicts?",
    "What are virtual environments and how do they work?",
    "Explain dependency isolation strategies.",
    "How do you package native extensions?",
    "What are common pitfalls in Python packaging?",
    "How do you distribute private Python packages?",
    "Explain how console_scripts entry points work.",
    "How do you manage backward compatibility in CLIs?",
    "What is the difference between click and typer?",
    "How do you design user-friendly CLI interfaces?",

    # =========================
    # Testing, Quality, Reliability
    # =========================
    "Explain the difference between unit, integration, and end-to-end tests.",
    "How does pytest fixture scoping work?",
    "What are property-based tests?",
    "How do you test async code in Python?",
    "Explain test isolation and determinism.",
    "How do you avoid flaky tests?",
    "What is mutation testing?",
    "Explain code coverage limitations.",
    "How do you test error handling paths?",
    "What are best practices for mocking?",
    "Explain snapshot testing.",
    "How do you test concurrency safely?",
    "What is fuzz testing?",
    "How do you structure large test suites?",
    "Explain test doubles: mocks, stubs, fakes.",
    "How do you measure test effectiveness?",
    "What are common pitfalls in test-driven development?",
    "Explain contract testing.",
    "How do you test CLI tools?",
    "What are best practices for regression testing?",

    # =========================
    # ML, Data, Systems (Alignment-Relevant)
    # =========================
    "Explain data leakage and how to prevent it.",
    "What is train/validation/test split and why does it matter?",
    "Explain bias-variance trade-off.",
    "How do you evaluate model generalization?",
    "What are common failure modes in ML pipelines?",
    "Explain feature leakage.",
    "What is concept drift?",
    "How do you handle class imbalance?",
    "Explain model calibration.",
    "What is overfitting and how do you detect it?",
    "How do you design reproducible ML experiments?",
    "Explain the difference between batch and online inference.",
    "What are the trade-offs between accuracy and latency?",
    "How do you version datasets?",
    "Explain ML pipeline orchestration.",
    "What are best practices for ML monitoring?",
    "How do you detect data distribution shifts?",
    "Explain offline vs online evaluation.",
    "What are common causes of silent ML failures?",
    "How do you design robust data validation pipelines?",
# Add these 200 student-style questions to your existing QUESTIONS list.
# (They’re phrased as “I’m confused / can you help me” prompts to elicit tutoring-style, professional answers.)
    # =========================
    # Python Basics (student tone)
    # =========================
    "I keep mixing up lists, tuples, and sets. Can you explain when to use each, with small examples?",
    "I don’t understand why my function changes a list outside the function. Can you explain mutability and pass-by-object-reference in Python?",
    "I’m confused about *args and **kwargs. Can you show how they work and when I should use them?",
    "Why does Python sometimes say UnboundLocalError for a variable I thought was global? Can you explain scope rules?",
    "I keep seeing __name__ == '__main__'. What exactly does it do and why is it useful?",
    "I don’t understand the difference between == and is. Can you explain with examples that show the pitfall?",
    "Why do default arguments sometimes ‘remember’ values across calls? Can you explain the mutable default argument bug?",
    "I’m confused about shallow copy vs deep copy. Can you show examples where shallow copy breaks?",
    "I don’t get why iterating a dict gives keys by default. Can you explain dict iteration patterns?",
    "What’s the difference between return and yield? I’m not sure when to write a generator.",
    "I’m confused about list comprehensions vs generator expressions. Can you show when each is better?",
    "I wrote a try/except/finally block and the behavior surprised me. Can you explain execution order?",
    "I don’t understand how context managers work. Can you show how to write a custom context manager?",
    "I’m getting UnicodeDecodeError reading a file. Can you explain encodings and a safe way to handle text files?",
    "Why does sorting sometimes behave unexpectedly with mixed types? Can you explain sorting rules and key functions?",
    "I’m confused about f-strings vs format vs % formatting. What’s recommended and why?",
    "Can you explain how Python imports modules and how PYTHONPATH affects it? I’m getting ModuleNotFoundError.",
    "I don’t understand how to structure a multi-file Python project. Can you show a simple example layout?",
    "I keep seeing TypeError: 'NoneType' object is not callable. What are common causes and how do I debug it?",
    "I’m confused about decorators. Can you explain what they are and show a practical example?",

    # =========================
    # OOP & Dataclasses (student tone)
    # =========================
    "I’m confused about classes vs functions. When should I make a class instead of just writing functions?",
    "What’s the difference between instance variables and class variables? I keep mixing them up.",
    "I don’t understand inheritance vs composition. Can you explain when composition is better?",
    "I keep seeing @property. Why use it instead of just a method?",
    "How do __repr__ and __str__ differ? When should I implement each?",
    "I’m not sure how super() works. Can you explain it with a multi-inheritance example?",
    "I’m confused about abstract base classes. When should I use abc.ABC?",
    "Can you explain dataclasses vs pydantic models? I’m not sure which to use for config/data objects.",
    "I keep seeing Protocol in typing. What problem does it solve compared to inheritance?",
    "I don’t understand how to design clean interfaces in Python. Can you show a small example with dependency injection?",

    # =========================
    # Typing & Static Analysis (student tone)
    # =========================
    "I’m confused about Optional[T] and None. When should I use Optional and when should I raise errors?",
    "What’s the difference between List[str] and list[str]? Which should I use?",
    "How do generics work in Python typing? Can you show a small example with TypeVar?",
    "I don’t understand Union vs | syntax. Can you explain modern typing best practices?",
    "Mypy complains about invariance for List. Why is List invariant and what should I do instead?",
    "I’m confused about Callable typing. How do I type a function that takes another function?",
    "How do I type a dictionary with specific keys? Should I use TypedDict or dataclasses?",
    "I keep seeing Any everywhere. When is Any acceptable and when does it defeat the purpose of typing?",
    "How do I type a function that returns an iterator or generator?",
    "Can you explain runtime type checking vs static type checking and how they relate?",

    # =========================
    # Files, JSON, Serialization (student tone)
    # =========================
    "I’m not sure how to safely read a huge JSONL file without loading it all in memory. Can you show a streaming approach?",
    "How do I write robust CSV parsing code when there are missing columns and weird encodings?",
    "I’m confused about json.dumps vs json.dump. When do I use each?",
    "What’s the difference between pickle and JSON? When is pickle unsafe?",
    "How do I serialize dataclasses cleanly to JSON?",
    "I’m getting issues with datetime in JSON. What’s a good way to serialize and parse timestamps?",
    "How do I design a file format for logs that’s easy to parse and append to?",
    "I’m unsure how to validate JSON input. Should I use jsonschema, pydantic, or something else?",
    "How do I handle partial failures when writing files (e.g., crash mid-write) to avoid corruption?",
    "I want to merge two big JSONL datasets and deduplicate them. What’s a memory-safe approach?",

    # =========================
    # HTTP, APIs, Networking (student tone)
    # =========================
    "I need to call an API and retry on failures. Can you show exponential backoff with jitter and explain why jitter matters?",
    "I’m confused about status codes. Which ones should I retry and which ones should I fail fast on?",
    "How do I design a Python client library for a REST API in a clean way?",
    "I’m not sure how to add timeouts correctly to requests. What timeouts should I set and why?",
    "What’s the difference between connect timeout and read timeout?",
    "I’m confused about pagination. How do I handle cursor-based vs page-based pagination robustly?",
    "How should I handle rate limiting (429) in a Python API client?",
    "What’s the correct way to do authentication (API keys vs OAuth) from a Python script?",
    "I’m not sure how to safely log HTTP responses without leaking secrets. What should I redact?",
    "How do I test HTTP code without actually hitting the network?",

    # =========================
    # Concurrency / Async (student tone)
    # =========================
    "I’m confused about threading vs asyncio. When should I choose one over the other?",
    "Why does my asyncio program ‘hang’ sometimes? How do I debug event loop issues?",
    "I’m not sure how to add timeouts to asyncio tasks. Can you show patterns for cancellation-safe timeouts?",
    "How do I limit concurrency in asyncio so I don’t open too many connections?",
    "I tried to use a blocking library inside async code. What’s the right way to handle blocking calls?",
    "I’m confused about async generators. When would I use async for instead of a normal for loop?",
    "How do I share state safely between asyncio tasks?",
    "What’s the difference between multiprocessing and multithreading for CPU-heavy code?",
    "I’m not sure how to design a producer/consumer pipeline in Python. Can you show a clean approach?",
    "My concurrent code is flaky. How can I write tests that are deterministic?",

    # =========================
    # Debugging, Profiling, Performance (student tone)
    # =========================
    "I’m not sure how to debug a Python program effectively. Can you outline a systematic debugging workflow?",
    "How do I profile Python code to find bottlenecks? When should I use cProfile vs py-spy?",
    "I’m confused about why my code is slow. How do I reason about algorithmic complexity in Python?",
    "My program uses too much memory. How do I track memory usage and find leaks?",
    "Can you explain why vectorized NumPy code is faster than Python loops?",
    "I’m not sure how to speed up pandas operations. What are the common anti-patterns?",
    "When does caching help performance and when does it make things worse?",
    "How do I measure latency vs throughput correctly when benchmarking a script?",
    "I’m confused about microbenchmarks. Why are they often misleading?",
    "How do I optimize I/O-heavy Python programs?",

    # =========================
    # Packaging, Environments, CLI (student tone)
    # =========================
    "I’m trying to make a CLI tool but I’m confused about project structure. Can you show a recommended folder layout for argparse/typer?",
    "What’s the difference between pyproject.toml and setup.py? Which should I use in 2025?",
    "How do console_scripts entry points work and how do I add one for my CLI?",
    "I’m not sure how to manage dependencies. Should I use pip-tools, poetry, or uv? What are the trade-offs?",
    "Why does my package work locally but fail in a fresh venv? How do I debug packaging issues?",
    "I’m confused about editable installs. When should I use pip install -e?",
    "How do I ship a CLI tool so users can install it and run it globally?",
    "How do I design subcommands cleanly in Typer or Click?",
    "What are best practices for CLI error messages and exit codes?",
    "How do I add configuration files to a CLI tool and support overrides via flags/env vars?",

    # =========================
    # Logging, Observability (student tone)
    # =========================
    "I’m not sure how to use Python logging properly. Can you show a good setup for libraries vs applications?",
    "How do I structure logs so they’re useful in production (JSON logs, correlation IDs, etc.)?",
    "I’m confused about log levels. What should be INFO vs DEBUG vs WARNING?",
    "How do I avoid logging secrets like API keys and tokens?",
    "What’s the difference between metrics, logs, and traces?",
    "How do I add basic tracing to a Python service?",
    "How do I handle log rotation for a long-running process?",
    "I’m not sure how to log exceptions correctly without losing stack traces.",
    "How do I make a CLI tool produce both human-readable and machine-readable output?",
    "How should I think about observability for batch jobs?",

    # =========================
    # Databases & Persistence (student tone)
    # =========================
    "I’m confused about transactions. Can you explain ACID and how it applies in PostgreSQL?",
    "How do I prevent SQL injection in Python?",
    "What’s the difference between an ORM and writing raw SQL?",
    "I’m not sure how to design database indexes. Can you explain how to choose indexes for a query pattern?",
    "How do I handle database migrations safely?",
    "What is connection pooling and why do I need it?",
    "I’m confused about isolation levels. How do they affect concurrency and consistency?",
    "How do I design idempotent writes to a database?",
    "What’s the difference between upsert and insert-then-update?",
    "How do I handle retries with databases without creating duplicates?",

    # =========================
    # Security (student tone)
    # =========================
    "I’m not sure what input validation I should do in a Python API. Can you outline best practices?",
    "How do I store secrets safely for a Python app (API keys, DB passwords)?",
    "I’m confused about TLS/HTTPS. What does certificate validation do and why does it matter?",
    "How should I sanitize logs and error messages to avoid leaking sensitive data?",
    "What are common Python security pitfalls (pickle, eval, yaml.load, etc.)?",
    "How do I safely parse untrusted JSON or YAML?",
    "What’s the difference between authentication and authorization?",
    "How do I implement secure password hashing in Python?",
    "I’m confused about CSRF vs XSS. Can you explain them with practical mitigation steps?",
    "How do I design a safe file upload endpoint?",

    # =========================
    # Web APIs (FastAPI) (student tone)
    # =========================
    "I’m trying to build a FastAPI service. How should I structure the project for scalability?",
    "How do I validate request data in FastAPI with Pydantic models?",
    "What’s the difference between sync and async endpoints in FastAPI?",
    "How do I add authentication (e.g., JWT) to a FastAPI app cleanly?",
    "I’m confused about dependency injection in FastAPI. Can you explain it with an example?",
    "How do I handle background tasks safely in FastAPI?",
    "How do I implement rate limiting for an API?",
    "How do I design error responses and exception handlers consistently?",
    "What’s a good strategy for versioning an API (v1/v2) in FastAPI?",
    "How do I write tests for FastAPI endpoints efficiently?",

    # =========================
    # Data Engineering (student tone)
    # =========================
    "I’m processing a huge dataset and my script crashes. How do I redesign it to be streaming and memory-safe?",
    "How do I do deduplication at scale using MinHash/LSH and what parameters matter?",
    "I’m confused about ETL vs ELT. When should I transform before loading vs after?",
    "How do I validate data quality (null checks, schema checks) in a pipeline?",
    "How do I design a robust deduplication key when records are messy?",
    "What are common failure modes in data pipelines and how do I make them resilient?",
    "I’m not sure how to handle slowly changing dimensions. Can you explain SCD types?",
    "How do I do incremental processing (only new data) safely?",
    "How do I choose partition keys when writing Parquet datasets?",
    "What’s the difference between batching and streaming in practice?",

    # =========================
    # ML / Evaluation (student tone)
    # =========================
    "I’m confused about data leakage in machine learning. Can you give concrete examples and how to avoid them?",
    "How do I split data correctly for time series problems?",
    "Why can a model have high AUC but be poorly calibrated?",
    "I’m not sure how to choose between accuracy, precision, recall, and F1. Can you explain trade-offs?",
    "How do I evaluate an imbalanced classifier properly?",
    "What is concept drift and how do I detect it in production?",
    "I’m confused about cross-validation. When is k-fold appropriate and when is it not?",
    "How do I design reproducible ML experiments with seeds and versioned data?",
    "What are common failure modes of ML systems in production and how do I monitor them?",
    "How do I design an offline evaluation that predicts online performance?",

    # =========================
    # LLM / Alignment (student tone)
    # =========================
    "I’m trying to create a DPO dataset. What exactly should prompt/chosen/rejected contain for a chat model?",
    "Why is it important to strip system/user text out of the chosen/rejected fields in preference datasets?",
    "I’m confused about the difference between SFT and DPO. When should I do each?",
    "How do I generate two diverse candidate answers without changing the system prompt?",
    "What are common pitfalls when using a larger model as a judge for preference data?",
    "How do I reduce position bias and verbosity bias in LLM judging?",
    "I’m not sure how to detect low-quality preference pairs. What heuristics should I apply?",
    "How do I avoid training a model to imitate judge quirks instead of true quality?",
    "What are best practices for formatting DPO data for TRL’s DPOTrainer?",
    "How do I evaluate whether DPO improved the model in a reliable way?",

    # =========================
    # Software Design (student tone)
    # =========================
    "I’m confused about layering in software design. What’s a clean architecture for a Python app?",
    "How do I design a plugin system in Python cleanly?",
    "What’s the difference between a library and an application in terms of design?",
    "How do I write code that is easy to refactor later?",
    "I’m not sure how to separate concerns in a CLI + library project. Can you show a pattern?",
    "What’s a good way to handle configuration (defaults, env vars, CLI flags) without making a mess?",
    "I’m confused about dependency injection in Python. Can you show a practical example?",
    "How do I design stable public APIs for a Python package?",
    "What’s the best way to handle errors across layers (domain vs infra)?",
    "How do I organize code so that unit tests don’t require network access?",

    # =========================
    # DevOps / Deployment (student tone)
    # =========================
    "I’m confused about Docker images vs containers. Can you explain and show a minimal Dockerfile for a Python CLI?",
    "How do I create a reproducible Python environment for deployment?",
    "What’s the difference between CI and CD, and what should a basic CI pipeline test for a Python project?",
    "I’m not sure how to manage environment variables securely in deployment.",
    "How do I handle database migrations in CI/CD safely?",
    "What’s the difference between blue/green and rolling deployments?",
    "How do I monitor a batch job in production (alerts, retries, dead letter queues)?",
    "What is idempotency and why does it matter for retries in distributed systems?",
    "I’m confused about scaling. When does vertical scaling fail and when should I use horizontal scaling?",
    "How do I handle graceful shutdown in a deployed Python service?",

]



# -----------------------------
# Helpers
# -----------------------------
def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_exists_dir(path: str, must_contain: Optional[List[str]] = None):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist:\n  {path}")
    if not p.is_dir():
        raise NotADirectoryError(f"Path is not a directory:\n  {path}")
    if must_contain:
        missing = [f for f in must_contain if not (p / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing required files in:\n  {path}\nMissing: {missing}")


def make_openai_client() -> OpenAI:
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def strip_chat_tokens(text: str) -> str:
    toks = ["<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|return|>"]
    for t in toks:
        text = text.replace(t, "")
    return text.strip()


def robust_extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # first balanced {...}
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        start = i
        depth = 0
        in_str = False
        esc = False
        j = i
        while j < n:
            ch = text[j]
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
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = text[start : j + 1]
                        try:
                            obj = json.loads(cand)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
            j += 1
        i = start + 1
    return None


@torch.no_grad()
def generate(model, tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=(TEMPERATURE > 0),
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    return strip_chat_tokens(decoded)


def judge_pair(client: OpenAI, question: str, ans_a: str, ans_b: str) -> Dict[str, Any]:
    """
    IMPORTANT: gpt-5-mini does NOT support temperature=0.0.
    So we do NOT pass temperature at all.
    """

    judge_system = (
        "You are an expert Python software engineer evaluating two answers. "
        "Select the better answer for an academic, developer, professional response. "
        "Prioritize correctness, clarity, completeness, best practices, and safe guidance. "
        "Do NOT prefer length; prefer signal. "
        "Respond ONLY with valid JSON."
    )

    user_msg = f"""
PROMPT:
{question}

ANSWER_A:
{ans_a}

ANSWER_B:
{ans_b}

Return JSON exactly like:
{{
  "winner": "A" | "B",
  "scores": {{"A": 0-10, "B": 0-10}},
  "justification": "1-3 sentences, technical"
}}
""".strip()

    # If your endpoint supports JSON schema for this model, use it; otherwise fallback.
    schema = {
        "type": "object",
        "properties": {
            "winner": {"type": "string", "enum": ["A", "B"]},
            "scores": {
                "type": "object",
                "properties": {
                    "A": {"type": "integer", "minimum": 0, "maximum": 10},
                    "B": {"type": "integer", "minimum": 0, "maximum": 10},
                },
                "required": ["A", "B"],
                "additionalProperties": False,
            },
            "justification": {"type": "string"},
        },
        "required": ["winner", "scores", "justification"],
        "additionalProperties": False,
    }

    for attempt in range(1, 6):
        try:
            # Try schema mode first (best)
            try:
                resp = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": judge_system},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "judge", "schema": schema, "strict": True},
                    },
                )
                text = resp.choices[0].message.content.strip()
                obj = robust_extract_json(text)
                if obj and obj.get("winner") in ("A", "B"):
                    return obj
            except Exception:
                # schema not supported; fall through to plain JSON
                pass

            # Plain JSON fallback (no temperature)
            resp2 = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": judge_system},
                    {"role": "user", "content": user_msg},
                ],
            )
            text2 = resp2.choices[0].message.content.strip()
            obj2 = robust_extract_json(text2)
            if not obj2:
                raise ValueError("Judge did not return valid JSON:\n" + text2[:500])
            if obj2.get("winner") not in ("A", "B"):
                raise ValueError("Judge JSON missing winner A/B:\n" + json.dumps(obj2)[:500])
            return obj2

        except Exception as e:
            if attempt == 5:
                raise
            backoff = (1.5 ** attempt) + random.random()
            print(f"[{now_ts()}] Judge attempt {attempt} failed: {e} | retry in {backoff:.1f}s")
            time.sleep(backoff)

    raise RuntimeError("Unreachable")


# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    ensure_exists_dir(ADAPTER_DIR, must_contain=["adapter_config.json"])

    print(f"🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.")
    print(f"[{now_ts()}] Loading base model ONCE with Unsloth:\n  {BASE_MODEL}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        load_in_4bit=True,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
    )

    print(f"[{now_ts()}] Attaching adapter:\n  {ADAPTER_DIR}")
    if hasattr(model, "load_adapter"):
        model.load_adapter(ADAPTER_DIR)
    else:
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)

    model = FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[{now_ts()}] Model ready ✅ (base + adapter on one instance)")

    client = make_openai_client()

    qs = QUESTIONS[:NUM_QUESTIONS]
    buffer: List[Dict[str, Any]] = []

    print(f"[{now_ts()}] Starting pilot={NUM_QUESTIONS} | flush_every={FLUSH_EVERY}")
    print(f"[{now_ts()}] Output JSONL:\n  {OUT_PATH}")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for idx, q in enumerate(qs, 1):
            print("\n" + "=" * 120)
            print(f"[{now_ts()}] QUESTION {idx}/{NUM_QUESTIONS}: {q}")
            print("=" * 120)

            ans_a = generate(model, tokenizer, SYSTEM_PROMPT_A, q)
            ans_b = generate(model, tokenizer, SYSTEM_PROMPT_B, q)

            verdict = judge_pair(client, q, ans_a, ans_b)

            winner = verdict["winner"]
            chosen = ans_a if winner == "A" else ans_b
            rejected = ans_b if winner == "A" else ans_a

            record = {
                "prompt": q,
                "chosen": chosen,
                "rejected": rejected,
                "meta": {
                    "answer_A": ans_a,
                    "answer_B": ans_b,
                    "system_prompt_A": SYSTEM_PROMPT_A,
                    "system_prompt_B": SYSTEM_PROMPT_B,
                    "judge_model": JUDGE_MODEL,
                    "judge_verdict": verdict,
                    "base_model": BASE_MODEL,
                    "adapter_dir": ADAPTER_DIR,
                    "gen": {
                        "max_seq_length": MAX_SEQ_LENGTH,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "top_k": TOP_K,
                        "seed": SEED,
                    },
                    "timestamp": now_ts(),
                },
            }

            buffer.append(record)

            if idx % FLUSH_EVERY == 0:
                print(f"[{now_ts()}] Flushing {len(buffer)} records...")
                for r in buffer:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                buffer.clear()

        if buffer:
            print(f"[{now_ts()}] Final flush {len(buffer)} records...")
            for r in buffer:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            buffer.clear()

    print(f"\n[{now_ts()}] DONE ✅ wrote:\n  {OUT_PATH}\n")


if __name__ == "__main__":
    main()
