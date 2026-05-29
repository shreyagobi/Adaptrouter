# benchmark/profiler.py
import time
import psutil
import subprocess
import json
from dataclasses import dataclass, field
from typing import Optional
import requests

@dataclass
class InferenceProfile:
    query: str
    model: str
    # Time splits
    router_classify_ms: float = 0.0
    model_load_ms: float = 0.0       # cold start cost
    time_to_first_token_ms: float = 0.0   # TTFT — most important metric
    generation_ms: float = 0.0       # token generation phase
    total_ms: float = 0.0
    # Token stats
    prompt_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    # Memory
    vram_used_mb: float = 0.0
    ram_used_mb: float = 0.0
    # Routing decision
    routed_to: str = ""
    confidence: float = 0.0

def profile_inference(query: str, model: str = "llama3.1:8b") -> InferenceProfile:
    p = InferenceProfile(query=query, model=model)

    # RAM snapshot before
    mem_before = psutil.virtual_memory().used / 1024**2

    # ── 1. Time the router classifier ────────────────────────────────────────
    t0 = time.perf_counter()
    # (import your actual router here)
    # result_label = router.classify(query)
    p.router_classify_ms = (time.perf_counter() - t0) * 1000

    # ── 2. First-token latency vs generation latency via Ollama stream ────────
    payload = {
        "model": model,
        "prompt": query,
        "stream": True,
        "options": {
            "num_predict": 512,
            "temperature": 0.1,
        }
    }

    t_request = time.perf_counter()
    first_token_received = False
    output_tokens = 0
    full_response = ""

    with requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        stream=True,
        timeout=120
    ) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)

            if not first_token_received:
                p.time_to_first_token_ms = (time.perf_counter() - t_request) * 1000
                first_token_received = True
                t_generation_start = time.perf_counter()

            if chunk.get("response"):
                full_response += chunk["response"]
                output_tokens += 1

            if chunk.get("done"):
                p.generation_ms = (time.perf_counter() - t_generation_start) * 1000
                # Ollama gives us eval stats in the done chunk
                p.prompt_tokens      = chunk.get("prompt_eval_count", 0)
                p.output_tokens      = chunk.get("eval_count", output_tokens)
                eval_duration_ns     = chunk.get("eval_duration", 1)
                p.tokens_per_second  = (
                    p.output_tokens / (eval_duration_ns / 1e9)
                    if eval_duration_ns > 0 else 0
                )
                break

    p.total_ms   = (time.perf_counter() - t_request) * 1000
    p.ram_used_mb = psutil.virtual_memory().used / 1024**2 - mem_before

    return p


def run_benchmark():
    """
    Run a structured set of queries and print a breakdown table.
    This tells you exactly where time is going.
    """
    test_cases = [
        # (query, expected_route, complexity_label)
        ("What is the capital of France?",                          "8b",  "trivial"),
        ("Define gradient descent in one sentence.",                "8b",  "simple"),
        ("Explain how transformers work with attention mechanism.",  "70b", "medium"),
        ("Compare BERT vs GPT architectures for classification.",   "70b", "complex"),
        ("Walk me through backpropagation step by step.",           "70b", "complex"),
    ]

    print(f"\n{'Query':<45} {'Route':<6} {'TTFT ms':<10} {'Gen ms':<10} "
          f"{'tok/s':<8} {'Total ms':<10} {'RAM MB':<8}")
    print("─" * 110)

    for query, expected, label in test_cases:
        p = profile_inference(query, model="llama3.1:8b")
        flag = "✓" if expected in p.routed_to else "✗"
        print(
            f"{query[:44]:<45} {flag:<6} {p.time_to_first_token_ms:<10.0f} "
            f"{p.generation_ms:<10.0f} {p.tokens_per_second:<8.1f} "
            f"{p.total_ms:<10.0f} {p.ram_used_mb:<8.1f}"
        )

if __name__ == "__main__":
    run_benchmark()