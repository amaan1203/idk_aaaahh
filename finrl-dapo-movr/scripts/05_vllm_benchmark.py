"""
scripts/05_vllm_benchmark.py — HuggingFace vs vLLM Throughput Benchmark
========================================================================
Measures tokens/sec throughput for HuggingFace generate() vs vLLM
PagedAttention across batch sizes [1, 2, 4, 8, 16, 32].

Run: python scripts/05_vllm_benchmark.py

Requires: A10G GPU (24 GB VRAM) for 7B model, tmux
Outputs: results/vllm_benchmark.json
"""

import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()
import os

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
PORT = int(os.environ.get("VLLM_PORT", "8000"))
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
N_WARMUP = 3
N_BENCH = 10
MAX_NEW_TOKENS = 64

DUMMY_PROMPT = (
    "Stock: AAPL, Date: 2022-03-15, Price change (7-day): +1.23%, "
    "Sentiment: 3.8/5, Risk: 2.1/5, Technical (MACD): 0.0042. "
    "Should you BUY or SELL tomorrow?"
)

print(f"\n=== Script 05: vLLM Throughput Benchmark ===")
print(f"  Model: {MODEL}")
print(f"  Batch sizes: {BATCH_SIZES}")
print(f"  Warmup: {N_WARMUP} | Benchmark: {N_BENCH} iters\n")


def benchmark_generator(generator, generate_fn_name, batch_sizes):
    """Time generate_batch() for each batch size."""
    results = {
        "throughput_tok_per_sec": [],
        "latency_ms_mean": [],
        "latency_ms_std": [],
    }

    for bs in batch_sizes:
        prompts = [DUMMY_PROMPT] * bs
        generate_fn = getattr(generator, generate_fn_name)

        # Warmup
        for _ in range(N_WARMUP):
            generate_fn(prompts, n=1, max_tokens=MAX_NEW_TOKENS, temperature=0.8)

        # Benchmark
        times_ms = []
        for _ in range(N_BENCH):
            t0 = time.perf_counter()
            generate_fn(prompts, n=1, max_tokens=MAX_NEW_TOKENS, temperature=0.8)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times_ms.append(elapsed_ms)

        mean_ms = np.mean(times_ms)
        std_ms = np.std(times_ms)
        total_tokens = bs * MAX_NEW_TOKENS
        tps = (total_tokens / (mean_ms / 1000))

        results["throughput_tok_per_sec"].append(float(tps))
        results["latency_ms_mean"].append(float(mean_ms))
        results["latency_ms_std"].append(float(std_ms))
        print(f"  batch={bs:2d} | {tps:8.1f} tok/s | {mean_ms:.1f}±{std_ms:.1f} ms")

    return results


# ── 1. HuggingFace Benchmark ─────────────────────────────────────────────────
print("[1/2] HuggingFace generate() benchmark:")
from src.inference.hf_generator import HFGenerator

hf_gen = HFGenerator(model_name=MODEL, dtype="bfloat16")
hf_gen.load()
hf_results = benchmark_generator(hf_gen, "generate_batch", BATCH_SIZES)
hf_gen.unload()
print()

# ── 2. vLLM Benchmark ────────────────────────────────────────────────────────
print("[2/2] vLLM benchmark:")
from src.inference.vllm_generator import VLLMGenerator

vllm_gen = VLLMGenerator(model_name=MODEL, port=PORT)
vllm_gen.start_server(timeout=180)
vllm_results = benchmark_generator(vllm_gen, "generate_batch", BATCH_SIZES)
vllm_gen.stop_server()

# ── Compute speedup ──────────────────────────────────────────────────────────
speedup = [
    v / h for v, h in zip(
        vllm_results["throughput_tok_per_sec"],
        hf_results["throughput_tok_per_sec"],
    )
]
peak_speedup = max(speedup)

benchmark_output = {
    "model": MODEL,
    "max_new_tokens": MAX_NEW_TOKENS,
    "batch_sizes": BATCH_SIZES,
    "hf": hf_results,
    "vllm": vllm_results,
    "speedup_ratio": speedup,
    "peak_speedup": float(peak_speedup),
    "timestamp": datetime.utcnow().isoformat() + "Z",
}

with open(RESULTS_DIR / "vllm_benchmark.json", "w") as f:
    json.dump(benchmark_output, f, indent=2)

print("\n" + "=" * 60)
print("vLLM Speedup Summary")
print("=" * 60)
for bs, sp in zip(BATCH_SIZES, speedup):
    print(f"  batch={bs:2d} | speedup: {sp:.2f}x")
print(f"\n  Peak speedup: {peak_speedup:.2f}x at batch={BATCH_SIZES[speedup.index(peak_speedup)]}")
print("=" * 60)
print(f"\n[DONE] Results saved to {RESULTS_DIR}/vllm_benchmark.json")
print("Next: python scripts/06_generate_all_plots.py")
