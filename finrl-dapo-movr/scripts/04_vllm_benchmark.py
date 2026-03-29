"""
scripts/04_vllm_benchmark.py — HuggingFace vs vLLM Throughput Benchmark
=========================================================================
Full throughput benchmark comparing HuggingFace generate() vs vLLM PagedAttention.
Requires A10G GPU with 24 GB VRAM.

Run: python scripts/04_vllm_benchmark.py
Requires: A10G GPU
Outputs:
  results/vllm_benchmark.json
  results/timing_log.json (appended)
"""

import sys
import json
import time
import torch
import yaml
from datetime import timezone, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load config ────────────────────────────────────────────────────────────
cfg_path = Path("configs/vllm_benchmark.yaml")
if not cfg_path.exists():
    print(f"[ERROR] Missing {cfg_path}"); sys.exit(1)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

MODEL       = cfg.get("model", "Qwen/Qwen2.5-7B-Instruct")
BATCH_SIZES = cfg.get("batch_sizes", [1, 2, 4, 8, 16, 32])
MAX_TOKENS  = cfg.get("max_new_tokens", 64)
N_WARMUP    = cfg.get("n_warmup_iters", 3)
N_BENCH     = cfg.get("n_bench_iters", 10)
TEMP        = cfg.get("temperature", 0.8)
DUMMY_PRMPT = cfg.get("dummy_prompt",
    "Stock: AAPL, Date: 2022-03-15, Price change (7-day): +1.23%, "
    "Sentiment score (1-5): 3.8, Risk score (1-5): 2.1, "
    "MACD: 0.0042, RSI: 58.3. BUY or SELL?").strip()

print(f"\n=== Script 04: vLLM Benchmark ===")
print(f"  Model: {MODEL}")
print(f"  Batch sizes: {BATCH_SIZES}")
print(f"  Max new tokens: {MAX_TOKENS}")


def bench_generator(gen, gen_name, use_sync=True):
    results = {"throughput_tok_per_sec": [], "latency_ms": []}
    for bs in BATCH_SIZES:
        prompts = [DUMMY_PRMPT] * bs
        # Warmup
        for _ in range(N_WARMUP):
            gen.generate_batch(prompts, n=1, max_tokens=MAX_TOKENS, temperature=TEMP)
        if use_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        # Benchmark
        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            gen.generate_batch(prompts, n=1, max_tokens=MAX_TOKENS, temperature=TEMP)
        if use_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / N_BENCH
        throughput = bs * MAX_TOKENS / elapsed
        latency_ms = elapsed * 1000
        results["throughput_tok_per_sec"].append(round(throughput, 2))
        results["latency_ms"].append(round(latency_ms, 2))
        print(f"  [{gen_name}] bs={bs:3d}  {throughput:7.1f} tok/s  {latency_ms:7.1f} ms")
    return results


# ── Part A: HuggingFace ────────────────────────────────────────────────────
print("\n[Part A] HuggingFace generate()")
from src.inference.hf_generator import HFGenerator
hf = HFGenerator(model_name=MODEL, device="cuda", dtype="bfloat16")
hf.load()
hf_results = bench_generator(hf, "HF", use_sync=True)
hf.unload()

# ── Part B: vLLM ──────────────────────────────────────────────────────────
print("\n[Part B] vLLM PagedAttention")
from src.inference.vllm_generator import VLLMGenerator
vllm = VLLMGenerator(
    model_name=MODEL, port=cfg.get("port", 8000),
    gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
    max_model_len=cfg.get("max_model_len", 2048),
    dtype=cfg.get("dtype", "bfloat16"),
)
vllm.start_server()
vllm_results = bench_generator(vllm, "vLLM", use_sync=False)
vllm.stop_server()

# ── Compute speedup ────────────────────────────────────────────────────────
speedup = [
    round(v / h, 2) if h > 0 else 0.0
    for v, h in zip(vllm_results["throughput_tok_per_sec"], hf_results["throughput_tok_per_sec"])
]
peak_speedup = max(speedup)
mean_speedup = sum(speedup) / len(speedup)

# ── Save ───────────────────────────────────────────────────────────────────
benchmark_data = {
    "model": MODEL,
    "max_new_tokens": MAX_TOKENS,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "batch_sizes": BATCH_SIZES,
    "hf": hf_results,
    "vllm": vllm_results,
    "speedup_ratio": speedup,
    "peak_speedup": peak_speedup,
    "mean_speedup": round(mean_speedup, 2),
}
with open(RESULTS_DIR / "vllm_benchmark.json", "w") as f:
    json.dump(benchmark_data, f, indent=2)

# ── Append timing ──────────────────────────────────────────────────────────
timing_path = RESULTS_DIR / "timing_log.json"
timing_data = []
if timing_path.exists():
    with open(timing_path) as f:
        timing_data = json.load(f)
timing_data.append({
    "script": "04_vllm_benchmark", "config": MODEL,
    "peak_speedup": peak_speedup, "mean_speedup": mean_speedup,
    "timestamp": datetime.now(timezone.utc).isoformat(),
})
with open(timing_path, "w") as f:
    json.dump(timing_data, f, indent=2)

# ── Summary table ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'Batch':>6} | {'HF (tok/s)':>12} | {'vLLM (tok/s)':>13} | {'Speedup':>8}")
print("─" * 60)
for bs, ht, vt, sp in zip(BATCH_SIZES, hf_results["throughput_tok_per_sec"],
                           vllm_results["throughput_tok_per_sec"], speedup):
    print(f"{bs:>6} | {ht:>12.1f} | {vt:>13.1f} | {sp:>7.1f}×")
print("=" * 60)
print(f"Peak speedup: {peak_speedup:.1f}×   Mean speedup: {mean_speedup:.1f}×")
print(f"\n[DONE] Results saved to {RESULTS_DIR}/vllm_benchmark.json")
print("Next: python scripts/05_generate_all_plots.py")
