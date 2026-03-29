"""
scripts/04_llm_call_baseline.py — Zero-Shot LLM Inference Baseline
===================================================================
Runs zero-shot trading signal inference using Groq's free-tier API
(llama-3.3-70b-versatile). No training — pure LLM zero-shot prompting.

Falls back to local Qwen2.5-1.5B if GROQ_API_KEY is not set.

Run: python scripts/04_llm_call_baseline.py

Requires: dataset/*.csv (run 00_download_data.py first)
Outputs:
  results/llm_call_predictions.csv
  results/llm_call_backtest.csv
  results/llm_call_metrics.json
"""

import sys
import time
import json
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()

torch.manual_seed(42); np.random.seed(42); random.seed(42)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

data_needed = Path("dataset/trade_data_2019_2023.csv")
if not data_needed.exists():
    print(f"[ERROR] Missing {data_needed}. Run: python scripts/00_download_data.py")
    sys.exit(1)

print("\n=== Script 04: Zero-Shot LLM Call Baseline ===")
print("    LLM: Groq llama-3.3-70b-versatile (free tier)")
print("    Fallback: Qwen/Qwen2.5-1.5B-Instruct (local HF)\n")

from src.data.data_loader import get_train_test_split
from src.inference.llm_call_baseline import run_llm_call_baseline
from src.evaluation.backtest import run_backtest

_, test_df = get_train_test_split()
print(f"  Test set: {len(test_df):,} rows")

# Run LLM inference (500-sample cost-controlled evaluation)
t0 = time.perf_counter()
predictions_df = run_llm_call_baseline(
    test_df=test_df,
    sample_size=500,
    output_path=str(RESULTS_DIR / "llm_call_predictions.csv"),
    rate_limit_delay=0.2,
    random_seed=42,
)
inference_time = time.perf_counter() - t0

# Simulate portfolio from predictions
print("\nRunning portfolio backtest from LLM signals...")
portfolio_df, metrics = run_backtest(
    predictions_df=predictions_df,
    price_df=test_df,
    initial_capital=1_000_000,
    method_name="llm_call",
)

metrics["inference_time_seconds"] = inference_time
metrics["n_samples"] = len(predictions_df)
metrics["accuracy"] = float(predictions_df["correct"].mean())

with open(RESULTS_DIR / "llm_call_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 60)
print("LLM Call Baseline Results")
print("=" * 60)
print(f"  Directional accuracy:   {metrics['accuracy']*100:.1f}%")
print(f"  Cumulative return:      {metrics['cumulative_return']*100:.2f}%")
print(f"  Sharpe ratio:           {metrics['sharpe_ratio']:.3f}")
print(f"  Max drawdown:           {metrics['max_drawdown']*100:.2f}%")
print(f"  Inference time:         {inference_time:.1f}s ({len(predictions_df)} samples)")
print("=" * 60)
print(f"\n[DONE] Results saved to {RESULTS_DIR}/")
print("Next: python scripts/05_vllm_benchmark.py")
