"""
scripts/03_llm_call_baseline.py — Zero-Shot Claude Inference Baseline
======================================================================
Zero-shot Claude inference on 500 test samples. No GPU required.
Overwrites the placeholder in results/llm_call_metrics.json.

Run: python scripts/03_llm_call_baseline.py
Requires: ANTHROPIC_API_KEY in .env, dataset/trade_data_2019_2023.csv
Outputs:
  results/llm_call_predictions.csv
  results/llm_call_metrics.json (overwrites placeholder from script 01)
  results/timing_log.json (appended)

Estimated cost: ~500 calls × ~50 tokens × ~$0.000003/token ≈ $0.08
"""

import sys
import time
import json
from datetime import timezone, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()

import os
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("[ERROR] ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    sys.exit(1)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Check prerequisites ────────────────────────────────────────────────────
for fname in ["trade_data_2019_2023.csv",
              "trade_data_deepseek_sentiment_2019_2023.csv",
              "trade_data_deepseek_risk_2019_2023.csv"]:
    path = Path("dataset") / fname
    if not path.exists():
        print(f"[ERROR] Missing {path}. Run: python scripts/00_download_data.py")
        sys.exit(1)

print("\n=== Script 03: LLM Call Baseline (Zero-Shot Claude) ===\n")

# ── Load and merge test data ───────────────────────────────────────────────
price = pd.read_csv("dataset/trade_data_2019_2023.csv")
sentiment = pd.read_csv("dataset/trade_data_deepseek_sentiment_2019_2023.csv")
risk = pd.read_csv("dataset/trade_data_deepseek_risk_2019_2023.csv")

# Normalise column names
for df in [price, sentiment, risk]:
    if "tic" in df.columns and "ticker" not in df.columns:
        df.rename(columns={"tic": "ticker"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])

# Find sentiment/risk columns
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

sent_col = find_col(sentiment, ["sentiment", "score", "deepseek_sentiment"])
risk_col = find_col(risk, ["risk", "deepseek_risk", "risk_score"])

test_df = price.copy()
if sent_col:
    sent_data = sentiment[["date", "ticker", sent_col]].rename(columns={sent_col: "sentiment"})
    test_df = test_df.merge(sent_data, on=["date", "ticker"], how="left")
else:
    test_df["sentiment"] = 3.0

if risk_col:
    risk_data = risk[["date", "ticker", risk_col]].rename(columns={risk_col: "risk"})
    test_df = test_df.merge(risk_data, on=["date", "ticker"], how="left")
else:
    test_df["risk"] = 3.0

test_df["sentiment"] = test_df["sentiment"].fillna(3.0)
test_df["risk"] = test_df["risk"].fillna(3.0)

# true_action: 1 if next-day close > open
if "close" in test_df.columns and "open" in test_df.columns:
    test_df["true_action"] = (test_df["close"] > test_df["open"]).astype(int)
else:
    test_df["true_action"] = 1

# price_change_7d
if "close" in test_df.columns:
    test_df["price_change_7d"] = test_df.groupby("ticker")["close"].pct_change(7).fillna(0.0)
else:
    test_df["price_change_7d"] = 0.0

# macd proxy
if "close" in test_df.columns:
    ema12 = test_df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=12).mean())
    ema26 = test_df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=26).mean())
    test_df["macd"] = ema12 - ema26
else:
    test_df["macd"] = 0.0

print(f"Test set: {len(test_df):,} rows | Unique tickers: {test_df['ticker'].nunique()}")

# ── Run LLM inference ──────────────────────────────────────────────────────
from src.inference.llm_call_baseline import run

t0 = time.perf_counter()
preds_df = run(test_df, sample_size=500, out_dir=str(RESULTS_DIR))
t_end = time.perf_counter()
elapsed = t_end - t0

# ── Backtest from predictions ──────────────────────────────────────────────
from src.evaluation.backtest import run_backtest
from src.evaluation.metrics import compute_all

portfolio_df, daily_returns = run_backtest(
    predictions_df=preds_df,
    price_df=test_df,
    initial_capital=1_000_000,
    method_name="llm_call",
)

metrics = compute_all(daily_returns)

# ── Save results (overwrites placeholder from script 01) ──────────────────
accuracy = preds_df["correct"].mean() if "correct" in preds_df.columns else None
result = {
    "method": "LLM Call (Zero-Shot, claude-sonnet-4-6)",
    "source": "This work",
    "test_period": "2019-01-01 to 2023-12-31",
    "accuracy": accuracy,
    "sample_size": len(preds_df),
    "training_time_hrs": 0.0,
    "ram_usage_gb": 0.0,
    **metrics,
}
with open(RESULTS_DIR / "llm_call_metrics.json", "w") as f:
    json.dump(result, f, indent=2)

# ── Append timing ──────────────────────────────────────────────────────────
timing_path = RESULTS_DIR / "timing_log.json"
timing_data = []
if timing_path.exists():
    with open(timing_path) as f:
        timing_data = json.load(f)
timing_data.append({
    "script": "03_llm_call_baseline", "config": "claude-sonnet-4-6",
    "duration_sec": elapsed,
    "timestamp": datetime.now(timezone.utc).isoformat(),
})
with open(timing_path, "w") as f:
    json.dump(timing_data, f, indent=2)

print(f"\n  Accuracy: {accuracy:.3f}" if accuracy else "\n  Accuracy: N/A")
print(f"  Cumulative Return: {(metrics.get('cumulative_return', 0) or 0)*100:.1f}%")
print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
print(f"  Time taken: {elapsed:.0f}s")
print(f"\n[DONE] Results saved to {RESULTS_DIR}/")
print("Next: python scripts/04_vllm_benchmark.py  (A10G GPU required)")
print("      python scripts/05_generate_all_plots.py  (CPU, anytime)")
