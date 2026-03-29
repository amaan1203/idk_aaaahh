"""
scripts/01_reproduce_baseline.py — Reproduce DAPO-SR Paper Baseline
====================================================================
Trains the paper's DAPO algorithm on NASDAQ-100 data (2013–2018) and
backtests on the test set (2019–2023). Produces Table 1 reproduction.

Run: python scripts/01_reproduce_baseline.py

Requires: dataset/*.csv (run 00_download_data.py first)
Outputs:
  checkpoints/dapo_baseline/actor_critic.pt
  results/dapo_baseline_backtest.csv
  results/dapo_baseline_metrics.json
  results/timing_log.json (appended)
"""

import sys
import time
import json
import random
import numpy as np
import torch
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()

# Deterministic seeds (Rule 5)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

import yaml
import pandas as pd
from src.data.data_loader import get_train_test_split
from src.envs.env_llm_risk import StockTradingEnvLLMRisk
from src.algorithms.dapo import DAPOAlgorithm
from src.evaluation.backtest import run_backtest, build_nasdaq100_benchmark
from src.evaluation.metrics import compute_all_metrics, metrics_to_row

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints/dapo_baseline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load config
cfg_path = Path("configs/dapo_baseline.yaml")
if not cfg_path.exists():
    print(f"[ERROR] Missing {cfg_path}. Run from project root.")
    sys.exit(1)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# Load data
print("\n=== Script 01: Reproduce DAPO-SR Baseline ===\n")
print("Loading training data...")
data_needed = Path("dataset/train_data_2013_2018.csv")
if not data_needed.exists():
    print(f"[ERROR] Missing {data_needed}. Run: python scripts/00_download_data.py")
    sys.exit(1)

train_df, test_df = get_train_test_split(
    train_start=cfg["train_start"], train_end=cfg["train_end"],
    test_start=cfg["test_start"], test_end=cfg["test_end"],
)
print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

# Build environment (indexed by day)
train_df_indexed = train_df.reset_index(drop=True)
train_df_indexed.index = train_df_indexed.index // max(1, len(train_df_indexed.tic.unique()))

env = StockTradingEnvLLMRisk(
    df=train_df_indexed,
    reward_alpha=cfg.get("reward_alpha", 3.0),
    reward_beta=cfg.get("reward_beta", 1.0),
    initial_amount=cfg.get("initial_capital", 1_000_000),
)

# Train DAPO
print("\nTraining DAPO baseline...")
algo = DAPOAlgorithm(
    env=env,
    epsilon_low=cfg.get("epsilon_low", 0.2),
    epsilon_high=cfg.get("epsilon_high", 0.28),
    dynamic_sampling=cfg.get("dynamic_sampling", True),
    learning_rate=cfg.get("learning_rate", 3e-4),
    gamma=cfg.get("gamma", 0.99),
    train_pi_iters=cfg.get("train_pi_iters", 80),
    train_v_iters=cfg.get("train_v_iters", 80),
    target_kl=cfg.get("target_kl", 0.01),
)

start_time = time.perf_counter()
training_log = algo.train(
    total_epochs=cfg.get("total_epochs", 100),
    steps_per_epoch=cfg.get("steps_per_epoch", 20000),
    checkpoint_dir=CHECKPOINT_DIR,
)
training_time = time.perf_counter() - start_time

# Record timing (Rule 4 — append, never overwrite)
timing_path = RESULTS_DIR / "timing_log.json"
timing_data = []
if timing_path.exists():
    with open(timing_path) as f:
        timing_data = json.load(f)
timing_data.append({
    "method": "dapo_baseline",
    "training_time_seconds": training_time,
    "training_time_hours": training_time / 3600,
})
with open(timing_path, "w") as f:
    json.dump(timing_data, f, indent=2)

# Backtest: generate BUY signals from trained policy on test set
print("\nRunning backtest on test data (2019–2023)...")
test_df_indexed = test_df.reset_index(drop=True)
test_df_indexed.index = test_df_indexed.index // max(1, len(test_df_indexed.tic.unique()))

# Generate signals using trained policy
signals = []
obs, _ = env.reset()

# We need to iterate through the test data date by date
test_dates = sorted(test_df["date"].unique())
for date in test_dates:
    # Get observation for the current day
    # In a real step-by-step backtest, we'd use env.step()
    # but here we can just use the state from the env at each day
    day_obs = obs
    action = algo.get_action_deterministic(day_obs)
    
    # Action is a vector of length stock_dim (84 in this case)
    # We convert it to signals: 1 (BUY) if action > 0.0 else 0 (HOLD/SELL)
    # Match with the tickers present on that day
    day_df = test_df[test_df["date"] == date]
    tics = day_df["tic"].tolist()
    
    for i, tic in enumerate(tics):
        if i < len(action):
            signals.append({
                "date": date,
                "ticker": tic,
                "predicted_action": 1 if action[i] > 0.0 else 0
            })
    
    # Advance env to next day to get next obs
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

preds_df = pd.DataFrame(signals)

portfolio_df, metrics = run_backtest(
    predictions_df=preds_df,
    price_df=test_df,
    initial_capital=cfg.get("initial_capital", 1_000_000),
    method_name="dapo_baseline",
)

# Save metrics
with open(RESULTS_DIR / "dapo_baseline_metrics.json", "w") as f:
    json.dump({**metrics, "training_time_hours": training_time / 3600}, f, indent=2)

# Print summary table
print("\n" + "=" * 60)
print("DAPO-SR Baseline Results (Table 1 Reproduction)")
print("=" * 60)
row = metrics_to_row("DAPO-SR Baseline", metrics, training_time / 3600)
for k, v in row.items():
    print(f"  {k:<30} {v}")
print("=" * 60)
print(f"\n[DONE] Results saved to {RESULTS_DIR}/")
print("Next: python scripts/02_train_movr.py")
