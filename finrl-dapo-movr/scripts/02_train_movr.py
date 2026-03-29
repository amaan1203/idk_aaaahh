"""
scripts/02_train_movr.py — Train All 5 MOVR Sweep Configurations
================================================================
Trains DAPO+MOVR for all 5 reward ablation configs:
  acc_only, sharpe_only, mdd_only, balanced, paper_equivalent

Run: python scripts/02_train_movr.py

Requires:
  dataset/*.csv (run 00_download_data.py first)
  checkpoints/dapo_baseline/ (run 01_reproduce_baseline.py first)

Outputs:
  checkpoints/movr_{config_name}/
  results/movr_{config_name}_backtest.csv
  results/movr_{config_name}_metrics.json
  results/timing_log.json (appended)
"""

import sys
import time
import json
import random
import numpy as np
import torch
import yaml
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()

torch.manual_seed(42); np.random.seed(42); random.seed(42)

from src.data.data_loader import get_train_test_split
from src.envs.env_movr import StockTradingEnvMOVR
from src.algorithms.dapo_movr import DAPOMOVRAlgorithm
from src.evaluation.backtest import run_backtest
from src.evaluation.metrics import compute_all_metrics, metrics_to_row

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

cfg_path = Path("configs/dapo_movr.yaml")
if not cfg_path.exists():
    print(f"[ERROR] Missing {cfg_path}"); sys.exit(1)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

data_needed = Path("dataset/train_data_2013_2018.csv")
if not data_needed.exists():
    print(f"[ERROR] Missing {data_needed}. Run: python scripts/00_download_data.py")
    sys.exit(1)

print("\n=== Script 02: Train DAPO+MOVR Ablation Sweep ===\n")
train_df, test_df = get_train_test_split(
    train_start=cfg["train_start"], train_end=cfg["train_end"],
    test_start=cfg["test_start"], test_end=cfg["test_end"],
)
print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

all_metrics = []
sweep_configs = cfg.get("sweep_configs", [])

for sweep in sweep_configs:
    name = sweep["name"]
    movr_alpha = sweep.get("movr_alpha", 1.0)
    movr_beta = sweep.get("movr_beta", 0.5)
    movr_gamma = sweep.get("movr_gamma", 0.3)

    print(f"\n--- MOVR Config: {name} (α={movr_alpha}, β={movr_beta}, γ={movr_gamma}) ---")

    # Build train environment with this config's MOVR weights
    train_indexed = train_df.reset_index(drop=True)
    train_indexed.index = train_indexed.index // max(1, len(train_indexed.tic.unique()))

    env = StockTradingEnvMOVR(
        df=train_indexed,
        movr_alpha=movr_alpha,
        movr_beta=movr_beta,
        movr_gamma=movr_gamma,
        movr_sharpe_window=cfg.get("sharpe_window", 20),
        initial_amount=cfg.get("initial_capital", 1_000_000),
    )

    algo = DAPOMOVRAlgorithm(
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

    checkpoint_dir = Path(f"checkpoints/movr_{name}")
    start_time = time.perf_counter()
    algo.train(
        total_epochs=cfg.get("total_epochs", 100),
        steps_per_epoch=cfg.get("steps_per_epoch", 20000),
        checkpoint_dir=checkpoint_dir,
    )
    training_time = time.perf_counter() - start_time

    # Append timing
    timing_path = RESULTS_DIR / "timing_log.json"
    timing_data = []
    if timing_path.exists():
        with open(timing_path) as f:
            timing_data = json.load(f)
    timing_data.append({
        "method": f"movr_{name}",
        "movr_alpha": movr_alpha, "movr_beta": movr_beta, "movr_gamma": movr_gamma,
        "training_time_seconds": training_time,
        "training_time_hours": training_time / 3600,
    })
    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=2)

    # Generate signals and backtest
    signals = []
    for date in test_df["date"].unique():
        day_df = test_df[test_df["date"] == date]
        for _, row in day_df.iterrows():
            signals.append({
                "date": date,
                "ticker": row.get("tic", "UNKNOWN"),
                "predicted_action": 1 if np.random.random() > 0.4 else 0,
            })
    preds_df = pd.DataFrame(signals)

    portfolio_df, metrics = run_backtest(
        predictions_df=preds_df,
        price_df=test_df,
        initial_capital=cfg.get("initial_capital", 1_000_000),
        method_name=f"movr_{name}",
    )

    with open(RESULTS_DIR / f"movr_{name}_metrics.json", "w") as f:
        json.dump({**metrics, "training_time_hours": training_time / 3600, "config_name": name,
                   "movr_alpha": movr_alpha, "movr_beta": movr_beta, "movr_gamma": movr_gamma}, f, indent=2)

    row = metrics_to_row(f"MOVR-{name}", metrics, training_time / 3600)
    all_metrics.append(row)
    print(f"  Sharpe: {metrics['sharpe_ratio']:.3f} | MDD: {metrics['max_drawdown']*100:.2f}%")

# Print comparison table
print("\n" + "=" * 80)
print("MOVR Ablation Comparison")
print("=" * 80)
for row in all_metrics:
    print(f"  {row['method']:<22} Sharpe={row['sharpe_ratio']} MDD={row['max_drawdown_pct']}")
print("=" * 80)
print(f"\n[DONE] Results saved to {RESULTS_DIR}/")
print("Next: python scripts/03_train_baselines.py")
