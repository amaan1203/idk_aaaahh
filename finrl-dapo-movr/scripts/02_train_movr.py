"""
scripts/02_train_movr.py — Train All MOVR Configs + GRPO Vanilla
=================================================================
Trains DAPO+MOVR for each of the 5 configs in configs/dapo_movr.yaml,
plus GRPO vanilla (symmetric clip, no dynamic sampling) as config 0.
This is the ONLY heavy GPU script in the project.

Run: python scripts/02_train_movr.py
Requires: dataset/train_data_2013_2018.csv (run 00_download_data.py first)
Outputs:
  checkpoints/{config_name}/actor_critic.pt
  results/{config_name}_metrics.json
  results/{config_name}_backtest.csv
  results/timing_log.json (appended)
"""

import sys
import time
import json
import random
from datetime import timezone, datetime
from pathlib import Path

import numpy as np
import torch
import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()

# ── Deterministic seeds ────────────────────────────────────────────────────
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from src.data.data_loader import get_train_test_split
from src.envs.env_movr import StockTradingEnvMOVR
from src.envs.env_llm_risk import StockTradingEnvLLMRisk
from src.algorithms.dapo import DAPOAlgorithm
from src.algorithms.dapo_movr import DAPOMOVRAlgorithm
from src.evaluation.backtest import run_backtest
from src.evaluation.metrics import compute_all

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load config ────────────────────────────────────────────────────────────
cfg_name = sys.argv[1] if len(sys.argv) > 1 else "configs/dapo_movr.yaml"
cfg_path = Path(cfg_name)
if not cfg_path.exists():
    print(f"[ERROR] Missing {cfg_path}"); sys.exit(1)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

data_needed = Path("dataset/train_data_2013_2018.csv")
if not data_needed.exists():
    print(f"[ERROR] Missing {data_needed}. Run: python scripts/00_download_data.py")
    sys.exit(1)

print("\n=== Script 02: Train MOVR Configs + GRPO Vanilla ===\n")
train_df, test_df = get_train_test_split(
    train_start=cfg["train_start"], train_end=cfg["train_end"],
    test_start=cfg["test_start"],  test_end=cfg["test_end"],
)
print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")


def append_timing(entry: dict):
    timing_path = RESULTS_DIR / "timing_log.json"
    timing_data = []
    if timing_path.exists():
        with open(timing_path) as f:
            timing_data = json.load(f)
    timing_data.append(entry)
    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=2)


def run_config(name, movr_alpha, movr_beta, movr_gamma, is_grpo_vanilla=False):
    print(f"\n{'─'*60}")
    print(f"Config: {name}  (α={movr_alpha}, β={movr_beta}, γ={movr_gamma})")

    # Build training env
    train_indexed = train_df.reset_index(drop=True)
    train_indexed.index = train_indexed.index // max(1, len(train_indexed.tic.unique()))

    if is_grpo_vanilla:
        env = StockTradingEnvLLMRisk(
            df=train_indexed,
            initial_amount=cfg.get("initial_capital", 1_000_000),
            reward_alpha=0.0,  # disable sentiment weighting for vanilla GRPO
            reward_beta=0.0,   # disable risk weighting for vanilla GRPO
        )
    else:
        env = StockTradingEnvMOVR(
            df=train_indexed,
            movr_alpha=movr_alpha,
            movr_beta=movr_beta,
            movr_gamma=movr_gamma,
            movr_sharpe_window=cfg.get("sharpe_window", 20),
            initial_amount=cfg.get("initial_capital", 1_000_000),
        )

    algo_class = DAPOAlgorithm if is_grpo_vanilla else DAPOMOVRAlgorithm
    algo = algo_class(
        env=env,
        epsilon_low=cfg.get("epsilon_low", 0.2),
        epsilon_high=cfg.get("epsilon_high", 0.28),
        dynamic_sampling=False if is_grpo_vanilla else cfg.get("dynamic_sampling", True),
        symmetric_clip=is_grpo_vanilla,
        learning_rate=cfg.get("learning_rate", 3e-4),
        gamma=cfg.get("gamma", 0.99),
        train_pi_iters=cfg.get("train_pi_iters", 80),
        train_v_iters=cfg.get("train_v_iters", 80),
        target_kl=cfg.get("target_kl", 0.01),
    )

    checkpoint_dir = Path(f"checkpoints/{name}")
    t_start = time.perf_counter()
    algo.train(
        total_epochs=cfg.get("total_epochs", 100),
        steps_per_epoch=cfg.get("steps_per_epoch", 20000),
        checkpoint_dir=checkpoint_dir,
    )
    t_end = time.perf_counter()
    training_time = t_end - t_start

    # Generate signals via inference
    signals = []
    obs, _ = env.reset()
    test_dates = sorted(test_df["date"].unique())

    for date in test_dates:
        action = algo.get_action_deterministic(obs)
        day_df = test_df[test_df["date"] == date]
        tics = day_df["tic"].tolist()
        for i, tic in enumerate(tics):
            if i < len(action):
                signals.append({
                    "date": date, "ticker": tic,
                    "predicted_action": 1 if action[i] > 0.0 else 0
                })
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    preds_df = pd.DataFrame(signals)
    portfolio_df, daily_returns = run_backtest(
        predictions_df=preds_df,
        price_df=test_df,
        initial_capital=cfg.get("initial_capital", 1_000_000),
        method_name=name,
    )

    metrics = compute_all(daily_returns)

    try:
        import psutil
        ram_gb = psutil.Process().memory_info().rss / 1e9
    except ImportError:
        ram_gb = None

    result = {
        "method": f"GRPO Vanilla" if is_grpo_vanilla else f"DAPO+MOVR ({name})",
        "source": "This work",
        "test_period": "2019-01-01 to 2023-12-31",
        "movr_alpha": movr_alpha,
        "movr_beta": movr_beta,
        "movr_gamma": movr_gamma,
        "training_time_hrs": training_time / 3600,
        "ram_usage_gb": ram_gb,
        **metrics,
    }
    with open(RESULTS_DIR / f"{name}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    append_timing({
        "script": "02_train_movr", "config": name,
        "duration_sec": training_time,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    cr_pct = (metrics.get("cumulative_return", 0) or 0) * 100
    sr = metrics.get("sharpe_ratio", 0) or 0
    mdd = (metrics.get("max_drawdown", 0) or 0) * 100
    print(f"  Results → CumRet: {cr_pct:.1f}% | Sharpe: {sr:.3f} | MDD: {mdd:.1f}% | Time: {training_time/3600:.2f}h")
    return result


# ── GRPO Vanilla ────────────────────────────────────────────────────────────
print("\n[0/5] GRPO Vanilla (symmetric clip, no dynamic sampling)")
grpo_result = run_config("grpo_vanilla", 0.0, 0.0, 0.0, is_grpo_vanilla=True)

# ── MOVR Sweep ─────────────────────────────────────────────────────────────
all_results = {"grpo_vanilla": grpo_result}
sweep_configs = cfg.get("sweep_configs", [])
for idx, sweep in enumerate(sweep_configs):
    name = sweep["name"]
    print(f"\n[{idx+1}/{len(sweep_configs)}] MOVR Config: {name}")
    result = run_config(
        name=name,
        movr_alpha=sweep.get("movr_alpha", 1.0),
        movr_beta=sweep.get("movr_beta", 0.5),
        movr_gamma=sweep.get("movr_gamma", 0.3),
    )
    all_results[name] = result

# ── Comparison table ────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"{'Config':<22} {'CumRet%':>9} {'Sharpe':>8} {'MDD%':>8} {'Time(h)':>8}")
print("─" * 80)

# Load DAPO-SR target from seeded results
dapo_target_path = RESULTS_DIR / "dapo_sr_2019_metrics.json"
if dapo_target_path.exists():
    with open(dapo_target_path) as f:
        dapo_target = json.load(f)
    cr = (dapo_target.get("cumulative_return") or 0) * 100
    sr = dapo_target.get("sharpe_ratio") or float("nan")
    mdd = (dapo_target.get("max_drawdown") or 0) * 100
    t = dapo_target.get("training_time_hrs") or 0
    print(f"  {'DAPO-SR (paper) †':<20} {cr:>9.1f} {str(sr):>8} {mdd:>8.1f} {t:>8.2f}")
    print("─" * 80)

best_sharpe = -999
best_name = ""
for name, r in all_results.items():
    cr = (r.get("cumulative_return") or 0) * 100
    sr = r.get("sharpe_ratio") or 0
    mdd = (r.get("max_drawdown") or 0) * 100
    t = r.get("training_time_hrs") or 0
    marker = " ★" if name == "balanced" else ""
    print(f"  {name+marker:<20} {cr:>9.1f} {sr:>8.3f} {mdd:>8.1f} {t:>8.2f}")
    if sr > best_sharpe:
        best_sharpe = sr
        best_name = name

print("=" * 80)
print(f"Best Sharpe: {best_name} ({best_sharpe:.3f})")
print(f"\n[DONE] Results saved to {RESULTS_DIR}/")
print("Next: python scripts/03_llm_call_baseline.py (requires ANTHROPIC_API_KEY)")
print("      python scripts/04_vllm_benchmark.py (requires A10G GPU)")
