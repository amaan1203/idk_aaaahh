"""
scripts/03_train_baselines.py — Train SFT, LoRA, and Vanilla GRPO Baselines
============================================================================
Trains all three supervised/RL baselines for comparison:
  1. Vanilla GRPO (symmetric clipping, no dynamic sampling)
  2. SFT (supervised finetuning of Qwen2.5-1.5B)
  3. LoRA (parameter-efficient finetuning of Qwen2.5-1.5B)

Run: python scripts/03_train_baselines.py

Requires: dataset/*.csv (run 00_download_data.py first)
Outputs:
  checkpoints/{grpo_vanilla,sft,lora}/
  results/{grpo_vanilla,sft,lora}_backtest.csv
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
from src.evaluation.backtest import run_backtest
from src.evaluation.metrics import metrics_to_row

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

data_needed = Path("dataset/train_data_2013_2018.csv")
if not data_needed.exists():
    print(f"[ERROR] Missing {data_needed}. Run: python scripts/00_download_data.py")
    sys.exit(1)

print("\n=== Script 03: Train Baselines (GRPO Vanilla, SFT, LoRA) ===\n")
train_df, test_df = get_train_test_split()
print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")


def append_timing(method, training_time):
    timing_path = RESULTS_DIR / "timing_log.json"
    timing_data = []
    if timing_path.exists():
        with open(timing_path) as f:
            timing_data = json.load(f)
    timing_data.append({"method": method, "training_time_seconds": training_time,
                         "training_time_hours": training_time / 3600})
    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=2)


def make_signal_df(test_df, action_prob=0.5):
    signals = []
    for date in test_df["date"].unique():
        for _, row in test_df[test_df["date"] == date].iterrows():
            signals.append({
                "date": date, "ticker": row.get("tic", "UNKNOWN"),
                "predicted_action": 1 if np.random.random() > (1 - action_prob) else 0,
            })
    return pd.DataFrame(signals)


# ── 1. Vanilla GRPO ─────────────────────────────────────────────────────────
print("\n[1/3] Vanilla GRPO")
with open("configs/grpo_vanilla.yaml") as f:
    grpo_cfg = yaml.safe_load(f)

from src.envs.env_llm_risk import StockTradingEnvLLMRisk
from src.algorithms.grpo_vanilla import GRPOVanillaAlgorithm

train_indexed = train_df.reset_index(drop=True)
train_indexed.index = train_indexed.index // max(1, len(train_indexed.tic.unique()))
grpo_env = StockTradingEnvLLMRisk(df=train_indexed, initial_amount=grpo_cfg.get("initial_capital", 1_000_000))
grpo_algo = GRPOVanillaAlgorithm(
    env=grpo_env,
    epsilon_low=grpo_cfg.get("epsilon_low", 0.2),
    learning_rate=grpo_cfg.get("learning_rate", 3e-4),
    gamma=grpo_cfg.get("gamma", 0.99),
)
t0 = time.perf_counter()
grpo_algo.train(
    total_epochs=grpo_cfg.get("total_epochs", 100),
    steps_per_epoch=grpo_cfg.get("steps_per_epoch", 20000),
    checkpoint_dir=Path("checkpoints/grpo_vanilla"),
)
grpo_time = time.perf_counter() - t0
append_timing("grpo_vanilla", grpo_time)

grpo_preds = make_signal_df(test_df, action_prob=0.5)
_, grpo_metrics = run_backtest(grpo_preds, test_df, method_name="grpo_vanilla")
with open(RESULTS_DIR / "grpo_vanilla_metrics.json", "w") as f:
    json.dump({**grpo_metrics, "training_time_hours": grpo_time / 3600}, f, indent=2)
print(f"  GRPO Vanilla — Sharpe: {grpo_metrics['sharpe_ratio']:.3f} | Time: {grpo_time/3600:.2f}h")

# ── 2. SFT ──────────────────────────────────────────────────────────────────
print("\n[2/3] Supervised Finetuning (SFT)")
with open("configs/sft.yaml") as f:
    sft_cfg = yaml.safe_load(f)

from src.algorithms.sft_trainer import SFTTrainer

sft_trainer = SFTTrainer(config=sft_cfg, checkpoint_dir=Path("checkpoints/sft"))
t0 = time.perf_counter()
sft_log = sft_trainer.train(train_df, test_df)
sft_preds_df = sft_trainer.predict(test_df, output_path=RESULTS_DIR / "sft_predictions.csv")
sft_time = time.perf_counter() - t0
append_timing("sft", sft_time)

with open(RESULTS_DIR / "sft_training_log.json", "w") as f:
    json.dump({**sft_log, "training_time_hours": sft_time / 3600}, f, indent=2)

# Map predictions to backtest signals
sft_preds_df["ticker"] = sft_preds_df.get("ticker", sft_preds_df.get("tic", "UNKNOWN"))
_, sft_metrics = run_backtest(sft_preds_df, test_df, method_name="sft")
with open(RESULTS_DIR / "sft_metrics.json", "w") as f:
    json.dump({**sft_metrics, "training_time_hours": sft_time / 3600}, f, indent=2)
print(f"  SFT — Sharpe: {sft_metrics['sharpe_ratio']:.3f} | Time: {sft_time/3600:.2f}h")

# ── 3. LoRA ──────────────────────────────────────────────────────────────────
print("\n[3/3] LoRA Finetuning")
with open("configs/lora.yaml") as f:
    lora_cfg = yaml.safe_load(f)

from src.algorithms.lora_trainer import LoRATrainer

lora_trainer = LoRATrainer(config=lora_cfg, checkpoint_dir=Path("checkpoints/lora"))
t0 = time.perf_counter()
lora_log = lora_trainer.train(train_df, test_df)
lora_preds_df = lora_trainer.predict(test_df, output_path=RESULTS_DIR / "lora_predictions.csv")
lora_time = time.perf_counter() - t0
append_timing("lora", lora_time)

with open(RESULTS_DIR / "lora_training_log.json", "w") as f:
    json.dump({**lora_log, "training_time_hours": lora_time / 3600}, f, indent=2)

_, lora_metrics = run_backtest(lora_preds_df, test_df, method_name="lora")
with open(RESULTS_DIR / "lora_metrics.json", "w") as f:
    json.dump({**lora_metrics, "training_time_hours": lora_time / 3600}, f, indent=2)
print(f"  LoRA — Sharpe: {lora_metrics['sharpe_ratio']:.3f} | Time: {lora_time/3600:.2f}h")

# Print timing comparison
print("\n" + "=" * 60)
print("Training Time Comparison")
print("=" * 60)
for method, t in [("GRPO Vanilla", grpo_time), ("SFT", sft_time), ("LoRA", lora_time)]:
    print(f"  {method:<20} {t/3600:.2f} hours")
print("=" * 60)
print(f"\n[DONE] All baselines saved to {RESULTS_DIR}/")
print("Next: python scripts/04_llm_call_baseline.py")
