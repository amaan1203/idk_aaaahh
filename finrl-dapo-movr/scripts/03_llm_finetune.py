"""
scripts/03_llm_finetune.py — LLM Fine-Tuning with GRPO / DAPO / DAPO+MOVR
==========================================================================
Trains a smol base model three times, each with a different
reward function.
"""

import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import get_train_test_split, build_prompt_dataset
from src.rewards.movr_reward_fn import (
    grpo_vanilla_reward,
    dapo_reward,
    dapo_movr_reward,
)
from src.evaluation.metrics import compute_all

# ── Reproducibility ────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── Config ─────────────────────────────────────────────────────────────────
cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/dapo_nifty.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME   = cfg.get("llm_model", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_STEPS    = cfg.get("llm_max_steps", 200)
LR           = 5e-6 
NIFTY_BASELINE_CUMRET = 22.24

# ── Data ───────────────────────────────────────────────────────────────────
print("\n=== Script 03: LLM Fine-Tuning (GRPO / DAPO / DAPO+MOVR) ===\n")
train_df, test_df = get_train_test_split(
    train_start=cfg["train_start"], train_end=cfg["train_end"],
    test_start=cfg["test_start"],  test_end=cfg["test_end"],
    data_path=cfg.get("data_path"),
)
print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

print("  Building prompt dataset...")
train_records = build_prompt_dataset(train_df, lookback_days=5)
test_records  = build_prompt_dataset(test_df,  lookback_days=5)
print(f"  Train prompts: {len(train_records):,} | Test prompts: {len(test_records):,}")

# ── HuggingFace Datasets ───────────────────────────────────────────────────
try:
    from datasets import Dataset
except ImportError:
    print("[ERROR] 'datasets' not installed. Run: pip install datasets")
    sys.exit(1)

hf_train = Dataset.from_list([
    {
        "prompt": r["prompt"], 
        "future_return": r["future_return"],
        "return_history": r["return_history"]
    }
    for r in train_records
])


# ── Model loader ────────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_path: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[ERROR] 'transformers' not installed. Run: pip install transformers")
        sys.exit(1)

    print(f"  Loading from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,   # Use float32 for stability on MacOS
        device_map="auto",
    )
    return model, tokenizer


# ── Training function ───────────────────────────────────────────────────────
def run_llm_finetune(method_name: str, alpha=1.0, beta=0.5, gamma=0.3):
    checkpoint_dir = Path(f"checkpoints/llm_{method_name.replace(' ', '_').lower()}")
    
    if checkpoint_dir.exists() and any(checkpoint_dir.glob("*.safetensors")):
        print(f"  [SKIP] Existing checkpoint found at {checkpoint_dir}.")
        model, tokenizer = load_model_and_tokenizer(str(checkpoint_dir))
        return model, tokenizer, 0

    try:
        from trl import GRPOTrainer, GRPOConfig
    except ImportError:
        print("[ERROR] 'trl' not installed. Run: pip install trl")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"Method: {method_name}")

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    def reward_wrapper(prompts, completions, **kwargs):
        future_returns = kwargs.get("future_return", [0.0] * len(completions))
        return_history = kwargs.get("return_history", [[]] * len(completions))
        
        if method_name == "GRPO Vanilla":
            return grpo_vanilla_reward(prompts, completions, future_returns=future_returns)
        elif method_name == "DAPO":
            return dapo_reward(prompts, completions, future_returns=future_returns)
        else:
            return dapo_movr_reward(
                prompts, completions,
                alpha=alpha, beta=beta, gamma=gamma,
                future_returns=future_returns,
                return_history=return_history,
            )

    training_args = GRPOConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=1,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        generation_batch_size=8,
        learning_rate=LR,
        logging_steps=5,
        save_steps=MAX_STEPS,
        report_to="none",
        seed=42,
        max_completion_length=16,
        num_generations=8,
        temperature=0.7,
        bf16=False,
        fp16=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_wrapper],
        args=training_args,
        train_dataset=hf_train,
    )

    t_start = time.perf_counter()
    trainer.train()
    t_elapsed = time.perf_counter() - t_start

    trainer.save_model(str(checkpoint_dir))
    print(f"  Checkpoint saved → {checkpoint_dir} ({t_elapsed/60:.1f} min)")

    return model, tokenizer, t_elapsed


def run_inference_and_backtest(model, tokenizer, method_name: str):
    result_path = RESULTS_DIR / f"llm_{method_name.replace(' ', '_').lower()}_metrics.json"
    
    if result_path.exists():
        print(f"  [SKIP] Inference results already exist {result_path}.")
        with open(result_path) as f:
            return json.load(f)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Target device: {device}")
    model = model.to(device)
    
    subsampled_test = test_records[::100]
    print(f"  Running inference on {len(subsampled_test):,} prompts...")
    model.eval()

    inference_batch_size = 8
    portfolio_returns = []
    correct: int = 0
    total = len(subsampled_test)
    
    for i in range(0, total, inference_batch_size):
        batch = subsampled_test[i : i + inference_batch_size]
        prompts = [r["prompt"] for r in batch]
        
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=10, do_sample=False
            )
            if device == "mps":
                torch.mps.empty_cache()
        
        for j, rec in enumerate(batch):
            completion = outputs[j][inputs["input_ids"].shape[-1]:]
            decoded = tokenizer.decode(completion, skip_special_tokens=True).upper()
            
            if "BUY" in decoded:
                action = "BUY"
            elif "SELL" in decoded:
                action = "SELL"
            else:
                action = "HOLD"

            fr = rec["future_return"]
            port_r: float = 0.0
            if action == "BUY":
                port_r = fr
            elif action == "SELL":
                port_r = -fr
            
            portfolio_returns.append(port_r)
            if port_r > 0:
                correct += 1
        
        if (i // inference_batch_size) % 50 == 0:
            print(f"    Processed {min(i + inference_batch_size, total):,} / {total:,}...")

    metrics = compute_all(portfolio_returns)
    metrics["directional_accuracy"] = correct / max(len(portfolio_returns), 1)

    result = {
        "method": method_name,
        **metrics,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    cr  = (metrics.get("cumulative_return",   0) or 0) * 100
    sr  = metrics.get("sharpe_ratio",          0) or 0
    mdd = (metrics.get("max_drawdown",         0) or 0) * 100
    acc = metrics.get("directional_accuracy",  0) * 100
    print(f"  CumRet: {cr:.1f}% | Sharpe: {sr:.3f} | MDD: {mdd:.1f}% | DirAcc: {acc:.1f}%")
    return metrics


# ── Main loop ──────────────────────────────────────────────────────────────
run_configs = [
    {
        "name": "GRPO Vanilla",
        "alpha": 0.0, "beta": 0.0, "gamma": 0.0,
    },
    {
        "name": "DAPO",
        "alpha": 0.0, "beta": 0.0, "gamma": 0.0,
    },
    {
        "name": "DAPO+MOVR",
        "alpha": cfg.get("sweep_configs", [{}])[0].get("movr_alpha", 1.0),
        "beta":  cfg.get("sweep_configs", [{}])[0].get("movr_beta",  0.5),
        "gamma": cfg.get("sweep_configs", [{}])[0].get("movr_gamma", 0.3),
    },
]

all_results = {}
for run in run_configs:
    model, tokenizer, elapsed = run_llm_finetune(
        method_name=run["name"],
        alpha=run["alpha"], beta=run["beta"], gamma=run["gamma"],
    )
    metrics = run_inference_and_backtest(model, tokenizer, run["name"])
    all_results[run["name"]] = metrics
    del model

# ── Final comparison table ─────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"{'Method':<22} {'CumRet%':>9} {'Sharpe':>8} {'MDD%':>8} {'DirAcc%':>9}")
print("─" * 80)
print(f"  {'Nifty BnH Baseline':<20} {NIFTY_BASELINE_CUMRET:>9.1f} {'0.323':>8} {'-39.2':>8} {'—':>9}")
print("─" * 80)

for name, m in all_results.items():
    cr  = (m.get("cumulative_return", 0) or 0) * 100
    sr  = float(m.get("sharpe_ratio",  0) or 0)
    mdd = (m.get("max_drawdown",       0) or 0) * 100
    acc = (m.get("directional_accuracy", 0) or 0) * 100
    display_name = str(name) + " ★" if name == "DAPO+MOVR" else str(name)
    print(f"  {display_name:<22} {cr:>9.1f} {sr:>8.3f} {mdd:>8.1f} {acc:>9.1f}")

print("=" * 80)
print(f"\n[DONE] Results saved to {RESULTS_DIR}/")
