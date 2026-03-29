"""
src/algorithms/sft_trainer.py — Supervised Finetuning Trainer
=============================================================
Trains a Qwen2.5-1.5B-Instruct model for binary stock direction classification.
Corresponds to: SFT baseline in the benchmarking comparison.

Label construction:
  true_action = 1 if next_day_close > open else 0  (directional correctness)

Prompt format (same as DeepStock):
  Given: stock={ticker}, price_change_7d={val:.2%}, sentiment={s}/5, risk={r}/5
  Predict: will the price go UP or DOWN tomorrow?
  Answer:

Inputs: CSV dataframes from dataset/, configs/sft.yaml
Outputs: checkpoints/sft/, results/sft_training_log.json, results/sft_predictions.csv
"""

import json
import time
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset


def build_prompt(row: pd.Series) -> str:
    """Build the DeepStock-compatible prompt for a single row."""
    ticker = row.get("tic", row.get("ticker", "UNKNOWN"))
    price_change = float(row.get("price_change_7d", 0.0))
    sentiment = float(row.get("sentiment", 3.0))
    risk = float(row.get("risk", 3.0))
    return (
        f"Given: stock={ticker}, price_change_7d={price_change:.2%}, "
        f"sentiment={sentiment:.1f}/5, risk={risk:.1f}/5\n"
        f"Predict: will the price go UP or DOWN tomorrow?\n"
        f"Answer:"
    )


class StockDataset(Dataset):
    """PyTorch dataset for SFT/LoRA training."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts = [build_prompt(row) for _, row in df.iterrows()]
        self.labels = df["true_action"].astype(int).tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.prompts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def prepare_dataframes(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Add true_action column: 1 if next-day close > open, else 0.
    Uses a simple forward-shift of close vs open.
    """
    for df in [train_df, test_df]:
        if "true_action" not in df.columns:
            if "close" in df.columns and "open" in df.columns:
                df["true_action"] = (df["close"] > df["open"]).astype(int)
            else:
                df["true_action"] = 1  # fallback
        if "price_change_7d" not in df.columns:
            if "close" in df.columns:
                df["price_change_7d"] = df["close"].pct_change(7).fillna(0.0)
            else:
                df["price_change_7d"] = 0.0
    return train_df, test_df


class SFTTrainer:
    """
    Supervised finetuning trainer for stock direction prediction.

    Parameters
    ----------
    config : dict loaded from configs/sft.yaml
    checkpoint_dir : where to save the model (default checkpoints/sft/)
    """

    def __init__(
        self,
        config: dict,
        checkpoint_dir: Path = Path("checkpoints/sft"),
    ):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.base_model = config.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        print(f"Loading {self.base_model} for SFT...", end=" ", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=self.config.get("num_labels", 2),
            torch_dtype=torch.bfloat16 if self.config.get("bf16", True) else torch.float32,
            device_map="auto",
        )
        print("done.")

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """
        Train the model and return training metrics.

        Returns
        -------
        dict with keys: training_log, training_time_seconds
        """
        train_df, test_df = prepare_dataframes(train_df, test_df)
        self._load_model()

        train_dataset = StockDataset(train_df, self.tokenizer, self.config.get("max_seq_length", 256))
        eval_dataset = StockDataset(test_df, self.tokenizer, self.config.get("max_seq_length", 256))

        training_args = TrainingArguments(
            output_dir=str(self.checkpoint_dir),
            num_train_epochs=self.config.get("num_train_epochs", 3),
            per_device_train_batch_size=self.config.get("per_device_train_batch_size", 16),
            per_device_eval_batch_size=self.config.get("per_device_eval_batch_size", 32),
            learning_rate=self.config.get("learning_rate", 2e-5),
            warmup_ratio=self.config.get("warmup_ratio", 0.1),
            weight_decay=self.config.get("weight_decay", 0.01),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            bf16=self.config.get("bf16", True),
            fp16=self.config.get("fp16", False),
            logging_dir=str(self.checkpoint_dir / "logs"),
            logging_steps=50,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        start_time = time.perf_counter()
        train_result = trainer.train()
        training_time = time.perf_counter() - start_time

        # Save model
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(self.checkpoint_dir))
        self.tokenizer.save_pretrained(str(self.checkpoint_dir))

        training_log = {
            "algorithm": "sft",
            "base_model": self.base_model,
            "training_time_seconds": training_time,
            "training_time_hours": training_time / 3600,
            "train_loss": train_result.training_loss,
            "train_samples": len(train_df),
        }

        return training_log

    def predict(self, test_df: pd.DataFrame, output_path: Path = Path("results/sft_predictions.csv")) -> pd.DataFrame:
        """Run inference on test_df and save predictions."""
        if self.model is None:
            raise RuntimeError("Call train() or load_checkpoint() before predict()")

        test_df, _ = prepare_dataframes(test_df, test_df)
        self.model.eval()

        predictions = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="SFT inference"):
            prompt = build_prompt(row)
            enc = self.tokenizer(
                prompt,
                max_length=self.config.get("max_seq_length", 256),
                return_tensors="pt",
                truncation=True,
            ).to(self.model.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            predicted = logits.argmax(-1).item()
            true_action = int(row.get("true_action", 0))
            predictions.append({
                "date": row.get("date", None),
                "ticker": row.get("tic", row.get("ticker", "UNKNOWN")),
                "predicted_action": predicted,
                "true_action": true_action,
                "correct": int(predicted == true_action),
            })

        results_df = pd.DataFrame(predictions)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        accuracy = results_df["correct"].mean()
        print(f"SFT test accuracy: {accuracy:.3f}")
        return results_df
