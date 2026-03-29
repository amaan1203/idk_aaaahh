"""
src/algorithms/lora_trainer.py — LoRA Finetuning Trainer
=========================================================
Identical to sft_trainer.py but wraps the model with PEFT LoRA adapters
before training. Only the LoRA adapter weights are updated during training.

Corresponds to: LoRA baseline in the benchmarking comparison.

Inputs: CSV dataframes from dataset/, configs/lora.yaml
Outputs: checkpoints/lora/, results/lora_training_log.json, results/lora_predictions.csv
"""

import time
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.algorithms.sft_trainer import SFTTrainer, StockDataset, prepare_dataframes
from transformers import Trainer, TrainingArguments
from tqdm import tqdm


class LoRATrainer(SFTTrainer):
    """
    LoRA finetuning trainer — wraps Qwen2.5-1.5B with PEFT LoRA adapters.

    Only the LoRA adapter matrices (r=8, alpha=16) are updated during training.
    All other model parameters are frozen. This dramatically reduces trainable
    parameters and memory footprint.
    """

    def _load_model(self):
        print(f"Loading {self.base_model} with LoRA...", end=" ", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=self.config.get("num_labels", 2),
            torch_dtype=torch.bfloat16 if self.config.get("bf16", True) else torch.float32,
            device_map="auto",
        )

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.config.get("lora_r", 8),
            lora_alpha=self.config.get("lora_alpha", 16),
            lora_dropout=self.config.get("lora_dropout", 0.1),
            target_modules=self.config.get("target_modules", ["q_proj", "v_proj"]),
            bias=self.config.get("bias", "none"),
        )

        self.model = get_peft_model(base_model, lora_config)
        trainable, total = self.model.get_nb_trainable_parameters()
        print(f"done. Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """Train LoRA adapters only."""
        start_time = time.perf_counter()
        training_log = super().train(train_df, test_df)
        training_time = time.perf_counter() - start_time

        # Override timing (parent also times, but LoRA time includes _load_model overhead)
        training_log["algorithm"] = "lora"
        training_log["lora_r"] = self.config.get("lora_r", 8)
        training_log["lora_alpha"] = self.config.get("lora_alpha", 16)
        return training_log
