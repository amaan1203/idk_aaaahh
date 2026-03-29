# FinRL-DAPO-MOVR: Implementation Plan
### For autonomous codebase generation via Claude Code

**Project**: Comparative Benchmarking of LLM Finetuning Techniques on Financial Time Series  
**Base paper**: *A New DAPO Algorithm for Stock Trading* — arXiv:2505.06408 (IEEE IDS 2025)  
**Base repo**: https://github.com/Ruijian-Zha/FinRL-DAPO-SR  
**Novel contributions**: Multi-Objective Verifiable Reward (MOVR / Fin-RLVR) + vLLM rollout acceleration  
**Target runtime**: Lightning.ai Studio — A10G GPU (24 GB VRAM), Ubuntu 22.04  

---

## 0. Project overview and what you are building

You are building a research codebase that does three things simultaneously:
make sure i dont require any paid API keys to run and generate results for this project now. so use OPENSOURCE MODELS only. DONT USE ANTHROPIC API KEY FOR LLM CALLS. USE OPENSOURCE LLM FOR THAT. 

1. **Reproduces** the FinRL-DAPO-SR paper baseline on NASDAQ-100 data.
2. **Extends** it with a novel Multi-Objective Verifiable Reward (MOVR) function that replaces the paper's single exponentiated reward with a three-component vector reward: `R = α·accuracy + β·ΔSharpe − γ·drawdown_penalty`. Every component is verifiable from market data — no human labels.
3. **Benchmarks** it against four baselines (Zero-shot LLM call, SFT, LoRA, vanilla GRPO) and produces publication-quality plots for all comparisons.

In parallel it benchmarks vLLM-served rollout generation against HuggingFace `generate()` and produces a throughput / speedup curve — the systems contribution.

The final deliverable is a runnable codebase that a user drops onto a Lightning.ai A10G Studio, runs `bash setup.sh`, and then runs each numbered script in order.

---

## 1. Repository layout to generate

Generate exactly this directory and file structure. Do not add or rename anything.

```
finrl-dapo-movr/
│
├── setup.sh                          # one-shot environment bootstrap
├── requirements.txt                  # pinned Python dependencies
├── .env.example                      # template for secrets (ANTHROPIC_API_KEY, HF_TOKEN)
│
├── configs/
│   ├── dapo_baseline.yaml            # paper's original hyperparameters
│   ├── dapo_movr.yaml                # MOVR reward hyperparameters (α, β, γ sweep)
│   ├── grpo_vanilla.yaml             # symmetric-clip GRPO (no DAPO mods)
│   ├── sft.yaml                      # supervised finetuning config
│   └── lora.yaml                     # LoRA finetuning config
│
├── src/
│   ├── __init__.py
│   │
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── env_base.py               # copy of original env_stocktrading.py
│   │   ├── env_llm_risk.py           # copy of original env_stocktrading_llm_risk.py
│   │   └── env_movr.py               # NEW: env with MOVR reward (extends env_llm_risk.py)
│   │
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── dapo.py                   # copy of original dapo_algorithm.py (untouched)
│   │   ├── dapo_movr.py              # NEW: DAPO with MOVR reward hook
│   │   ├── grpo_vanilla.py           # DAPO with symmetric ε (standard GRPO behaviour)
│   │   ├── sft_trainer.py            # NEW: supervised finetuning trainer
│   │   └── lora_trainer.py           # NEW: LoRA finetuning trainer
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── hf_generator.py           # HuggingFace generate() wrapper
│   │   ├── vllm_generator.py         # NEW: vLLM OpenAI-compatible API wrapper
│   │   └── llm_call_baseline.py      # NEW: zero-shot Claude API baseline
│   │
│   ├── rewards/
│   │   ├── __init__.py
│   │   └── movr.py                   # NEW: MOVR reward function (standalone, testable)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_datasets.py      # downloads all 6 CSVs from HuggingFace
│   │   └── data_loader.py            # unified train/test split loader
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                # Sharpe, MDD, Calmar, annualised return
│       └── backtest.py               # portfolio simulation from signal CSV
│
├── scripts/
│   ├── 00_download_data.py           # downloads datasets, verifies checksums
│   ├── 01_reproduce_baseline.py      # trains paper's DAPO, runs backtest, saves results
│   ├── 02_train_movr.py              # trains all 5 MOVR configs, saves results
│   ├── 03_train_baselines.py         # trains SFT, LoRA, vanilla GRPO
│   ├── 04_llm_call_baseline.py       # zero-shot Claude inference on test set
│   ├── 05_vllm_benchmark.py          # HF vs vLLM throughput benchmark
│   └── 06_generate_all_plots.py      # reads all results CSVs, generates all 8 plots
│
├── plots/                            # auto-created, all PNGs saved here
├── results/                          # auto-created, all CSVs and JSONs saved here
├── checkpoints/                      # auto-created, model checkpoints saved here
│
└── notebooks/
    └── exploration.ipynb             # optional scratch notebook
```

---

## 2. Environment bootstrap — `setup.sh`

Generate `setup.sh` with exactly the following behaviour. It must be idempotent (safe to run twice).

```bash
#!/bin/bash
set -e

echo "=== FinRL-DAPO-MOVR setup ==="

# 1. System packages
sudo apt-get update -q
sudo apt-get install -y git wget curl tmux htop

# 2. Python dependencies (pinned for reproducibility)
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create output directories
mkdir -p plots results checkpoints logs

# 4. Copy env template if .env does not exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from template. Fill in ANTHROPIC_API_KEY and HF_TOKEN before running scripts."
fi

echo "=== Setup complete. Run: python scripts/00_download_data.py ==="
```

---

## 3. `requirements.txt` — pinned dependencies

```
# Core RL + FinRL
torch==2.2.1
gymnasium==0.29.1
stable-baselines3==2.3.0
finrl==0.3.7

# HuggingFace stack
transformers==4.40.2
datasets==2.19.1
accelerate==0.29.3
peft==0.10.0
huggingface_hub==0.23.0

# vLLM (inference acceleration)
vllm==0.4.2
openai==1.26.0          # vLLM uses OpenAI-compatible API

# Anthropic (LLM-call baseline)
anthropic==0.26.0

# Data and numerics
pandas==2.2.2
numpy==1.26.4
scipy==1.13.0
scikit-learn==1.4.2
yfinance==0.2.38

# Plotting
matplotlib==3.9.0
seaborn==0.13.2

# Utilities
python-dotenv==1.0.1
tqdm==4.66.4
pyyaml==6.0.1
```

---

## 4. Config files

### `configs/dapo_baseline.yaml`

This exactly replicates the paper's settings.

```yaml
# Paper: arXiv:2505.06408 — Table 1 settings
algorithm: dapo
model: MlpPolicy
env: env_llm_risk                  # uses paper's sentiment-risk env

# DAPO-specific
epsilon_low: 0.2                   # asymmetric clipping lower bound
epsilon_high: 0.28                 # asymmetric clipping upper bound (decoupled)
dynamic_sampling: true             # filter groups where all rewards are equal

# Reward (paper's exponentiated sentiment-risk formula)
reward_alpha: 3.0                  # sentiment exponent
reward_beta: 1.0                   # risk exponent

# Training
total_epochs: 100
steps_per_epoch: 20000
learning_rate: 3.0e-4
gamma: 0.99
clip_ratio: 0.2
train_pi_iters: 80
train_v_iters: 80
target_kl: 0.01

# Data
train_start: "2013-01-01"
train_end: "2018-12-31"
test_start: "2019-01-01"
test_end: "2023-12-31"
initial_capital: 1000000
```

### `configs/dapo_movr.yaml`

```yaml
# Novel contribution: Multi-Objective Verifiable Reward (MOVR / Fin-RLVR)
algorithm: dapo_movr
model: MlpPolicy
env: env_movr                      # uses new MOVR reward env

# DAPO-specific (same as baseline)
epsilon_low: 0.2
epsilon_high: 0.28
dynamic_sampling: true

# MOVR reward components — these are swept in 02_train_movr.py
# R = alpha * accuracy_signal + beta * delta_sharpe - gamma * drawdown_penalty
movr_alpha: 1.0
movr_beta: 0.5
movr_gamma: 0.3
sharpe_window: 20                  # rolling window (days) for delta_sharpe computation

# MOVR sweep configs (used by 02_train_movr.py — overrides movr_* above)
sweep_configs:
  - name: acc_only
    movr_alpha: 1.0
    movr_beta: 0.0
    movr_gamma: 0.0
  - name: sharpe_only
    movr_alpha: 0.0
    movr_beta: 1.0
    movr_gamma: 0.0
  - name: mdd_only
    movr_alpha: 0.0
    movr_beta: 0.0
    movr_gamma: 1.0
  - name: balanced
    movr_alpha: 1.0
    movr_beta: 0.5
    movr_gamma: 0.3
  - name: paper_equivalent
    movr_alpha: 3.0
    movr_beta: 1.0
    movr_gamma: 0.0

# Training (same as baseline)
total_epochs: 100
steps_per_epoch: 20000
learning_rate: 3.0e-4
gamma: 0.99
train_pi_iters: 80
train_v_iters: 80
target_kl: 0.01

train_start: "2013-01-01"
train_end: "2018-12-31"
test_start: "2019-01-01"
test_end: "2023-12-31"
initial_capital: 1000000
```

### `configs/grpo_vanilla.yaml`

```yaml
# GRPO vanilla — DAPO with symmetric clipping, no dynamic sampling
# Represents the DeepStock / pre-DAPO approach
algorithm: grpo_vanilla
model: MlpPolicy
env: env_llm_risk

epsilon_low: 0.2
epsilon_high: 0.2                  # symmetric = standard GRPO
dynamic_sampling: false            # disabled = all samples used

reward_alpha: 3.0
reward_beta: 1.0

total_epochs: 100
steps_per_epoch: 20000
learning_rate: 3.0e-4
gamma: 0.99
train_pi_iters: 80
train_v_iters: 80
target_kl: 0.01

train_start: "2013-01-01"
train_end: "2018-12-31"
test_start: "2019-01-01"
test_end: "2023-12-31"
initial_capital: 1000000
```

### `configs/sft.yaml`

```yaml
algorithm: sft
base_model: "Qwen/Qwen2.5-1.5B-Instruct"   # small enough for T4 or CPU
task: sequence_classification
num_labels: 2                               # buy=1, sell=0

num_train_epochs: 3
per_device_train_batch_size: 16
per_device_eval_batch_size: 32
learning_rate: 2.0e-5
warmup_ratio: 0.1
weight_decay: 0.01
max_seq_length: 256
fp16: false
bf16: true

train_start: "2013-01-01"
train_end: "2018-12-31"
test_start: "2019-01-01"
test_end: "2023-12-31"
```

### `configs/lora.yaml`

```yaml
algorithm: lora
base_model: "Qwen/Qwen2.5-1.5B-Instruct"
task: sequence_classification
num_labels: 2

# LoRA-specific
lora_r: 8
lora_alpha: 16
lora_dropout: 0.1
target_modules: ["q_proj", "v_proj"]
bias: none

num_train_epochs: 3
per_device_train_batch_size: 16
per_device_eval_batch_size: 32
learning_rate: 2.0e-4
warmup_ratio: 0.1
max_seq_length: 256
bf16: true

train_start: "2013-01-01"
train_end: "2018-12-31"
test_start: "2019-01-01"
test_end: "2023-12-31"
```

---

## 5. Source files — detailed specifications

### 5.1 `src/rewards/movr.py` — MOVR reward function

This is the core novel contribution. Implement as a standalone, unit-testable class.

```python
"""
Multi-Objective Verifiable Reward (MOVR) — Fin-RLVR
====================================================
Extends the FinRL-DAPO-SR paper's single exponentiated reward with a
three-component vector reward. Each component is verifiable from market
data — no human labels required. This directly extends DAPO's RLVR
philosophy to multi-dimensional financial objectives.

R(t) = alpha * accuracy_signal(t)
     + beta  * delta_sharpe(t)
     - gamma * drawdown_penalty(t)

Component definitions:
  accuracy_signal:  +1 if portfolio return > 0 else -1
                    (directional correctness, same as DeepStock binary reward)
  delta_sharpe:     change in rolling Sharpe ratio over `sharpe_window` days
                    (rewards actions that improve risk-adjusted performance)
  drawdown_penalty: current drawdown from rolling peak, clipped to [0, 1]
                    (penalises large equity drops proportionally)
"""

import numpy as np
from collections import deque
from typing import Optional


class MOVRReward:
    """
    Stateful MOVR reward calculator.

    Must be reset() at the start of each episode.
    Call compute(portfolio_return) at each step.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        sharpe_window: int = 20,
        annualise_sharpe: bool = True,
        trading_days_per_year: int = 252,
    ):
        """
        Parameters
        ----------
        alpha : weight on accuracy signal component
        beta  : weight on delta-Sharpe component
        gamma : weight on drawdown penalty component
        sharpe_window : rolling window length (days) for Sharpe computation
        annualise_sharpe : if True, multiply daily Sharpe by sqrt(trading_days_per_year)
        trading_days_per_year : used for annualisation
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sharpe_window = sharpe_window
        self.annualise = annualise_sharpe
        self.ann_factor = np.sqrt(trading_days_per_year) if annualise_sharpe else 1.0

        self._returns: deque = deque(maxlen=sharpe_window + 1)
        self._portfolio_values: list = []
        self._prev_sharpe: Optional[float] = None

    def reset(self) -> None:
        """Call at the start of each episode."""
        self._returns.clear()
        self._portfolio_values.clear()
        self._prev_sharpe = None

    def _compute_sharpe(self, returns_window) -> float:
        """Annualised Sharpe ratio from a list of daily returns."""
        arr = np.array(returns_window, dtype=np.float64)
        if len(arr) < 2:
            return 0.0
        std = arr.std() + 1e-8
        return (arr.mean() / std) * self.ann_factor

    def compute(
        self,
        portfolio_return: float,
        portfolio_value: float,
    ) -> float:
        """
        Compute MOVR reward for one timestep.

        Parameters
        ----------
        portfolio_return : percentage return this step (e.g. 0.012 for +1.2%)
        portfolio_value  : absolute portfolio value this step

        Returns
        -------
        float : scalar MOVR reward
        """
        # --- Component 1: accuracy signal ---
        accuracy_signal = 1.0 if portfolio_return > 0.0 else -1.0

        # --- Component 2: delta Sharpe ---
        self._returns.append(portfolio_return)
        if len(self._returns) >= self.sharpe_window:
            current_sharpe = self._compute_sharpe(list(self._returns))
            if self._prev_sharpe is None:
                delta_sharpe = 0.0
            else:
                delta_sharpe = current_sharpe - self._prev_sharpe
            self._prev_sharpe = current_sharpe
        else:
            delta_sharpe = 0.0

        # --- Component 3: drawdown penalty ---
        self._portfolio_values.append(portfolio_value)
        peak = max(self._portfolio_values)
        drawdown = max(0.0, (peak - portfolio_value) / (peak + 1e-8))
        # clip to [0, 1] to prevent extreme penalisation
        drawdown_penalty = min(drawdown, 1.0)

        # --- MOVR composite ---
        reward = (
            self.alpha * accuracy_signal
            + self.beta * delta_sharpe
            - self.gamma * drawdown_penalty
        )

        return float(reward)

    def get_state(self) -> dict:
        """Return internal state for logging/debugging."""
        return {
            "n_returns_stored": len(self._returns),
            "prev_sharpe": self._prev_sharpe,
            "peak_value": max(self._portfolio_values) if self._portfolio_values else None,
            "current_value": self._portfolio_values[-1] if self._portfolio_values else None,
        }
```

Also write a `test_movr.py` at the project root that runs 5 assertion-based unit tests:
- reward is positive for consistent positive returns
- reward penalises large drawdowns
- reward increases when Sharpe improves
- reset() clears state correctly
- all-zero weights return 0.0

### 5.2 `src/envs/env_movr.py` — trading environment with MOVR reward

Copy `env_stocktrading_llm_risk.py` from the base repo exactly, then make these three surgical changes only. Do not change anything else.

**Change 1**: Add import at top of file:
```python
from src.rewards.movr import MOVRReward
```

**Change 2**: In `__init__`, add after all existing `self.*` assignments:
```python
        # MOVR reward calculator — injected via config
        self.movr_alpha = movr_alpha      # new __init__ parameter, default 1.0
        self.movr_beta  = movr_beta       # new __init__ parameter, default 0.5
        self.movr_gamma = movr_gamma      # new __init__ parameter, default 0.3
        self.movr_sharpe_window = movr_sharpe_window  # new __init__ parameter, default 20
        self._movr = MOVRReward(
            alpha=self.movr_alpha,
            beta=self.movr_beta,
            gamma=self.movr_gamma,
            sharpe_window=self.movr_sharpe_window,
        )
```

**Change 3**: In the `step()` method, locate the line that computes `reward` (the portfolio return change). Replace only that reward assignment with:
```python
        # Original portfolio return (unchanged)
        portfolio_return = (self.portfolio_value / self.portfolio_value_memory[-1]) - 1.0

        # MOVR reward replaces single-objective portfolio return reward
        reward = self._movr.compute(
            portfolio_return=portfolio_return,
            portfolio_value=self.portfolio_value,
        )
```

And in `reset()`, add:
```python
        self._movr.reset()
```

**Change 4**: In `__init__` signature, add the four new parameters with defaults:
```python
def __init__(self, df, ..., movr_alpha=1.0, movr_beta=0.5, movr_gamma=0.3, movr_sharpe_window=20):
```

### 5.3 `src/algorithms/grpo_vanilla.py`

Copy `dapo_algorithm.py` from the base repo exactly. Then make exactly two changes:

**Change 1**: Find the clipping computation (where `epsilon_low` and `epsilon_high` are used). Add a flag at top of class `__init__`:
```python
self.symmetric_clip = symmetric_clip  # new param, default True for vanilla GRPO
```

**Change 2**: In the loss computation, when `self.symmetric_clip is True`, override:
```python
        if self.symmetric_clip:
            epsilon_high = epsilon_low    # force symmetric clipping
```

**Change 3**: Find the dynamic sampling filter. Add:
```python
        if not self.dynamic_sampling:
            # vanilla GRPO: keep all samples, no filtering
            pass  # skip the filter
        else:
            # DAPO: filter groups where all rewards are equal
            ...   # existing filter code
```

### 5.4 `src/algorithms/sft_trainer.py`

Implement a complete supervised finetuning trainer that:

- Loads `Qwen/Qwen2.5-1.5B-Instruct` via HuggingFace `transformers`
- Builds a binary classification dataset from the NASDAQ training CSV where the label is `1` if next-day close > open else `0`
- Input prompt format (same as DeepStock):
  ```
  Given: stock={ticker}, price_change_7d={val:.2%}, sentiment={s}/5, risk={r}/5
  Predict: will the price go UP or DOWN tomorrow?
  Answer:
  ```
- Uses `Trainer` from HuggingFace with `TrainingArguments` loaded from `configs/sft.yaml`
- Saves model to `checkpoints/sft/`
- Saves training loss curve to `results/sft_training_log.json`
- After training, runs inference on the test set and saves predictions to `results/sft_predictions.csv` with columns: `date, ticker, predicted_action, true_action, correct`

### 5.5 `src/algorithms/lora_trainer.py`

Identical to `sft_trainer.py` but wraps the model with `peft.get_peft_model()` using `LoraConfig` from `configs/lora.yaml` before training. Saves to `checkpoints/lora/`.

### 5.6 `src/inference/vllm_generator.py`

```python
"""
vLLM rollout generator — drop-in replacement for HuggingFace generate().

Usage:
    generator = VLLMGenerator(model_name="Qwen/Qwen2.5-7B-Instruct", port=8000)
    generator.start_server()         # launches vLLM server in background tmux session
    outputs = generator.generate(prompts, n=16, max_tokens=64)
    generator.stop_server()
"""

import subprocess
import time
import os
import requests
from typing import List
from openai import OpenAI


class VLLMGenerator:
    """
    Wraps vLLM's OpenAI-compatible API for batch rollout generation.
    
    Parameters
    ----------
    model_name : HuggingFace model identifier
    port       : port to run vLLM server on
    gpu_memory_utilization : fraction of GPU memory vLLM may use (default 0.85)
    max_model_len : max context length (default 2048)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        port: int = 8000,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 2048,
        dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.base_url = f"http://localhost:{port}/v1"
        self._client = None
        self._server_pid = None

    def start_server(self, timeout: int = 120) -> None:
        """
        Launch vLLM server in a background tmux session named 'vllm'.
        Blocks until server is healthy or timeout is exceeded.
        """
        cmd = (
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {self.model_name} "
            f"--dtype {self.dtype} "
            f"--max-model-len {self.max_model_len} "
            f"--gpu-memory-utilization {self.gpu_memory_utilization} "
            f"--port {self.port}"
        )
        # Kill existing tmux session if present
        subprocess.run("tmux kill-session -t vllm 2>/dev/null", shell=True)
        subprocess.run(f"tmux new-session -d -s vllm '{cmd}'", shell=True, check=True)
        print(f"vLLM server starting on port {self.port}... ", end="", flush=True)

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"http://localhost:{self.port}/health", timeout=2)
                if r.status_code == 200:
                    print("ready.")
                    break
            except Exception:
                pass
            time.sleep(3)
        else:
            raise TimeoutError(f"vLLM server did not start within {timeout}s")

        self._client = OpenAI(base_url=self.base_url, api_key="token-unused")

    def stop_server(self) -> None:
        """Kill the vLLM tmux session."""
        subprocess.run("tmux kill-session -t vllm 2>/dev/null", shell=True)
        print("vLLM server stopped.")

    def is_running(self) -> bool:
        try:
            r = requests.get(f"http://localhost:{self.port}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompts: List[str],
        n: int = 1,
        max_tokens: int = 64,
        temperature: float = 0.8,
    ) -> List[List[str]]:
        """
        Generate `n` completions for each prompt in `prompts`.

        Returns
        -------
        List[List[str]] : outer list indexed by prompt, inner list by completion
        """
        if self._client is None:
            raise RuntimeError("Call start_server() before generate()")

        results = []
        for prompt in prompts:
            response = self._client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                n=n,
                temperature=temperature,
            )
            completions = [choice.text for choice in response.choices]
            results.append(completions)
        return results

    def generate_batch(
        self,
        prompts: List[str],
        n: int = 1,
        max_tokens: int = 64,
        temperature: float = 0.8,
    ) -> List[List[str]]:
        """
        Send all prompts in a single batched request for maximum throughput.
        Use this for the vLLM benchmark — it exploits PagedAttention properly.
        """
        if self._client is None:
            raise RuntimeError("Call start_server() before generate_batch()")

        response = self._client.completions.create(
            model=self.model_name,
            prompt=prompts,       # list = batched request
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
        )
        # Group choices back by prompt
        results = [[] for _ in prompts]
        for i, choice in enumerate(response.choices):
            results[i % len(prompts)].append(choice.text)
        return results
```

### 5.7 `src/inference/hf_generator.py`

```python
"""
HuggingFace generate() wrapper — used as the baseline for vLLM benchmarking.
API is identical to VLLMGenerator so the benchmark script can swap them freely.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List


class HFGenerator:
    """
    Parameters
    ----------
    model_name : HuggingFace model identifier
    device     : 'cuda' or 'cpu'
    dtype      : torch dtype string ('bfloat16', 'float16', 'float32')
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                            "float32": torch.float32}[dtype]
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load tokenizer and model into GPU memory."""
        print(f"Loading {self.model_name} with HuggingFace... ", end="", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )
        self.model.eval()
        print("loaded.")

    def unload(self) -> None:
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        self.model = None
        self.tokenizer = None

    def generate(
        self,
        prompts: List[str],
        n: int = 1,
        max_tokens: int = 64,
        temperature: float = 0.8,
    ) -> List[List[str]]:
        """Generate `n` completions per prompt. Same API as VLLMGenerator."""
        if self.model is None:
            raise RuntimeError("Call load() before generate()")

        results = []
        for prompt in prompts:
            inputs = self.tokenizer(
                [prompt] * n,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            completions = [
                self.tokenizer.decode(ids[input_len:], skip_special_tokens=True)
                for ids in output_ids
            ]
            results.append(completions)
        return results

    def generate_batch(
        self,
        prompts: List[str],
        n: int = 1,
        max_tokens: int = 64,
        temperature: float = 0.8,
    ) -> List[List[str]]:
        """Batched generation — all prompts in one forward pass."""
        if self.model is None:
            raise RuntimeError("Call load() before generate_batch()")

        all_prompts = prompts * n
        inputs = self.tokenizer(
            all_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        all_completions = [
            self.tokenizer.decode(ids[input_len:], skip_special_tokens=True)
            for ids in output_ids
        ]
        # Group by prompt
        results = []
        for i in range(len(prompts)):
            results.append(all_completions[i::len(prompts)])
        return results
```

### 5.8 `src/inference/llm_call_baseline.py`

```python
"""
Zero-shot LLM call baseline using Claude claude-sonnet-4-6.
No training. Prompt is constructed from state features in the same
format as the FinRL-DAPO-SR paper's training prompts.

Requires ANTHROPIC_API_KEY in environment.
"""

import os
import time
import pandas as pd
from typing import Optional
import anthropic
from tqdm import tqdm


SYSTEM_PROMPT = """You are an expert quantitative trading agent. 
You analyse stock market data and make buy/sell decisions.
Be concise. Respond with exactly one word: BUY or SELL."""


def build_prompt(row: pd.Series) -> str:
    return (
        f"Stock: {row.get('ticker', 'UNKNOWN')}\n"
        f"Date: {row.get('date', 'N/A')}\n"
        f"Price change (7-day): {row.get('price_change_7d', 0.0):.2%}\n"
        f"Sentiment score (1–5): {row.get('sentiment', 3.0):.1f}\n"
        f"Risk score (1–5): {row.get('risk', 3.0):.1f}\n"
        f"Technical indicator (MACD): {row.get('macd', 0.0):.4f}\n"
        f"\nBased on these signals, should you BUY or SELL this stock tomorrow?"
    )


def run_llm_call_baseline(
    test_df: pd.DataFrame,
    sample_size: int = 500,
    output_path: str = "results/llm_call_predictions.csv",
    model: str = "claude-sonnet-4-6",
    rate_limit_delay: float = 0.5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Run zero-shot inference on a sample of test data.

    Parameters
    ----------
    test_df       : test dataframe with columns: date, ticker, price_change_7d,
                    sentiment, risk, macd, true_action (1=up, 0=down)
    sample_size   : number of rows to evaluate (cost-controlled)
    output_path   : where to save predictions CSV
    model         : Anthropic model string
    rate_limit_delay : seconds to sleep between API calls
    random_seed   : for reproducible sampling

    Returns
    -------
    DataFrame with columns: date, ticker, predicted_action, true_action, correct
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sample = test_df.sample(min(sample_size, len(test_df)), random_state=random_seed)

    predictions = []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="LLM call baseline"):
        prompt = build_prompt(row)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=5,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.content[0].text.strip().upper()
            predicted = 1 if "BUY" in answer else 0
        except Exception as e:
            print(f"API error: {e}")
            predicted = 0   # default to sell on error

        true_action = int(row.get("true_action", 0))
        predictions.append({
            "date": row.get("date"),
            "ticker": row.get("ticker"),
            "predicted_action": predicted,
            "true_action": true_action,
            "correct": int(predicted == true_action),
        })
        time.sleep(rate_limit_delay)

    results_df = pd.DataFrame(predictions)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)

    accuracy = results_df["correct"].mean()
    print(f"LLM call baseline accuracy: {accuracy:.3f} ({results_df['correct'].sum()}/{len(results_df)})")
    return results_df
```

### 5.9 `src/evaluation/metrics.py`

```python
"""
Financial performance metrics — all computed from a daily returns series.
"""

import numpy as np
import pandas as pd
from typing import Union


def compute_all_metrics(
    daily_returns: Union[pd.Series, np.ndarray],
    trading_days_per_year: int = 252,
) -> dict:
    """
    Compute a full suite of metrics from a daily returns series.

    Parameters
    ----------
    daily_returns : series of daily percentage returns (e.g. 0.012 for +1.2%)

    Returns
    -------
    dict with keys:
        cumulative_return, annualised_return, sharpe_ratio,
        max_drawdown, calmar_ratio, sortino_ratio,
        information_ratio (vs zero), rachev_ratio, cvar_5pct,
        win_rate, n_days
    """
    r = np.array(daily_returns, dtype=np.float64)
    n = len(r)

    # Cumulative return
    cum_return = (1 + r).prod() - 1

    # Annualised return (CAGR)
    ann_return = (1 + cum_return) ** (trading_days_per_year / n) - 1

    # Sharpe ratio (annualised)
    sharpe = (r.mean() / (r.std() + 1e-8)) * np.sqrt(trading_days_per_year)

    # Max drawdown
    cum_curve = (1 + r).cumprod()
    rolling_peak = np.maximum.accumulate(cum_curve)
    drawdowns = (cum_curve - rolling_peak) / (rolling_peak + 1e-8)
    max_drawdown = drawdowns.min()  # negative number

    # Calmar ratio
    calmar = ann_return / (abs(max_drawdown) + 1e-8)

    # Sortino ratio (downside deviation)
    downside = r[r < 0]
    downside_std = downside.std() + 1e-8 if len(downside) > 0 else 1e-8
    sortino = (r.mean() / downside_std) * np.sqrt(trading_days_per_year)

    # CVaR at 5% (expected loss in worst 5% of days)
    cvar_threshold = np.percentile(r, 5)
    cvar_5 = r[r <= cvar_threshold].mean() if (r <= cvar_threshold).any() else cvar_threshold

    # Rachev ratio (expected gain top 5% / expected loss bottom 5%)
    top5 = r[r >= np.percentile(r, 95)].mean() if (r >= np.percentile(r, 95)).any() else 0
    bot5 = abs(cvar_5) + 1e-8
    rachev = top5 / bot5

    # Win rate
    win_rate = (r > 0).mean()

    return {
        "cumulative_return": float(cum_return),
        "annualised_return": float(ann_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar),
        "sortino_ratio": float(sortino),
        "cvar_5pct": float(cvar_5),
        "rachev_ratio": float(rachev),
        "win_rate": float(win_rate),
        "n_days": int(n),
    }
```

### 5.10 `src/evaluation/backtest.py`

Implement a `run_backtest(predictions_df, price_df, initial_capital)` function that:
- Takes a `predictions_df` with columns `date, ticker, predicted_action` (1=buy, 0=sell)
- Takes a `price_df` with columns `date, ticker, close`
- Simulates: on each day, allocate capital equally across all buy signals; hold cash for sell signals
- Returns a `pd.DataFrame` with columns: `date, portfolio_value, daily_return, cumulative_return`
- Also returns the NASDAQ-100 index as a comparison series (read from `dataset/trade_data_2019_2023.csv` — the `close` column averaged across all NASDAQ-100 tickers)
- Saves both to `results/{method}_backtest.csv`

### 5.11 `src/data/download_datasets.py`

```python
"""
Downloads all 6 required CSVs from HuggingFace Hub to ./dataset/.
Skips files that already exist and match expected size.
"""

import os
from huggingface_hub import hf_hub_download
import shutil

REPO_ID = "benstaf/nasdaq_2013_2023"
REPO_TYPE = "dataset"
DEST_DIR = "dataset"

REQUIRED_FILES = [
    "train_data_2013_2018.csv",
    "train_data_deepseek_risk_2013_2018.csv",
    "train_data_deepseek_sentiment_2013_2018.csv",
    "trade_data_2019_2023.csv",
    "trade_data_deepseek_risk_2019_2023.csv",
    "trade_data_deepseek_sentiment_2019_2023.csv",
]


def download_all(dest_dir: str = DEST_DIR, force: bool = False) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for filename in REQUIRED_FILES:
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path) and not force:
            print(f"  already exists: {filename}")
            continue
        print(f"  downloading: {filename} ... ", end="", flush=True)
        cached = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type=REPO_TYPE,
        )
        shutil.copy(cached, dest_path)
        size_mb = os.path.getsize(dest_path) / 1e6
        print(f"done ({size_mb:.1f} MB)")

    print(f"\nAll datasets in ./{dest_dir}/")


if __name__ == "__main__":
    download_all()
```

---

## 6. Scripts — detailed specifications

### `scripts/00_download_data.py`

Calls `src.data.download_datasets.download_all()`. Then prints the shape of each CSV. Exits with code 1 if any file is missing or empty.

### `scripts/01_reproduce_baseline.py`

```
WHAT IT DOES:
1. Loads configs/dapo_baseline.yaml
2. Loads train data from dataset/train_data_deepseek_*.csv using src.data.data_loader
3. Instantiates env_llm_risk.py environment with paper's default settings
4. Instantiates dapo.py algorithm with paper's hyperparameters
5. Records training start time
6. Runs 100 epochs of training
7. Records training end time, saves to results/timing_log.json
8. Saves model checkpoint to checkpoints/dapo_baseline/
9. Runs backtest on 2019-2023 test data using src.evaluation.backtest
10. Saves results to results/dapo_baseline_backtest.csv
11. Computes metrics using src.evaluation.metrics, saves to results/dapo_baseline_metrics.json
12. Prints a summary table to stdout matching Table 1 from the paper
```

### `scripts/02_train_movr.py`

```
WHAT IT DOES:
1. Loads configs/dapo_movr.yaml — reads sweep_configs list
2. For each of the 5 sweep configs (acc_only, sharpe_only, mdd_only, balanced, paper_equivalent):
   a. Instantiates env_movr.py with that config's alpha/beta/gamma
   b. Instantiates dapo_movr.py algorithm
   c. Records start time
   d. Trains 100 epochs
   e. Records end time, appends to results/timing_log.json
   f. Saves checkpoint to checkpoints/movr_{config_name}/
   g. Runs backtest, saves results/movr_{config_name}_backtest.csv
   h. Computes metrics, saves results/movr_{config_name}_metrics.json
3. After all 5 configs, prints a comparison table of all metrics
```

### `scripts/03_train_baselines.py`

```
WHAT IT DOES:
1. Trains vanilla GRPO:
   - Load configs/grpo_vanilla.yaml
   - Use grpo_vanilla.py (symmetric clip, no dynamic sampling)
   - Train 100 epochs, save to checkpoints/grpo_vanilla/
   - Backtest, save results/grpo_vanilla_backtest.csv

2. Trains SFT:
   - Load configs/sft.yaml
   - Use sft_trainer.py
   - Train, save to checkpoints/sft/
   - Run test inference, save results/sft_predictions.csv
   - Simulate portfolio from predictions, save results/sft_backtest.csv

3. Trains LoRA:
   - Load configs/lora.yaml
   - Use lora_trainer.py
   - Train, save to checkpoints/lora/
   - Run test inference, save results/lora_predictions.csv
   - Simulate portfolio from predictions, save results/lora_backtest.csv

4. Print timing comparison for all three
```

### `scripts/04_llm_call_baseline.py`

```
WHAT IT DOES:
1. Load dataset/trade_data_2019_2023.csv and dataset/trade_data_deepseek_*.csv
2. Merge into a unified test DataFrame with columns:
   date, ticker, price_change_7d, sentiment, risk, macd, true_action
   Where true_action = 1 if next_day_close > open else 0
3. Call src.inference.llm_call_baseline.run_llm_call_baseline() with sample_size=500
4. Simulate portfolio from predictions using src.evaluation.backtest
5. Save results/llm_call_backtest.csv and results/llm_call_metrics.json
6. Print accuracy and key metrics
```

### `scripts/05_vllm_benchmark.py`

```
WHAT IT DOES (complete throughput benchmark):

1. Define benchmark parameters:
   BATCH_SIZES = [1, 2, 4, 8, 16, 32]
   N_WARMUP_ITERS = 3
   N_BENCH_ITERS = 10
   MAX_NEW_TOKENS = 64
   MODEL = "Qwen/Qwen2.5-7B-Instruct"

   Dummy prompt (same length as real trading prompts):
   "Stock: AAPL, Date: 2022-03-15, Price change (7-day): +1.23%,
    Sentiment: 3.8/5, Risk: 2.1/5, Technical (MACD): 0.0042.
    Should you BUY or SELL tomorrow?"

2. HuggingFace benchmark:
   a. Load model with HFGenerator
   b. For each batch_size in BATCH_SIZES:
      - Warmup N_WARMUP_ITERS
      - Time N_BENCH_ITERS of generate_batch(prompts=[dummy]*batch_size)
      - Record: mean_time_ms, std_time_ms, tokens_per_second
   c. Free GPU memory (del model, torch.cuda.empty_cache())

3. vLLM benchmark:
   a. Start VLLMGenerator server (blocks until healthy)
   b. For each batch_size in BATCH_SIZES:
      - Warmup N_WARMUP_ITERS
      - Time N_BENCH_ITERS of generate_batch(prompts=[dummy]*batch_size)
      - Record: mean_time_ms, std_time_ms, tokens_per_second
   c. Stop server

4. Save full benchmark results to results/vllm_benchmark.json:
   {
     "model": MODEL,
     "max_new_tokens": MAX_NEW_TOKENS,
     "batch_sizes": [...],
     "hf": {
       "throughput_tok_per_sec": [...],
       "latency_ms_mean": [...],
       "latency_ms_std": [...]
     },
     "vllm": {
       "throughput_tok_per_sec": [...],
       "latency_ms_mean": [...],
       "latency_ms_std": [...]
     },
     "speedup_ratio": [...],
     "peak_speedup": float,
     "timestamp": "ISO8601"
   }

5. Print summary table with speedup ratios per batch size
```

### `scripts/06_generate_all_plots.py`

This script reads all results CSVs and JSONs and generates all 8 plots. It must not re-run any training — only plotting. All plots saved to `plots/` at 150 DPI.

Implement each plot as a separate function `plot_N_name(ax_or_fig)` so they can be called independently. Call all 8 from `main()`.

#### Plot 1 — Baseline reproduction
Two subplots side by side.
- Left: cumulative return curve for DAPO-SR vs NASDAQ-100 index, 2019–2023. Read from `results/dapo_baseline_backtest.csv`. Y-axis as percentage. Annotate final return value on the line.
- Right: drawdown curve for DAPO-SR. Fill under the curve with red alpha=0.3. Annotate max drawdown value.

#### Plot 2 — MOVR reward ablation
Three subplots.
- Left: 5 cumulative return lines (one per MOVR config) + NASDAQ-100 dashed. Each line labelled directly on the plot (no legend box — use `ax.annotate` at end of each line).
- Middle: grouped bar chart of Sharpe ratio for all 5 configs. Highlight `balanced` bar in a distinct colour.
- Right: grouped bar chart of Max Drawdown (%) for all 5 configs. Note: lower is better. Add a horizontal dashed line at the paper baseline's MDD for comparison.

#### Plot 3 — vLLM throughput
Two subplots.
- Left: dual-line chart of HF vs vLLM throughput (tokens/sec) vs batch size. Add shaded error bands using std. Log scale on Y axis.
- Right: bar chart of speedup ratio (vLLM / HF) per batch size. Colour bars by speedup magnitude (green = high speedup). Annotate each bar with exact ratio.

#### Plot 4 — Full comparison table
A matplotlib table rendered as a figure. Rows = 6 methods (LLM call, SFT, LoRA, GRPO vanilla, DAPO-SR baseline, DAPO+MOVR balanced). Columns = Cumulative Return, Sharpe, Max Drawdown, Calmar, Training Time (hrs). Colour cells using a green-yellow-red colormap per column (best = green). Bold the DAPO+MOVR row.

#### Plot 5 — All methods cumulative return
Single chart. Six lines + NASDAQ-100 dashed. Use a consistent colour scheme: grays for weak baselines, orange for GRPO, blue for paper's DAPO, bold dark teal for DAPO+MOVR. Grid, legend. Annotate final values on right margin.

#### Plot 6 — Radar chart
Spider/radar chart with 5 axes: Cumulative Return, Sharpe Ratio, Inverted MDD, Calmar Ratio, Win Rate. Normalise each axis 0–1 across all methods. Plot all 6 methods. Fill DAPO+MOVR polygon with alpha=0.15. Include a legend.

#### Plot 7 — Training time vs performance scatter
X axis: training time in hours. Y axis: cumulative return %. One scatter point per method, sized by Sharpe ratio (larger = higher Sharpe). Annotate each point with method name. Draw a Pareto frontier line connecting non-dominated points. Add a shaded region labelled "ideal: fast + profitable".

#### Plot 8 — Training time reduction bar chart
Horizontal bar chart. Methods on Y axis in order of training time. Bars coloured by category (gray = prior work, blue = ours). Include a vertical dashed line at the paper's CPPO-DeepSeek baseline (7.5 hrs). Annotate each bar with exact hours.

---

## 7. `.env.example`

```bash
# Copy this to .env and fill in your values before running scripts
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HF_TOKEN=your_huggingface_token_here           # needed for gated models on HF Hub
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct            # change if using a different model
VLLM_PORT=8000
```

---

## 8. Lightning.ai deployment instructions

Generate a file `LIGHTNING_SETUP.md` at the project root with these exact instructions:

```markdown
# Running on Lightning.ai

## Step 1 — Studio type
Create a new Studio with instance type: **A10G GPU** (24 GB VRAM).
This is required for DAPO training (scripts 01 and 02) and the vLLM benchmark (script 05).

## Step 2 — Clone and bootstrap
In the Studio terminal:
    git clone https://github.com/YOUR_USERNAME/finrl-dapo-movr.git
    cd finrl-dapo-movr
    cp .env.example .env
    # Edit .env and add your ANTHROPIC_API_KEY and HF_TOKEN
    bash setup.sh

## Step 3 — Download data
    python scripts/00_download_data.py

## Step 4 — Run scripts in order (GPU required for 01, 02, 03, 05)
    python scripts/01_reproduce_baseline.py
    python scripts/02_train_movr.py
    python scripts/03_train_baselines.py
    python scripts/04_llm_call_baseline.py
    python scripts/05_vllm_benchmark.py

## Step 5 — Generate all plots (can run on CPU instance to save cost)
Switch to a CPU Studio instance, then:
    python scripts/06_generate_all_plots.py
    # All 8 plots saved to ./plots/

## GPU cost saving tips
- Scripts 01–03 need the A10G. Pause between runs if not chaining them.
- Script 04 (LLM call) uses Anthropic API — no GPU needed, use CPU Studio.
- Script 05 (vLLM benchmark) needs A10G. Run as one session with 01–03.
- Script 06 (plots) needs zero GPU. Always run on CPU Studio.
- Scripts 03 SFT and LoRA training use Qwen2.5-1.5B — these can run on T4 (16 GB).

## vLLM server management
Script 05 starts and stops the vLLM server automatically.
If you need to manually manage it:
    tmux new -s vllm
    python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000
    # Ctrl+B then D to detach from tmux
    tmux kill-session -t vllm   # to stop

## Expected outputs
After all scripts complete:
    plots/     — 8 PNG files (plot1 through plot8)
    results/   — backtest CSVs and metrics JSONs for all 6 methods
    results/timing_log.json — all training time measurements
    checkpoints/ — saved model weights for all trained methods
```

---

## 9. Implementation rules for the code generator

These rules apply to every file you generate. Follow them without exception.

**Rule 1 — Do not modify the base repo files directly.**  
`src/envs/env_base.py` and `src/envs/env_llm_risk.py` are exact copies of the originals from https://github.com/Ruijian-Zha/FinRL-DAPO-SR. `src/algorithms/dapo.py` is an exact copy of `dapo_algorithm.py`. Do not change these files. All modifications go into the `_movr` and `_vanilla` variants.

**Rule 2 — Every script is runnable standalone.**  
Each `scripts/XX_*.py` must be executable as `python scripts/XX_name.py` with no positional arguments. All config is read from the appropriate YAML file in `configs/` and the `.env` file.

**Rule 3 — Graceful failure with informative errors.**  
Every script must check that its required input files exist before starting and print a clear error message naming the missing file and which prior script generates it.

**Rule 4 — Timing is mandatory.**  
Every training script records wall-clock time using `time.perf_counter()`. All timing results are appended to `results/timing_log.json` (not overwritten — appended).

**Rule 5 — Results are deterministic across runs.**  
Set `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)` at the top of every training script.

**Rule 6 — The plotting script never imports training code.**  
`scripts/06_generate_all_plots.py` only reads CSV and JSON files from `results/`. It never imports `src/algorithms/` or `src/envs/`. This ensures it runs on a CPU Studio without GPU dependencies.

**Rule 7 — All plots use a consistent visual style.**  
At the top of `scripts/06_generate_all_plots.py`, set:
```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})
METHOD_COLORS = {
    "llm_call":        "#adb5bd",
    "sft":             "#6c757d",
    "lora":            "#495057",
    "grpo_vanilla":    "#fd7e14",
    "dapo_baseline":   "#4895ef",
    "dapo_movr":       "#1a6b3c",   # bold — this is "our method"
    "nasdaq_100":      "#000000",
}
```

**Rule 8 — Each source file starts with a module docstring.**  
Every `.py` file must begin with a triple-quoted docstring stating: what the file does, which paper section it corresponds to (if applicable), and what its inputs and outputs are.

**Rule 9 — No hardcoded absolute paths.**  
All paths are relative to the project root. Use `pathlib.Path` throughout. Never write `/home/user/...` or any absolute path.

**Rule 10 — The MOVR reward is fully decoupled.**  
`src/rewards/movr.py` must have zero imports from `src/envs/` or `src/algorithms/`. It is a pure Python class that can be unit-tested without FinRL, PyTorch, or any trading dependency installed.

---

## 10. Verification checklist

After generating the full codebase, verify:

- [ ] `setup.sh` exists and is executable
- [ ] `requirements.txt` has all packages listed in section 3
- [ ] All 5 YAML configs exist in `configs/`
- [ ] All 10 source files exist in `src/`
- [ ] All 7 scripts exist in `scripts/`
- [ ] `src/rewards/movr.py` imports only `numpy` and standard library
- [ ] `test_movr.py` exists at project root and contains 5 test functions
- [ ] `LIGHTNING_SETUP.md` exists at project root
- [ ] `.env.example` exists at project root
- [ ] `scripts/06_generate_all_plots.py` defines exactly 8 plot functions named `plot_1_*` through `plot_8_*`
- [ ] No script hardcodes an absolute path
- [ ] Every training script appends to `results/timing_log.json`
- [ ] `src/envs/env_movr.py` inherits from `env_llm_risk.py` and only modifies the reward computation

---

*Base paper*: Ruijian Zha, Bojun Liu. "A New DAPO Algorithm for Stock Trading." IEEE IDS 2025. arXiv:2505.06408  
*Novel extensions*: MOVR / Fin-RLVR reward + vLLM rollout acceleration benchmark  
*Course*: Comparative Benchmarking of LLM Finetuning Techniques on Financial Time Series