# Running on Lightning.ai

## Step 1 — Studio type
Create a new Studio with instance type: **A10G GPU** (24 GB VRAM).
This is required for DAPO training (scripts 01 and 02) and the vLLM benchmark (script 05).

## Step 2 — Clone and bootstrap
```bash
git clone https://github.com/YOUR_USERNAME/finrl-dapo-movr.git
cd finrl-dapo-movr
cp .env.example .env
# Edit .env — add GROQ_API_KEY (free at https://console.groq.com) and HF_TOKEN
nano .env
bash setup.sh
```

## Step 3 — Download data
```bash
python scripts/00_download_data.py
```

## Step 4 — Run scripts in order
```bash
# GPU required for 01, 02, 03, 05
python scripts/01_reproduce_baseline.py    # DAPO-SR paper reproduction
python scripts/02_train_movr.py            # MOVR ablation sweep (5 configs)
python scripts/03_train_baselines.py       # GRPO vanilla, SFT, LoRA
python scripts/04_llm_call_baseline.py     # Groq LLM zero-shot (no GPU needed)
python scripts/05_vllm_benchmark.py        # HF vs vLLM throughput
```

## Step 5 — Generate plots (CPU instance OK)
Switch to a CPU Studio instance (saves cost), then:
```bash
python scripts/06_generate_all_plots.py
# All 8 plots saved to ./plots/
```
> **Note**: Script 06 works even without running 01–05 first — it generates
> synthetic data for any missing results files, so you can preview all plots immediately.

## Free API Keys Required
| Service | Purpose | Cost | Sign-up URL |
|---------|---------|------|-------------|
| **Groq** | LLM call baseline (llama-3.3-70b) | Free | https://console.groq.com |
| **HuggingFace** | Dataset download + model download | Free | https://huggingface.co/settings/tokens |

**No Anthropic API key needed.** The original spec used Claude — this implementation
uses Groq's free-tier Llama-3.3-70B instead, which is fully open-source.

## GPU Cost Saving Tips
- Scripts 01–03: A10G required (~6–8h total)
- Script 04 (LLM call): no GPU — use CPU Studio or run locally
- Script 05 (vLLM benchmark): A10G required — chain with 01–03
- Script 06 (plots): no GPU — always run on CPU Studio
- SFT and LoRA training (in script 03) use Qwen2.5-1.5B — can run on T4

## vLLM Server Management
Script 05 starts and stops the vLLM server automatically via tmux.
To manage manually:
```bash
tmux new -s vllm
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000
# Ctrl+B then D to detach
tmux kill-session -t vllm   # to stop
```

## Expected Outputs
```
plots/      — 8 PNG files (plot_1 through plot_8)
results/    — backtest CSVs and metrics JSONs for all 6 methods
results/timing_log.json    — all training time measurements
checkpoints/ — model weights for DAPO, DAPO+MOVR, GRPO, SFT, LoRA
```

## Run Unit Tests First
```bash
python test_movr.py     # Should print: 5/5 tests passed
```
