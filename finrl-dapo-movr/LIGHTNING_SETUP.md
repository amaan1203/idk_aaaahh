# Running on Lightning.ai

## Instance selection

| Script | Instance | Approx. cost |
|--------|----------|--------------|
| 00, 01, 03, 05 | CPU (any) | ~$0 |
| 02 (MOVR training) | **A10G GPU 24 GB** | ~$2–4/hr |
| 04 (vLLM benchmark) | **A10G GPU 24 GB** | ~$1–2/hr |

Pause the A10G between scripts 02 and 04 if not running back-to-back.

## Step-by-step

    # 1. Clone and bootstrap (CPU instance)
    git clone https://github.com/YOUR_USERNAME/finrl-dapo-movr.git
    cd finrl-dapo-movr
    cp .env.example .env
    nano .env            # add ANTHROPIC_API_KEY and HF_TOKEN
    bash setup.sh

    # 2. Download data (CPU instance, ~5 min)
    python scripts/00_download_data.py

    # 3. Seed published numbers (CPU instance, <1 sec)
    python scripts/01_seed_published_results.py

    # 4. Train MOVR + GRPO Vanilla (A10G GPU, ~3-6 hrs for all 6 configs)
    python scripts/02_train_movr.py

    # 5. LLM call baseline (CPU instance, ~10 min, ~$0.10 API cost)
    python scripts/03_llm_call_baseline.py

    # 6. vLLM benchmark (A10G GPU, ~30 min)
    python scripts/04_vllm_benchmark.py

    # 7. Generate all plots (CPU instance, ~2 min)
    python scripts/05_generate_all_plots.py

## After step 3, plots 1, 4, 7, 8 already render with published data.
## No GPU needed to see those results.

## Unit tests (no GPU, no network, <5 sec)

    python test_movr.py

## Managing vLLM manually

    tmux new -s vllm
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct --port 8000
    # Ctrl+B then D to detach
    curl http://localhost:8000/health    # check
    tmux kill-session -t vllm           # stop

## Expected outputs

    plots/   → 8 PNG files (plot1_*.png through plot8_*.png)
    results/ → metrics JSONs (all methods), vllm_benchmark.json, timing_log.json
    results/ → backtest CSVs (MOVR configs, LLM call)
    checkpoints/ → 6 model checkpoints (5 MOVR + GRPO vanilla)

## What runs where — summary

| Script | GPU? | Data source | Est. time |
|--------|------|-------------|-----------|
| `00_download_data.py` | No | HuggingFace Hub | ~5 min |
| `01_seed_published_results.py` | No | Hardcoded from paper | <1 sec |
| `02_train_movr.py` | **YES — A10G** | Our training | ~3–6 hrs |
| `03_llm_call_baseline.py` | No | Anthropic API (~$0.10) | ~10 min |
| `04_vllm_benchmark.py` | **YES — A10G** | Our benchmark | ~30 min |
| `05_generate_all_plots.py` | No | reads `results/` | ~2 min |

**Total GPU time: ~4–7 hours on one A10G.**
After script 01, you can already generate plots 1, 4, 7, and 8
using only published data — no GPU needed.

## Cost saving tips

- Scripts 01, 03, 05 use zero GPU. Always run on CPU instance.
- Script 03 uses ~$0.10 Anthropic API cost (500 samples × ~50 tokens).
- Script 04 needs A10G. Chain it with script 02 to avoid startup overhead.
- After all scripts, switch to CPU instance for plot generation (script 05).
