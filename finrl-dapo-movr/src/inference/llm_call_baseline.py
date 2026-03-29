"""
src/inference/llm_call_baseline.py — Zero-Shot Open-Source LLM Baseline
=======================================================================
Uses Groq's FREE-TIER API (llama-3.3-70b-versatile) as the zero-shot LLM
call baseline. NO PAID API KEYS REQUIRED — Groq offers a generous free tier.

Get your free key at: https://console.groq.com

Falls back to local HuggingFace inference if GROQ_API_KEY is not set.

Prompt format is identical to the DeepStock paper for comparability.
No training — pure inference from pre-trained LLM.

Inputs: test dataframe with market features, GROQ_API_KEY env var
Outputs: results/llm_call_predictions.csv with columns:
         date, ticker, predicted_action, true_action, correct
"""

import os
import time
import pandas as pd
from typing import Optional
from tqdm import tqdm
from pathlib import Path


SYSTEM_PROMPT = """You are an expert quantitative trading agent.
You analyse stock market data and make buy/sell decisions.
Be concise. Respond with exactly one word: BUY or SELL."""


def build_prompt(row: pd.Series) -> str:
    """Build a DeepStock-compatible trading prompt from a data row."""
    ticker = row.get("tic", row.get("ticker", "UNKNOWN"))
    price_change = float(row.get("price_change_7d", 0.0))
    sentiment = float(row.get("sentiment", 3.0))
    risk = float(row.get("risk", 3.0))
    macd = float(row.get("macd", 0.0))
    return (
        f"Stock: {ticker}\n"
        f"Price change (7-day): {price_change:.2%}\n"
        f"Sentiment score (1-5): {sentiment:.1f}\n"
        f"Risk score (1-5): {risk:.1f}\n"
        f"Technical indicator (MACD): {macd:.4f}\n"
        f"\nBased on these signals, should you BUY or SELL this stock tomorrow?"
    )


def _call_groq(client, prompt: str, model: str, rate_limit_delay: float) -> int:
    """Call Groq API and parse BUY/SELL response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip().upper()
        predicted = 1 if "BUY" in answer else 0
    except Exception as e:
        print(f"\n[WARNING] Groq API error: {e}. Defaulting to SELL.")
        predicted = 0
    time.sleep(rate_limit_delay)
    return predicted


def _call_hf_local(model_name: str, prompt: str) -> int:
    """Fallback: use local HuggingFace model for inference."""
    try:
        import torch
        from transformers import pipeline

        pipe = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=5,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        output = pipe(f"{SYSTEM_PROMPT}\n\n{prompt}")[0]["generated_text"]
        predicted = 1 if "BUY" in output.upper() else 0
        del pipe
        torch.cuda.empty_cache()
        return predicted
    except Exception as e:
        print(f"[WARNING] Local HF inference failed: {e}")
        return 0


def run_llm_call_baseline(
    test_df: pd.DataFrame,
    sample_size: int = 500,
    output_path: str = "results/llm_call_predictions.csv",
    model: str = "llama-3.3-70b-versatile",   # Groq free-tier model
    fallback_model: str = "Qwen/Qwen2.5-1.5B-Instruct",  # local HF fallback
    rate_limit_delay: float = 0.2,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Run zero-shot inference on a sample of test data.

    PRIMARY: Groq API (llama-3.3-70b-versatile) — free tier, no signup cost
    FALLBACK: Local HuggingFace model if GROQ_API_KEY not set

    Parameters
    ----------
    test_df       : test dataframe with market features
    sample_size   : number of rows to evaluate (rate-limit safe)
    output_path   : where to save predictions CSV
    model         : Groq model name (default: llama-3.3-70b-versatile, free)
    fallback_model: HF model to use if Groq key not available
    rate_limit_delay : seconds between API calls
    random_seed   : for reproducible sampling

    Returns
    -------
    DataFrame with columns: date, ticker, predicted_action, true_action, correct
    """
    sample = test_df.sample(min(sample_size, len(test_df)), random_state=random_seed)

    # Decide inference backend
    groq_key = os.environ.get("GROQ_API_KEY", "")
    use_groq = bool(groq_key)

    if use_groq:
        from groq import Groq
        client = Groq(api_key=groq_key)
        print(f"[LLM Baseline] Using Groq free-tier ({model}) — {len(sample)} samples")
    else:
        client = None
        print(f"[LLM Baseline] GROQ_API_KEY not set. Using local {fallback_model}.")
        import torch
        from transformers import pipeline
        pipe = pipeline(
            "text-generation",
            model=fallback_model,
            max_new_tokens=10,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    predictions = []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="LLM call baseline"):
        prompt = build_prompt(row)
        true_action = int(row.get("true_action", 0))

        if use_groq:
            predicted = _call_groq(client, prompt, model, rate_limit_delay)
        else:
            # Use the already-loaded local pipeline
            try:
                output = pipe(f"{SYSTEM_PROMPT}\n\n{prompt}")[0]["generated_text"]
                predicted = 1 if "BUY" in output.upper().split(prompt)[-1] else 0
            except Exception:
                predicted = 0

        predictions.append({
            "date": row.get("date", None),
            "ticker": row.get("tic", row.get("ticker", "UNKNOWN")),
            "predicted_action": predicted,
            "true_action": true_action,
            "correct": int(predicted == true_action),
        })

    if not use_groq and "pipe" in dir():
        import torch
        del pipe
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    accuracy = results_df["correct"].mean()
    n = results_df["correct"].sum()
    total = len(results_df)
    print(f"[LLM Baseline] Accuracy: {accuracy:.3f} ({n}/{total})")
    return results_df
