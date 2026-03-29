"""
src/evaluation/backtest.py — Portfolio Backtester
=================================================
Simulates a long-only portfolio from signal predictions.
Compares against NASDAQ-100 buy-and-hold benchmark.

Inputs:
  predictions_df : date, ticker, predicted_action (1=buy, 0=sell)
  price_df       : date, ticker, close
  initial_capital : starting portfolio value

Outputs:
  DataFrame: date, portfolio_value, daily_return, cumulative_return
  Saves results to results/{method}_backtest.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from src.evaluation.metrics import compute_all


def run_backtest(
    predictions_df: pd.DataFrame,
    price_df: pd.DataFrame,
    initial_capital: float = 1_000_000,
    method_name: str = "method",
    output_dir: Path = Path("results"),
) -> Tuple[pd.DataFrame, list]:
    """
    Simulate a long-only equal-weight portfolio from signal predictions.

    On each day:
    - Allocate capital equally across all tickers with predicted_action=1 (BUY)
    - Hold cash for SELL signals (no shorting)

    Parameters
    ----------
    predictions_df : must have columns: date, ticker (or tic), predicted_action
    price_df       : must have columns: date, ticker (or tic), close
    initial_capital : starting cash
    method_name    : used for output file naming
    output_dir     : directory to save backtest CSV

    Returns
    -------
    (portfolio_df, daily_returns_list)
    where portfolio_df has: date, portfolio_value, daily_return, cumulative_return
    """
    pred = predictions_df.copy()
    prices = price_df.copy()

    # Normalise column names
    for df in [pred, prices]:
        if "tic" in df.columns and "ticker" not in df.columns:
            df.rename(columns={"tic": "ticker"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])

    dates = sorted(pred["date"].unique())
    portfolio_value = initial_capital
    portfolio_history = []

    # Pre-calculate daily returns for all tickers
    prices = prices.sort_values(["ticker", "date"])
    prices["daily_return"] = prices.groupby("ticker")["close"].pct_change().fillna(0.0)

    for date in dates:
        day_preds = pred[pred["date"] == date]
        day_prices = prices[prices["date"] == date]

        buy_signals = day_preds[day_preds["predicted_action"] == 1]["ticker"].tolist()
        n_buys = len(buy_signals)

        if n_buys == 0:
            daily_return = 0.0
        else:
            # Average daily return of tickers with buy signals
            bought_prices = day_prices[day_prices["ticker"].isin(buy_signals)]
            if len(bought_prices) == 0:
                daily_return = 0.0
            else:
                daily_return = bought_prices["daily_return"].mean()

        portfolio_value *= (1 + daily_return)
        portfolio_history.append({
            "date": date,
            "portfolio_value": portfolio_value,
            "daily_return": daily_return,
            "n_buy_signals": n_buys,
        })

    portfolio_df = pd.DataFrame(portfolio_history)
    portfolio_df["cumulative_return"] = (
        portfolio_df["portfolio_value"] / initial_capital - 1
    )

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio_df.to_csv(output_dir / f"{method_name}_backtest.csv", index=False)

    daily_returns = portfolio_df["daily_return"].tolist()
    return portfolio_df, daily_returns



def build_nasdaq100_benchmark(
    price_df: pd.DataFrame,
    initial_capital: float = 1_000_000,
) -> pd.DataFrame:
    """
    Construct NASDAQ-100 buy-and-hold benchmark from price data.
    Returns the same format as run_backtest() output.
    """
    prices = price_df.copy()
    if "tic" in prices.columns:
        prices.rename(columns={"tic": "ticker"}, inplace=True)
    prices["date"] = pd.to_datetime(prices["date"])

    if "close" not in prices.columns:
        return pd.DataFrame(columns=["date", "portfolio_value", "daily_return", "cumulative_return"])

    # Average close across all tickers per day
    daily_avg = prices.groupby("date")["close"].mean().sort_index()
    daily_return = daily_avg.pct_change().fillna(0.0)

    portfolio_value = initial_capital * (1 + daily_return).cumprod()
    benchmark_df = pd.DataFrame({
        "date": daily_return.index,
        "portfolio_value": portfolio_value.values,
        "daily_return": daily_return.values,
    })
    benchmark_df["cumulative_return"] = portfolio_value.values / initial_capital - 1
    return benchmark_df
