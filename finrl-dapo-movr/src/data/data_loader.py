"""
src/data/data_loader.py — Unified Data Loader
=============================================
Loads and merges the 6 NASDAQ-100 CSVs into train/test splits
ready for all algorithms (DAPO, SFT, LoRA, LLM baseline).

Inputs: ./dataset/*.csv files (from 00_download_data.py)
Outputs: merged pandas DataFrames for train and test periods
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


DATASET_DIR = Path("dataset")


def load_train_data() -> pd.DataFrame:
    """
    Load and merge training CSVs (2013–2018).

    Returns merged DataFrame with columns:
    date, tic, open, high, low, close, volume, sentiment, risk,
    price_change_7d, true_action
    """
    price = pd.read_csv(DATASET_DIR / "train_data_2013_2018.csv")
    risk = pd.read_csv(DATASET_DIR / "train_data_deepseek_risk_2013_2018.csv")
    sentiment = pd.read_csv(DATASET_DIR / "train_data_deepseek_sentiment_2013_2018.csv")
    return _merge_and_enrich(price, risk, sentiment)


def load_test_data() -> pd.DataFrame:
    """
    Load and merge test CSVs (2019–2023).

    Returns merged DataFrame with same columns as load_train_data().
    """
    price = pd.read_csv(DATASET_DIR / "trade_data_2019_2023.csv")
    risk = pd.read_csv(DATASET_DIR / "trade_data_deepseek_risk_2019_2023.csv")
    sentiment = pd.read_csv(DATASET_DIR / "trade_data_deepseek_sentiment_2019_2023.csv")
    return _merge_and_enrich(price, risk, sentiment)


def _merge_and_enrich(
    price: pd.DataFrame,
    risk: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge price, risk, and sentiment dataframes and add engineered features.
    """
    # Standardise column names
    price.columns = [c.lower().strip() for c in price.columns]
    risk.columns = [c.lower().strip() for c in risk.columns]
    sentiment.columns = [c.lower().strip() for c in sentiment.columns]

    # Identify join keys (date + ticker)
    date_col = _find_col(price, ["date", "datadate", "timestamp"])
    tic_col = _find_col(price, ["tic", "ticker", "symbol"])

    # Rename to standard names
    if date_col and date_col != "date":
        price.rename(columns={date_col: "date"}, inplace=True)
    if tic_col and tic_col != "tic":
        price.rename(columns={tic_col: "tic"}, inplace=True)

    # Build merge keys common to all three frames
    merge_keys = ["date", "tic"]

    df = price.copy()

    # Merge sentiment
    sent_col = _find_col(sentiment, ["sentiment", "score", "deepseek_sentiment"])
    if sent_col and all(k in sentiment.columns for k in ["date", "tic"]):
        # Rename to 'sentiment' if it was something else
        sent_data = sentiment[["date", "tic", sent_col]].copy()
        sent_data.rename(columns={sent_col: "sentiment"}, inplace=True)
        df = df.merge(sent_data, on=merge_keys, how="left")
    elif sent_col:
        # Fallback: just take the values if keys don't match (less ideal)
        df["sentiment"] = sentiment[sent_col].values[:len(df)] if len(sentiment) >= len(df) else 3.0
    else:
        df["sentiment"] = 3.0

    # Merge risk
    risk_col = _find_col(risk, ["risk", "deepseek_risk", "risk_score"])
    if risk_col and all(k in risk.columns for k in ["date", "tic"]):
        # Rename to 'risk' if it was something else
        risk_data = risk[["date", "tic", risk_col]].copy()
        risk_data.rename(columns={risk_col: "risk"}, inplace=True)
        df = df.merge(risk_data, on=merge_keys, how="left")
    elif risk_col:
        # Fallback
        df["risk"] = risk[risk_col].values[:len(df)] if len(risk) >= len(df) else 3.0
    else:
        df["risk"] = 3.0

    # Fill missing sentiment/risk with neutral values
    df["sentiment"] = df["sentiment"].fillna(3.0)
    df["risk"] = df["risk"].fillna(3.0)

    # Sort by date and ticker
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["date", "tic"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Add engineered features
    df = _add_features(df)

    return df


def _find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price_change_7d, macd, and true_action columns."""
    # price_change_7d: 7-day trailing return per ticker
    if "close" in df.columns:
        df["price_change_7d"] = df.groupby("tic")["close"].pct_change(7).fillna(0.0)
        # true_action: 1 if next-day close > open else 0
        if "open" in df.columns:
            df["true_action"] = (df["close"] > df["open"]).astype(int)
        else:
            df["true_action"] = 1
        # MACD: EMA12 - EMA26
        df["macd"] = (
            df.groupby("tic")["close"].transform(lambda x: x.ewm(span=12).mean())
            - df.groupby("tic")["close"].transform(lambda x: x.ewm(span=26).mean())
        ).fillna(0.0)
    else:
        df["price_change_7d"] = 0.0
        df["macd"] = 0.0
        df["true_action"] = 1

    return df


def get_train_test_split(
    train_start: str = "2013-01-01",
    train_end: str = "2018-12-31",
    test_start: str = "2019-01-01",
    test_end: str = "2023-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and return (train_df, test_df)."""
    train_df = load_train_data()
    test_df = load_test_data()

    train_df = train_df[
        (train_df["date"] >= train_start) & (train_df["date"] <= train_end)
    ].reset_index(drop=True)

    test_df = test_df[
        (test_df["date"] >= test_start) & (test_df["date"] <= test_end)
    ].reset_index(drop=True)

    return train_df, test_df
