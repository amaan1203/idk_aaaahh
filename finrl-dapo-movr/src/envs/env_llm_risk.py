"""
src/envs/env_llm_risk.py — LLM-Risk Stock Trading Environment
=============================================================
Adapted from FinRL-DAPO-SR base repo (https://github.com/Ruijian-Zha/FinRL-DAPO-SR).
Corresponds to: env_stocktrading_llm_risk.py in the original repo.

Extends the base environment with LLM-generated sentiment and risk features.
The reward uses the paper's exponentiated sentiment-risk formula:
  R = portfolio_return * exp(alpha * sentiment) * exp(-beta * risk)

Inputs: OHLCV + sentiment + risk dataframe
Outputs: gymnasium environment with enriched reward signal
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional
from src.envs.env_base import StockTradingEnv


class StockTradingEnvLLMRisk(StockTradingEnv):
    """
    Stock trading environment augmented with LLM-generated sentiment and risk scores.
    Implements the paper's reward: R = daily_return * exp(alpha*sentiment - beta*risk)

    Additional columns required in df: 'sentiment', 'risk' (both 1–5 scale)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        reward_alpha: float = 3.0,   # sentiment exponent (paper Table 1)
        reward_beta: float = 1.0,    # risk exponent (paper Table 1)
        **kwargs,
    ):
        super().__init__(df, **kwargs)
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta

        # Validate that sentiment/risk columns exist
        required_cols = {"sentiment", "risk"}
        available = set(df.columns.tolist())
        if not required_cols.issubset(available):
            missing = required_cols - available
            # Add dummy columns if missing (fallback for testing)
            for col in missing:
                df[col] = 3.0  # neutral default
            self.df = df
            print(f"[WARNING] Missing columns {missing}: using neutral defaults.")

    def _compute_llm_reward_multiplier(self) -> float:
        """
        Compute the sentiment-risk multiplier from current step's data.
        Returns exp(alpha * sentiment_norm - beta * risk_norm)
        where sentiment/risk are normalised from [1,5] to [0,1].
        """
        if isinstance(self.data, pd.DataFrame):
            sentiment = self.data["sentiment"].mean() if "sentiment" in self.data.columns else 3.0
            risk = self.data["risk"].mean() if "risk" in self.data.columns else 3.0
        else:
            sentiment = getattr(self.data, "sentiment", 3.0)
            risk = getattr(self.data, "risk", 3.0)

        # Normalise from [1, 5] to [0, 1]
        sentiment_norm = (sentiment - 1.0) / 4.0
        risk_norm = (risk - 1.0) / 4.0

        multiplier = np.exp(self.reward_alpha * sentiment_norm - self.reward_beta * risk_norm)
        return float(multiplier)

    def step(self, actions):
        obs, base_reward, terminated, truncated, info = super().step(actions)

        # Apply LLM sentiment-risk multiplier to the reward
        if not terminated:
            multiplier = self._compute_llm_reward_multiplier()
            enhanced_reward = base_reward * multiplier
            info["llm_multiplier"] = multiplier
            return obs, enhanced_reward, terminated, truncated, info

        return obs, base_reward, terminated, truncated, info
