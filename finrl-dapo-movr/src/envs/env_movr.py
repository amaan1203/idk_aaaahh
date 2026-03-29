"""
src/envs/env_movr.py — Stock Trading Environment with MOVR Reward
=================================================================
Novel contribution of the FinRL-DAPO-MOVR project.
Corresponds to: Section 3.2 of the paper draft (MOVR / Fin-RLVR extension).

Extends env_llm_risk.py by replacing the exponentiated single-objective
reward with the Multi-Objective Verifiable Reward (MOVR):
  R(t) = alpha * accuracy_signal(t)
        + beta  * delta_sharpe(t)
        - gamma * drawdown_penalty(t)

All three components are verifiable from market data — no human labels required.

Inputs: OHLCV + sentiment + risk dataframe + MOVR hyperparameters
Outputs: gymnasium environment with MOVR reward signal
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.envs.env_llm_risk import StockTradingEnvLLMRisk
from src.rewards.movr import MOVRReward


class StockTradingEnvMOVR(StockTradingEnvLLMRisk):
    """
    Stock trading environment that replaces the paper's single-objective
    sentiment-risk reward with the MOVR three-component reward.

    Additional parameters beyond StockTradingEnvLLMRisk
    ----------------------------------------------------
    movr_alpha : weight on accuracy signal component (default 1.0)
    movr_beta  : weight on delta-Sharpe component (default 0.5)
    movr_gamma : weight on drawdown penalty component (default 0.3)
    movr_sharpe_window : rolling window for delta-Sharpe (default 20)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        movr_alpha: float = 1.0,
        movr_beta: float = 0.5,
        movr_gamma: float = 0.3,
        movr_sharpe_window: int = 20,
        **kwargs,
    ):
        super().__init__(df, **kwargs)

        # MOVR reward calculator — injected via config
        self.movr_alpha = movr_alpha
        self.movr_beta = movr_beta
        self.movr_gamma = movr_gamma
        self.movr_sharpe_window = movr_sharpe_window

        self._movr = MOVRReward(
            alpha=self.movr_alpha,
            beta=self.movr_beta,
            gamma=self.movr_gamma,
            sharpe_window=self.movr_sharpe_window,
        )

    def step(self, actions):
        # Run parent step to get observations and portfolio update
        # (we bypass the parent reward by calling grandparent directly)
        obs, _base_reward, terminated, truncated, info = super(StockTradingEnvLLMRisk, self).step(actions)

        if not terminated:
            # Original portfolio return (unchanged)
            if len(self.portfolio_value_memory) >= 2:
                prev_value = self.portfolio_value_memory[-2]
                portfolio_return = (self.portfolio_value / (prev_value + 1e-8)) - 1.0
            else:
                portfolio_return = 0.0

            # MOVR reward replaces single-objective portfolio return reward
            reward = self._movr.compute(
                portfolio_return=portfolio_return,
                portfolio_value=self.portfolio_value,
            )

            info["movr_state"] = self._movr.state_dict()
            info["portfolio_return"] = portfolio_return
            return obs, reward, terminated, truncated, info

        return obs, 0.0, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Reset MOVR state at episode start
        self._movr.reset()
        return obs, info
