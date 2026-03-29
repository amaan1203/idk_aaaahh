"""
src/envs/env_base.py — Base Stock Trading Environment
======================================================
Adapted from FinRL-DAPO-SR base repo (https://github.com/Ruijian-Zha/FinRL-DAPO-SR).
Corresponds to: env_stocktrading.py in the original repo.

This is a standard FinRL gymnasium-compatible stock trading environment.
Inputs: OHLCV price dataframe
Outputs: gymnasium environment (obs, reward, done, truncated, info)
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class StockTradingEnv(gym.Env):
    """
    A simple stock trading environment based on FinRL's StockTradingEnv.

    Observation space: [portfolio_value_norm, *stock_prices_norm, *stock_holdings]
    Action space: continuous [-1, 1] per stock (sell=-1, hold=0, buy=1)
    Reward: daily portfolio return
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int = 30,
        hmax: int = 100,
        initial_amount: float = 1_000_000,
        buy_cost_pct: float = 1e-3,
        sell_cost_pct: float = 1e-3,
        reward_scaling: float = 1e-4,
        state_space: int = 181,
        action_space: int = 30,
        tech_indicator_list: Optional[list] = None,
        turbulence_threshold: Optional[float] = None,
        risk_indicator_col: str = "turbulence",
        print_verbosity: int = 10,
        day: int = 0,
        random_seed: int = 42,
    ):
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space_dim = action_space
        self.tech_indicator_list = tech_indicator_list or []
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.print_verbosity = print_verbosity
        self.day = day
        self.random_seed = random_seed

        # Action space: continuous per stock
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_space_dim,), dtype=np.float32
        )

        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32
        )

        # Time tracking
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.portfolio_value_memory = [self.initial_amount]
        self.returns_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.cost = 0
        self.trades = 0
        self.episode = 0

        # State
        self.state = self._initiate_state()

    def _get_date(self):
        if "date" in self.df.columns.tolist():
            return self.data["date"].iloc[0] if isinstance(self.data, pd.DataFrame) else self.data["date"]
        return self.day

    def _initiate_state(self):
        if len(self.df.tic.unique()) > 1:
            state = (
                [self.initial_amount]
                + self.data.close.values.tolist()
                + [0] * self.stock_dim
                + sum(
                    [self.data[ti].values.tolist() for ti in self.tech_indicator_list],
                    [],
                )
            )
        else:
            state = (
                [self.initial_amount]
                + [self.data.close]
                + [0] * self.stock_dim
                + [self.data[ti] for ti in self.tech_indicator_list]
            )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            state = (
                [self.portfolio_value]
                + self.data.close.values.tolist()
                + list(self.holdings)
                + sum(
                    [self.data[ti].values.tolist() for ti in self.tech_indicator_list],
                    [],
                )
            )
        else:
            state = (
                [self.portfolio_value]
                + [self.data.close]
                + list(self.holdings)
                + [self.data[ti] for ti in self.tech_indicator_list]
            )
        return state

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            return np.array(self.state, dtype=np.float32), 0.0, True, False, {}

        prev_value = self.portfolio_value
        # Simplified: treat actions as buy/sell signals proportional to capital
        self.day += 1
        self.data = self.df.loc[self.day, :]

        portfolio_return = np.random.normal(0.0005, 0.01)  # placeholder; real impl reads prices
        self.portfolio_value = prev_value * (1 + portfolio_return)

        reward = portfolio_return * self.reward_scaling
        self.state = self._update_state() if hasattr(self, "holdings") else self.state
        self.portfolio_value_memory.append(self.portfolio_value)
        self.returns_memory.append(portfolio_return)
        self.rewards_memory.append(reward)

        return np.array(self.state, dtype=np.float32), reward, self.terminal, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.portfolio_value_memory = [self.initial_amount]
        self.returns_memory = []
        self.actions_memory = []
        self.rewards_memory = []
        self.cost = 0
        self.trades = 0
        self.state = self._initiate_state()
        return np.array(self.state, dtype=np.float32), {}

    def render(self, mode="human"):
        return self.state
