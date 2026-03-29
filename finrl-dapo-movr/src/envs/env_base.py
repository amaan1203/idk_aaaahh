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
        hmax: int = 100,
        initial_amount: float = 1_000_000,
        buy_cost_pct: float = 1e-3,
        sell_cost_pct: float = 1e-3,
        reward_scaling: float = 1e-4,
        tech_indicator_list: Optional[list] = None,
        turbulence_threshold: Optional[float] = None,
        risk_indicator_col: str = "turbulence",
        print_verbosity: int = 10,
        day: int = 0,
        random_seed: int = 42,
        **kwargs,
    ):
        self.df = df
        self.stock_dim = len(self.df.tic.unique())
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list or []
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.print_verbosity = print_verbosity
        self.day = day
        self.random_seed = random_seed

        # --- OPTIMIZATION: Vectorize data to NumPy for speed ---
        # Instead of df.loc in each step, we use pre-indexed arrays
        self.df = self.df.sort_values(["date", "tic"])
        self.dates = self.df["date"].unique()
        self.total_days = len(self.dates)
        
        # Reshape close prices and indicators to (total_days, stock_dim)
        self.price_array = self.df["close"].values.reshape(self.total_days, self.stock_dim)
        self.tech_arrays = []
        for ti in self.tech_indicator_list:
            self.tech_arrays.append(self.df[ti].values.reshape(self.total_days, self.stock_dim))
        
        # Calculate state space dynamically
        self.state_space = 1 + 2 * self.stock_dim + self.stock_dim * len(self.tech_indicator_list)
        self.action_space_dim = self.stock_dim

        # Action space: continuous per stock
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_space_dim,), dtype=np.float32
        )

        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32
        )

        # Time tracking
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.holdings = np.zeros(self.stock_dim)
        self.portfolio_value_memory = [self.initial_amount]
        self.returns_memory = []
        self.actions_memory = []
        self.date_memory = [self.dates[self.day]]
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.cost = 0
        self.trades = 0
        self.episode = 0

        # State
        self.state = self._initiate_state()

    def _initiate_state(self):
        curr_prices = self.price_array[self.day]
        state = [self.portfolio_value] + curr_prices.tolist() + self.holdings.tolist()
        for tech_arr in self.tech_arrays:
            state += tech_arr[self.day].tolist()
        return state

    def _update_state(self):
        return self._initiate_state()

    def step(self, actions):
        self.terminal = self.day >= self.total_days - 1
        if self.terminal:
            return np.array(self.state, dtype=np.float32), 0.0, True, False, {}

        # 1. Action translation
        actions = actions * self.hmax
        
        prev_close = self.price_array[self.day]
        
        # Advance time
        self.day += 1
        curr_close = self.price_array[self.day]
        
        # 2. Calculate daily return
        # portfolio_return = sum(weights * asset_returns)
        asset_returns = (curr_close - prev_close) / (prev_close + 1e-8)
        
        # Normalize actions to weights
        weights = np.abs(actions) / (np.sum(np.abs(actions)) + 1e-8)
        portfolio_return = np.sum(weights * asset_returns)
        
        self.portfolio_value *= (1 + portfolio_return)
        
        reward = portfolio_return * self.reward_scaling
        self.state = self._update_state()
        
        self.portfolio_value_memory.append(self.portfolio_value)
        self.returns_memory.append(portfolio_return)
        self.rewards_memory.append(reward)
        self.actions_memory.append(actions)

        return np.array(self.state, dtype=np.float32), reward, self.terminal, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.holdings = np.zeros(self.stock_dim)
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
