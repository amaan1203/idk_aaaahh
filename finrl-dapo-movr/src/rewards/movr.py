"""
src/rewards/movr.py
===================
Multi-Objective Verifiable Reward (MOVR) — Fin-RLVR

Extends the paper's single exponentiated reward:
    r'_t = r_t * (S^alpha / R^beta)          [arXiv:2505.06408, eq. 3]

With an additive three-component verifiable reward:
    R(t) = alpha * accuracy_signal(t)
          + beta  * delta_sharpe(t)
          - gamma * drawdown_penalty(t)

Each component is verifiable from market data — no human labels.

Inputs per step:  portfolio_return (float), portfolio_value (float)
Output per step:  scalar reward (float)

Dependencies: numpy, Python stdlib only.
This module is ZERO ML dependencies — fully unit-testable on any machine.

Unit tests: python test_movr.py (project root)
"""

import numpy as np
from collections import deque
from typing import Optional


class MOVRReward:
    """
    Stateful per-episode MOVR reward calculator.
    Call reset() at episode start.
    Call compute(portfolio_return, portfolio_value) at each step.

    Parameters
    ----------
    alpha         : weight on accuracy_signal component (>= 0)
    beta          : weight on delta_sharpe component (>= 0)
    gamma         : weight on drawdown_penalty component (>= 0)
    sharpe_window : rolling window length (days) for Sharpe computation
    trading_days_per_year : used for Sharpe annualisation (default 252)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        sharpe_window: int = 20,
        trading_days_per_year: int = 252,
    ):
        if alpha < 0 or beta < 0 or gamma < 0:
            raise ValueError("All MOVR weights (alpha, beta, gamma) must be >= 0.")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sharpe_window = sharpe_window
        self.ann_factor = np.sqrt(trading_days_per_year)
        self._returns: deque = deque(maxlen=sharpe_window + 1)
        self._values: list = []
        self._prev_sharpe: Optional[float] = None

    def reset(self) -> None:
        """Clear all episode state. Must be called at the start of each episode."""
        self._returns.clear()
        self._values.clear()
        self._prev_sharpe = None

    def compute(self, portfolio_return: float, portfolio_value: float) -> float:
        """
        Compute MOVR scalar reward for one timestep.

        Parameters
        ----------
        portfolio_return : fractional return this step (e.g. 0.012 = +1.2%)
        portfolio_value  : absolute portfolio value (e.g. 1_012_000.0)

        Returns
        -------
        float : MOVR = alpha*acc + beta*delta_sharpe - gamma*drawdown
        """
        acc      = self._accuracy_signal(portfolio_return)
        dsharpe  = self._delta_sharpe(portfolio_return)
        drawdown = self._drawdown_penalty(portfolio_value)
        return float(self.alpha * acc + self.beta * dsharpe - self.gamma * drawdown)

    def state_dict(self) -> dict:
        """Return internal state for logging/debugging."""
        return {
            "n_returns":   len(self._returns),
            "prev_sharpe": self._prev_sharpe,
            "peak_value":  max(self._values) if self._values else None,
            "last_value":  self._values[-1] if self._values else None,
            "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _accuracy_signal(self, r: float) -> float:
        return 1.0 if r > 0.0 else -1.0

    def _delta_sharpe(self, r: float) -> float:
        self._returns.append(r)
        if len(self._returns) < self.sharpe_window:
            return 0.0
        arr = np.array(self._returns, dtype=np.float64)
        current = (arr.mean() / (arr.std() + 1e-8)) * self.ann_factor
        if self._prev_sharpe is None:
            self._prev_sharpe = current
            return 0.0
        delta = current - self._prev_sharpe
        self._prev_sharpe = current
        return float(delta)

    def _drawdown_penalty(self, value: float) -> float:
        self._values.append(value)
        peak = max(self._values)
        return float(np.clip((peak - value) / (peak + 1e-8), 0.0, 1.0))
