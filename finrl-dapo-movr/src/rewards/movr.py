"""
src/rewards/movr.py — Multi-Objective Verifiable Reward (MOVR) / Fin-RLVR
=========================================================================
Core novel contribution of the FinRL-DAPO-MOVR research project.

Extends the FinRL-DAPO-SR paper's (arXiv:2505.06408) single exponentiated
sentiment-risk reward with a three-component vector reward. Every component
is verifiable from market data — no human labels required.

Formula:
    R(t) = alpha * accuracy_signal(t)
          + beta  * delta_sharpe(t)
          - gamma * drawdown_penalty(t)

Inputs: portfolio_return (daily %), portfolio_value (absolute)
Outputs: scalar MOVR reward at each timestep

This module has ZERO dependencies outside numpy and the standard library.
Unit tests: test_movr.py at project root.
"""

import numpy as np
from collections import deque
from typing import Optional


class MOVRReward:
    """
    Stateful MOVR reward calculator.

    Must be reset() at the start of each episode.
    Call compute(portfolio_return, portfolio_value) at each step.

    Parameters
    ----------
    alpha : weight on accuracy signal component
    beta  : weight on delta-Sharpe component
    gamma : weight on drawdown penalty component
    sharpe_window : rolling window length (days) for Sharpe computation
    annualise_sharpe : if True, multiply daily Sharpe by sqrt(trading_days_per_year)
    trading_days_per_year : used for annualisation (default 252)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        sharpe_window: int = 20,
        annualise_sharpe: bool = True,
        trading_days_per_year: int = 252,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sharpe_window = sharpe_window
        self.annualise = annualise_sharpe
        self.ann_factor = np.sqrt(trading_days_per_year) if annualise_sharpe else 1.0

        self._returns: deque = deque(maxlen=sharpe_window + 1)
        self._portfolio_values: list = []
        self._prev_sharpe: Optional[float] = None

    def reset(self) -> None:
        """Call at the start of each episode."""
        self._returns.clear()
        self._portfolio_values.clear()
        self._prev_sharpe = None

    def _compute_sharpe(self, returns_window) -> float:
        """Annualised Sharpe ratio from a list of daily returns."""
        arr = np.array(returns_window, dtype=np.float64)
        if len(arr) < 2:
            return 0.0
        std = arr.std() + 1e-8
        return float((arr.mean() / std) * self.ann_factor)

    def compute(
        self,
        portfolio_return: float,
        portfolio_value: float,
    ) -> float:
        """
        Compute MOVR reward for one timestep.

        Parameters
        ----------
        portfolio_return : percentage return this step (e.g. 0.012 for +1.2%)
        portfolio_value  : absolute portfolio value this step

        Returns
        -------
        float : scalar MOVR reward
        """
        # --- Component 1: accuracy signal ---
        accuracy_signal = 1.0 if portfolio_return > 0.0 else -1.0

        # --- Component 2: delta Sharpe ---
        self._returns.append(portfolio_return)
        if len(self._returns) >= self.sharpe_window:
            current_sharpe = self._compute_sharpe(list(self._returns))
            if self._prev_sharpe is None:
                delta_sharpe = 0.0
            else:
                delta_sharpe = current_sharpe - self._prev_sharpe
            self._prev_sharpe = current_sharpe
        else:
            delta_sharpe = 0.0

        # --- Component 3: drawdown penalty ---
        self._portfolio_values.append(portfolio_value)
        peak = max(self._portfolio_values)
        drawdown = max(0.0, (peak - portfolio_value) / (peak + 1e-8))
        # clip to [0, 1] to prevent extreme penalisation
        drawdown_penalty = min(drawdown, 1.0)

        # --- MOVR composite ---
        reward = (
            self.alpha * accuracy_signal
            + self.beta * delta_sharpe
            - self.gamma * drawdown_penalty
        )

        return float(reward)

    def get_state(self) -> dict:
        """Return internal state for logging/debugging."""
        return {
            "n_returns_stored": len(self._returns),
            "prev_sharpe": self._prev_sharpe,
            "peak_value": max(self._portfolio_values) if self._portfolio_values else None,
            "current_value": self._portfolio_values[-1] if self._portfolio_values else None,
        }
