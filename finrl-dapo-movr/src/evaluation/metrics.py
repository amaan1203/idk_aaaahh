"""
src/evaluation/metrics.py — Financial Performance Metrics
=========================================================
All metrics computed from a daily returns series.
No external dependencies beyond numpy and pandas.

Inputs: array or Series of daily percentage returns
Outputs: dict with Sharpe, MDD, Calmar, Sortino, CVaR, etc.
"""

import numpy as np
import pandas as pd
from typing import Union


def compute_all_metrics(
    daily_returns: Union[pd.Series, np.ndarray],
    trading_days_per_year: int = 252,
) -> dict:
    """
    Compute a full suite of financial performance metrics.

    Parameters
    ----------
    daily_returns : series of daily percentage returns (e.g. 0.012 = +1.2%)
    trading_days_per_year : annualisation factor (default 252)

    Returns
    -------
    dict with keys: cumulative_return, annualised_return, sharpe_ratio,
                    max_drawdown, calmar_ratio, sortino_ratio,
                    cvar_5pct, rachev_ratio, win_rate, n_days
    """
    r = np.array(daily_returns, dtype=np.float64)
    n = len(r)
    if n == 0:
        return {k: 0.0 for k in [
            "cumulative_return", "annualised_return", "sharpe_ratio",
            "max_drawdown", "calmar_ratio", "sortino_ratio",
            "cvar_5pct", "rachev_ratio", "win_rate", "n_days",
        ]}

    # Cumulative return
    cum_return = float((1 + r).prod() - 1)

    # Annualised return (CAGR)
    ann_return = float((1 + cum_return) ** (trading_days_per_year / n) - 1)

    # Sharpe ratio (annualised, risk-free rate = 0)
    sharpe = float((r.mean() / (r.std() + 1e-8)) * np.sqrt(trading_days_per_year))

    # Max drawdown
    cum_curve = (1 + r).cumprod()
    rolling_peak = np.maximum.accumulate(cum_curve)
    drawdowns = (cum_curve - rolling_peak) / (rolling_peak + 1e-8)
    max_drawdown = float(drawdowns.min())   # negative number

    # Calmar ratio
    calmar = float(ann_return / (abs(max_drawdown) + 1e-8))

    # Sortino ratio (penalises only downside deviation)
    downside = r[r < 0]
    downside_std = float(downside.std() + 1e-8) if len(downside) > 0 else 1e-8
    sortino = float((r.mean() / downside_std) * np.sqrt(trading_days_per_year))

    # CVaR at 5% (expected loss in worst 5% of days)
    p5 = np.percentile(r, 5)
    cvar_5 = float(r[r <= p5].mean()) if (r <= p5).any() else float(p5)

    # Rachev ratio (upside / downside tail)
    p95 = np.percentile(r, 95)
    top5_mean = float(r[r >= p95].mean()) if (r >= p95).any() else 0.0
    rachev = float(top5_mean / (abs(cvar_5) + 1e-8))

    # Win rate
    win_rate = float((r > 0).mean())

    return {
        "cumulative_return": cum_return,
        "annualised_return": ann_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "sortino_ratio": sortino,
        "cvar_5pct": cvar_5,
        "rachev_ratio": rachev,
        "win_rate": win_rate,
        "n_days": int(n),
    }


def metrics_to_row(method_name: str, metrics: dict, training_time_hours: float = 0.0) -> dict:
    """Format metrics dict as a flat row for comparison tables."""
    return {
        "method": method_name,
        "cumulative_return_pct": f"{metrics['cumulative_return']*100:.2f}%",
        "sharpe_ratio": f"{metrics['sharpe_ratio']:.3f}",
        "max_drawdown_pct": f"{metrics['max_drawdown']*100:.2f}%",
        "calmar_ratio": f"{metrics['calmar_ratio']:.3f}",
        "sortino_ratio": f"{metrics['sortino_ratio']:.3f}",
        "win_rate_pct": f"{metrics['win_rate']*100:.1f}%",
        "training_time_hours": f"{training_time_hours:.2f}h",
    }
