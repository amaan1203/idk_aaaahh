"""
src/evaluation/metrics.py
==========================
Financial performance metrics from a daily returns series.
No ML dependencies. numpy only.

Inputs:  list or ndarray of daily fractional returns (e.g. 0.012 = +1.2%)
Outputs: dict of performance metrics

Corresponds to: evaluation methodology in arXiv:2505.06408
"""

import numpy as np
from typing import Union


def compute_all(
    daily_returns: Union[list, np.ndarray],
    trading_days: int = 252,
) -> dict:
    """
    Compute a full suite of financial performance metrics.

    Parameters
    ----------
    daily_returns : list or ndarray of daily fractional returns
    trading_days  : trading days per year for annualisation (default 252)

    Returns
    -------
    dict with keys:
        cumulative_return, annualised_return, sharpe_ratio,
        max_drawdown, calmar_ratio, sortino_ratio,
        rachev_ratio, cvar_5pct, win_rate, n_days
    """
    r = np.array(daily_returns, dtype=np.float64)
    n, eps = len(r), 1e-8

    if n == 0:
        return {k: 0.0 for k in [
            "cumulative_return", "annualised_return", "sharpe_ratio",
            "max_drawdown", "calmar_ratio", "sortino_ratio",
            "rachev_ratio", "cvar_5pct", "win_rate", "n_days"
        ]}

    cum  = float((1 + r).prod() - 1)
    ann  = float((1 + cum) ** (trading_days / n) - 1)
    sharpe = float((r.mean() / (r.std() + eps)) * np.sqrt(trading_days))

    # Max drawdown
    curve = (1 + r).cumprod()
    mdd   = float(((curve - np.maximum.accumulate(curve)) /
                   (np.maximum.accumulate(curve) + eps)).min())
    calmar = float(ann / (abs(mdd) + eps))

    # Sortino ratio (downside deviation)
    down = r[r < 0]
    sortino = float((r.mean() / ((down.std() if len(down) > 0 else eps) + eps))
                    * np.sqrt(trading_days))

    # CVaR at 5% (expected loss in worst 5% of days)
    cvar_t = np.percentile(r, 5)
    cvar_5 = float(r[r <= cvar_t].mean()) if (r <= cvar_t).any() else float(cvar_t)

    # Rachev ratio (expected gain top 5% / expected loss bottom 5%)
    top5 = r[r >= np.percentile(r, 95)]
    rachev = float((top5.mean() if len(top5) > 0 else 0.0) / (abs(cvar_5) + eps))

    return {
        "cumulative_return": cum,
        "annualised_return": ann,
        "sharpe_ratio":      sharpe,
        "max_drawdown":      mdd,
        "calmar_ratio":      calmar,
        "sortino_ratio":     sortino,
        "cvar_5pct":         cvar_5,
        "rachev_ratio":      rachev,
        "win_rate":          float((r > 0).mean()),
        "n_days":            int(n),
    }


def metrics_to_row(metrics: dict, method_name: str = "") -> dict:
    """
    Format metrics dict as a flat row for DataFrame concatenation.
    Adds a 'method' key if method_name is provided.
    """
    row = {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()}
    if method_name:
        row["method"] = method_name
    return row
