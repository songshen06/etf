"""Performance metrics shared by research portfolio backtest and tooling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance_summary(
    equity: pd.Series,
    daily_returns: pd.Series | None = None,
    trading_days_per_year: int = 252,
    risk_free_rate_annual: float = 0.0,
) -> dict:
    eq = equity.astype(float)
    if eq.empty or not np.isfinite(eq.iloc[0]) or eq.iloc[0] == 0:
        return {
            "total_return": float("nan"),
            "annualized_return": float("nan"),
            "max_drawdown": float("nan"),
            "sharpe_ratio": float("nan"),
            "n_days": int(len(eq)),
        }

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    n = len(eq)
    years = n / trading_days_per_year
    if years > 0 and total_return > -1:
        annualized_return = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0)
    else:
        annualized_return = float("nan")

    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_drawdown = float(dd.min())

    if daily_returns is None:
        daily_returns = eq.pct_change().dropna()
    else:
        daily_returns = daily_returns.dropna()

    rf_daily = risk_free_rate_annual / trading_days_per_year
    excess = daily_returns - rf_daily
    std = float(excess.std(ddof=1))
    if std > 1e-12 and len(excess) > 1:
        sharpe = float(np.sqrt(trading_days_per_year) * excess.mean() / std)
    else:
        sharpe = float("nan")

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "n_days": n,
    }
