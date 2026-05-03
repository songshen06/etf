"""Lightweight helpers for displaying / exporting research output."""

from __future__ import annotations

from typing import Any

import pandas as pd


def metrics_dataframe(metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([metrics]).T.rename(columns={0: "value"})


def trades_dataframe(trades: list[Any]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_date",
                "exit_date",
                "signal_tier",
                "weight",
                "entry_price",
                "exit_price",
                "stock_return",
                "portfolio_return",
                "holding_days",
            ]
        )
    rows = []
    for t in trades:
        rows.append(
            {
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "signal_tier": t.signal_tier,
                "weight": t.weight,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stock_return": t.stock_return,
                "portfolio_return": t.portfolio_return,
                "holding_days": t.holding_days,
            }
        )
    return pd.DataFrame(rows)
