"""Tabular presentation helpers (percent formatting for event-study tables)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def format_ratio_as_percent(value: float | None, *, digits: int = 2) -> str:
    """0–1 比率 → 可读百分比字符串（如 3.25%）；无效则 —。"""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v * 100.0:.{digits}f}%"


def recommendation_candidates_as_percent_df(df: pd.DataFrame) -> pd.DataFrame:
    """推荐候选表：与事件研究一致，将 mean_return / win_rate / std 转为百分比列名。"""
    out = df.copy()
    pct_cols = {
        "mean_return_60": "mean_return_60_%",
        "mean_return_120": "mean_return_120_%",
        "win_rate_60": "win_rate_60_%",
        "std_60": "std_60_%",
    }
    for c, new_name in pct_cols.items():
        if c not in out.columns:
            continue
        out[new_name] = np.where(
            pd.isna(out[c]),
            np.nan,
            (pd.to_numeric(out[c], errors="coerce") * 100.0).round(2),
        )
        out = out.drop(columns=[c])
    return out


def event_study_as_percent_df(tbl: pd.DataFrame) -> pd.DataFrame:
    """Ratios 0–1 → percentage columns for display / CSV export."""
    out = tbl.copy()
    for c in ("win_rate", "mean_return", "median_return", "std"):
        if c in out.columns:
            out[c] = np.where(
                pd.isna(out[c]),
                np.nan,
                (pd.to_numeric(out[c], errors="coerce") * 100.0).round(2),
            )
    return out.rename(
        columns={
            "win_rate": "win_rate_%",
            "mean_return": "mean_return_%",
            "median_return": "median_return_%",
            "std": "std_%",
        }
    )
