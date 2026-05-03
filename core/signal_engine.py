"""
Signal construction: full-sample or rolling quantiles.

NEG: momentum bottom third | LOW: bias bottom third | HIGH: volume top third.
Also assigns hierarchical signal_tier in {0,1,2,3} for backtests.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

SignalMode = Literal["full_sample", "rolling"]


def _bias_col(bias_ma_window: int) -> str:
    return f"bias_ma{int(bias_ma_window)}"


def neg_low_high_masks(
    m: pd.Series,
    b: pd.Series,
    v: pd.Series,
    *,
    mode: SignalMode,
    rolling_window: int,
    quantile_low: float,
    quantile_high: float,
    rolling_min_periods: int | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    NEG / LOW / HIGH boolean masks (same thresholds as ``apply_signals`` / research).

    LOW uses bottom ``quantile_low`` of bias; HIGH uses top ``1 - quantile_high``
    slice via ``>= quantile(quantile_high)`` on volume_ratio (research volume_rule).
    """
    m = pd.to_numeric(m, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    v = pd.to_numeric(v, errors="coerce")

    if mode == "full_sample":
        qm = m.quantile(quantile_low)
        qb = b.quantile(quantile_low)
        qv = v.quantile(quantile_high)
        neg = m <= qm
        low = b <= qb
        high = v >= qv
    elif mode == "rolling":
        w = int(rolling_window)
        if w < 20:
            raise ValueError("rolling_window must be >= 20 for rolling quantiles")
        mp = int(w if rolling_min_periods is None else rolling_min_periods)
        qm = m.rolling(w, min_periods=mp).quantile(quantile_low)
        qb = b.rolling(w, min_periods=mp).quantile(quantile_low)
        qv = v.rolling(w, min_periods=mp).quantile(quantile_high)
        neg = m <= qm
        low = b <= qb
        high = v >= qv
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return neg, low, high


def assign_research_signal_flags(
    df: pd.DataFrame,
    *,
    signal_mode: SignalMode,
    signal_rolling_window: int,
    quantile_low: float,
    quantile_high: float,
    rolling_min_periods: int | None = None,
) -> pd.DataFrame:
    """Append NEG, LOW, HIGH using columns ``momentum``, ``bias``, ``volume_ratio``."""
    out = df.copy()
    for col in ("momentum", "bias", "volume_ratio"):
        if col not in out.columns:
            raise KeyError(f"Missing {col}; run compute_research_features first")
    neg, low, high = neg_low_high_masks(
        out["momentum"],
        out["bias"],
        out["volume_ratio"],
        mode=signal_mode,
        rolling_window=signal_rolling_window,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
        rolling_min_periods=rolling_min_periods,
    )
    out["NEG"] = neg.fillna(False).to_numpy(dtype=bool)
    out["LOW"] = low.fillna(False).to_numpy(dtype=bool)
    out["HIGH"] = high.fillna(False).to_numpy(dtype=bool)
    return out


def research_tier_mask(df: pd.DataFrame, tier: str) -> pd.Series:
    if tier == "NEG":
        return df["NEG"]
    if tier == "NEG_LOW":
        return df["NEG"] & df["LOW"]
    if tier == "NEG_LOW_HIGH":
        return df["NEG"] & df["LOW"] & df["HIGH"]
    raise ValueError(f"Unknown tier: {tier}")


def apply_signals(
    df: pd.DataFrame,
    *,
    mode: SignalMode,
    bias_ma_window: int,
    momentum_col: str = "momentum_10",
    volume_col: str = "volume_ratio_20",
    rolling_window: int = 252,
    quantile_low: float = 0.33,
    quantile_high: float = 0.67,
    rolling_min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Requires precomputed columns: momentum_* , bias_ma{bias_ma_window}, volume_ratio_* (see volume_col).

    Appends NEG, LOW, HIGH (bool) and signal_tier (0–3).
    """
    out = df.copy()
    bcol = _bias_col(bias_ma_window)
    if bcol not in out.columns:
        raise KeyError(f"Missing {bcol}; run add_indicators with bias_windows including {bias_ma_window}")
    if momentum_col not in out.columns:
        raise KeyError(f"Missing {momentum_col}")
    if volume_col not in out.columns:
        raise KeyError(f"Missing {volume_col}")

    m = pd.to_numeric(out[momentum_col], errors="coerce")
    b = pd.to_numeric(out[bcol], errors="coerce")
    v = pd.to_numeric(out[volume_col], errors="coerce")

    neg, low, high = neg_low_high_masks(
        m,
        b,
        v,
        mode=mode,
        rolling_window=rolling_window,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
        rolling_min_periods=rolling_min_periods,
    )

    out["NEG"] = neg.fillna(False).to_numpy(dtype=bool)
    out["LOW"] = low.fillna(False).to_numpy(dtype=bool)
    out["HIGH"] = high.fillna(False).to_numpy(dtype=bool)

    n = len(out)
    t = np.zeros(n, dtype=np.int32)
    neg_a = out["NEG"].to_numpy()
    low_a = out["LOW"].to_numpy()
    high_a = out["HIGH"].to_numpy()
    for i in range(n):
        if not neg_a[i]:
            continue
        if low_a[i] and high_a[i]:
            t[i] = 3
        elif low_a[i]:
            t[i] = 2
        else:
            t[i] = 1
    out["signal_tier"] = t
    return out
