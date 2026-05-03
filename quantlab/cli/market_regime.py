from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def rolling_bucket_rank_series(values: pd.Series, *, window: int, n_buckets: int = 5) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    w = int(window)
    nb = int(n_buckets)
    if nb != 5:
        raise ValueError("only n_buckets=5 supported")
    mp = max(60, min(w, w // 2))
    q20 = s.rolling(w, min_periods=mp).quantile(0.2)
    q40 = s.rolling(w, min_periods=mp).quantile(0.4)
    q60 = s.rolling(w, min_periods=mp).quantile(0.6)
    q80 = s.rolling(w, min_periods=mp).quantile(0.8)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    fin = s.notna() & q20.notna() & q40.notna() & q60.notna() & q80.notna()
    if not fin.any():
        return out
    out.loc[fin & (s <= q20)] = 1.0
    out.loc[fin & (s > q20) & (s <= q40)] = 2.0
    out.loc[fin & (s > q40) & (s <= q60)] = 3.0
    out.loc[fin & (s > q60) & (s <= q80)] = 4.0
    out.loc[fin & (s > q80)] = 5.0
    return out


def detect_market_state_series(momentum_q: pd.Series, *, window: int = 20) -> pd.DataFrame:
    mq = pd.to_numeric(momentum_q, errors="coerce")
    w = int(window)
    high = (mq >= 4).astype(float)
    low = (mq <= 2).astype(float)
    ratio_high = high.rolling(w, min_periods=w).mean()
    ratio_low = low.rolling(w, min_periods=w).mean()
    raw_state = pd.Series(pd.NA, index=mq.index, dtype="string")
    raw_state.loc[ratio_high >= 0.6] = "TREND"
    raw_state.loc[(ratio_high < 0.6) & (ratio_low >= 0.6)] = "DOWN"
    raw_state.loc[(ratio_high < 0.6) & (ratio_low < 0.6) & ratio_high.notna() & ratio_low.notna()] = "RANGE"
    return pd.DataFrame(
        {
            "ratio_high": ratio_high,
            "ratio_low": ratio_low,
            "raw_state": raw_state,
        }
    )


def apply_state_persistence(raw_state: pd.Series, *, min_persist_days: int = 10) -> pd.Series:
    mp = int(min_persist_days)
    rs = raw_state.astype("string")
    out = pd.Series(pd.NA, index=rs.index, dtype="string")
    current: str | None = None
    candidate: str | None = None
    streak = 0
    for i, v in enumerate(rs.tolist()):
        s = None if v is None or str(v) == "<NA>" else str(v)
        if s is None:
            out.iloc[i] = current if current is not None else pd.NA
            continue
        if current is None:
            current = s
            candidate = None
            streak = 0
            out.iloc[i] = current
            continue
        if s == current:
            candidate = None
            streak = 0
            out.iloc[i] = current
            continue
        if candidate is None or s != candidate:
            candidate = s
            streak = 1
        else:
            streak += 1
        if streak >= mp:
            current = candidate
            candidate = None
            streak = 0
        out.iloc[i] = current
    return out


def detect_market_state(df: pd.DataFrame, *, window: int = 20, min_persist_days: int = 10) -> pd.DataFrame:
    if "momentum_q" not in df.columns:
        raise KeyError("missing momentum_q")
    base = detect_market_state_series(df["momentum_q"], window=window)
    stable = apply_state_persistence(base["raw_state"], min_persist_days=min_persist_days)
    base["stable_state"] = stable
    return base

