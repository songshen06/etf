"""Technical indicators used by the signal engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .signal_dimensions import BIAS_MA_WINDOWS, MOMENTUM_WINDOWS, VOLUME_MA_WINDOWS


def momentum_n(close: pd.Series, window: int) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    return c / c.shift(window) - 1.0


def bias_vs_ma(close: pd.Series, window: int) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    ma = c.rolling(window, min_periods=window).mean()
    return (c - ma) / ma


def volume_ratio(volume: pd.Series, ma_window: int = 20) -> pd.Series:
    v = pd.to_numeric(volume, errors="coerce")
    denom = v.rolling(ma_window, min_periods=1).mean()
    return v / denom


def add_indicators(
    df: pd.DataFrame,
    *,
    momentum_windows: tuple[int, ...] = MOMENTUM_WINDOWS,
    volume_ma_windows: tuple[int, ...] = VOLUME_MA_WINDOWS,
    bias_windows: tuple[int, ...] = BIAS_MA_WINDOWS,
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    为信号三维度预计算列：

    - ``momentum_{n}``：n 日收益
    - ``bias_ma{w}``：相对 w 日均线乖离
    - ``volume_ratio_{k}``：量 / k 日均量
    """
    out = df.copy()
    out[close_col] = pd.to_numeric(out[close_col], errors="coerce")
    out[volume_col] = pd.to_numeric(out[volume_col], errors="coerce")

    for mw in momentum_windows:
        out[f"momentum_{int(mw)}"] = momentum_n(out[close_col], int(mw))

    for w in bias_windows:
        out[f"bias_ma{int(w)}"] = bias_vs_ma(out[close_col], int(w))

    for vw in volume_ma_windows:
        out[f"volume_ratio_{int(vw)}"] = volume_ratio(out[volume_col], ma_window=int(vw))

    return out


def momentum_10(df: pd.DataFrame, *, bias_windows: tuple[int, ...] = BIAS_MA_WINDOWS) -> pd.DataFrame:
    """向后兼容：仅补全常用动量/量比窗口 + 全套乖离均线。"""
    return add_indicators(
        df,
        momentum_windows=(10,),
        volume_ma_windows=(20,),
        bias_windows=bias_windows,
    )


def normalize_bias_to_decimal(s: pd.Series) -> pd.Series:
    """
    Normalize stored bias to a decimal rate consistent with (close - MA) / MA.

    If typical magnitudes look like percent points (|median| > 1), scale by /100.
    """
    x = pd.to_numeric(s, errors="coerce")
    med = x.abs().median()
    if med is not None and np.isfinite(float(med)) and float(med) > 1.0:
        x = x / 100.0
    return x


def compute_research_features(
    df: pd.DataFrame,
    *,
    momentum_window: int,
    bias_ma_window: int,
    volume_ma_window: int = 20,
    close_col: str = "close",
    volume_col: str = "volume",
    use_precomputed_bias: bool = False,
    recompute_bias: bool = True,
    precomputed_bias_col: str = "bias_rate",
) -> pd.DataFrame:
    """
    Research-style columns: momentum, bias, volume_ratio.

    Default matches historical behavior (always recompute bias from MA). When
    ``use_precomputed_bias`` and not ``recompute_bias``, reads ``precomputed_bias_col``
    and normalizes scale via ``normalize_bias_to_decimal``.
    """
    out = df.copy()
    close = pd.to_numeric(out[close_col], errors="coerce")
    out[close_col] = close
    out["momentum"] = momentum_n(close, momentum_window)
    if use_precomputed_bias and (not recompute_bias) and precomputed_bias_col in out.columns:
        out["bias"] = normalize_bias_to_decimal(out[precomputed_bias_col])
    else:
        out["bias"] = bias_vs_ma(close, bias_ma_window)
    vol = pd.to_numeric(out[volume_col], errors="coerce")
    out["volume_ratio"] = volume_ratio(vol, ma_window=volume_ma_window)
    return out


def add_analyzer_indicators(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    """
    Columns used by signal-quality-analyzer: ``close_norm``, ``momentum_5`` / ``momentum_10``,
    ``ma5``, ``momentum_ma5``, ``bias_rate`` (vs MA120), ``volume_norm``, ``volume_ma20``, ``volume_ratio``.
    """
    group_col: str | None = "code" if "code" in df.columns else None

    def _one(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(date_col).copy()
        close_col = "close"
        for col in ("close", "price", "收盘"):
            if col in g.columns:
                close_col = col
                break
        if close_col not in g.columns:
            raise ValueError("No 'close' or 'price' column found in data.")
        g["close_norm"] = pd.to_numeric(g[close_col], errors="coerce")
        g["momentum_5"] = momentum_n(g["close_norm"], 5)
        g["momentum_10"] = momentum_n(g["close_norm"], 10)
        g["ma5"] = g["close_norm"].rolling(window=5).mean()
        g["momentum_ma5"] = g["ma5"] / g["ma5"].shift(5) - 1.0
        g["bias_rate"] = bias_vs_ma(g["close_norm"], 120)
        vol_col = "volume" if "volume" in g.columns else ("成交量" if "成交量" in g.columns else None)
        if vol_col is not None:
            g["volume_norm"] = pd.to_numeric(g[vol_col], errors="coerce")
            g["volume_ma20"] = g["volume_norm"].rolling(window=20).mean()
            g["volume_ratio"] = volume_ratio(g["volume_norm"], ma_window=20)
        else:
            g["volume_ratio"] = np.nan
        return g

    if group_col is not None:
        return df.groupby(group_col, group_keys=False).apply(_one)
    return _one(df)
