"""Load → validate → indicators → signals (reused by CLI, Streamlit, notebooks)."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from .data_loader import load_etf_sqlite
from .data_validation import DataValidationResult
from .indicators import add_indicators
from .signal_dimensions import (
    BIAS_MA_WINDOWS,
    MOMENTUM_WINDOWS,
    VOLUME_MA_WINDOWS,
    momentum_column,
    volume_ratio_column,
)
from .quantile_buckets import assign_global_quantile_bucket_columns
from .signal_engine import SignalMode, apply_signals


def prepare_indicator_panel(db_path: str, etf_code: str) -> tuple[pd.DataFrame, DataValidationResult]:
    """Validated OHLCV + full indicator grid (no NEG/LOW/HIGH tier flags)."""
    df, vres = load_etf_sqlite(db_path, etf_code)
    df = add_indicators(
        df,
        momentum_windows=MOMENTUM_WINDOWS,
        volume_ma_windows=VOLUME_MA_WINDOWS,
        bias_windows=BIAS_MA_WINDOWS,
    )
    return df, vres


BiasSource = Literal["recompute", "db"]


def prepare_research_frame(
    db_path: str,
    etf_code: str,
    *,
    signal_mode: SignalMode,
    bias_ma_window: int,
    momentum_window: int = 10,
    volume_ma_window: int = 20,
    rolling_window: int = 252,
    quantile_low: float = 0.33,
    quantile_high: float = 0.67,
    bias_source: BiasSource = "recompute",
) -> tuple[pd.DataFrame, DataValidationResult]:
    df, vres = prepare_indicator_panel(db_path, etf_code)
    if bias_source == "db":
        from .indicators import normalize_bias_to_decimal
        from .signal_dimensions import bias_column

        if "bias_rate" not in df.columns:
            raise ValueError(
                "bias_source='db' requires column bias_rate in SQLite for this ETF "
                f"(etf_code={etf_code!r}). Use bias_source='recompute' or add the column."
            )
        bcol = bias_column(bias_ma_window)
        df = df.copy()
        df[bcol] = normalize_bias_to_decimal(df["bias_rate"])
    df = apply_signals(
        df,
        mode=signal_mode,
        bias_ma_window=bias_ma_window,
        momentum_col=momentum_column(momentum_window),
        volume_col=volume_ratio_column(volume_ma_window),
        rolling_window=rolling_window,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )
    assign_global_quantile_bucket_columns(
        df,
        bias_ma=bias_ma_window,
        momentum_window=momentum_window,
        volume_ma_window=volume_ma_window,
        inplace=True,
    )
    return df, vres
