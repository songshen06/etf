"""
Global equal-frequency quantile bucket columns (single source of truth).

Assigned once on the full research frame; filtering and breakdowns must not re-qcut.
"""

from __future__ import annotations

import pandas as pd

from quantlab.filters.quantile_filter import (
    DEFAULT_N_BUCKETS,
    assign_equal_frequency_quantile_labels,
)

from .signal_dimensions import bias_column, momentum_column, volume_ratio_column

BIAS_BUCKET_COL = "bias_bucket"
MOMENTUM_BUCKET_COL = "momentum_bucket"
VOLUME_RATIO_BUCKET_COL = "volume_ratio_bucket"
DAILY_CHANGE_BUCKET_COL = "daily_change_bucket"


def assign_global_quantile_bucket_columns(
    df: pd.DataFrame,
    *,
    bias_ma: int,
    momentum_window: int,
    volume_ma_window: int,
    n_buckets: int = DEFAULT_N_BUCKETS,
    close_col: str = "close",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add ``bias_bucket``, ``momentum_bucket``, ``volume_ratio_bucket``, ``daily_change_bucket``
    via ``assign_equal_frequency_quantile_labels`` on the **entire** frame index.
    """
    out = df if inplace else df.copy()
    nb = int(n_buckets)
    bcol = bias_column(int(bias_ma))
    mcol = momentum_column(int(momentum_window))
    vcol = volume_ratio_column(int(volume_ma_window))
    for c in (bcol, mcol, vcol):
        if c not in out.columns:
            raise KeyError(f"missing column {c!r}; cannot assign global quantile buckets")
    out[BIAS_BUCKET_COL] = assign_equal_frequency_quantile_labels(
        pd.to_numeric(out[bcol], errors="coerce"),
        n_buckets=nb,
    )
    out[MOMENTUM_BUCKET_COL] = assign_equal_frequency_quantile_labels(
        pd.to_numeric(out[mcol], errors="coerce"),
        n_buckets=nb,
    )
    out[VOLUME_RATIO_BUCKET_COL] = assign_equal_frequency_quantile_labels(
        pd.to_numeric(out[vcol], errors="coerce"),
        n_buckets=nb,
    )
    close = pd.to_numeric(out[close_col], errors="coerce")
    daily_ch = close / close.shift(1) - 1.0
    out[DAILY_CHANGE_BUCKET_COL] = assign_equal_frequency_quantile_labels(
        daily_ch,
        n_buckets=nb,
    )
    return out


def path_quality_bucket_column(
    feature_name: str,
    *,
    bias_ma: int,
    momentum_window: int,
    volume_ma_window: int,
) -> str | None:
    """Logical feature name → global bucket column, or ``None`` if unknown."""
    fn = str(feature_name).strip().lower().replace("-", "_")
    if fn == "bias_rate":
        return BIAS_BUCKET_COL
    if fn == "momentum":
        return MOMENTUM_BUCKET_COL
    if fn == "volume_ratio":
        return VOLUME_RATIO_BUCKET_COL
    if fn == "daily_change":
        return DAILY_CHANGE_BUCKET_COL
    return None
