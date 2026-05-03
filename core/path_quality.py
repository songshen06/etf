"""
Path-quality research: among origin-state days, which raw features co-occur with
reaching a target state within a horizon (no trades, no exits).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from quantlab.filters.quantile_filter import (
    QuantileRange,
    bucket_cell_to_rank,
    mask_in_quantile_range,
    normalize_bucket_label,
)

from .quantile_buckets import (
    BIAS_BUCKET_COL,
    path_quality_bucket_column,
)
from .signal_dimensions import bias_column
from .state_transition import origin_state_mask

TargetMode = Literal["ever", "final"]

# CLI / JSON feature names → resolved against research-frame columns
KNOWN_FEATURES: tuple[str, ...] = ("bias_rate", "momentum", "volume_ratio", "daily_change")

_MAX_Q = 5
_ORDERED_Q_LABELS: tuple[str, ...] = tuple(f"Q{i}" for i in range(1, _MAX_Q + 1))


def state_matches(state_value: str, target_pattern: str) -> bool:
    """Exact match or ``target_pattern + '_'`` prefix (same as from-state logic)."""
    fs = str(target_pattern).strip()
    if not fs:
        return False
    s = str(state_value).strip()
    return s == fs or s.startswith(fs + "_")


def _add_daily_change(df: pd.DataFrame, *, close_col: str = "close") -> pd.Series:
    c = pd.to_numeric(df[close_col], errors="coerce")
    return c / c.shift(1) - 1.0


def resolve_feature_series(
    df: pd.DataFrame,
    feature_name: str,
    *,
    bias_ma: int,
    momentum_window: int,
    volume_ma_window: int,
    close_col: str = "close",
) -> tuple[pd.Series, str]:
    """
    Return (series aligned to df index after reset, resolved_label).

    ``resolved_label`` is the logical feature name for JSON (same as input for known names).
    """
    from .signal_dimensions import bias_column, momentum_column, volume_ratio_column

    fn = str(feature_name).strip().lower().replace("-", "_")
    if fn == "bias_rate":
        col = bias_column(int(bias_ma))
        if col not in df.columns:
            raise KeyError(f"missing column {col!r} for bias_rate (bias_ma={bias_ma})")
        return pd.to_numeric(df[col], errors="coerce"), "bias_rate"
    if fn == "momentum":
        col = momentum_column(int(momentum_window))
        if col not in df.columns:
            raise KeyError(f"missing column {col!r} for momentum")
        return pd.to_numeric(df[col], errors="coerce"), "momentum"
    if fn == "volume_ratio":
        col = volume_ratio_column(int(volume_ma_window))
        if col not in df.columns:
            raise KeyError(f"missing column {col!r} for volume_ratio")
        return pd.to_numeric(df[col], errors="coerce"), "volume_ratio"
    if fn == "daily_change":
        return _add_daily_change(df, close_col=close_col), "daily_change"
    raise ValueError(
        f"unknown feature {feature_name!r}; allowed: {', '.join(KNOWN_FEATURES)}"
    )


def _assert_em_idx_bias_buckets_in_allowed_range(
    d: pd.DataFrame,
    em_idx: np.ndarray,
    qrange: QuantileRange,
) -> None:
    allowed = qrange.labels()
    col = d[BIAS_BUCKET_COL]
    for ii in em_idx:
        i = int(ii)
        lab = normalize_bucket_label(col.iloc[i])
        if lab is None:
            raise ValueError(
                "bias quantile filter: origin row has missing/invalid bias_bucket; "
                "ensure prepare_research_frame assigned global bucket columns"
            )
        if lab not in allowed:
            raise ValueError(
                f"bias quantile filter invariant violated: row {i} has bias_bucket={lab!r}, "
                f"allowed={sorted(allowed)}"
            )


def build_path_labeled_samples(
    df: pd.DataFrame,
    *,
    from_state: str,
    target_state: str,
    horizon: int,
    target_mode: TargetMode,
    close_col: str = "close",
    date_col: str = "date",
    state_col: str = "state",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Valid ``from_state`` origins with full horizon and finite forward return.

    Returns ``(d, em_idx, hit_e, fwd_e)`` where ``d`` is date-sorted reset frame;
    ``em_idx`` are row indices into ``d``; ``hit_e`` / ``fwd_e`` align to those rows.
    """
    H = int(horizon)
    if H < 1:
        raise ValueError("horizon must be >= 1")

    d = df.sort_values(date_col).reset_index(drop=True)
    n = len(d)
    state = d[state_col].astype(str).to_numpy()
    close = pd.to_numeric(d[close_col], errors="coerce").to_numpy(dtype=float)

    origin = origin_state_mask(pd.Series(state), from_state).to_numpy()
    origin &= state != "MISSING"
    full_h = np.arange(n, dtype=np.int32) + H < n
    valid = origin & full_h

    hit = np.zeros(n, dtype=np.int8)
    fwd = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if not valid[i]:
            continue
        if not (np.isfinite(close[i]) and close[i] > 0 and np.isfinite(close[i + H])):
            continue
        fwd[i] = float(close[i + H] / close[i] - 1.0)
        if target_mode == "final":
            hit[i] = 1 if state_matches(str(state[i + H]), target_state) else 0
        else:
            seen = False
            for j in range(i + 1, i + H + 1):
                if state_matches(str(state[j]), target_state):
                    seen = True
                    break
            hit[i] = 1 if seen else 0

    eval_mask = valid & np.isfinite(fwd)
    em_idx = np.where(eval_mask)[0]
    hit_e = hit[em_idx].astype(np.int8)
    fwd_e = fwd[em_idx]
    return d, em_idx, hit_e, fwd_e


def filter_labeled_samples_by_bias_quantile(
    d: pd.DataFrame,
    em_idx: np.ndarray,
    hit_e: np.ndarray,
    fwd_e: np.ndarray,
    *,
    bias_ma: int,
    qrange: QuantileRange | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only labeled-sample rows whose **global** ``bias_bucket`` lies in ``qrange``."""
    if qrange is None:
        return em_idx, hit_e, fwd_e
    _ = bias_ma  # reserved for API parity with callers using a specific MA column
    if BIAS_BUCKET_COL not in d.columns:
        raise KeyError(
            f"missing {BIAS_BUCKET_COL!r}; run prepare_research_frame (or "
            "assign_global_quantile_bucket_columns) before path-quality with --bias-q"
        )
    sub = d[BIAS_BUCKET_COL].iloc[em_idx].to_numpy(object)
    keep = mask_in_quantile_range(sub, qrange)
    out_idx = em_idx[keep]
    out_hit = hit_e[keep]
    out_fwd = fwd_e[keep]
    _assert_em_idx_bias_buckets_in_allowed_range(d, out_idx, qrange)
    return out_idx, out_hit, out_fwd


def _bucket_labels_at_indices(d: pd.DataFrame, bucket_col: str, em_idx: np.ndarray) -> np.ndarray:
    """Object array of normalized ``Q*`` or ``None`` for missing (safe for boolean masks)."""
    raw = d[bucket_col].iloc[em_idx].to_numpy(object)
    out = np.empty(len(raw), dtype=object)
    for i, v in enumerate(raw):
        lab = normalize_bucket_label(v)
        out[i] = lab
    return out


def compute_path_quality(
    df: pd.DataFrame,
    *,
    from_state: str,
    target_state: str,
    horizon: int,
    target_mode: TargetMode,
    feature_names: tuple[str, ...],
    bucket_n: int,
    bias_ma: int,
    momentum_window: int,
    volume_ma_window: int,
    close_col: str = "close",
    date_col: str = "date",
    state_col: str = "state",
    bias_quantile_range: QuantileRange | None = None,
) -> dict[str, Any]:
    """
    Build hit labels and per-feature **global** quantile bucket stats among valid origin rows.

    Valid origin: ``from_state`` match (prefix rules), state != MISSING, and ``i + horizon < n``
    so the full window and forward return to t+H are defined.

    Breakdown buckets use precomputed ``*_bucket`` columns (see ``assign_global_quantile_bucket_columns``);
    ``bucket_n`` is ignored (kept for API stability; global buckets use five quantiles by default).
    """
    _ = bucket_n
    d, em_idx, hit_e, fwd_e = build_path_labeled_samples(
        df,
        from_state=from_state,
        target_state=target_state,
        horizon=horizon,
        target_mode=target_mode,
        close_col=close_col,
        date_col=date_col,
        state_col=state_col,
    )
    em_idx, hit_e, fwd_e = filter_labeled_samples_by_bias_quantile(
        d,
        em_idx,
        hit_e,
        fwd_e,
        bias_ma=bias_ma,
        qrange=bias_quantile_range,
    )
    total_samples = int(len(em_idx))
    hit_count = int(hit_e.sum())
    hit_rate = float(hit_count / total_samples) if total_samples > 0 else 0.0
    mean_forward_return = float(np.mean(fwd_e)) if total_samples > 0 else 0.0

    feature_breakdowns: list[dict[str, Any]] = []

    for raw_name in feature_names:
        _series, feat_label = resolve_feature_series(
            d,
            raw_name,
            bias_ma=bias_ma,
            momentum_window=momentum_window,
            volume_ma_window=volume_ma_window,
            close_col=close_col,
        )
        bucket_col = path_quality_bucket_column(
            raw_name,
            bias_ma=bias_ma,
            momentum_window=momentum_window,
            volume_ma_window=volume_ma_window,
        )
        if bucket_col is None:
            raise ValueError(
                f"unknown feature {raw_name!r}; allowed: {', '.join(KNOWN_FEATURES)}"
            )
        if bucket_col not in d.columns:
            raise KeyError(
                f"missing column {bucket_col!r}; run prepare_research_frame before path-quality"
            )

        labels_at = _bucket_labels_at_indices(d, bucket_col, em_idx)
        rows_out: list[dict[str, Any]] = []
        for lab in _ORDERED_Q_LABELS:
            m = np.fromiter(
                (labels_at[j] == lab for j in range(len(labels_at))),
                dtype=bool,
                count=len(labels_at),
            )
            cnt = int(m.sum())
            if cnt == 0:
                continue
            h_sub = hit_e[m]
            f_sub = fwd_e[m]
            hc = int(h_sub.sum())
            hr = float(hc / cnt)
            mean_fr = float(np.mean(f_sub))
            wr = float(np.mean(f_sub > 0))
            rows_out.append(
                {
                    "bucket": lab,
                    "count": cnt,
                    "hit_count": hc,
                    "hit_rate": hr,
                    "mean_forward_return": mean_fr,
                    "win_rate_forward": wr,
                }
            )
        m_na = np.array([x is None for x in labels_at], dtype=bool)
        cnt_na = int(m_na.sum())
        if cnt_na > 0:
            h_sub = hit_e[m_na]
            f_sub = fwd_e[m_na]
            hc = int(h_sub.sum())
            hr = float(hc / cnt_na)
            mean_fr = float(np.mean(f_sub))
            wr = float(np.mean(f_sub > 0))
            rows_out.append(
                {
                    "bucket": "NA",
                    "count": cnt_na,
                    "hit_count": hc,
                    "hit_rate": hr,
                    "mean_forward_return": mean_fr,
                    "win_rate_forward": wr,
                }
            )

        feature_breakdowns.append({"feature": feat_label, "buckets": rows_out})

    return {
        "total_samples": total_samples,
        "hit_count": hit_count,
        "hit_rate": hit_rate,
        "mean_forward_return": mean_forward_return,
        "feature_breakdowns": feature_breakdowns,
    }


def ranks_at_indices_for_rules(
    d: pd.DataFrame,
    bucket_col: str,
    em_idx: np.ndarray,
) -> np.ndarray:
    """1..5 ranks for global bucket column at ``em_idx``; ``-1`` if unknown/NA."""
    raw = d[bucket_col].iloc[em_idx].to_numpy(object)
    out = np.full(len(raw), -1, dtype=np.int16)
    for i, v in enumerate(raw):
        r = bucket_cell_to_rank(v)
        if r is not None:
            out[i] = int(r)
    return out
