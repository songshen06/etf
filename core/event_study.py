"""Event-study statistics from signal date (close) to future close."""

from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np
import pandas as pd

from .state_quality import add_forward_returns

OutputFormat = Literal["table", "research"]

__all__ = ["OutputFormat", "add_forward_returns", "run_event_study", "tier_event_studies"]


@overload
def run_event_study(
    df: pd.DataFrame,
    signal_mask: pd.Series,
    horizons: tuple[int, ...],
    *,
    output_format: Literal["table"] = "table",
    close_col: str = "close",
    tier: str | None = None,
    path_stats_max_days: int | None = None,
) -> pd.DataFrame: ...


@overload
def run_event_study(
    df: pd.DataFrame,
    signal_mask: pd.Series,
    horizons: tuple[int, ...],
    *,
    output_format: Literal["research"],
    close_col: str = "close",
    tier: str | None = None,
    path_stats_max_days: int | None = None,
) -> dict[str, Any]: ...


def run_event_study(
    df: pd.DataFrame,
    signal_mask: pd.Series,
    horizons: tuple[int, ...],
    *,
    output_format: OutputFormat = "table",
    close_col: str = "close",
    tier: str | None = None,
    path_stats_max_days: int | None = None,
) -> pd.DataFrame | dict[str, Any]:
    """
    Overlapping events: each True row in ``signal_mask`` is one sample.

    Forward return = close[t+h]/close[t] - 1 (signal-day close to horizon close).

    * ``output_format="table"`` — one row per horizon (``mean_return`` column name;
      filters non-finite returns before stats; UI / charts).
    * ``output_format="research"`` — JSON-shaped dict including optional path stats;
      horizon stats use the full return array (historical research parity).
    """
    if output_format == "research":
        if tier is None or path_stats_max_days is None:
            raise ValueError('output_format="research" requires tier and path_stats_max_days')
        return _event_study_research_dict(
            df,
            signal_mask,
            horizons,
            tier=tier,
            path_stats_max_days=path_stats_max_days,
            close_col=close_col,
        )
    return _event_study_table_df(
        df,
        signal_mask,
        horizons,
        close_col=close_col,
        filter_nonfinite=True,
    )


def _event_study_table_df(
    df: pd.DataFrame,
    signal: pd.Series,
    horizons: tuple[int, ...],
    *,
    close_col: str,
    filter_nonfinite: bool,
) -> pd.DataFrame:
    d = df.sort_values("date").reset_index(drop=True) if "date" in df.columns else df.reset_index(drop=True)
    mask = signal.reindex(d.index).fillna(False).to_numpy(dtype=bool)
    close = pd.to_numeric(d[close_col], errors="coerce").to_numpy(dtype=float)
    n = len(close)
    idx = np.flatnonzero(mask)

    rows: list[dict] = []
    for h in horizons:
        h = int(h)
        rets: list[float] = []
        for i in idx:
            if i + h < n:
                rets.append(close[i + h] / close[i] - 1.0 if close[i] > 0 else float("nan"))
        arr = np.array(rets, dtype=float)
        if filter_nonfinite:
            arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            rows.append(
                {
                    "horizon": h,
                    "n": 0,
                    "win_rate": np.nan,
                    "mean_return": np.nan,
                    "median_return": np.nan,
                    "std": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "horizon": h,
                    "n": int(len(arr)),
                    "win_rate": float((arr > 0).mean()),
                    "mean_return": float(arr.mean()),
                    "median_return": float(np.median(arr)),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _event_study_research_dict(
    df: pd.DataFrame,
    signal: pd.Series,
    event_study_horizons: tuple[int, ...],
    *,
    tier: str,
    path_stats_max_days: int,
    close_col: str,
) -> dict[str, Any]:
    d = df.sort_values("date").reset_index(drop=True) if "date" in df.columns else df.reset_index(drop=True)
    mask = signal.reindex(d.index).fillna(False)
    close = d[close_col].to_numpy(dtype=float)
    n = len(close)
    idx = np.flatnonzero(mask.to_numpy())

    horizons_out: dict[str, dict[str, float | int]] = {}
    for h in event_study_horizons:
        rets: list[float] = []
        for i in idx:
            if i + h < n:
                rets.append(close[i + h] / close[i] - 1.0)
        arr = np.array(rets, dtype=float)
        if len(arr) == 0:
            horizons_out[str(h)] = {
                "n": 0,
                "win_rate": float("nan"),
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
            }
        else:
            horizons_out[str(h)] = {
                "n": int(len(arr)),
                "win_rate": float((arr > 0).mean()),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            }

    P = int(path_stats_max_days)
    path_mean: list[float] = []
    path_n: list[int] = []
    for dday in range(0, P + 1):
        rets = [close[i + dday] / close[i] - 1.0 for i in idx if i + dday < n]
        path_n.append(len(rets))
        path_mean.append(float(np.mean(rets)) if rets else float("nan"))

    return {
        "kind": "signal_quality_event_study",
        "tier": tier,
        "signal_day_count": int(mask.sum()),
        "forward_from": "signal_day_close",
        "forward_to": "close_at_t_plus_h",
        "overlap_allowed": True,
        "horizons": horizons_out,
        "path_stats": {
            "max_day": P,
            "mean_cumulative_return_by_day": path_mean,
            "sample_size_by_day": path_n,
        },
    }


def tier_event_studies(
    df: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (20, 60, 120),
) -> dict[str, pd.DataFrame]:
    """Run :func:`run_event_study` in ``table`` mode for NEG-only, NEG+LOW, NEG+LOW+HIGH tiers."""
    if "NEG" not in df.columns:
        raise KeyError("DataFrame must have NEG/LOW/HIGH from apply_signals")
    low = df["LOW"]
    high = df["HIGH"]
    neg = df["NEG"]
    return {
        "NEG": run_event_study(df, neg, horizons, output_format="table"),
        "NEG_LOW": run_event_study(df, neg & low, horizons, output_format="table"),
        "NEG_LOW_HIGH": run_event_study(df, neg & low & high, horizons, output_format="table"),
    }
