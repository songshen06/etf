"""
Raw entry-signal diagnostics (EOD condition), independent of exits and realized trades.

Mirrors the ``active`` logic in :func:`portfolio_backtest.run_portfolio_backtest`:
- **Fixed tier** (``entry_signal_tier``): same boolean as ``research_tier_mask``.
- **Hierarchical** (``entry_signal_tier is None``): bar is active iff ``signal_tier >= 1`` and
  ``weights_by_tier[tier] > 0``.
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np
import pandas as pd

from .signal_engine import research_tier_mask


def _iso_date(x: Any) -> str:
    ts = pd.Timestamp(x)
    return ts.date().isoformat()


def compute_raw_entry_active(
    df: pd.DataFrame,
    *,
    weights_by_tier: dict[int, float],
    entry_signal_tier: str | None,
    tier_col: str = "signal_tier",
) -> tuple[np.ndarray, pd.Series]:
    """Return ``(active_bool_len_n, dates_series_aligned)`` on sorted-by-date frame."""
    d = df.sort_values("date").reset_index(drop=True)
    dates = d["date"]
    n = len(d)
    active = np.zeros(n, dtype=bool)
    if entry_signal_tier is not None:
        em = research_tier_mask(d, entry_signal_tier)
        active = em.astype(bool).fillna(False).to_numpy(dtype=bool)
    else:
        tier = d[tier_col].to_numpy(dtype=int)
        for i in range(n):
            st = int(tier[i])
            if st >= 1:
                w = float(weights_by_tier.get(st, 0.0))
                if w > 0:
                    active[i] = True
    return active, dates


def _compress_regimes(dates: pd.Series, active: np.ndarray) -> list[dict[str, Any]]:
    regimes: list[dict[str, Any]] = []
    n = len(active)
    i = 0
    while i < n:
        if not bool(active[i]):
            i += 1
            continue
        j = i + 1
        while j < n and bool(active[j]):
            j += 1
        regimes.append(
            {
                "start_date": _iso_date(dates.iloc[i]),
                "end_date": _iso_date(dates.iloc[j - 1]),
                "duration_days": int(j - i),
            }
        )
        i = j
    return regimes


def _persistence_summary(regimes: list[dict[str, Any]]) -> dict[str, Any]:
    durs = [int(r["duration_days"]) for r in regimes]
    if not durs:
        return {
            "regime_count": 0,
            "avg_duration_days": None,
            "max_duration_days": 0,
            "median_duration_days": None,
        }
    return {
        "regime_count": len(durs),
        "avg_duration_days": float(statistics.mean(durs)),
        "max_duration_days": int(max(durs)),
        "median_duration_days": float(statistics.median(durs)),
    }


def _interpretation_note(
    *,
    entry_signal_tier: str | None,
    signal_mode: str,
    bias_ma: int,
    strategy_profile: str,
) -> str:
    if entry_signal_tier is not None:
        return (
            f"EOD raw entry = research_tier_mask('{entry_signal_tier}') "
            f"on apply_signals frame (mode={signal_mode}, MA{bias_ma}, profile={strategy_profile})."
        )
    return (
        f"EOD raw entry = signal_tier in {{1,2,3}} with positive layer weight (hierarchical; "
        f"mode={signal_mode}, MA{bias_ma}, profile={strategy_profile})."
    )


def build_entry_signal_diagnostics(
    df: pd.DataFrame,
    *,
    weights_by_tier: dict[int, float],
    entry_signal_tier: str | None,
    include_raw_dates: bool,
    signal_mode: str,
    bias_ma: int,
    strategy_profile: str,
) -> dict[str, Any]:
    """Plain dict for :class:`~core.schemas.EntrySignalDiagnosticsBlock`."""
    active, dates = compute_raw_entry_active(
        df,
        weights_by_tier=weights_by_tier,
        entry_signal_tier=entry_signal_tier,
    )
    count = int(np.sum(active))
    raw_dates: list[str] | None = None
    if include_raw_dates:
        raw_dates = [_iso_date(dates.iloc[i]) for i in range(len(active)) if bool(active[i])]
    regimes = _compress_regimes(dates, active)
    summary = _persistence_summary(regimes)
    return {
        "definition_note": _interpretation_note(
            entry_signal_tier=entry_signal_tier,
            signal_mode=signal_mode,
            bias_ma=bias_ma,
            strategy_profile=strategy_profile,
        ),
        "raw_entry_days_count": count,
        "raw_entry_dates": raw_dates,
        "entry_regimes": regimes,
        "entry_persistence_summary": summary,
    }
