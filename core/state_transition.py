"""
State transition analysis on daily ternary state sequences (no trades, no exit rules).

Uses the same ``state`` string as ``assign_ternary_states`` (signal-quality / state-rank).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


def origin_state_mask(state_series: pd.Series, from_state: str) -> pd.Series:
    """
    Rows where composite ``state`` matches ``from_state`` exactly or as a prefix.

    Prefix rule: ``from_state + "_"`` so ``NEG_LOW`` matches ``NEG_LOW_HIGH`` but not
    ``NEG_NEU_LOW``.
    """
    fs = str(from_state).strip()
    if not fs:
        raise ValueError("from_state must be a non-empty string")
    s = state_series.astype(str)
    exact = s == fs
    prefixed = s.str.startswith(fs + "_")
    return exact | prefixed


def _entropy_nats_from_counts(counts: dict[str, int]) -> float:
    tot = sum(counts.values())
    if tot <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / tot
        h -= p * math.log(p)
    return float(h)


def infer_state_pattern(states: pd.Series) -> str:
    for x in states.astype(str).unique():
        if x == "MISSING":
            continue
        if str(x).count("_") >= 2:
            return "MOM_BIAS_VOL"
        if str(x).count("_") == 1:
            return "MOM_BIAS"
    return "MOM_BIAS_VOL"


def compute_state_transitions(
    df: pd.DataFrame,
    *,
    from_state: str,
    horizons: tuple[int, ...],
    top_k: int | None = None,
    close_col: str = "close",
    date_col: str = "date",
    state_col: str = "state",
) -> dict[str, Any]:
    """
    For each row t matching ``from_state`` (non-MISSING), for horizon h take state at t+h
    and forward return close[t+h]/close[t]-1.

    Returns a dict suitable for ``StateTransitionResponse`` construction.
    """
    if state_col not in df.columns:
        raise ValueError(f"DataFrame must contain column {state_col!r}")
    if close_col not in df.columns:
        raise ValueError(f"DataFrame must contain column {close_col!r}")

    d = df.sort_values(date_col).reset_index(drop=True)
    close = pd.to_numeric(d[close_col], errors="coerce").to_numpy(dtype=float)
    state = d[state_col].astype(str).to_numpy()
    n = len(d)

    omask = origin_state_mask(pd.Series(state), from_state).to_numpy()
    omask &= state != "MISSING"
    total_samples = int(omask.sum())

    horizons_out: dict[str, Any] = {}
    for h in horizons:
        hh = int(h)
        if hh < 1:
            raise ValueError(f"horizon must be >= 1, got {hh}")

        counts: dict[str, int] = defaultdict(int)
        returns_by: dict[str, list[float]] = defaultdict(list)
        n_valid = 0

        for i in range(n):
            if not omask[i]:
                continue
            j = i + hh
            if j >= n:
                continue
            if not (np.isfinite(close[i]) and np.isfinite(close[j]) and close[i] > 0):
                continue
            to_st = str(state[j])
            r = float(close[j] / close[i] - 1.0)
            counts[to_st] += 1
            returns_by[to_st].append(r)
            n_valid += 1

        entropy = _entropy_nats_from_counts(dict(counts))

        rows: list[dict[str, Any]] = []
        for to_st, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            rs = returns_by[to_st]
            mean_ret = float(np.mean(rs)) if rs else None
            win_rate = float(sum(1 for x in rs if x > 0) / len(rs)) if rs else None
            prob = float(c) / float(n_valid) if n_valid > 0 else 0.0
            rows.append(
                {
                    "to_state": to_st,
                    "prob": prob,
                    "count": int(c),
                    "mean_return": mean_ret,
                    "win_rate": win_rate,
                }
            )

        if top_k is not None and top_k > 0:
            rows = rows[: int(top_k)]

        horizons_out[str(hh)] = {
            "n_valid": int(n_valid),
            "entropy_nats": float(entropy),
            "transitions": rows,
        }

    return {
        "total_samples": total_samples,
        "state_pattern": infer_state_pattern(d[state_col]),
        "horizons": horizons_out,
    }
