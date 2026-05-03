"""
Generic quantile bucket range parsing and row filtering (Q1..Q5 style labels).

Used by path-quality, path-rule mining, and backtest entry gating.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Canonical research buckets (aligned with default --bucket-n 5).
MAX_QUANTILE_BUCKET = 5
DEFAULT_N_BUCKETS = 5

_SINGLE_Q = re.compile(r"^Q([1-5])$", re.IGNORECASE)
_RANGE_Q = re.compile(r"^Q([1-5])-Q([1-5])$", re.IGNORECASE)


@dataclass(frozen=True)
class QuantileRange:
    """Inclusive 1-based quantile indices (Q1 = 1, Q5 = 5)."""

    start: int
    end: int

    def labels(self) -> frozenset[str]:
        """Canonical ``Q{start}``..``Q{end}`` inclusive (e.g. Q3-Q4 → {Q3, Q4})."""

        return frozenset(f"Q{i}" for i in range(self.start, self.end + 1))

    def __post_init__(self) -> None:
        if not (1 <= self.start <= MAX_QUANTILE_BUCKET):
            raise ValueError(f"start must be in 1..{MAX_QUANTILE_BUCKET}, got {self.start}")
        if not (1 <= self.end <= MAX_QUANTILE_BUCKET):
            raise ValueError(f"end must be in 1..{MAX_QUANTILE_BUCKET}, got {self.end}")
        if self.start > self.end:
            raise ValueError(f"start must be <= end, got Q{self.start}-Q{self.end}")


def parse_quantile_range(value: str | None) -> QuantileRange | None:
    """
    Parse ``Q1``, ``Q1-Q2``, ``q2-q4`` into a range. Whitespace stripped.

    Returns ``None`` if ``value`` is None or empty (no filter).

    Raises ``ValueError`` for invalid tokens (Q0, Q6, reversed range, commas, etc.).
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    compact = re.sub(r"\s+", "", s)
    if "," in compact or ";" in compact:
        raise ValueError(f"non-contiguous or multi-token quantile spec not allowed: {value!r}")
    m1 = _SINGLE_Q.match(compact)
    if m1:
        k = int(m1.group(1))
        return QuantileRange(k, k)
    m2 = _RANGE_Q.match(compact)
    if m2:
        a, b = int(m2.group(1)), int(m2.group(2))
        if a > b:
            raise ValueError(f"quantile range must be low-to-high, got {value!r}")
        return QuantileRange(a, b)
    raise ValueError(
        f"invalid quantile range {value!r}; use Q1..Q5 or contiguous Qa-Qb (e.g. Q1-Q2, Q2-Q4)"
    )


def maybe_parse_quantile_range(value: str | None) -> QuantileRange | None:
    """Alias of ``parse_quantile_range`` for call sites that emphasize optionality."""
    return parse_quantile_range(value)


def normalize_bucket_label(value: Any) -> str | None:
    """
    Map a cell value to canonical ``Q1``..``Q5``, or ``None`` if unknown / NA.

    Accepts strings ``q3``, ``Q3``, or numeric ``1``..``5`` / ``1.0`` (integer floats).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        k = int(value)
        if 1 <= k <= MAX_QUANTILE_BUCKET:
            return f"Q{k}"
        return None
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            return normalize_bucket_label(int(value))
        return None
    t = str(value).strip().upper()
    if not t:
        return None
    m = _SINGLE_Q.match(t)
    if m:
        return f"Q{int(m.group(1))}"
    return None


def bucket_label_to_rank(label: str | None) -> int | None:
    """``Q3`` -> 3; invalid -> None."""
    if label is None:
        return None
    n = normalize_bucket_label(label)
    if n is None:
        return None
    return int(n[1:])


def bucket_cell_to_rank(value: Any) -> int | None:
    """Normalize a dataframe cell (label or 1..5) to rank, or None."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        k = int(value)
        return k if 1 <= k <= MAX_QUANTILE_BUCKET else None
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        if value == int(value):
            return bucket_cell_to_rank(int(value))
        return None
    return bucket_label_to_rank(str(value))


def assign_equal_frequency_quantile_labels(
    values: pd.Series,
    *,
    n_buckets: int = DEFAULT_N_BUCKETS,
) -> pd.Series:
    """
    Equal-count quantile labels ``Q1``..``Qk`` on finite ``values`` (``pd.qcut``).

    Non-finite values become ``pd.NA``. ``k`` may be ``< n_buckets`` if
    ``duplicates='drop'`` collapses bins.
    """
    nb = int(n_buckets)
    if nb < 2:
        raise ValueError("n_buckets must be >= 2")
    s = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    fin = s.notna() & np.isfinite(s.to_numpy(dtype=float))
    if not fin.any():
        return out
    x = s[fin].to_numpy(dtype=float)
    try:
        cat = pd.qcut(pd.Series(x), q=nb, labels=False, duplicates="drop")
    except (ValueError, TypeError):
        cat = pd.Series(np.zeros(len(x), dtype=int))
    c = pd.to_numeric(cat, errors="coerce").to_numpy()
    u = np.sort(np.unique(c[np.isfinite(c)].astype(int)))
    rank_map = {int(old): new + 1 for new, old in enumerate(u)}
    ix = s.index[fin.to_numpy()]
    for j, idx in enumerate(ix):
        cc = c[j]
        if not np.isfinite(cc):
            continue
        rk = rank_map.get(int(cc))
        if rk is not None:
            out.loc[idx] = f"Q{rk}"
    return out


def mask_in_quantile_range(labels: pd.Series | np.ndarray, qrange: QuantileRange) -> np.ndarray:
    """Boolean mask: True where label rank is within ``[qrange.start, qrange.end]``."""
    if isinstance(labels, np.ndarray):
        out = np.zeros(len(labels), dtype=bool)
        for i, v in enumerate(labels):
            r = bucket_cell_to_rank(v)
            out[i] = r is not None and qrange.start <= r <= qrange.end
        return out
    arr = labels.to_numpy(object)
    out = np.zeros(len(arr), dtype=bool)
    for i, v in enumerate(arr):
        r = bucket_cell_to_rank(v)
        out[i] = r is not None and qrange.start <= r <= qrange.end
    return out


def filter_by_quantile_range(
    df: pd.DataFrame,
    bucket_col: str,
    qrange: QuantileRange,
    *,
    copy: bool = False,
) -> pd.DataFrame:
    """Keep rows whose ``bucket_col`` cell maps to a rank inside ``qrange`` (drop NA / unknown)."""
    if bucket_col not in df.columns:
        raise KeyError(f"missing column {bucket_col!r}")
    base = df.copy() if copy else df
    labs = base[bucket_col]
    keep = mask_in_quantile_range(labs, qrange)
    out = base.loc[keep]
    return out.copy() if copy else out


def maybe_filter_by_quantile_range(
    df: pd.DataFrame,
    bucket_col: str,
    qrange_str: str | None,
    *,
    copy: bool = False,
) -> pd.DataFrame:
    """If ``qrange_str`` parses, return filtered frame; else return ``df`` (optionally copied)."""
    qr = parse_quantile_range(qrange_str)
    if qr is None:
        return df.copy() if copy else df
    return filter_by_quantile_range(df, bucket_col, qr, copy=copy)


def suppress_entries_outside_quantile_range(
    df: pd.DataFrame,
    *,
    value_series: pd.Series,
    qrange: QuantileRange,
    tier_col: str = "signal_tier",
    n_buckets: int = DEFAULT_N_BUCKETS,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Zero ``tier_col`` on rows where ``value_series`` falls outside ``qrange``,
    but only where ``tier_col > 0`` (entry candidates). Does not add columns.
    """
    out = df.copy() if copy else df
    if tier_col not in out.columns:
        raise KeyError(f"missing column {tier_col!r}")
    labels = assign_equal_frequency_quantile_labels(value_series, n_buckets=n_buckets)
    ok = mask_in_quantile_range(labels, qrange)
    tier = pd.to_numeric(out[tier_col], errors="coerce").fillna(0).to_numpy(dtype=int)
    clear = (tier > 0) & ~ok
    out.loc[clear, tier_col] = 0
    return out
