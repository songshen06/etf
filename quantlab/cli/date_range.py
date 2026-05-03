from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


def parse_cli_date(value: str | None) -> Optional[datetime.date]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s10 = s[:10].replace("/", "-")
    if len(s10) == 10 and s10[4] == "-" and s10[7] == "-":
        return datetime.date.fromisoformat(s10)
    s8 = s.replace("-", "")[:8]
    if len(s8) == 8 and s8.isdigit():
        y, m, d = int(s8[0:4]), int(s8[4:6]), int(s8[6:8])
        return datetime.date(y, m, d)
    raise ValueError(f"invalid date: {value!r} (expected YYYY-MM-DD)")


def date_to_str(d: datetime.date | None) -> str:
    return d.isoformat() if d is not None else "NA"


def get_db_date_bounds(df: pd.DataFrame) -> Tuple[Optional[datetime.date], Optional[datetime.date]]:
    if df is None or df.empty or "date" not in df.columns:
        return None, None
    dt = pd.to_datetime(df["date"], errors="coerce")
    if dt.notna().sum() == 0:
        return None, None
    mn = dt.min()
    mx = dt.max()
    if pd.isna(mn) or pd.isna(mx):
        return None, None
    return mn.date(), mx.date()


@dataclass(frozen=True)
class EffectiveRange:
    requested_start: datetime.date
    requested_end: datetime.date
    effective_start: datetime.date
    effective_end: datetime.date


def compute_effective_range(
    *,
    user_start: datetime.date | None,
    user_end: datetime.date | None,
    inception_date: datetime.date | None,
    db_start: datetime.date | None,
    db_end: datetime.date | None,
) -> EffectiveRange:
    if db_start is None or db_end is None:
        raise ValueError("DB date bounds are unavailable")

    req_start = user_start or db_start
    req_end = user_end or db_end
    if req_end < req_start:
        raise ValueError("end_date must be >= start_date")

    eff_start = req_start
    if inception_date is not None and inception_date > eff_start:
        eff_start = inception_date
    if db_start > eff_start:
        eff_start = db_start

    eff_end = req_end
    if db_end < eff_end:
        eff_end = db_end
    if eff_end < eff_start:
        eff_end = eff_start

    return EffectiveRange(
        requested_start=req_start,
        requested_end=req_end,
        effective_start=eff_start,
        effective_end=eff_end,
    )


def filter_df_by_effective_range(df: pd.DataFrame, eff: EffectiveRange) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return df
    dt = pd.to_datetime(df["date"], errors="coerce")
    mask = dt.notna() & (dt.dt.date >= eff.effective_start) & (dt.dt.date <= eff.effective_end)
    return df.loc[mask].copy()


def compute_warmup_start_date(
    df: pd.DataFrame,
    *,
    eval_start: datetime.date,
    warmup_days: int,
) -> Optional[datetime.date]:
    w = int(warmup_days)
    if w <= 0 or df is None or df.empty or "date" not in df.columns:
        return None
    dt = pd.to_datetime(df["date"], errors="coerce")
    ok = dt.notna()
    if ok.sum() == 0:
        return None
    idx = df.index[ok].to_list()
    dt_ok = dt.loc[ok].reset_index(drop=True)
    i0 = int((dt_ok.dt.date >= eval_start).idxmax()) if (dt_ok.dt.date >= eval_start).any() else len(dt_ok) - 1
    i_w = max(0, i0 - w)
    return dt_ok.iloc[i_w].date()


def build_calc_df(
    df: pd.DataFrame,
    *,
    eff: EffectiveRange,
    warmup_days: int,
) -> tuple[pd.DataFrame, Optional[datetime.date]]:
    df_all = df.sort_values("date") if df is not None and "date" in df.columns else df
    warmup_start = compute_warmup_start_date(df_all, eval_start=eff.effective_start, warmup_days=warmup_days)
    if warmup_start is None:
        return filter_df_by_effective_range(df_all, eff), None
    dt = pd.to_datetime(df_all["date"], errors="coerce")
    mask = dt.notna() & (dt.dt.date >= warmup_start) & (dt.dt.date <= eff.effective_end)
    return df_all.loc[mask].copy(), warmup_start
