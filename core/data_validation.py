"""OHLCV validation, open fallback, and issue reporting for the data health UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    count: int | None = None


@dataclass
class DataValidationResult:
    issues: list[ValidationIssue] = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0
    open_fallback_rows: int = 0

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]


def collect_invalid_rows(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    close_col: str = "close",
    open_col: str = "open",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Rows flagged before cleaning: bad close, duplicate dates, bad open/volume.
    Used for Data Health display; does not apply fixes.
    """
    if date_col not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    c = pd.to_numeric(d[close_col], errors="coerce") if close_col in d.columns else pd.Series(np.nan, index=d.index)
    bad_close = c.isna() | (c <= 0) | ~np.isfinite(c)
    dup = d[date_col].duplicated(keep=False)
    o = pd.to_numeric(d[open_col], errors="coerce") if open_col in d.columns else pd.Series(np.nan, index=d.index)
    bad_open = o.isna() | (o <= 0) | ~np.isfinite(o)
    v = pd.to_numeric(d[volume_col], errors="coerce") if volume_col in d.columns else pd.Series(np.nan, index=d.index)
    bad_vol = v.isna() | (v < 0) | ~np.isfinite(v)

    reasons: list[str] = []
    for i in range(len(d)):
        parts: list[str] = []
        if bool(bad_close.iloc[i]):
            parts.append("bad_close")
        if bool(dup.iloc[i]):
            parts.append("duplicate_date")
        if open_col in d.columns and bool(bad_open.iloc[i]):
            parts.append("bad_open")
        if bool(bad_vol.iloc[i]):
            parts.append("bad_volume")
        reasons.append(";".join(parts))

    d["_invalid_reason"] = reasons
    flagged = d["_invalid_reason"].str.len() > 0
    return d.loc[flagged].copy()


def validate_ohlcv_panel(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    close_col: str = "close",
    open_col: str | None = "open",
    volume_col: str = "volume",
    fix_open_from_close: bool = True,
) -> tuple[pd.DataFrame, DataValidationResult]:
    res = DataValidationResult()
    res.rows_in = len(df)
    out = df.copy()
    if date_col not in out.columns:
        res.add(ValidationIssue("error", "missing_date", f"Column {date_col!r} missing"))
        return out, res

    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col)

    dup = out[date_col].duplicated(keep=False)
    if dup.any():
        n = int(dup.sum())
        res.add(ValidationIssue("warning", "duplicate_dates", "Duplicate dates collapsed (keep last)", n))
        out = out.drop_duplicates(subset=[date_col], keep="last")

    c = pd.to_numeric(out[close_col], errors="coerce")
    bad_close = c.isna() | (c <= 0) | ~np.isfinite(c)
    if bad_close.any():
        n = int(bad_close.sum())
        res.add(ValidationIssue("error", "bad_close", "Rows removed: close NaN/<=0/non-finite", n))
        out = out.loc[~bad_close].copy()
        c = pd.to_numeric(out[close_col], errors="coerce")

    out[close_col] = c

    if open_col and open_col in out.columns:
        o = pd.to_numeric(out[open_col], errors="coerce")
        bad_open = o.isna() | (o <= 0) | ~np.isfinite(o)
        n_bad = int(bad_open.sum())
        if n_bad and fix_open_from_close:
            res.open_fallback_rows = n_bad
            res.add(
                ValidationIssue(
                    "warning",
                    "open_fallback_close",
                    "Invalid open replaced with close",
                    n_bad,
                )
            )
            o = o.where(~bad_open, c)
        out[open_col] = o
    elif open_col:
        out[open_col] = out[close_col]

    if volume_col in out.columns:
        v = pd.to_numeric(out[volume_col], errors="coerce")
        bad_vol = v.isna() | (v < 0) | ~np.isfinite(v)
        if bad_vol.any():
            n = int(bad_vol.sum())
            res.add(ValidationIssue("warning", "bad_volume", "Volume NaN/negative/non-finite set NaN", n))
            v = v.where(~bad_vol, np.nan)
        med = v.median()
        if med and np.isfinite(med):
            huge = v > med * 100
            if huge.any():
                nh = int(huge.sum())
                res.add(ValidationIssue("warning", "volume_spike", "Volume > 100x median (rows flagged)", nh))
        out[volume_col] = v

    res.rows_out = len(out)
    return out.reset_index(drop=True), res


def write_data_quality_report(result: DataValidationResult, path: Path, *, title: str = "Data quality") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Summary",
        "",
        f"- Rows in: **{result.rows_in}**",
        f"- Rows out: **{result.rows_out}**",
        f"- Open fallback to close: **{result.open_fallback_rows}** rows",
        "",
        "## Issues",
        "",
    ]
    if not result.issues:
        lines.append("_No issues._")
    else:
        for iss in result.issues:
            cnt = f" (n={iss.count})" if iss.count is not None else ""
            lines.append(f"- **{iss.severity.upper()}** `[{iss.code}]` {iss.message}{cnt}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def issues_to_records(issues: list[ValidationIssue]) -> list[dict[str, Any]]:
    return [
        {"severity": i.severity, "code": i.code, "message": i.message, "count": i.count}
        for i in issues
    ]
