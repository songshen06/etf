"""Load ETF daily rows from SQLite into a validated, sorted panel."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .data_validation import DataValidationResult, validate_ohlcv_panel, write_data_quality_report


def load_etf_sqlite(
    db_path: str | Path,
    etf_code: str,
    *,
    table: str = "etf_daily_metrics",
    code_col: str = "etf_code",
    date_col: str = "trade_date",
    report_path: Path | None = None,
) -> tuple[pd.DataFrame, DataValidationResult]:
    """
    Columns: date, open, close, volume.

    Open is built from prev_close when present; invalid open is fixed after validation.
    """
    path = Path(db_path).expanduser().resolve()
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        table_cols = {row[1] for row in cur.fetchall()}
        bias_sql = ", bias_rate" if "bias_rate" in table_cols else ""
        q = f"""
        SELECT {date_col} AS date, price AS close, volume, prev_close{bias_sql}
        FROM {table}
        WHERE {code_col} = ?
        ORDER BY {date_col}
        """
        raw = pd.read_sql_query(q, conn, params=(str(etf_code),))
    finally:
        conn.close()

    if raw.empty:
        raise ValueError(f"No rows for {code_col}={etf_code!r} in {path}")

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw["date"])
    df["close"] = pd.to_numeric(raw["close"], errors="coerce")
    pc = pd.to_numeric(raw["prev_close"], errors="coerce")
    df["open"] = pc
    df["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    if "bias_rate" in raw.columns:
        df["bias_rate"] = pd.to_numeric(raw["bias_rate"], errors="coerce")

    cleaned, vres = validate_ohlcv_panel(
        df, date_col="date", close_col="close", open_col="open", volume_col="volume"
    )
    if report_path is not None:
        write_data_quality_report(vres, Path(report_path), title=f"Data quality — {etf_code}")
    return cleaned, vres




def list_etf_options(
    db_path: str | Path,
    *,
    table: str = "etf_daily_metrics",
    code_col: str = "etf_code",
    date_col: str = "trade_date",
) -> list[tuple[str, str]]:
    """Dynamic ETF options from DB: (etf_code, etf_name).

    - Source of truth: current SQLite table rows
    - Drop empty codes
    - For duplicated codes with multiple names, prefer latest non-empty name
    - Fallback name = code
    """
    path = Path(db_path).expanduser().resolve()
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        col_names = {row[1] for row in cur.fetchall()}
        if code_col not in col_names:
            return []

        if "etf_name" in col_names and date_col in col_names:
            q = f"""
            SELECT {code_col} AS etf_code, etf_name, {date_col} AS _d
            FROM {table}
            ORDER BY {date_col} DESC
            """
            df = pd.read_sql_query(q, conn)
            if df.empty:
                return []
            df["etf_code"] = df["etf_code"].fillna("").astype(str).str.strip()
            df["etf_name"] = df["etf_name"].fillna("").astype(str).str.strip()
            df = df[df["etf_code"].ne("")]
            if df.empty:
                return []
            df = df.drop_duplicates(subset=["etf_code"], keep="first")
            name_series = df["etf_name"].where(df["etf_name"].ne(""), df["etf_code"])
            pairs = list(zip(df["etf_code"].tolist(), name_series.tolist()))
            return sorted(((c, (n or c)) for c, n in pairs), key=lambda x: x[0])

        q = f"SELECT DISTINCT {code_col} AS etf_code FROM {table} ORDER BY {code_col}"
        s = pd.read_sql_query(q, conn)["etf_code"].fillna("").astype(str).str.strip()
        codes = [c for c in s.tolist() if c]
        return [(c, c) for c in codes]
    finally:
        conn.close()


def etf_universe_from_db(db_path: str | Path) -> list[dict[str, str]]:
    """用于 discover-entry-map：从当前库表生成 ``[{"code","name"}, ...]``，与 sidebar 选项同源。"""
    opts = list_etf_options(db_path)
    return [{"code": c, "name": n} for c, n in opts]


def list_etf_codes(db_path: str | Path, *, table: str = "etf_daily_metrics", code_col: str = "etf_code") -> list[str]:
    path = Path(db_path).expanduser().resolve()
    conn = sqlite3.connect(str(path))
    try:
        q = f"SELECT DISTINCT {code_col} FROM {table} ORDER BY {code_col}"
        s = pd.read_sql_query(q, conn)[code_col].astype(str).tolist()
    finally:
        conn.close()
    return s


def etf_name_map(
    db_path: str | Path,
    *,
    table: str = "etf_daily_metrics",
    code_col: str = "etf_code",
    date_col: str = "trade_date",
) -> dict[str, str]:
    """
    Map etf_code -> latest non-empty etf_name from DB.

    If `etf_name` column is missing, returns {code: code} for each distinct code.
    """
    path = Path(db_path).expanduser().resolve()
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        col_names = {row[1] for row in cur.fetchall()}
        codes = list_etf_codes(path, table=table, code_col=code_col)
        if "etf_name" not in col_names:
            return {c: c for c in codes}
        q = f"""
        SELECT {code_col} AS etf_code, etf_name, {date_col} AS _d
        FROM {table}
        ORDER BY {date_col} DESC
        """
        df = pd.read_sql_query(q, conn)
        if df.empty:
            return {c: c for c in codes}
        df["etf_code"] = df["etf_code"].astype(str)
        df["etf_name"] = df["etf_name"].fillna("").astype(str).str.strip()
        df = df.drop_duplicates(subset=["etf_code"], keep="first")
        by_row = df["etf_name"].where(df["etf_name"].ne(""), df["etf_code"])
        m = dict(zip(df["etf_code"], by_row))
        for c in codes:
            m.setdefault(c, c)
        return m
    finally:
        conn.close()
