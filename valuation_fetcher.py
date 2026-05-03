from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class IndexValuationRow:
    trade_date: str
    pe_ttm: float | None
    dividend_yield: float | None
    source_index_code: str | None = None
    source_index_name: str | None = None
    raw_payload: str | None = None


def _pick_first_numeric(row: pd.Series, candidates: list[str]) -> float | None:
    for k in candidates:
        if k not in row:
            continue
        v = pd.to_numeric(pd.Series([row.get(k)]), errors="coerce").iloc[0]
        if pd.notna(v):
            return float(v)
    return None


def _row_to_raw_payload(row: pd.Series) -> str:
    payload: dict[str, Any] = {}
    for k, v in row.to_dict().items():
        if v is None or pd.isna(v):
            payload[str(k)] = None
            continue
        if isinstance(v, (str, int, float, bool)):
            payload[str(k)] = v
            continue
        payload[str(k)] = str(v)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def fetch_csindex_index_valuation(index_code: str) -> pd.DataFrame:
    try:
        import akshare as ak
    except Exception as e:
        raise RuntimeError(f"akshare not available: {e}") from e

    df = ak.stock_zh_index_value_csindex(symbol=str(index_code))
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(
            columns=[
                "trade_date",
                "pe_ttm",
                "dividend_yield",
                "source_index_code",
                "source_index_name",
                "raw_payload",
            ]
        )

    cols = df.columns.tolist()
    date_col = "日期" if "日期" in cols else None
    if date_col is None:
        raise KeyError(f"missing date column in csindex valuation df: {cols}")

    out_rows: list[IndexValuationRow] = []
    for _, r in df.iterrows():
        dt = str(r.get(date_col) or "").strip()
        if not dt:
            continue

        pe = _pick_first_numeric(r, ["市盈率1", "市盈率", "PE_TTM", "PE"])
        div_y = _pick_first_numeric(r, ["股息率1", "股息率", "DIVIDEND_YIELD"])
        raw = _row_to_raw_payload(r)

        out_rows.append(
            IndexValuationRow(
                trade_date=dt,
                pe_ttm=pe,
                dividend_yield=div_y,
                source_index_code=str(r.get("指数代码")) if "指数代码" in cols else str(index_code),
                source_index_name=str(r.get("指数中文全称")) if "指数中文全称" in cols else None,
                raw_payload=raw,
            )
        )

    out = pd.DataFrame([x.__dict__ for x in out_rows])
    if not out.empty:
        out = out.dropna(subset=["trade_date"])
    return out


def fetch_index_valuation_multi_source(*, index_code: str, mode: str) -> tuple[pd.DataFrame, str, str | None]:
    m = str(mode or "standard").strip().lower()
    if m not in ("standard", "deep"):
        m = "standard"

    def _not_implemented(_: str) -> pd.DataFrame:
        raise NotImplementedError("deep valuation secondary source not implemented yet")

    if m == "standard":
        return fetch_csindex_index_valuation(index_code), "akshare_csindex_standard", None

    candidates: list[tuple[str, Any]] = [
        ("csindex_history_todo", _not_implemented),
        ("akshare_csindex_standard", fetch_csindex_index_valuation),
    ]
    fallback_note: str | None = None
    for tag, fn in candidates:
        try:
            df = fn(index_code)
            if tag == "akshare_csindex_standard":
                fallback_note = "deep mode fallback to standard source"
            return df, str(tag), fallback_note
        except NotImplementedError:
            continue
    return (
        pd.DataFrame(columns=["trade_date", "pe_ttm", "dividend_yield", "source_index_code", "source_index_name", "raw_payload"]),
        "none",
        "deep mode has no available sources",
    )
