from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


@dataclass(frozen=True)
class IndexValuationRow:
    trade_date: str
    pe_ttm: float | None
    dividend_yield: float | None
    source_index_code: str | None = None
    source_index_name: str | None = None
    raw_payload: str | None = None


@dataclass(frozen=True)
class DividendIndexIndicatorRow:
    trade_date: str
    index_code: str
    index_name: str | None
    pe1: float | None
    pe2: float | None
    dividend_yield_1: float | None
    dividend_yield_2: float | None
    source: str | None = None


@dataclass(frozen=True)
class MacroRateRow:
    trade_date: str
    indicator_name: str
    indicator_value: float
    source: str | None = None


def _norm_col(s: Any) -> str:
    raw = "" if s is None else str(s)
    raw = raw.strip().lower()
    out: list[str] = []
    for ch in raw:
        o = ord(ch)
        if "a" <= ch <= "z" or "0" <= ch <= "9":
            out.append(ch)
            continue
        if 0x4E00 <= o <= 0x9FFF:
            out.append(ch)
            continue
    return "".join(out)


def _coerce_trade_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (pd.NaT.__class__,)):
        return None
    if hasattr(value, "date") and callable(getattr(value, "date")):
        try:
            return value.date().isoformat()
        except Exception:
            pass
    if isinstance(value, (int, float)):
        v = float(value)
        if v != v:
            return None
        if v >= 20000:
            try:
                ts = pd.to_datetime(v, unit="D", origin="1899-12-30")
                return ts.date().isoformat()
            except Exception:
                return None
        s8 = str(int(v))
        if len(s8) == 8:
            return f"{s8[0:4]}-{s8[4:6]}-{s8[6:8]}"
        return None
    s = str(value).strip()
    if not s:
        return None
    s10 = s[:10].replace("/", "-")
    if len(s10) == 10 and s10[4] == "-" and s10[7] == "-":
        return s10
    s8 = s.replace("-", "").replace("/", "")[:8]
    if len(s8) == 8 and s8.isdigit():
        return f"{s8[0:4]}-{s8[4:6]}-{s8[6:8]}"
    return None


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


def fetch_csindex_indicator_xls(index_code: str) -> pd.DataFrame:
    code = str(index_code).strip()
    url = (
        "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/indicator/"
        f"{code}indicator.xls"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    import io

    def _empty() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "index_code",
                "index_name",
                "pe1",
                "pe2",
                "dividend_yield_1",
                "dividend_yield_2",
                "source",
            ]
        )

    def _read_with_header_guess() -> pd.DataFrame:
        last_err: Exception | None = None
        for hdr in (0, 1, 2, 3):
            try:
                df0 = pd.read_excel(io.BytesIO(resp.content), engine="xlrd", header=hdr)
            except Exception as e:
                last_err = e
                continue
            if df0 is None or getattr(df0, "empty", True):
                continue
            cols0 = [str(c).strip() for c in df0.columns.tolist()]
            if any(_norm_col(c) in ("日期", "交易日期", "trade_date", "date") for c in cols0):
                df0.columns = cols0
                return df0
            if sum(1 for c in cols0 if _norm_col(c).startswith("unnamed")) >= max(1, int(len(cols0) * 0.6)):
                try:
                    df_raw = pd.read_excel(io.BytesIO(resp.content), engine="xlrd", header=None)
                except Exception as e:
                    last_err = e
                    continue
                if df_raw is None or getattr(df_raw, "empty", True):
                    continue
                header_row_idx: int | None = None
                for i in range(min(15, len(df_raw))):
                    row = df_raw.iloc[i].astype(str).tolist()
                    if any("日期" in x or "交易日期" in x or "Date" in x for x in row):
                        header_row_idx = i
                        break
                if header_row_idx is None:
                    continue
                header = [str(x).strip() for x in df_raw.iloc[header_row_idx].tolist()]
                data = df_raw.iloc[header_row_idx + 1 :].copy()
                data.columns = header
                return data
        if last_err is not None:
            raise RuntimeError(f"read_excel failed (need xlrd for .xls): {last_err}") from last_err
        return _empty()

    df = _read_with_header_guess()
    if df is None or getattr(df, "empty", True):
        return _empty()

    cols = [str(c).strip() for c in df.columns.tolist()]
    df.columns = cols
    norm_to_raw = {_norm_col(c): c for c in cols if c}

    def pick(norm_candidates: list[str]) -> str | None:
        for k in norm_candidates:
            kk = _norm_col(k)
            if kk in norm_to_raw:
                return norm_to_raw[kk]
        for kk, raw_name in norm_to_raw.items():
            for cand in norm_candidates:
                c2 = _norm_col(cand)
                if c2 and c2 in kk:
                    return raw_name
        return None

    date_col = pick(["trade_date", "交易日期", "日期", "date", "日期date"])
    pe1_col = pick(["pe1", "市盈率1", "pettm1", "市盈率ttm1", "市盈率（1）", "市盈率(1)"])
    pe2_col = pick(["pe2", "市盈率2", "pettm", "pettm2", "市盈率ttm", "市盈率（2）", "市盈率(2)", "市盈率"])
    dy1_col = pick(["dividend_yield_1", "dividendyield1", "股息率1", "股息率（1）", "股息率(1)"])
    dy2_col = pick(["dividend_yield_2", "dividendyield2", "股息率2", "股息率（2）", "股息率(2)", "股息率"])
    name_col = pick(["index_name", "指数名称", "指数中文全称", "指数简称", "指数全称", "indexfullname"])

    if date_col is None:
        raise KeyError(f"missing date column in csindex indicator xls: {cols}")

    out_rows: list[DividendIndexIndicatorRow] = []
    for _, r in df.iterrows():
        d = _coerce_trade_date(r.get(date_col))
        if not d:
            continue
        pe1 = _pick_first_numeric(r, [pe1_col] if pe1_col else [])
        pe2 = _pick_first_numeric(r, [pe2_col] if pe2_col else [])
        dy1 = _pick_first_numeric(r, [dy1_col] if dy1_col else [])
        dy2 = _pick_first_numeric(r, [dy2_col] if dy2_col else [])
        idx_name = str(r.get(name_col)).strip() if name_col and r.get(name_col) is not None else None
        out_rows.append(
            DividendIndexIndicatorRow(
                trade_date=d,
                index_code=code,
                index_name=idx_name,
                pe1=pe1,
                pe2=pe2,
                dividend_yield_1=dy1,
                dividend_yield_2=dy2,
                source="csindex_indicator_xls",
            )
        )

    out = pd.DataFrame([x.__dict__ for x in out_rows])
    if not out.empty:
        out = out.dropna(subset=["trade_date"])
    return out


def fetch_cn10y_eastmoney_kline(*, secid: str = "171.CN10Y") -> pd.DataFrame:
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": str(secid),
        "klt": "101",
        "fqt": "1",
        "beg": "0",
        "end": "20500101",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    obj = r.json()
    data = (obj or {}).get("data") or {}
    klines = data.get("klines") or []
    rows: list[MacroRateRow] = []
    for line in klines:
        parts = str(line).split(",")
        if len(parts) < 3:
            continue
        trade_date = parts[0].strip()
        close = pd.to_numeric(pd.Series([parts[2]]), errors="coerce").iloc[0]
        if pd.isna(close):
            continue
        rows.append(
            MacroRateRow(
                trade_date=trade_date,
                indicator_name="CN10Y",
                indicator_value=float(close),
                source="eastmoney_push2his",
            )
        )
    return pd.DataFrame([x.__dict__ for x in rows])


def fetch_dividend_index_indicators_multi_source(*, index_code: str) -> tuple[pd.DataFrame, str, str | None]:
    code = str(index_code).strip()
    candidates: list[tuple[str, Any]] = [
        ("csindex_indicator_xls", fetch_csindex_indicator_xls),
        ("akshare_csindex_standard", lambda ic: _adapt_akshare_to_indicator(fetch_csindex_index_valuation(ic), ic)),
    ]
    for tag, fn in candidates:
        try:
            df = fn(code)
            if df is not None and not getattr(df, "empty", True):
                return df, tag, None
        except Exception:
            continue
    return (
        pd.DataFrame(
            columns=[
                "trade_date",
                "index_code",
                "index_name",
                "pe1",
                "pe2",
                "dividend_yield_1",
                "dividend_yield_2",
                "source",
            ]
        ),
        "none",
        "no available indicator sources",
    )


def _adapt_akshare_to_indicator(df: pd.DataFrame, index_code: str) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(
            columns=[
                "trade_date",
                "index_code",
                "index_name",
                "pe1",
                "pe2",
                "dividend_yield_1",
                "dividend_yield_2",
                "source",
            ]
        )
    out = df.copy()
    out["index_code"] = str(index_code)
    out = out.rename(columns={"source_index_name": "index_name"})
    out["pe1"] = None
    out["pe2"] = out.get("pe_ttm")
    out["dividend_yield_1"] = None
    out["dividend_yield_2"] = out.get("dividend_yield")
    out["source"] = "akshare:index_value_csindex"
    keep = ["trade_date", "index_code", "index_name", "pe1", "pe2", "dividend_yield_1", "dividend_yield_2", "source"]
    return out[keep]
