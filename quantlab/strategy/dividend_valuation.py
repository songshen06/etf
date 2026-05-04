from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class IndexValuationSnapshot:
    trade_date: dt.date
    index_code: str
    index_name: str | None
    pe1: float | None
    pe2: float | None
    dividend_yield_1: float | None
    dividend_yield_2: float | None
    source: str | None


@dataclass(frozen=True)
class MacroRateSnapshot:
    trade_date: dt.date
    indicator_name: str
    indicator_value: float
    source: str | None


@dataclass(frozen=True)
class DividendValuationReport:
    as_of: dt.date
    index_code: str
    index_name: str | None
    dividend_yield: float
    cn_10y_yield: float
    dividend_spread: float
    valuation_state: str
    valuation_explanation: str
    index_valuation_date: dt.date
    cn_10y_date: dt.date
    yield_source: str
    rate_source: str


@dataclass(frozen=True)
class DividendSignalDecision:
    etf_code: str
    etf_name: str | None
    as_of: dt.date
    index_code: str
    valuation_state: str
    dividend_spread: float
    dividend_yield: float
    cn_10y_yield: float
    bias_q: int | None
    momentum_q: int | None
    strategy_signal: str
    max_target_position: float
    target_position: float
    current_position: float
    total_position: float
    diff: float
    action: str
    suggested_step: float
    reason: str
    details: Dict[str, Any]


def ensure_dividend_valuation_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS index_valuation_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date DATE NOT NULL,
            index_code TEXT NOT NULL,
            index_name TEXT,
            pe1 REAL,
            pe2 REAL,
            dividend_yield_1 REAL,
            dividend_yield_2 REAL,
            source TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(trade_date, index_code)
        );

        CREATE TABLE IF NOT EXISTS macro_rate_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date DATE NOT NULL,
            indicator_name TEXT NOT NULL,
            indicator_value REAL NOT NULL,
            source TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(trade_date, indicator_name)
        );
        """
    )
    conn.commit()


def _parse_date(value: Any) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    s = str(value).strip()[:10].replace("/", "-")
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return dt.date.fromisoformat(s)
    s8 = str(value).replace("-", "").strip()[:8]
    return dt.date(int(s8[0:4]), int(s8[4:6]), int(s8[6:8]))


def _to_percent(x: float) -> float:
    v = float(x)
    if 0 < v < 0.5:
        return v * 100.0
    return v


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not (v == v):
        return None
    return v


def get_latest_index_valuation(
    conn: sqlite3.Connection, *, index_code: str, as_of: dt.date
) -> Tuple[IndexValuationSnapshot, str]:
    ensure_dividend_valuation_tables(conn)
    code = str(index_code)
    row = conn.execute(
        """
        SELECT trade_date, index_code, index_name, pe1, pe2, dividend_yield_1, dividend_yield_2, source
        FROM index_valuation_daily
        WHERE index_code = ? AND trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT 1
        """,
        (code, as_of.isoformat()),
    ).fetchone()

    if row is not None:
        snap = IndexValuationSnapshot(
            trade_date=_parse_date(row[0]),
            index_code=str(row[1]),
            index_name=str(row[2]) if row[2] is not None else None,
            pe1=_safe_float(row[3]),
            pe2=_safe_float(row[4]),
            dividend_yield_1=_safe_float(row[5]),
            dividend_yield_2=_safe_float(row[6]),
            source=str(row[7]) if row[7] is not None else None,
        )
        return snap, "index_valuation_daily"

    row2 = conn.execute(
        """
        SELECT trade_date, source_index_code, source_index_name, dividend_yield, pe_ttm, valuation_source
        FROM etf_valuation_daily
        WHERE source_index_code = ? AND trade_date <= ? AND dividend_yield IS NOT NULL
        ORDER BY trade_date DESC
        LIMIT 1
        """,
        (code, as_of.isoformat()),
    ).fetchone()
    if row2 is not None:
        snap = IndexValuationSnapshot(
            trade_date=_parse_date(row2[0]),
            index_code=str(row2[1]),
            index_name=str(row2[2]) if row2[2] is not None else None,
            pe1=None,
            pe2=_safe_float(row2[4]),
            dividend_yield_1=None,
            dividend_yield_2=_safe_float(row2[3]),
            source=str(row2[5]) if row2[5] is not None else None,
        )
        return snap, "etf_valuation_daily"

    raise ValueError(f"index valuation not found for index_code={code} as_of={as_of.isoformat()}")


def get_latest_macro_rate(
    conn: sqlite3.Connection, *, indicator_name: str, as_of: dt.date
) -> Tuple[MacroRateSnapshot, str]:
    ensure_dividend_valuation_tables(conn)
    name = str(indicator_name)
    row = conn.execute(
        """
        SELECT trade_date, indicator_name, indicator_value, source
        FROM macro_rate_daily
        WHERE indicator_name = ? AND trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT 1
        """,
        (name, as_of.isoformat()),
    ).fetchone()
    if row is None:
        raise ValueError(f"macro rate not found for indicator={name} as_of={as_of.isoformat()}")
    snap = MacroRateSnapshot(
        trade_date=_parse_date(row[0]),
        indicator_name=str(row[1]),
        indicator_value=float(row[2]),
        source=str(row[3]) if row[3] is not None else None,
    )
    return snap, "macro_rate_daily"


def compute_dividend_spread_state(spread: float) -> tuple[str, str]:
    s = float(spread)
    if s >= 3.0:
        return "VERY_CHEAP", "红利相对债券明显便宜，可提高目标仓位"
    if s >= 2.0:
        return "ATTRACTIVE", "红利具备配置价值，可结合 BIAS 低位分批建仓"
    if s >= 1.0:
        return "NEUTRAL", "红利估值中性，主要持有，不主动加仓"
    return "EXPENSIVE", "红利相对债券吸引力下降，应停止加仓或减仓"


def build_dividend_valuation_report(
    conn: sqlite3.Connection,
    *,
    index_code: str,
    as_of: dt.date,
    indicator_name: str = "CN10Y",
) -> DividendValuationReport:
    idx, idx_source = get_latest_index_valuation(conn, index_code=str(index_code), as_of=as_of)
    rate, rate_source = get_latest_macro_rate(conn, indicator_name=str(indicator_name), as_of=as_of)

    dy2 = _safe_float(idx.dividend_yield_2)
    dy1 = _safe_float(idx.dividend_yield_1)
    dividend_yield = dy2 if dy2 is not None else dy1
    if dividend_yield is None:
        raise ValueError(f"dividend_yield_1/dividend_yield_2 are both missing for index_code={index_code}")

    cn10y = _safe_float(rate.indicator_value)
    if cn10y is None:
        raise ValueError(f"cn_10y_yield missing for indicator={indicator_name}")

    dividend_yield = _to_percent(dividend_yield)
    cn10y = _to_percent(cn10y)

    spread = float(dividend_yield) - float(cn10y)
    state, expl = compute_dividend_spread_state(spread)

    return DividendValuationReport(
        as_of=as_of,
        index_code=str(index_code),
        index_name=idx.index_name,
        dividend_yield=float(dividend_yield),
        cn_10y_yield=float(cn10y),
        dividend_spread=float(spread),
        valuation_state=state,
        valuation_explanation=expl,
        index_valuation_date=idx.trade_date,
        cn_10y_date=rate.trade_date,
        yield_source=idx_source,
        rate_source=rate_source,
    )


def max_target_from_state(state: str) -> float:
    st = str(state).upper()
    if st == "VERY_CHEAP":
        return 0.70
    if st == "ATTRACTIVE":
        return 0.60
    if st == "NEUTRAL":
        return 0.40
    if st == "EXPENSIVE":
        return 0.20
    return 0.40


def compute_bias_momentum_buckets(
    df_with_indicators: pd.DataFrame, *, as_of: dt.date, rolling_window: int = 252
) -> tuple[int | None, int | None]:
    df = df_with_indicators.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[df["date"] <= as_of]
    if df.empty:
        return None, None
    if "bias_rate" not in df.columns:
        return None, None
    mom_col = "momentum_10" if "momentum_10" in df.columns else "momentum"
    if mom_col not in df.columns:
        return None, None

    from quantlab.filters.quantile_filter import assign_equal_frequency_quantile_labels, bucket_label_to_rank

    def _rank(series: pd.Series) -> int | None:
        s = pd.to_numeric(series, errors="coerce").tail(int(rolling_window))
        if len(s) == 0:
            return None
        labels = assign_equal_frequency_quantile_labels(s, int(5))
        lab = labels.iloc[-1] if len(labels) else None
        return bucket_label_to_rank(lab)

    return _rank(df["bias_rate"]), _rank(df[mom_col])


def strategy_signal_from_inputs(
    *,
    valuation_state: str,
    bias_q: int | None,
    momentum_q: int | None,
) -> str:
    vs = str(valuation_state).upper()
    if vs == "EXPENSIVE":
        return "HOLD_CASH_OR_TRIM"
    low_bias = bias_q in (1, 2)
    low_mom = momentum_q in (1, 2)
    mid_high_bias = bias_q in (3, 4, 5)
    mid_high_mom = momentum_q in (3, 4, 5)
    if low_bias and low_mom:
        return "BUILD_BASE"
    if low_bias and vs in ("VERY_CHEAP", "ATTRACTIVE"):
        return "ACCUMULATE"
    if mid_high_bias and mid_high_mom:
        return "HOLD_OR_TRIM"
    return "HOLD"


def decide_action(
    *,
    target_position: float,
    current_position: float,
    total_position: float,
    max_step: float = 0.10,
) -> tuple[str, float, float, str]:
    target = float(max(0.0, min(1.0, target_position)))
    current = float(max(0.0, min(1.0, current_position)))
    total = float(max(0.0, min(1.0, total_position)))

    diff = target - current
    if diff > 0 and total > 0.80:
        return "HOLD", 0.0, diff, "total_position>80%: block new buys; allow only HOLD/TRIM"
    if current >= target - 1e-12 and diff > 0:
        return "HOLD", 0.0, diff, "current_position already >= target_position"

    step = diff
    if step > max_step:
        step = max_step
    if step < -max_step:
        step = -max_step

    if diff > 0.10:
        return "BUY", float(step), diff, "diff>10% → BUY (cap 10% per step)"
    if diff > 0.03:
        return "SMALL_BUY", float(step), diff, "3%<diff<=10% → SMALL_BUY (cap 10% per step)"
    if diff >= -0.03:
        return "HOLD", 0.0, diff, "abs(diff)<=3% → HOLD"
    if diff >= -0.10:
        return "SMALL_TRIM", float(step), diff, "-10%<=diff<-3% → SMALL_TRIM (cap 10% per step)"
    return "TRIM", float(step), diff, "diff<-10% → TRIM (cap 10% per step)"


def run_dividend_signal(
    *,
    conn: sqlite3.Connection,
    etf_code: str,
    etf_name: str | None,
    index_code: str,
    as_of: dt.date,
    current_position: float,
    total_position: float,
    bias_q: int | None,
    momentum_q: int | None,
    indicator_name: str = "CN10Y",
) -> DividendSignalDecision:
    rpt = build_dividend_valuation_report(conn, index_code=index_code, as_of=as_of, indicator_name=indicator_name)
    max_target = max_target_from_state(rpt.valuation_state)
    sig = strategy_signal_from_inputs(
        valuation_state=rpt.valuation_state,
        bias_q=bias_q,
        momentum_q=momentum_q,
    )

    if sig in ("BUILD_BASE", "ACCUMULATE"):
        target = max_target
    else:
        target = min(float(current_position), max_target)

    action, step, diff, action_reason = decide_action(
        target_position=target,
        current_position=float(current_position),
        total_position=float(total_position),
        max_step=0.10,
    )

    reason = "; ".join(
        [
            f"valuation_state={rpt.valuation_state} (spread={rpt.dividend_spread:.2f}%)",
            f"bias_q={'NA' if bias_q is None else 'Q'+str(bias_q)}",
            f"momentum_q={'NA' if momentum_q is None else 'Q'+str(momentum_q)}",
            f"signal={sig}",
            action_reason,
        ]
    )

    return DividendSignalDecision(
        etf_code=str(etf_code),
        etf_name=etf_name,
        as_of=as_of,
        index_code=str(index_code),
        valuation_state=rpt.valuation_state,
        dividend_spread=float(rpt.dividend_spread),
        dividend_yield=float(rpt.dividend_yield),
        cn_10y_yield=float(rpt.cn_10y_yield),
        bias_q=bias_q,
        momentum_q=momentum_q,
        strategy_signal=sig,
        max_target_position=float(max_target),
        target_position=float(target),
        current_position=float(current_position),
        total_position=float(total_position),
        diff=float(diff),
        action=action,
        suggested_step=float(step),
        reason=reason,
        details={
            "index_name": rpt.index_name,
            "index_valuation_date": rpt.index_valuation_date.isoformat(),
            "cn_10y_date": rpt.cn_10y_date.isoformat(),
            "valuation_explanation": rpt.valuation_explanation,
            "yield_source": rpt.yield_source,
            "rate_source": rpt.rate_source,
        },
    )

