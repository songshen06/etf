from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date

from core.paths import resolve_db_path
from quantlab.strategy.dividend_valuation import build_dividend_valuation_report


def _parse_date(s: str) -> date:
    s = str(s).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return date.fromisoformat(s[:10])
    s8 = s.replace("-", "").replace("/", "")[:8]
    return date(int(s8[0:4]), int(s8[4:6]), int(s8[6:8]))


def cmd_dividend_valuation(ns: argparse.Namespace) -> int:
    db_path = resolve_db_path(ns.db_path)
    as_of = _parse_date(ns.as_of) if ns.as_of else None
    if as_of is None:
        print("quantlab: error: --as-of is required", file=sys.stderr)
        return 2

    try:
        with sqlite3.connect(db_path) as conn:
            rpt = build_dividend_valuation_report(
                conn,
                index_code=str(ns.index_code),
                as_of=as_of,
                indicator_name=str(ns.indicator_name),
            )
    except Exception as e:
        print(f"quantlab: error: {e}", file=sys.stderr)
        return 1

    if ns.format == "json":
        print(
            json.dumps(
                {
                    "date": rpt.as_of.isoformat(),
                    "index_code": rpt.index_code,
                    "index_name": rpt.index_name,
                    "dividend_yield": rpt.dividend_yield,
                    "cn_10y_yield": rpt.cn_10y_yield,
                    "dividend_spread": rpt.dividend_spread,
                    "valuation_state": rpt.valuation_state,
                    "interpretation": rpt.valuation_explanation,
                    "index_valuation_date": rpt.index_valuation_date.isoformat(),
                    "cn_10y_date": rpt.cn_10y_date.isoformat(),
                    "yield_source": rpt.yield_source,
                    "rate_source": rpt.rate_source,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print("Dividend Valuation Report")
    print(f"Date: {rpt.as_of.isoformat()}")
    idx_name = f" {rpt.index_name}" if rpt.index_name else ""
    print(f"Index: {rpt.index_code}{idx_name}")
    print(f"Dividend Yield: {rpt.dividend_yield:.2f}%")
    print(f"CN 10Y Yield: {rpt.cn_10y_yield:.2f}%")
    print(f"Dividend Spread: {rpt.dividend_spread:.2f}%")
    print(f"Valuation State: {rpt.valuation_state}")
    print("")
    print("Interpretation:")
    print(rpt.valuation_explanation)
    return 0


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "dividend-valuation",
        help="Dividend spread valuation: dividend_yield(index) - CN10Y",
    )
    p.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="SQLite path (default: ETF_DB_PATH env, else repo root etf_data.db, else db/etf_data.db)",
    )
    p.add_argument("--index-code", required=True, help="Index code, e.g. 000922 (中证红利)")
    p.add_argument("--as-of", required=True, help="Date, e.g. 2026-04-30")
    p.add_argument(
        "--indicator-name",
        default="CN10Y",
        help="Macro rate indicator name in macro_rate_daily (default: CN10Y)",
    )
    p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    p.set_defaults(_run=cmd_dividend_valuation)

