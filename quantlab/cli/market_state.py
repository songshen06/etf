import argparse
import sys
from typing import Any

import pandas as pd

from core.data_loader import etf_name_map, load_etf_sqlite
from core.indicators import add_analyzer_indicators
from core.paths import resolve_db_path
from quantlab.cli.date_range import build_calc_df, compute_effective_range, filter_df_by_effective_range, get_db_date_bounds, parse_cli_date
from quantlab.config.etf_metadata import get_inception_date
from quantlab.cli.market_regime import rolling_bucket_rank_series, detect_market_state


def cmd_market_state(ns: argparse.Namespace) -> int:
    db = resolve_db_path(ns.db_path)
    code = str(ns.etf)
    name = str(etf_name_map(db).get(code, code))

    df, _ = load_etf_sqlite(db, code)
    df = df.sort_values("date") if "date" in df.columns else df
    if df.empty:
        print("no data", file=sys.stderr)
        return 1
    db_start, db_end = get_db_date_bounds(df)
    if db_start is None or db_end is None:
        print("date bounds unavailable", file=sys.stderr)
        return 1
    eff = compute_effective_range(
        user_start=parse_cli_date(getattr(ns, "start_date", None)),
        user_end=parse_cli_date(getattr(ns, "end_date", None)),
        inception_date=get_inception_date(code),
        db_start=db_start,
        db_end=db_end,
    )
    df_calc, warmup_start = build_calc_df(df, eff=eff, warmup_days=int(getattr(ns, "warmup_days", 0) or 0))
    df_calc = add_analyzer_indicators(df_calc)
    df_calc = df_calc.sort_values("date") if "date" in df_calc.columns else df_calc
    df_eval = filter_df_by_effective_range(df_calc, eff)
    if df_eval.empty:
        print("no data in effective range", file=sys.stderr)
        return 1

    mom_col = "momentum_10" if "momentum_10" in df_calc.columns else "momentum"
    if mom_col not in df_calc.columns:
        print("missing momentum", file=sys.stderr)
        return 1
    df_calc = df_calc.copy()
    df_calc["momentum_q"] = rolling_bucket_rank_series(df_calc[mom_col], window=int(ns.rolling_window), n_buckets=5)
    ms = detect_market_state(df_calc[["momentum_q"]], window=int(ns.window), min_persist_days=int(ns.persist))
    last_idx = df_eval.index[-1]
    last = ms.loc[last_idx]
    date = str(pd.to_datetime(df_eval.iloc[-1]["date"]).date())
    st = str(last.get("stable_state") or "")
    rh = last.get("ratio_high")
    rl = last.get("ratio_low")

    print(f"[{code} {name}]")
    rs = getattr(ns, "start_date", None) or "数据库最早"
    re = getattr(ns, "end_date", None) or "数据库最新"
    print(f"requested_range: {rs} ~ {re}")
    print(f"effective_range: {eff.effective_start.isoformat()} ~ {eff.effective_end.isoformat()}")
    w = int(getattr(ns, "warmup_days", 0) or 0)
    if w > 0:
        cs = warmup_start.isoformat() if warmup_start is not None else eff.effective_start.isoformat()
        print(f"calc_range: {cs} ~ {eff.effective_end.isoformat()} (warmup_days={w})")
    print(f"date: {date}")
    print(f"market_state: {st}")
    if len(df_calc) < int(ns.window) + int(ns.persist):
        print(f"warning: 计算样本较短({len(df_calc)}行)，regime 可能不稳定", file=sys.stderr)
    if rh is not None and pd.notna(rh):
        print(f"ratio_high (mom_q>=4, {ns.window}d): {float(rh) * 100:.1f}%")
    else:
        print(f"ratio_high (mom_q>=4, {ns.window}d): NA")
    if rl is not None and pd.notna(rl):
        print(f"ratio_low  (mom_q<=2, {ns.window}d): {float(rl) * 100:.1f}%")
    else:
        print(f"ratio_low  (mom_q<=2, {ns.window}d): NA")
    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("market-state", help="Detect market regime from momentum_q (TREND/DOWN/RANGE)")
    p.add_argument("--etf", required=True, dest="etf", help="ETF code")
    p.add_argument("--db", required=True, dest="db_path", help="SQLite path")
    p.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD), inclusive")
    p.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD), inclusive")
    p.add_argument("--warmup-days", type=int, default=0, help="Warmup trading days before effective start for indicator stability")
    p.add_argument("--rolling-window", type=int, default=252, help="Rolling window for momentum_q buckets")
    p.add_argument("--window", type=int, default=20, help="Market-state window (days)")
    p.add_argument("--persist", type=int, default=20, help="Min days a state must persist before switching")
    p.set_defaults(_run=cmd_market_state)
