import argparse
import sys
from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.data_loader import etf_name_map, load_etf_sqlite
from core.indicators import add_analyzer_indicators
from core.paths import resolve_db_path
from quantlab.cli.date_range import build_calc_df, compute_effective_range, filter_df_by_effective_range, get_db_date_bounds, parse_cli_date
from quantlab.config.etf_metadata import get_inception_date
from quantlab.cli.market_regime import rolling_bucket_rank_series, detect_market_state
from quantlab.cli.recommend_strategy import (
    MIN_EXPOSURE,
    classify_etf,
    dividend_strategy,
    dividend_growth_strategy,
    mean_reversion_strategy,
    trend_following_strategy,
    defensive_strategy,
)


def _as_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not (v == v):
        return None
    return v


def _date_str(x: Any) -> str:
    try:
        return pd.to_datetime(x).date().isoformat()
    except Exception:
        return str(x)


def _action_kind(position_before: float, position_after: float) -> str | None:
    b = float(position_before)
    a = float(position_after)
    if abs(a - b) < 1e-12:
        return None
    if b <= 0 and a > 0:
        return "BUY"
    if a > b:
        return "ADD"
    if a <= 0 and b > 0:
        return "EXIT"
    return "REDUCE"


@dataclass
class TradeLogRow:
    date: str
    action: str
    position_before: float
    position_after: float
    reason: str
    market_state: str | None = None
    strategy: str | None = None
    price: float | None = None
    pnl_since_entry: float | None = None

    def format_line(self) -> str:
        pb = f"{self.position_before * 100:.1f}%"
        pa = f"{self.position_after * 100:.1f}%"
        parts = [f"{self.date} | {self.action:<6} | {pb:<4} → {pa:<4} | {self.reason}"]
        if self.market_state is not None:
            parts.append(f"market_state/市场状态={self.market_state}")
        if self.strategy is not None:
            parts.append(f"strategy/策略={self.strategy}")
        if self.price is not None:
            parts.append(f"price/价格={self.price:.3f}")
        if self.pnl_since_entry is not None:
            parts.append(f"pnl/盈亏={self.pnl_since_entry * 100:+.1f}%")
        return " | ".join(parts)


def route_decision(
    *,
    code: str,
    category: str,
    market_state: str | None,
    bias_q: int | None,
    momentum_q: int | None,
    current_position: float,
) -> tuple[str, dict[str, Any]]:
    if category == "DIVIDEND":
        if str(code) == "159209":
            return "dividend_growth_strategy", dividend_growth_strategy(
                bias_q=bias_q, momentum_q=momentum_q, current_position=current_position
            )
        return "dividend_strategy", dividend_strategy(bias_q=bias_q, momentum_q=momentum_q, current_position=current_position)
    if category == "CORE_INDEX":
        st = (market_state or "RANGE").upper()
        if st == "TREND":
            return "trend_following_strategy", trend_following_strategy(
                category="CORE_INDEX_OVERLAY", bias_q=bias_q, momentum_q=momentum_q, current_position=current_position
            )
        if st == "DOWN":
            return "defensive_strategy", defensive_strategy(category="CORE_INDEX_OVERLAY", current_position=current_position)
        return "mean_reversion_strategy", mean_reversion_strategy(
            category="CORE_INDEX_OVERLAY", bias_q=bias_q, momentum_q=momentum_q, current_position=current_position
        )
    raise ValueError(f"debug-trades supports DIVIDEND/CORE_INDEX only (got {category})")


def replay_trades(
    *,
    df: pd.DataFrame,
    code: str,
    category: str,
    rolling_window: int = 252,
    regime_window: int = 20,
    regime_persist: int = 20,
    core_exposure_floor: float = 0.4,
    tactical_overlay_max: float = 0.4,
    eval_start: str | None = None,
    eval_end: str | None = None,
    limit: int | None = None,
) -> list[TradeLogRow]:
    d = df.sort_values("date") if "date" in df.columns else df.copy()
    d = d.reset_index(drop=True)
    if d.empty:
        return []

    if "bias_rate" not in d.columns:
        raise ValueError("missing bias_rate")
    mom_col = "momentum_10" if "momentum_10" in d.columns else "momentum"
    if mom_col not in d.columns:
        raise ValueError("missing momentum")
    if "close" not in d.columns and "close_norm" not in d.columns and "price" not in d.columns:
        raise ValueError("missing price/close")

    bias_q_s = rolling_bucket_rank_series(d["bias_rate"], window=int(rolling_window), n_buckets=5)
    mom_q_s = rolling_bucket_rank_series(d[mom_col], window=int(rolling_window), n_buckets=5)
    bias_q_i = bias_q_s.round().astype("Int64")
    mom_q_i = mom_q_s.round().astype("Int64")
    ms = detect_market_state(
        pd.DataFrame({"momentum_q": mom_q_s}),
        window=int(regime_window),
        min_persist_days=int(regime_persist),
    )
    stable_state_s = ms["stable_state"]

    def _to_strategy_units(current_overlay: float, overlay_max: float, assumed_pos_max: float) -> float:
        om = float(overlay_max)
        pm = float(assumed_pos_max)
        if om <= 0 or pm <= 0:
            return float(current_overlay)
        return float(current_overlay) / om * pm

    def _from_strategy_units(pos_after: float, pos_max: float, overlay_max: float) -> float:
        om = float(overlay_max)
        pm = float(pos_max)
        if om <= 0 or pm <= 0:
            return 0.0
        return float(pos_after) / pm * om

    is_core = category == "CORE_INDEX"
    is_div_growth = category == "DIVIDEND" and str(code) == "159209"

    base_target = float(core_exposure_floor) if is_core else 0.0
    overlay_max = float(tactical_overlay_max) if is_core else 0.0
    if is_div_growth:
        base_target = 0.60
        if float(core_exposure_floor) >= 0.5:
            base_target = float(core_exposure_floor)
        overlay_default = max(0.0, 1.0 - base_target)
        overlay_max = overlay_default
        if float(tactical_overlay_max) > 0 and (base_target + float(tactical_overlay_max) <= 1.0 + 1e-9):
            overlay_max = float(tactical_overlay_max)
        base_floor = min(base_target, 0.60)

    base_active = False
    current_base = 0.0
    current_overlay = 0.0
    last_trade_i = -10_000
    entry_price: float | None = None
    logs: list[TradeLogRow] = []

    eval_start_d = parse_cli_date(eval_start) if eval_start is not None else None
    eval_end_d = parse_cli_date(eval_end) if eval_end is not None else None
    if eval_start_d is None or eval_end_d is None:
        db_start, db_end = get_db_date_bounds(d)
        if db_start is None or db_end is None:
            return []
        eval_start_d = db_start
        eval_end_d = db_end

    dt_all = pd.to_datetime(d["date"], errors="coerce")
    eval_mask = (dt_all.notna() & (dt_all.dt.date >= eval_start_d) & (dt_all.dt.date <= eval_end_d)).to_numpy(dtype=bool)
    if int(eval_mask.sum()) == 0:
        return []
    start_i = int((eval_mask).argmax())
    end_i = int((eval_mask).nonzero()[0].max())

    for i in range(start_i, end_i + 1):
        row = d.iloc[i]
        date = _date_str(row.get("date", i))
        price = _as_float(row.get("close", row.get("close_norm", row.get("price"))))
        bq = int(bias_q_i.iloc[i]) if pd.notna(bias_q_i.iloc[i]) else None
        mq = int(mom_q_i.iloc[i]) if pd.notna(mom_q_i.iloc[i]) else None

        market_state = None
        if is_core:
            market_state = str(stable_state_s.iloc[i]) if pd.notna(stable_state_s.iloc[i]) else "RANGE"
        if is_core:
            valid = bq is not None and mq is not None and (market_state is not None)
            if (not base_active) and valid and base_target > 0:
                base_active = True
                pb = current_base
                current_base = base_target
                logs.append(
                    TradeLogRow(
                        date=date,
                        action="BASE_BUY",
                        position_before=float(pb),
                        position_after=float(current_base),
                        reason="base activated after valid quantile/regime data",
                        market_state=market_state,
                        strategy="base_core_exposure",
                        price=float(price) if price is not None else None,
                        pnl_since_entry=None,
                    )
                )
            if not valid:
                continue

            cur = _to_strategy_units(current_overlay, overlay_max, 0.8)
            strategy_used, dec = route_decision(
                code=code,
                category=category,
                market_state=market_state,
                bias_q=bq,
                momentum_q=mq,
                current_position=cur,
            )
            desired_overlay = _from_strategy_units(float(dec["position_after"]), float(dec["position_max"]), overlay_max)
            if dec.get("signal") == "HOLD" or dec.get("action") == "HOLD" or ("HOLD" in str(dec.get("reason") or "")):
                desired_overlay = current_overlay
            desired_pos = float(desired_overlay)
        elif is_div_growth:
            current_total = current_base + current_overlay
            strategy_used = "dividend_growth_strategy"
            dec = dividend_growth_strategy(
                bias_q=bq,
                momentum_q=mq,
                current_position=current_total,
                base_target=base_target,
                overlay_max=overlay_max,
            )
            desired_total = float(dec["position_after"])
            if dec.get("signal") == "HOLD" or dec.get("action") == "HOLD" or ("HOLD" in str(dec.get("reason") or "")):
                desired_total = current_total

            if (not base_active) and current_total <= 1e-12 and desired_total > 1e-12:
                base_active = True
                pb = current_base
                current_base = float(min(base_target, desired_total))
                logs.append(
                    TradeLogRow(
                        date=date,
                        action="BASE_BUY",
                        position_before=float(pb),
                        position_after=float(current_base),
                        reason="DIVIDEND_GROWTH: build base",
                        market_state=None,
                        strategy="dividend_growth_base",
                        price=float(price) if price is not None else None,
                        pnl_since_entry=None,
                    )
                )

            if dec.get("signal") == "BASE_REDUCE" and base_active:
                pb = current_base
                current_base = base_floor
                current_overlay = 0.0
                logs.append(
                    TradeLogRow(
                        date=date,
                        action="BASE_REDUCE",
                        position_before=float(pb),
                        position_after=float(current_base),
                        reason=str(dec.get("reason") or "BASE_REDUCE"),
                        market_state=None,
                        strategy="dividend_growth_base",
                        price=float(price) if price is not None else None,
                        pnl_since_entry=None,
                    )
                )
                desired_total = current_base

            if base_active and desired_total < base_floor:
                desired_total = base_floor
            desired_pos = float(max(0.0, min(overlay_max, desired_total - current_base)))
        else:
            if bq is None or mq is None:
                continue
            strategy_used, dec = route_decision(
                code=code,
                category=category,
                market_state=None,
                bias_q=bq,
                momentum_q=mq,
                current_position=float(current_overlay),
            )
            desired_pos = float(dec["position_after"])
            if dec.get("signal") == "HOLD" or dec.get("action") == "HOLD" or ("HOLD" in str(dec.get("reason") or "")):
                desired_pos = current_overlay
        if abs(desired_pos) < 0.005:
            desired_pos = 0.0
        if abs(current_overlay) < 0.005:
            current_overlay = 0.0

        if is_core and abs(desired_pos - current_overlay) > 1e-12:
            min_hold = 15
            within_hold = (i - last_trade_i) < min_hold
            risk_exit = mq is not None and mq <= 2 and desired_pos < current_overlay
            if within_hold and not risk_exit:
                desired_pos = current_overlay
            else:
                last_trade_i = i

        action_kind = _action_kind(current_overlay, desired_pos)
        action = None if action_kind is None else f"TACTICAL_{action_kind}"
        if action is None:
            current_overlay = desired_pos
            continue

        if action_kind == "BUY":
            if entry_price is None and price is not None:
                entry_price = float(price)
        elif action_kind == "EXIT":
            entry_price = None

        pnl = None
        if entry_price is not None and price is not None and entry_price > 0:
            pnl = float(price) / float(entry_price) - 1.0

        reason = str(dec.get("reason") or "")
        reason = f"bias_q=Q{bq} momentum_q=Q{mq} | {reason}".strip(" |")
        if is_core and base_target > 0:
            reason = f"{reason} | base={int(base_target * 100)}% overlay_max={int(overlay_max * 100)}%"
        if is_div_growth and base_target > 0:
            reason = (
                f"{reason} | base_cur={int(current_base * 100)}% base_target={int(base_target * 100)}% "
                f"overlay_max={int(overlay_max * 100)}%"
            )
        total_before = current_base + current_overlay if (is_core or is_div_growth) else current_overlay
        total_after = current_base + desired_pos if (is_core or is_div_growth) else desired_pos
        reason = f"{reason} | total={total_before*100:.0f}%→{total_after*100:.0f}%"

        logs.append(
            TradeLogRow(
                date=date,
                action=action,
                position_before=float(current_overlay),
                position_after=float(desired_pos),
                reason=reason,
                market_state=market_state,
                strategy=strategy_used,
                price=float(price) if price is not None else None,
                pnl_since_entry=pnl,
            )
        )
        current_overlay = desired_pos

    if limit is not None and int(limit) > 0:
        return logs[-int(limit) :]
    return logs


def cmd_debug_trades(ns: argparse.Namespace) -> int:
    db = resolve_db_path(ns.db_path)
    code = str(ns.etf)
    category = classify_etf(code)
    if category not in ("DIVIDEND", "CORE_INDEX"):
        print(f"debug-trades supports DIVIDEND/CORE_INDEX only (got {category})", file=sys.stderr)
        return 2

    name = str(etf_name_map(db).get(code, code))
    df, _ = load_etf_sqlite(db, code)
    df = df.sort_values("date") if "date" in df.columns else df
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
    df = add_analyzer_indicators(df_calc)
    logs = replay_trades(
        df=df,
        code=code,
        category=category,
        rolling_window=int(ns.rolling_window),
        regime_window=int(ns.regime_window),
        regime_persist=int(ns.regime_persist),
        core_exposure_floor=float(ns.core_exposure_floor),
        tactical_overlay_max=float(ns.tactical_overlay_max),
        eval_start=eff.effective_start.isoformat(),
        eval_end=eff.effective_end.isoformat(),
        limit=ns.limit,
    )

    def _h(en: str, zh: str) -> str:
        return f"{en} / {zh}"

    print(f"[{code} {name}]")
    print(f"{_h('category', '分类')}: {category}")
    rs = getattr(ns, "start_date", None) or "数据库最早"
    re = getattr(ns, "end_date", None) or "数据库最新"
    print(f"{_h('requested_range', '用户输入区间')}: {rs} ~ {re}")
    print(f"{_h('effective_range', '实际生效区间')}: {eff.effective_start.isoformat()} ~ {eff.effective_end.isoformat()}")
    w = int(getattr(ns, "warmup_days", 0) or 0)
    if w > 0:
        cs = warmup_start.isoformat() if warmup_start is not None else eff.effective_start.isoformat()
        print(f"{_h('calc_range', '计算区间')}: {cs} ~ {eff.effective_end.isoformat()} (warmup_days={w})")
    print(f"{_h('trace', '动作追踪')}:")
    if not logs:
        print("(no actions) / （无动作）")
        return 0
    for r in logs:
        print(r.format_line())
    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("debug-trades", help="Chronological trace of strategy actions (BUY/ADD/REDUCE/EXIT)")
    p.add_argument("--etf", required=True, dest="etf", help="ETF code")
    p.add_argument("--db", required=True, dest="db_path", help="SQLite path")
    p.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD), inclusive")
    p.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD), inclusive")
    p.add_argument("--warmup-days", type=int, default=0, help="Warmup trading days before effective start for indicator stability")
    p.add_argument("--limit", type=int, default=None, help="Keep only last N actions")
    p.add_argument("--rolling-window", type=int, default=252, help="Rolling window for bias_q/momentum_q")
    p.add_argument("--regime-window", type=int, default=20, help="Market-state window for regime detection")
    p.add_argument("--regime-persist", type=int, default=20, help="Min days a regime must persist before switching")
    p.add_argument(
        "--core-exposure-floor",
        "--core-base-exposure",
        dest="core_exposure_floor",
        type=float,
        default=0.4,
        help="CORE_INDEX base_core_exposure (0 disables base layer). Not counted as tactical trades.",
    )
    p.add_argument("--tactical-overlay-max", type=float, default=0.4, help="CORE_INDEX tactical overlay max exposure")
    p.set_defaults(_run=cmd_debug_trades)
