import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.cli.date_range import (
    build_calc_df,
    compute_effective_range,
    date_to_str,
    filter_df_by_effective_range,
    get_db_date_bounds,
    parse_cli_date,
)
from quantlab.config.etf_metadata import get_inception_date
from quantlab.cli.recommend_strategy import (
    DIVIDEND,
    CORE_INDEX,
    SECTOR_ROTATION,
    HIGH_BETA,
    THEMATIC,
    DEFENSIVE,
    MIN_EXPOSURE,
    classify_etf,
    dividend_strategy,
    dividend_growth_strategy,
    mean_reversion_strategy,
    trend_following_strategy,
    defensive_strategy,
    bloody_chip_strategy,
    light_trading_strategy,
    hold_only_strategy,
)
from quantlab.cli.market_regime import rolling_bucket_rank_series, detect_market_state


CATEGORY_UNIVERSE: dict[str, list[str]] = {
    "DIVIDEND": list(DIVIDEND),
    "CORE_INDEX": list(CORE_INDEX),
    "SECTOR_ROTATION": list(SECTOR_ROTATION),
    "HIGH_BETA": list(HIGH_BETA),
    "THEMATIC": list(THEMATIC),
    "DEFENSIVE": list(DEFENSIVE),
}

CORE_INDEX_MATURE = ["510300", "510500"]
CORE_INDEX_LIGHT = ["159361"]


def _as_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _close_from_row(row: pd.Series) -> float | None:
    for k in ("close_norm", "close", "price", "收盘"):
        if k in row:
            v = _as_float(row.get(k))
            if v is not None:
                return float(v)
    return None


def _equity_curve(close: pd.Series, pos: pd.Series) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(pos, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(c)
    eq = np.ones(n, dtype=float)
    for i in range(1, n):
        if not (np.isfinite(c[i]) and np.isfinite(c[i - 1]) and c[i - 1] > 0):
            eq[i] = eq[i - 1]
            continue
        r = c[i] / c[i - 1] - 1.0
        expo = float(p[i])
        eq[i] = eq[i - 1] * (1.0 + expo * r)
    return pd.Series(eq, index=close.index)


def _max_drawdown(eq: pd.Series) -> float:
    x = pd.to_numeric(eq, errors="coerce").to_numpy(dtype=float)
    peak = -np.inf
    mdd = 0.0
    for v in x:
        if not np.isfinite(v):
            continue
        if v > peak:
            peak = v
        if peak > 0:
            dd = v / peak - 1.0
            if dd < mdd:
                mdd = float(dd)
    return float(mdd)


def _annualized_return(eq: pd.Series, *, periods_per_year: int = 252) -> float:
    x = pd.to_numeric(eq, errors="coerce").to_numpy(dtype=float)
    fin = x[np.isfinite(x)]
    if fin.size < 2:
        return 0.0
    total = fin[-1] / fin[0]
    n = fin.size - 1
    if n <= 0:
        return 0.0
    return float(total ** (periods_per_year / n) - 1.0)


def _sharpe(eq: pd.Series, *, periods_per_year: int = 252) -> float:
    x = pd.to_numeric(eq, errors="coerce").to_numpy(dtype=float)
    if x.size < 3:
        return 0.0
    rets = x[1:] / x[:-1] - 1.0
    rets = rets[np.isfinite(rets)]
    if rets.size < 20:
        return 0.0
    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    if sd <= 0:
        return 0.0
    return float(mu / sd * math.sqrt(periods_per_year))


def _trade_stats(pos: pd.Series, eq: pd.Series) -> dict[str, float]:
    p = pd.to_numeric(pos, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    x = pd.to_numeric(eq, errors="coerce").ffill().to_numpy(dtype=float)
    n = len(p)
    trade_count = 0
    add_count = 0
    reduce_count = 0
    entry_sizes: list[float] = []
    entries: list[int] = []
    exits: list[int] = []

    for i in range(1, n):
        d = float(p[i] - p[i - 1])
        if abs(d) < 1e-12:
            continue
        trade_count += 1
        if d > 0:
            add_count += 1
            entry_sizes.append(d)
        else:
            reduce_count += 1
        if p[i - 1] <= 0 and p[i] > 0:
            entries.append(i)
        if p[i - 1] > 0 and p[i] <= 0:
            exits.append(i)

    pairs = []
    j = 0
    for e in entries:
        while j < len(exits) and exits[j] <= e:
            j += 1
        if j < len(exits):
            pairs.append((e, exits[j]))
            j += 1
        else:
            if e < n - 1:
                pairs.append((e, n - 1))

    wins = 0
    holds = []
    for e, ex in pairs:
        if ex <= e:
            continue
        pnl = x[ex] / x[e] - 1.0 if x[e] > 0 else 0.0
        if pnl > 0:
            wins += 1
        holds.append(ex - e)

    win_rate = float(wins / len(pairs)) if pairs else 0.0
    avg_hold = float(np.mean(holds)) if holds else 0.0
    entry_size_mean = float(np.mean(entry_sizes)) if entry_sizes else 0.0
    return {
        "trade_count": float(trade_count),
        "win_rate": float(win_rate),
        "avg_holding_days": float(avg_hold),
        "entry_size_mean": float(entry_size_mean),
        "add_position_count": float(add_count),
        "reduce_position_count": float(reduce_count),
    }


def backtest_one(
    *,
    df: pd.DataFrame,
    code: str,
    name: str,
    category: str,
    rolling_window: int = 252,
    regime_window: int = 20,
    regime_persist: int = 20,
    core_exposure_floor: float = 0.4,
    tactical_overlay_max: float = 0.4,
    return_curves: bool = False,
    eval_start: Any = None,
    eval_end: Any = None,
) -> dict[str, Any]:
    if df.empty:
        raise ValueError("empty df")
    d = df.sort_values("date") if "date" in df.columns else df.copy()
    d = d.reset_index(drop=True)
    close = d.apply(lambda r: _close_from_row(r), axis=1)
    close = pd.to_numeric(close, errors="coerce")
    if close.notna().sum() < 60:
        raise ValueError("insufficient close history")
    if "bias_rate" not in d.columns:
        raise ValueError("missing bias_rate")
    mom_col = "momentum_10" if "momentum_10" in d.columns else "momentum"
    if mom_col not in d.columns:
        raise ValueError("missing momentum")

    bias_q = rolling_bucket_rank_series(d["bias_rate"], window=rolling_window, n_buckets=5)
    momentum_q = rolling_bucket_rank_series(d[mom_col], window=rolling_window, n_buckets=5)
    bias_q_i = bias_q.round().astype("Int64")
    momentum_q_i = momentum_q.round().astype("Int64")
    market_state_s = None
    if category == "CORE_INDEX":
        _tmp = pd.DataFrame({"momentum_q": momentum_q})
        ms = detect_market_state(_tmp, window=int(regime_window), min_persist_days=int(regime_persist))
        market_state_s = ms["stable_state"]

    eval_start_d = parse_cli_date(eval_start) if eval_start is not None else None
    eval_end_d = parse_cli_date(eval_end) if eval_end is not None else None
    if eval_start_d is None or eval_end_d is None:
        dt0, dt1 = get_db_date_bounds(d)
        if dt0 is None or dt1 is None:
            raise ValueError("date bounds unavailable")
        eval_start_d = dt0
        eval_end_d = dt1

    dt_all = pd.to_datetime(d["date"], errors="coerce")
    eval_mask = (dt_all.notna() & (dt_all.dt.date >= eval_start_d) & (dt_all.dt.date <= eval_end_d)).to_numpy(dtype=bool)
    if int(eval_mask.sum()) < 2:
        raise ValueError("insufficient eval range rows")
    start_i = int(np.argmax(eval_mask))
    end_i = int(np.where(eval_mask)[0].max())
    if end_i <= start_i:
        raise ValueError("insufficient eval range rows")

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

    total_pos = np.zeros(len(d), dtype=float)
    base_pos = np.zeros(len(d), dtype=float)
    overlay_pos = np.zeros(len(d), dtype=float)

    base_target = float(core_exposure_floor) if category == "CORE_INDEX" else 0.0
    overlay_max = float(tactical_overlay_max) if category == "CORE_INDEX" else 0.0
    base_active = False

    current_base = 0.0
    current_overlay = 0.0
    total_pos[0] = current_base + current_overlay
    pos_max = 1.0
    last_trade_i = -10_000
    for i in range(start_i, end_i):
        bq = int(bias_q_i.iloc[i]) if pd.notna(bias_q_i.iloc[i]) else None
        mq = int(momentum_q_i.iloc[i]) if pd.notna(momentum_q_i.iloc[i]) else None

        if category == "DIVIDEND":
            if code == "159209":
                base_target = 0.60
                if float(core_exposure_floor) >= 0.5:
                    base_target = float(core_exposure_floor)
                overlay_default = max(0.0, 1.0 - base_target)
                overlay_max = overlay_default
                if float(tactical_overlay_max) > 0 and (base_target + float(tactical_overlay_max) <= 1.0 + 1e-9):
                    overlay_max = float(tactical_overlay_max)
                base_floor = min(base_target, 0.60)
                current_total = current_base + current_overlay
                dec = dividend_growth_strategy(
                    bias_q=bq,
                    momentum_q=mq,
                    current_position=current_total,
                    base_target=base_target,
                    overlay_max=overlay_max,
                )
                desired_total = float(dec["position_after"])
                if (not base_active) and current_total <= 1e-12 and desired_total > 1e-12:
                    base_active = True
                    current_base = float(min(base_target, desired_total))

                if dec.get("signal") == "BASE_REDUCE" and base_active:
                    current_base = base_floor
                    current_overlay = 0.0
                    desired_total = current_base

                if base_active and desired_total < base_floor:
                    desired_total = base_floor
                current_overlay = float(max(0.0, min(overlay_max, desired_total - current_base)))
                total_pos[i + 1] = current_base + current_overlay
                base_pos[i + 1] = current_base
                overlay_pos[i + 1] = current_overlay
                pos_max = float(base_target + overlay_max)
                continue

            dec = dividend_strategy(bias_q=bq, momentum_q=mq, current_position=total_pos[i])
            desired_total = float(dec["position_after"])
            total_pos[i + 1] = desired_total
            base_pos[i + 1] = 0.0
            overlay_pos[i + 1] = desired_total
            pos_max = float(dec["position_max"])
            continue
        elif category == "CORE_INDEX":
            st = str(market_state_s.iloc[i]) if market_state_s is not None and pd.notna(market_state_s.iloc[i]) else "RANGE"
            valid = bq is not None and mq is not None and st in ("TREND", "RANGE", "DOWN")
            if (not base_active) and valid and base_target > 0:
                base_active = True
                current_base = base_target
            if not valid:
                total_pos[i + 1] = current_base + current_overlay
                base_pos[i + 1] = current_base
                overlay_pos[i + 1] = current_overlay
                continue

            if st == "TREND":
                cur = _to_strategy_units(current_overlay, overlay_max, 0.8)
                dec = trend_following_strategy(category="CORE_INDEX_OVERLAY", bias_q=bq, momentum_q=mq, current_position=cur)
            elif st == "DOWN":
                dec = defensive_strategy(category="CORE_INDEX_OVERLAY", current_position=current_overlay)
            else:
                cur = _to_strategy_units(current_overlay, overlay_max, 0.8)
                dec = mean_reversion_strategy(category="CORE_INDEX_OVERLAY", bias_q=bq, momentum_q=mq, current_position=cur)

            desired_overlay = _from_strategy_units(float(dec["position_after"]), float(dec["position_max"]), overlay_max)
            if dec.get("signal") == "HOLD" or dec.get("action") == "HOLD" or ("HOLD" in str(dec.get("reason") or "")):
                desired_overlay = current_overlay
        elif category == "SECTOR_ROTATION":
            dec = mean_reversion_strategy(category=category, bias_q=bq, momentum_q=mq, current_position=total_pos[i])
            desired_total = float(dec["position_after"])
            total_pos[i + 1] = desired_total
            base_pos[i + 1] = 0.0
            overlay_pos[i + 1] = desired_total
            pos_max = float(dec["position_max"])
            continue
        elif category == "HIGH_BETA":
            dec = bloody_chip_strategy(category=category, bias_q=bq, momentum_q=mq, current_position=total_pos[i])
            desired_total = float(dec["position_after"])
            total_pos[i + 1] = desired_total
            base_pos[i + 1] = 0.0
            overlay_pos[i + 1] = desired_total
            pos_max = float(dec["position_max"])
            continue
        elif category == "THEMATIC":
            dec = light_trading_strategy(bias_q=bq, momentum_q=mq, current_position=total_pos[i])
            desired_total = float(dec["position_after"])
            total_pos[i + 1] = desired_total
            base_pos[i + 1] = 0.0
            overlay_pos[i + 1] = desired_total
            pos_max = float(dec["position_max"])
            continue
        elif category == "DEFENSIVE":
            dec = hold_only_strategy(current_position=total_pos[i])
            desired_total = float(dec["position_after"])
            total_pos[i + 1] = desired_total
            base_pos[i + 1] = 0.0
            overlay_pos[i + 1] = desired_total
            pos_max = float(dec["position_max"])
            continue
        else:
            raise ValueError(f"unknown category {category}")

        desired_pos = float(desired_overlay)
        if category == "CORE_INDEX" and abs(desired_pos - current_overlay) > 1e-12:
            min_hold = 15
            within_hold = (i - last_trade_i) < min_hold
            risk_exit = (mq is not None and mq <= 2 and desired_pos < current_overlay)
            if within_hold and not risk_exit:
                desired_pos = current_overlay
            else:
                last_trade_i = i

        current_overlay = float(desired_pos)
        overlay_pos[i + 1] = current_overlay
        current_base = current_base if base_active else 0.0
        base_pos[i + 1] = current_base
        total_pos[i + 1] = current_base + current_overlay
        pos_max = float(current_base + overlay_max)

    total_pos_s = pd.Series(total_pos, index=d.index)
    base_pos_s = pd.Series(base_pos, index=d.index)
    overlay_pos_s = pd.Series(overlay_pos, index=d.index)
    close_eval = close.iloc[start_i : end_i + 1]
    total_pos_eval = total_pos_s.loc[close_eval.index]
    base_pos_eval = base_pos_s.loc[close_eval.index]
    overlay_pos_eval = overlay_pos_s.loc[close_eval.index]

    eq = _equity_curve(close_eval, total_pos_eval)
    base_eq = _equity_curve(close_eval, base_pos_eval)
    overlay_eq = _equity_curve(close_eval, overlay_pos_eval)
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    ann = _annualized_return(eq)
    mdd = _max_drawdown(eq)
    shr = _sharpe(eq)
    tstats = _trade_stats(overlay_pos_eval, overlay_eq)

    avg_exposure = float(np.mean(total_pos_eval.to_numpy(dtype=float)))
    max_exposure = float(np.max(total_pos_eval.to_numpy(dtype=float)))
    fully_ratio = float(np.mean((total_pos_eval.to_numpy(dtype=float) >= (pos_max - 1e-9)).astype(float))) if pos_max > 0 else 0.0
    ret_per_exp = float(total_return / avg_exposure) if avg_exposure > 1e-12 else 0.0
    dd_per_exp = float(abs(mdd) / avg_exposure) if avg_exposure > 1e-12 else 0.0

    base_return = float(close_eval.iloc[-1] / close_eval.iloc[0] - 1.0) if close_eval.iloc[0] and np.isfinite(close_eval.iloc[0]) else 0.0
    excess = float(total_return - base_return)
    base_layer_return = float(base_eq.iloc[-1] / base_eq.iloc[0] - 1.0)
    overlay_layer_return = float(overlay_eq.iloc[-1] / overlay_eq.iloc[0] - 1.0)

    out: dict[str, Any] = {
        "etf_code": code,
        "etf_name": name,
        "category": category,
        "status": "ok",
        "metrics": {
            "total_return": total_return,
            "annualized_return": ann,
            "max_drawdown": mdd,
            "sharpe_ratio": shr,
            "trade_count": int(tstats["trade_count"]),
            "win_rate": float(tstats["win_rate"]),
            "avg_holding_days": float(tstats["avg_holding_days"]),
        },
        "exposure": {
            "avg_exposure": avg_exposure,
            "max_exposure": max_exposure,
            "entry_size_mean": float(tstats["entry_size_mean"]),
            "add_position_count": int(tstats["add_position_count"]),
            "reduce_position_count": int(tstats["reduce_position_count"]),
            "fully_invested_days_ratio": fully_ratio,
            "return_per_exposure": ret_per_exp,
            "drawdown_per_exposure": dd_per_exp,
        },
        "baseline": {
            "baseline_return": base_return,
            "excess_return_vs_baseline": excess,
        },
        "layers": {
            "base_core_exposure": float(base_target),
            "tactical_overlay_max": float(overlay_max),
            "base_return": float(base_layer_return),
            "overlay_return": float(overlay_layer_return),
            "combined_return": float(total_return),
            "baseline_return": float(base_return),
            "excess_return": float(excess),
        },
    }
    if return_curves:
        dates = d.loc[start_i : end_i, "date"].astype(str).tolist() if "date" in d.columns else list(range(len(close_eval)))
        baseline_eq = (close_eval / close_eval.iloc[0]).ffill()
        out["curves"] = {
            "dates": dates,
            "base_eq": base_eq.tolist(),
            "overlay_eq": overlay_eq.tolist(),
            "combined_eq": eq.tolist(),
            "baseline_eq": baseline_eq.tolist(),
            "base_exposure": base_pos_eval.tolist(),
            "overlay_exposure": overlay_pos_eval.tolist(),
            "total_exposure": total_pos_eval.tolist(),
        }
    return out


def backtest_one_safe(
    *,
    df: pd.DataFrame,
    code: str,
    name: str,
    category: str,
    rolling_window: int,
    regime_window: int,
    regime_persist: int,
    core_exposure_floor: float,
    tactical_overlay_max: float,
    eval_start: Any = None,
    eval_end: Any = None,
) -> dict[str, Any]:
    try:
        return backtest_one(
            df=df,
            code=code,
            name=name,
            category=category,
            rolling_window=rolling_window,
            regime_window=regime_window,
            regime_persist=regime_persist,
            core_exposure_floor=core_exposure_floor,
            tactical_overlay_max=tactical_overlay_max,
            eval_start=eval_start,
            eval_end=eval_end,
        )
    except Exception as e:
        return {
            "etf_code": code,
            "etf_name": name,
            "category": category,
            "status": "error",
            "error": str(e),
            "metrics": {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "trade_count": 0,
                "win_rate": 0.0,
                "avg_holding_days": 0.0,
            },
            "exposure": {
                "avg_exposure": 0.0,
                "max_exposure": 0.0,
                "entry_size_mean": 0.0,
                "add_position_count": 0,
                "reduce_position_count": 0,
                "fully_invested_days_ratio": 0.0,
                "return_per_exposure": 0.0,
                "drawdown_per_exposure": 0.0,
            },
            "baseline": {
                "baseline_return": 0.0,
                "excess_return_vs_baseline": 0.0,
            },
        }


def _aggregate_category(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not rows:
        return {"n": 0, "n_ok": 0, "summary": {}}
    if not ok:
        return {"n": len(rows), "n_ok": 0, "summary": {}}
    m = [r["metrics"] for r in ok]
    e = [r["exposure"] for r in ok]
    b = [r["baseline"] for r in ok]

    def _mean(xs: list[float]) -> float:
        arr = np.array(xs, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else 0.0

    summary = {
        "n": len(rows),
        "n_ok": len(ok),
        "mean_total_return": _mean([x["total_return"] for x in m]),
        "mean_annualized_return": _mean([x["annualized_return"] for x in m]),
        "mean_max_drawdown": _mean([x["max_drawdown"] for x in m]),
        "mean_sharpe_ratio": _mean([x["sharpe_ratio"] for x in m]),
        "mean_avg_exposure": _mean([x["avg_exposure"] for x in e]),
        "mean_return_per_exposure": _mean([x["return_per_exposure"] for x in e]),
        "mean_drawdown_per_exposure": _mean([x["drawdown_per_exposure"] for x in e]),
        "mean_excess_vs_baseline": _mean([x["excess_return_vs_baseline"] for x in b]),
    }
    layers = [r.get("layers") or {} for r in ok]
    if any(layers):
        summary["mean_base_return"] = _mean([float(x.get("base_return", 0.0) or 0.0) for x in layers])
        summary["mean_overlay_return"] = _mean([float(x.get("overlay_return", 0.0) or 0.0) for x in layers])
        summary["mean_combined_return"] = _mean([float(x.get("combined_return", 0.0) or 0.0) for x in layers])
    return {"n": len(rows), "n_ok": len(ok), "summary": summary}


def run_report(
    *,
    db_path: str,
    category: str | None,
    user_start_date: Any,
    user_end_date: Any,
    warmup_days: int,
    rolling_window: int,
    regime_window: int,
    regime_persist: int,
    core_exposure_floor: float,
    tactical_overlay_max: float,
) -> dict[str, Any]:
    from core.data_loader import etf_name_map, load_etf_sqlite
    from core.indicators import add_analyzer_indicators
    from core.paths import resolve_db_path

    db = resolve_db_path(db_path)
    names = etf_name_map(db)
    cats = [category] if category is not None else list(CATEGORY_UNIVERSE.keys())
    results: dict[str, Any] = {"db_path": str(db), "categories": {}}

    for cat in cats:
        if cat not in CATEGORY_UNIVERSE:
            raise ValueError(f"unknown category: {cat}")
        if cat == "CORE_INDEX":
            blocks = [
                ("CORE_INDEX_MATURE", CORE_INDEX_MATURE, tactical_overlay_max),
                ("CORE_INDEX_LIGHT", CORE_INDEX_LIGHT, 0.0),
            ]
        else:
            blocks = [(cat, CATEGORY_UNIVERSE[cat], tactical_overlay_max)]

        for out_key, codes, overlay_max in blocks:
            rows = []
            for code in codes:
                name = str(names.get(str(code), str(code)))
                df, _ = load_etf_sqlite(db, code)
                db_start, db_end = get_db_date_bounds(df)
                if db_start is None or db_end is None:
                    rows.append(
                        {
                            "etf_code": code,
                            "etf_name": name,
                            "category": out_key,
                            "status": "error",
                            "error": "DB date bounds unavailable",
                            "requested_range": {"start": None, "end": None},
                            "effective_range": {"start": None, "end": None},
                            "metrics": {
                                "total_return": 0.0,
                                "annualized_return": 0.0,
                                "max_drawdown": 0.0,
                                "sharpe_ratio": 0.0,
                                "trade_count": 0,
                                "win_rate": 0.0,
                                "avg_holding_days": 0.0,
                            },
                            "exposure": {
                                "avg_exposure": 0.0,
                                "max_exposure": 0.0,
                                "entry_size_mean": 0.0,
                                "add_position_count": 0,
                                "reduce_position_count": 0,
                                "fully_invested_days_ratio": 0.0,
                                "return_per_exposure": 0.0,
                                "drawdown_per_exposure": 0.0,
                            },
                            "baseline": {"baseline_return": 0.0, "excess_return_vs_baseline": 0.0},
                        }
                    )
                    continue

                inc = get_inception_date(code)
                eff = compute_effective_range(
                    user_start=parse_cli_date(user_start_date),
                    user_end=parse_cli_date(user_end_date),
                    inception_date=inc,
                    db_start=db_start,
                    db_end=db_end,
                )
                df_calc, warmup_start = build_calc_df(df, eff=eff, warmup_days=int(warmup_days))
                df = add_analyzer_indicators(df_calc)

                if len(df) < int(rolling_window):
                    rows.append(
                        {
                            "etf_code": code,
                            "etf_name": name,
                            "category": out_key,
                            "status": "insufficient_history",
                            "error": f"INSUFFICIENT_HISTORY: calc_rows {len(df)} < rolling_window {int(rolling_window)}",
                            "requested_range": {"start": date_to_str(eff.requested_start), "end": date_to_str(eff.requested_end)},
                            "effective_range": {"start": date_to_str(eff.effective_start), "end": date_to_str(eff.effective_end)},
                            "calc_range": {
                                "start": date_to_str(warmup_start) if warmup_start is not None else date_to_str(eff.effective_start),
                                "end": date_to_str(eff.effective_end),
                            },
                            "metrics": {
                                "total_return": 0.0,
                                "annualized_return": 0.0,
                                "max_drawdown": 0.0,
                                "sharpe_ratio": 0.0,
                                "trade_count": 0,
                                "win_rate": 0.0,
                                "avg_holding_days": 0.0,
                            },
                            "exposure": {
                                "avg_exposure": 0.0,
                                "max_exposure": 0.0,
                                "entry_size_mean": 0.0,
                                "add_position_count": 0,
                                "reduce_position_count": 0,
                                "fully_invested_days_ratio": 0.0,
                                "return_per_exposure": 0.0,
                                "drawdown_per_exposure": 0.0,
                            },
                            "baseline": {"baseline_return": 0.0, "excess_return_vs_baseline": 0.0},
                        }
                    )
                    continue
                r = backtest_one_safe(
                    df=df,
                    code=code,
                    name=name,
                    category=cat,
                    rolling_window=rolling_window,
                    regime_window=regime_window,
                    regime_persist=regime_persist,
                    core_exposure_floor=core_exposure_floor,
                    tactical_overlay_max=float(overlay_max),
                    eval_start=eff.effective_start.isoformat(),
                    eval_end=eff.effective_end.isoformat(),
                )
                r["category"] = out_key
                r["requested_range"] = {"start": date_to_str(eff.requested_start), "end": date_to_str(eff.requested_end)}
                r["effective_range"] = {"start": date_to_str(eff.effective_start), "end": date_to_str(eff.effective_end)}
                r["calc_range"] = {
                    "start": date_to_str(warmup_start) if warmup_start is not None else date_to_str(eff.effective_start),
                    "end": date_to_str(eff.effective_end),
                }
                rows.append(r)
            agg = _aggregate_category(rows)
            results["categories"][out_key] = {
                "strategy": {
                    "DIVIDEND": "dividend_strategy",
                    "CORE_INDEX_MATURE": "core+overlay(regime_switching)",
                    "CORE_INDEX_LIGHT": "base_only(excluded_from_overlay_eval)",
                    "SECTOR_ROTATION": "mean_reversion_strategy",
                    "HIGH_BETA": "bloody_chip_strategy",
                    "THEMATIC": "light_trading_strategy",
                    "DEFENSIVE": "hold_only_strategy",
                }[out_key],
                "aggregate": agg,
                "members": rows,
                "commentary": "Includes exposure metrics to separate alpha vs higher exposure.",
            }

    return results


def run_report_single(
    *,
    db_path: str,
    etf: str,
    user_start_date: Any,
    user_end_date: Any,
    warmup_days: int,
    rolling_window: int,
    regime_window: int,
    regime_persist: int,
    core_exposure_floor: float,
    tactical_overlay_max: float,
) -> dict[str, Any]:
    from core.data_loader import etf_name_map, load_etf_sqlite
    from core.indicators import add_analyzer_indicators
    from core.paths import resolve_db_path

    db = resolve_db_path(db_path)
    names = etf_name_map(db)
    code = str(etf)
    name = str(names.get(code, code))

    cat = classify_etf(code)
    if cat == "CORE_INDEX":
        out_key = "CORE_INDEX_MATURE" if code in CORE_INDEX_MATURE else "CORE_INDEX_LIGHT"
        overlay_max = float(tactical_overlay_max) if out_key == "CORE_INDEX_MATURE" else 0.0
    else:
        out_key = str(cat)
        overlay_max = float(tactical_overlay_max)

    results: dict[str, Any] = {"db_path": str(db), "categories": {}}

    df, _ = load_etf_sqlite(db, code)
    db_start, db_end = get_db_date_bounds(df)
    if db_start is None or db_end is None:
        rows = [
            {
                "etf_code": code,
                "etf_name": name,
                "category": out_key,
                "status": "error",
                "error": "DB date bounds unavailable",
                "requested_range": {"start": None, "end": None},
                "effective_range": {"start": None, "end": None},
                "metrics": {
                    "total_return": 0.0,
                    "annualized_return": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "trade_count": 0,
                    "win_rate": 0.0,
                    "avg_holding_days": 0.0,
                },
                "exposure": {
                    "avg_exposure": 0.0,
                    "max_exposure": 0.0,
                    "entry_size_mean": 0.0,
                    "add_position_count": 0,
                    "reduce_position_count": 0,
                    "fully_invested_days_ratio": 0.0,
                    "return_per_exposure": 0.0,
                    "drawdown_per_exposure": 0.0,
                },
                "baseline": {"baseline_return": 0.0, "excess_return_vs_baseline": 0.0},
            }
        ]
        agg = _aggregate_category(rows)
        results["categories"][out_key] = {
            "strategy": {
                "DIVIDEND": "dividend_strategy",
                "CORE_INDEX_MATURE": "core+overlay(regime_switching)",
                "CORE_INDEX_LIGHT": "base_only(excluded_from_overlay_eval)",
                "SECTOR_ROTATION": "mean_reversion_strategy",
                "HIGH_BETA": "bloody_chip_strategy",
                "THEMATIC": "light_trading_strategy",
                "DEFENSIVE": "hold_only_strategy",
            }[out_key],
            "aggregate": agg,
            "members": rows,
            "commentary": "Includes exposure metrics to separate alpha vs higher exposure.",
        }
        return results

    inc = get_inception_date(code)
    eff = compute_effective_range(
        user_start=parse_cli_date(user_start_date),
        user_end=parse_cli_date(user_end_date),
        inception_date=inc,
        db_start=db_start,
        db_end=db_end,
    )
    df_calc, warmup_start = build_calc_df(df, eff=eff, warmup_days=int(warmup_days))
    df_ind = add_analyzer_indicators(df_calc)

    if len(df_ind) < int(rolling_window):
        r = {
            "etf_code": code,
            "etf_name": name,
            "category": out_key,
            "status": "insufficient_history",
            "error": f"INSUFFICIENT_HISTORY: calc_rows {len(df_ind)} < rolling_window {int(rolling_window)}",
            "requested_range": {"start": date_to_str(eff.requested_start), "end": date_to_str(eff.requested_end)},
            "effective_range": {"start": date_to_str(eff.effective_start), "end": date_to_str(eff.effective_end)},
            "calc_range": {
                "start": date_to_str(warmup_start) if warmup_start is not None else date_to_str(eff.effective_start),
                "end": date_to_str(eff.effective_end),
            },
            "metrics": {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "trade_count": 0,
                "win_rate": 0.0,
                "avg_holding_days": 0.0,
            },
            "exposure": {
                "avg_exposure": 0.0,
                "max_exposure": 0.0,
                "entry_size_mean": 0.0,
                "add_position_count": 0,
                "reduce_position_count": 0,
                "fully_invested_days_ratio": 0.0,
                "return_per_exposure": 0.0,
                "drawdown_per_exposure": 0.0,
            },
            "baseline": {"baseline_return": 0.0, "excess_return_vs_baseline": 0.0},
        }
    else:
        r = backtest_one_safe(
            df=df_ind,
            code=code,
            name=name,
            category=cat,
            rolling_window=rolling_window,
            regime_window=regime_window,
            regime_persist=regime_persist,
            core_exposure_floor=core_exposure_floor,
            tactical_overlay_max=float(overlay_max),
            eval_start=eff.effective_start.isoformat(),
            eval_end=eff.effective_end.isoformat(),
        )
        r["category"] = out_key
        r["requested_range"] = {"start": date_to_str(eff.requested_start), "end": date_to_str(eff.requested_end)}
        r["effective_range"] = {"start": date_to_str(eff.effective_start), "end": date_to_str(eff.effective_end)}
        r["calc_range"] = {
            "start": date_to_str(warmup_start) if warmup_start is not None else date_to_str(eff.effective_start),
            "end": date_to_str(eff.effective_end),
        }

    rows = [r]
    agg = _aggregate_category(rows)
    results["categories"][out_key] = {
        "strategy": {
            "DIVIDEND": "dividend_strategy",
            "CORE_INDEX_MATURE": "core+overlay(regime_switching)",
            "CORE_INDEX_LIGHT": "base_only(excluded_from_overlay_eval)",
            "SECTOR_ROTATION": "mean_reversion_strategy",
            "HIGH_BETA": "bloody_chip_strategy",
            "THEMATIC": "light_trading_strategy",
            "DEFENSIVE": "hold_only_strategy",
        }[out_key],
        "aggregate": agg,
        "members": rows,
        "commentary": "Includes exposure metrics to separate alpha vs higher exposure.",
    }
    return results


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _f(x: float) -> str:
    return f"{x:.4f}"


def _print_category(cat: str, block: dict[str, Any], *, detail: bool) -> None:
    agg = block["aggregate"]["summary"]
    print(f"== {cat} ({block['strategy']}) ==")
    n = int(block.get("aggregate", {}).get("n", 0) or 0)
    n_ok = int(block.get("aggregate", {}).get("n_ok", 0) or 0)
    if n:
        print(f"members / 成员: {n_ok}/{n} ok / 有效")
        if n_ok == 0:
            print(
                "note / 说明: no valid backtest samples in this range (common: effective samples < rolling_window). "
                "Try --detail to see per-ETF SKIP reasons, or reduce --rolling-window.\n"
                "说明: 当前区间内无可用回测样本（常见原因：有效样本 < rolling_window）。建议加 --detail 查看每只 ETF 的 SKIP 原因，或调小 --rolling-window。"
            )
        print("")
    if agg:
        print("Performance metrics / 绩效指标:")
        print(f"  total_return(mean) / 总收益(均值): {_pct(agg['mean_total_return'])}")
        print(f"  annualized_return(mean) / 年化收益(均值): {_pct(agg['mean_annualized_return'])}")
        print(f"  max_drawdown(mean) / 最大回撤(均值): {_pct(agg['mean_max_drawdown'])}")
        print(f"  sharpe_ratio(mean) / 夏普(均值): {_f(agg['mean_sharpe_ratio'])}")
        print("")
        if "mean_base_return" in agg:
            print("Layer returns (mean) / 分层收益(均值):")
            print(f"  base_return / 底仓收益: {_pct(agg['mean_base_return'])}")
            print(f"  overlay_return / 战术层收益: {_pct(agg['mean_overlay_return'])}")
            print(f"  combined_return / 合并收益: {_pct(agg['mean_combined_return'])}")
            print("")
        print("Position / exposure metrics / 仓位与暴露指标:")
        print(f"  avg_exposure(mean) / 平均暴露(均值): {_pct(agg['mean_avg_exposure'])}")
        print(f"  return_per_exposure(mean) / 单位暴露收益(均值): {_f(agg['mean_return_per_exposure'])}")
        print(f"  drawdown_per_exposure(mean) / 单位暴露回撤(均值): {_f(agg['mean_drawdown_per_exposure'])}")
        print("")
        print("Baseline comparison / 基准对比:")
        print(f"  excess_return(mean) / 超额收益(均值): {_pct(agg['mean_excess_vs_baseline'])}")
        print("")
    print(f"commentary / 说明: {block['commentary']}")
    print("")

    if not detail:
        return

    print("Member ETF breakdown:")
    for r in block["members"]:
        if r.get("status") != "ok":
            print(f"- {r['etf_code']} {r['etf_name']} (SKIP: {r.get('error')})")
            continue
        m = r["metrics"]
        e = r["exposure"]
        b = r["baseline"]
        layers = r.get("layers") or {}
        print(f"- {r['etf_code']} {r['etf_name']}")
        rr = r.get("requested_range") or {}
        er = r.get("effective_range") or {}
        cr = r.get("calc_range") or {}
        if rr or er:
            extra = ""
            if cr and cr.get("start") and cr.get("end") and (cr.get("start") != er.get("start")):
                extra = f" calc={cr.get('start','NA')}~{cr.get('end','NA')}"
            print(
                f"  range / 区间: requested={rr.get('start','NA')}~{rr.get('end','NA')} "
                f"effective={er.get('start','NA')}~{er.get('end','NA')}{extra}"
            )
        print(
            f"  perf / 绩效: total={_pct(m['total_return'])} ann={_pct(m['annualized_return'])} "
            f"mdd={_pct(m['max_drawdown'])} sharpe={_f(m['sharpe_ratio'])} "
            f"trades={m['trade_count']} win={_pct(m['win_rate'])} hold_days={m['avg_holding_days']:.1f}"
        )
        if layers:
            print(
                f"  layer_return / 分层收益: base={_pct(float(layers.get('base_return', 0.0) or 0.0))} "
                f"overlay={_pct(float(layers.get('overlay_return', 0.0) or 0.0))} "
                f"combined={_pct(float(layers.get('combined_return', 0.0) or 0.0))} "
                f"baseline={_pct(float(layers.get('baseline_return', 0.0) or 0.0))} "
                f"excess={_pct(float(layers.get('excess_return', 0.0) or 0.0))}"
            )
        print(
            f"  expo / 暴露: avg={_pct(e['avg_exposure'])} max={_pct(e['max_exposure'])} "
            f"entry_mean={_pct(e['entry_size_mean'])} add_n={e['add_position_count']} "
            f"reduce_n={e['reduce_position_count']} full_ratio={_pct(e['fully_invested_days_ratio'])} "
            f"ret/exp={_f(e['return_per_exposure'])} dd/exp={_f(e['drawdown_per_exposure'])}"
        )
        print(f"  baseline / 基准: buyhold={_pct(b['baseline_return'])} excess={_pct(b['excess_return_vs_baseline'])}")
    print("")


def cmd_backtest_strategy_report(ns: argparse.Namespace) -> int:
    try:
        if getattr(ns, "etf", None):
            results = run_report_single(
                db_path=ns.db_path,
                etf=str(ns.etf),
                user_start_date=getattr(ns, "start_date", None),
                user_end_date=getattr(ns, "end_date", None),
                warmup_days=int(getattr(ns, "warmup_days", 0) or 0),
                rolling_window=int(ns.rolling_window),
                regime_window=int(ns.regime_window),
                regime_persist=int(ns.regime_persist),
                core_exposure_floor=float(ns.core_exposure_floor),
                tactical_overlay_max=float(ns.tactical_overlay_max),
            )
        else:
            results = run_report(
                db_path=ns.db_path,
                category=ns.category,
                user_start_date=getattr(ns, "start_date", None),
                user_end_date=getattr(ns, "end_date", None),
                warmup_days=int(getattr(ns, "warmup_days", 0) or 0),
                rolling_window=int(ns.rolling_window),
                regime_window=int(ns.regime_window),
                regime_persist=int(ns.regime_persist),
                core_exposure_floor=float(ns.core_exposure_floor),
                tactical_overlay_max=float(ns.tactical_overlay_max),
            )
    except Exception as e:
        print(f"quantlab: error: {e}", file=sys.stderr)
        return 1

    if getattr(ns, "plot", False):
        supported_plot_categories = {"CORE_INDEX", "DIVIDEND"}
        if ns.category is not None and str(ns.category) not in supported_plot_categories:
            print(
                "quantlab: plot currently supports categories: CORE_INDEX, DIVIDEND. "
                "Use --category CORE_INDEX or DIVIDEND, or omit --plot.\n"
                "quantlab: --plot 当前仅支持品类：CORE_INDEX、DIVIDEND。请使用 --category CORE_INDEX 或 DIVIDEND，或不传 --plot。",
                file=sys.stderr,
            )
            return 2

        try:
            import plotly.graph_objects as go
        except Exception as e:
            print(f"quantlab: plot requires plotly ({e})", file=sys.stderr)
            return 2

        from core.data_loader import load_etf_sqlite
        from core.indicators import add_analyzer_indicators

        out_dir = Path(ns.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_blocks: list[tuple[str, str, dict[str, Any]]] = []
        if ns.category is None or str(ns.category) == "CORE_INDEX":
            if "CORE_INDEX_MATURE" in results["categories"]:
                plot_blocks.append(("CORE_INDEX_MATURE", "CORE_INDEX", results["categories"]["CORE_INDEX_MATURE"]))
        if ns.category is None or str(ns.category) == "DIVIDEND":
            if "DIVIDEND" in results["categories"]:
                plot_blocks.append(("DIVIDEND", "DIVIDEND", results["categories"]["DIVIDEND"]))

        if not plot_blocks:
            print(
                "quantlab: plot skipped (no supported category blocks in results). "
                "Try --category CORE_INDEX or DIVIDEND.\n"
                "quantlab: 跳过绘图（结果中没有可绘图的品类）。建议使用 --category CORE_INDEX 或 DIVIDEND。",
                file=sys.stderr,
            )
            return 2

        for block_key, category_key, block in plot_blocks:
            for r in block["members"]:
                if r.get("status") != "ok":
                    continue
                etf_code = r["etf_code"]
                etf_name = r["etf_name"]
                df, _ = load_etf_sqlite(results["db_path"], etf_code)  # type: ignore[arg-type]
                er = r.get("effective_range") or {}
                cr = r.get("calc_range") or {}
                try:
                    eff_end = parse_cli_date(er.get("end"))
                except Exception:
                    eff_end = None
                try:
                    calc_start = parse_cli_date(cr.get("start"))
                except Exception:
                    calc_start = None
                if calc_start is not None and eff_end is not None:
                    df_start, df_end = get_db_date_bounds(df)
                    if df_start is not None and df_end is not None:
                        eff = compute_effective_range(
                            user_start=calc_start,
                            user_end=eff_end,
                            inception_date=get_inception_date(etf_code),
                            db_start=df_start,
                            db_end=df_end,
                        )
                        df = filter_df_by_effective_range(df, eff)
                df = add_analyzer_indicators(df)
                curves = backtest_one(
                    df=df,
                    code=etf_code,
                    name=etf_name,
                    category=category_key,
                    rolling_window=int(ns.rolling_window),
                    regime_window=int(ns.regime_window),
                    regime_persist=int(ns.regime_persist),
                    core_exposure_floor=float(ns.core_exposure_floor),
                    tactical_overlay_max=float(ns.tactical_overlay_max),
                    return_curves=True,
                    eval_start=er.get("start"),
                    eval_end=er.get("end"),
                ).get("curves") or {}
                dates = curves.get("dates") or []
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=curves.get("combined_eq"), name="combined", line={"width": 2}))
                fig.add_trace(
                    go.Scatter(x=dates, y=curves.get("baseline_eq"), name="baseline", line={"width": 2, "dash": "dot"})
                )
                fig.add_trace(go.Scatter(x=dates, y=curves.get("base_eq"), name="base", line={"width": 1, "dash": "dash"}))
                fig.add_trace(
                    go.Scatter(x=dates, y=curves.get("overlay_eq"), name="overlay", line={"width": 1, "dash": "dashdot"})
                )
                fig.update_layout(
                    title=f"{block_key} | {etf_code} {etf_name} | equity (base/overlay/combined vs baseline)",
                    xaxis_title="date",
                    yaxis_title="equity",
                    legend={"orientation": "h"},
                )
                fig.write_html(out_dir / f"{block_key.lower()}_{etf_code}_equity.html", include_plotlyjs="cdn")

            rows = [x for x in block["members"] if x.get("status") == "ok"]
            if not rows:
                continue
            x = [f"{r['etf_code']}" for r in rows]
            base_r = [float((r.get("layers") or {}).get("base_return", 0.0) or 0.0) * 100 for r in rows]
            ov_r = [float((r.get("layers") or {}).get("overlay_return", 0.0) or 0.0) * 100 for r in rows]
            comb_r = [float((r.get("layers") or {}).get("combined_return", 0.0) or 0.0) * 100 for r in rows]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=x, y=base_r, name="base_return(%)"))
            fig.add_trace(go.Bar(x=x, y=ov_r, name="overlay_return(%)"))
            fig.add_trace(go.Bar(x=x, y=comb_r, name="combined_return(%)"))
            fig.update_layout(
                barmode="group",
                title=f"{block_key} | layer return comparison",
                xaxis_title="ETF",
                yaxis_title="return (%)",
            )
            fig.write_html(out_dir / f"{block_key.lower()}_layer_returns.html", include_plotlyjs="cdn")

            eff_x = [float(r["exposure"]["return_per_exposure"]) for r in rows]
            eff_y = [float(r["exposure"]["drawdown_per_exposure"]) for r in rows]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eff_x, y=eff_y, mode="markers+text", text=x, textposition="top center"))
            fig.update_layout(
                title=f"{block_key} | exposure efficiency (higher x, lower y is better)",
                xaxis_title="return_per_exposure",
                yaxis_title="drawdown_per_exposure",
            )
            fig.write_html(out_dir / f"{block_key.lower()}_exposure_efficiency.html", include_plotlyjs="cdn")

        plotted_keys = {k for k, _, _ in plot_blocks}

        if "CORE_INDEX_MATURE" in plotted_keys:
            md_lines = []
            md_lines.append("# CORE_INDEX v1.0 Summary")
            md_lines.append("")
            md_lines.append("## CORE_INDEX_MATURE")
            for code in CORE_INDEX_MATURE:
                r = None
                for row in results["categories"].get("CORE_INDEX_MATURE", {}).get("members", []):
                    if row.get("etf_code") == code:
                        r = row
                        break
                if r and r.get("status") == "ok":
                    layers = r.get("layers") or {}
                    md_lines.append(f"- **{code} {r.get('etf_name','')}** validated")
                    md_lines.append(
                        f"  - combined_return: {layers.get('combined_return', 0.0):.2%}, "
                        f"baseline_return: {layers.get('baseline_return', 0.0):.2%}, "
                        f"excess_return: {layers.get('excess_return', 0.0):.2%}"
                    )
                    md_lines.append(
                        f"  - base_return: {layers.get('base_return', 0.0):.2%}, overlay_return: {layers.get('overlay_return', 0.0):.2%}"
                    )
                else:
                    md_lines.append(f"- **{code}** not available in report (error)")
            md_lines.append("")
            md_lines.append("## CORE_INDEX_LIGHT")
            for code in CORE_INDEX_LIGHT:
                r = None
                for row in results["categories"].get("CORE_INDEX_LIGHT", {}).get("members", []):
                    if row.get("etf_code") == code:
                        r = row
                        break
                note = f"{r.get('error')}" if r and r.get("error") else "short history / not mature for overlay validation"
                md_lines.append(f"- **{code}** excluded due to short history: {note}")
            md_path = out_dir / "core_index_v1_summary.md"
            md_path.write_text("\n".join(md_lines), encoding="utf-8")

        if "DIVIDEND" in plotted_keys:
            md_lines = []
            md_lines.append("# DIVIDEND v1.0 Summary")
            md_lines.append("")
            md_lines.append("## Members")
            for r in results["categories"].get("DIVIDEND", {}).get("members", []):
                if r.get("status") != "ok":
                    md_lines.append(f"- **{r.get('etf_code','')} {r.get('etf_name','')}** (SKIP: {r.get('error')})")
                    continue
                layers = r.get("layers") or {}
                code = str(r.get("etf_code") or "")
                name = str(r.get("etf_name") or "")
                md_lines.append(f"- **{code} {name}**")
                md_lines.append(
                    f"  - combined_return: {float(layers.get('combined_return', 0.0) or 0.0):.2%}, "
                    f"baseline_return: {float(layers.get('baseline_return', 0.0) or 0.0):.2%}, "
                    f"excess_return: {float(layers.get('excess_return', 0.0) or 0.0):.2%}"
                )
                md_lines.append(f"  - chart: dividend_{code}_equity.html")
            md_path = out_dir / "dividend_v1_summary.md"
            md_path.write_text("\n".join(md_lines), encoding="utf-8")

    if ns.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    def _h(en: str, zh: str) -> str:
        return f"{en} / {zh}"

    print(_h("Backtest Strategy Report", "策略回测报告"))
    print(f"{_h('db', '数据库')}: {results['db_path']}")
    rs = getattr(ns, "start_date", None) or "数据库最早"
    re = getattr(ns, "end_date", None) or "数据库最新"
    print(f"{_h('requested_range', '用户输入区间')}: {rs} ~ {re}")
    w = int(getattr(ns, "warmup_days", 0) or 0)
    if w > 0:
        print(f"{_h('warmup_days', '预热天数')}: {w} (used for indicator warmup; excluded from performance window) / 用于指标预热，不计入绩效统计区间")
    print("")
    for cat, block in results["categories"].items():
        _print_category(cat, block, detail=bool(getattr(ns, "detail", False) or getattr(ns, "etf", None)))
    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "backtest-strategy-report",
        help="Backtest category strategies and compare performance vs exposure",
    )
    p.add_argument("--db", dest="db_path", default=None, help="SQLite path")
    p.add_argument(
        "--category",
        type=str,
        choices=["DIVIDEND", "CORE_INDEX", "SECTOR_ROTATION", "HIGH_BETA", "THEMATIC", "DEFENSIVE"],
        default=None,
        help="Filter by category. Choices: DIVIDEND | CORE_INDEX | SECTOR_ROTATION | HIGH_BETA | THEMATIC | DEFENSIVE. Omit for all.",
    )
    p.add_argument(
        "--etf",
        type=str,
        default=None,
        help="Single-ETF mode: run report for one ETF code only. Category is auto-detected; --category is not required.",
    )
    p.add_argument("--detail", action="store_true", help="Print per-ETF breakdown")
    p.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD), inclusive")
    p.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD), inclusive")
    p.add_argument("--warmup-days", type=int, default=0, help="Warmup trading days before effective start for indicator stability")
    p.add_argument("--rolling-window", type=int, default=252, help="Rolling window for bias_q/momentum_q")
    p.add_argument("--regime-window", type=int, default=20, help="Market-state window for CORE_INDEX regime detection")
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
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate plotly HTML charts and markdown summary to --out-dir (currently CORE_INDEX, DIVIDEND)",
    )
    p.add_argument("--out-dir", type=str, default="outputs/core_index_v1", help="Output directory for plots/markdown")
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.set_defaults(_run=cmd_backtest_strategy_report)
