from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sqlite3


class MarketRegime(str, Enum):
    AGGRESSIVE = "AGGRESSIVE"
    BALANCED = "BALANCED"
    DEFENSIVE = "DEFENSIVE"


@dataclass
class RegimeDetection:
    trade_date: str
    regime: MarketRegime
    raw_regime: MarketRegime
    bias: float
    bias_percentile: float
    momentum: float
    breadth: float
    reason: str


PORTFOLIO_WEIGHTS = {
    MarketRegime.AGGRESSIVE: {
        "159361": 0.60,
        "159209": 0.30,
        "515080": 0.10,
    },
    MarketRegime.BALANCED: {
        "159361": 0.35,
        "159209": 0.30,
        "515080": 0.35,
    },
    MarketRegime.DEFENSIVE: {
        "159361": 0.15,
        "159209": 0.30,
        "515080": 0.55,
    },
}

WEIGHT_RANGES = {
    MarketRegime.AGGRESSIVE: {
        "159361": (0.40, 0.60),
        "159209": (0.30, 0.30),
    },
    MarketRegime.BALANCED: {
        "159361": (0.30, 0.40),
        "159209": (0.30, 0.30),
    },
    MarketRegime.DEFENSIVE: {
        "159361": (0.10, 0.30),
        "159209": (0.30, 0.30),
    },
}

ETF_NAMES = {
    "159361": "A500 ETF",
    "159209": "中证红利质量 ETF",
    "515080": "中证红利 ETF",
}


def _load_market_proxy(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
        SELECT
            trade_date, proxy_name, proxy_value
        FROM market_proxy_daily
        WHERE proxy_name IN (
            'market_composite_bias', 'market_momentum_score', 'market_breadth_score'
        )
        ORDER BY trade_date
    """
    df = pd.read_sql_query(query, conn)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    pivot = df.pivot(index="trade_date", columns="proxy_name", values="proxy_value").reset_index()
    pivot.columns.name = None
    return pivot


def _compute_bias_percentile(bias_series: pd.Series) -> pd.Series:
    return bias_series.rank(pct=True, method="min")


def _rolling_percentile_min_rank(
    values: pd.Series,
    *,
    window: int = 252,
    min_periods: int = 60,
) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    out = np.full(len(s), np.nan, dtype=float)
    x = s.to_numpy(dtype=float)
    for i in range(len(x)):
        lo = max(0, i - int(window) + 1)
        w = x[lo : i + 1]
        w = w[np.isfinite(w)]
        if w.size < int(min_periods):
            continue
        v = x[i]
        if not np.isfinite(v):
            continue
        rank_min = int(np.sum(w < v)) + 1
        out[i] = float(rank_min) / float(w.size)
    return pd.Series(out, index=values.index)


def compute_confirmed_regime_series(
    raw_regimes: pd.Series,
    *,
    confirm_days: int = 5,
) -> pd.Series:
    s = raw_regimes.copy()
    if s.empty:
        return s
    converted: list[MarketRegime | None] = []
    for x in s.tolist():
        if isinstance(x, MarketRegime):
            converted.append(x)
            continue
        if pd.isna(x):
            converted.append(None)
            continue
        converted.append(MarketRegime(str(x)))
    confirmed = MarketRegime.BALANCED
    streak = 0
    out: list[MarketRegime] = []
    for r in converted:
        if r is None:
            streak = 0
            out.append(confirmed)
            continue
        if r == confirmed:
            streak = 0
        else:
            streak += 1
            if streak >= int(confirm_days):
                confirmed = r
                streak = 0
        out.append(confirmed)
    return pd.Series(out, index=s.index)


def detect_regime_single(
    bias: float,
    bias_percentile: float,
    momentum: float,
    breadth: float,
) -> Tuple[MarketRegime, str]:
    aggressive_count = 0
    defensive_count = 0

    aggressive_conditions = []
    defensive_conditions = []

    if pd.notna(bias_percentile) and bias_percentile < 0.30:
        aggressive_count += 1
        aggressive_conditions.append("bias分位<0.30")
    if breadth < 0.40:
        aggressive_count += 1
        aggressive_conditions.append("breadth<0.40")
    if momentum < 0:
        aggressive_count += 1
        aggressive_conditions.append("momentum<0")

    if pd.notna(bias_percentile) and bias_percentile > 0.80:
        defensive_count += 1
        defensive_conditions.append("bias分位>0.80")
    if breadth > 0.70:
        defensive_count += 1
        defensive_conditions.append("breadth>0.70")
    if momentum > 0.50:
        defensive_count += 1
        defensive_conditions.append("momentum>0.50")

    if aggressive_count >= 2:
        reason = f"进攻信号满足 {aggressive_count}/3: {', '.join(aggressive_conditions)}"
        return MarketRegime.AGGRESSIVE, reason
    elif defensive_count >= 2:
        reason = f"防守信号满足 {defensive_count}/3: {', '.join(defensive_conditions)}"
        return MarketRegime.DEFENSIVE, reason
    else:
        reason = f"无显著信号：进攻{aggressive_count}，防守{defensive_count}，均未达到2/3阈值"
        return MarketRegime.BALANCED, reason


def detect_regime_history(proxy_df: pd.DataFrame) -> pd.DataFrame:
    proxy_df = proxy_df.copy().sort_values("trade_date").reset_index(drop=True)
    proxy_df["bias_percentile"] = _rolling_percentile_min_rank(
        proxy_df["market_composite_bias"],
        window=252,
        min_periods=60,
    )

    regimes = []
    reasons = []
    for _, row in proxy_df.iterrows():
        r, reason = detect_regime_single(
            bias=row["market_composite_bias"],
            bias_percentile=row["bias_percentile"],
            momentum=row["market_momentum_score"],
            breadth=row["market_breadth_score"],
        )
        if pd.isna(row["bias_percentile"]):
            reason = f"{reason}（bias分位=NA: 历史窗口不足）"
        regimes.append(r)
        reasons.append(reason)

    proxy_df["regime"] = regimes
    proxy_df["reason"] = reasons
    return proxy_df


def get_latest_regime(conn: sqlite3.Connection, *, confirm_days: int = 5) -> RegimeDetection:
    proxy_df = _load_market_proxy(conn)
    proxy_df = detect_regime_history(proxy_df)
    confirmed = compute_confirmed_regime_series(proxy_df["regime"], confirm_days=int(confirm_days))
    latest = proxy_df.iloc[-1]
    confirmed_latest = confirmed.iloc[-1]
    confirmed_latest = confirmed_latest if isinstance(confirmed_latest, MarketRegime) else MarketRegime(str(confirmed_latest))
    raw_latest = latest["regime"] if isinstance(latest["regime"], MarketRegime) else MarketRegime(str(latest["regime"]))
    return RegimeDetection(
        trade_date=str(latest["trade_date"]),
        regime=confirmed_latest,
        raw_regime=raw_latest,
        bias=latest["market_composite_bias"],
        bias_percentile=latest["bias_percentile"],
        momentum=latest["market_momentum_score"],
        breadth=latest["market_breadth_score"],
        reason=latest["reason"],
    )


def get_portfolio_weights(regime: MarketRegime) -> Dict[str, float]:
    return PORTFOLIO_WEIGHTS[regime].copy()


def _load_etf_returns(conn: sqlite3.Connection, etf_codes: List[str]) -> pd.DataFrame:
    codes_str = ", ".join(f"'{c}'" for c in etf_codes)
    query = f"""
        SELECT trade_date, etf_code, daily_change
        FROM etf_daily_metrics
        WHERE etf_code IN ({codes_str})
        ORDER BY trade_date
    """
    df = pd.read_sql_query(query, conn)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    pivot = df.pivot(index="trade_date", columns="etf_code", values="daily_change").reset_index()
    pivot.columns.name = None
    return pivot


def get_target_weight_range(regime: MarketRegime) -> Dict[str, Tuple[float, float]]:
    return WEIGHT_RANGES[regime].copy()


def adjust_position(
    current_weight: float,
    target_min: float,
    target_max: float,
    max_step: float,
) -> float:
    if target_min <= current_weight <= target_max:
        return current_weight
    target = max(target_min, min(target_max, current_weight))
    delta = target - current_weight
    if abs(delta) > max_step:
        return current_weight + max_step * (1 if delta > 0 else -1)
    return target


def backtest_portfolio(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
    rebalance_freq: str = "monthly",
    regime_confirm_days: int = 5,
    max_step: float = 0.1,
) -> Dict[str, Any]:
    etf_codes = ["159361", "159209", "515080"]
    proxy_df = _load_market_proxy(conn)
    proxy_df = detect_regime_history(proxy_df)

    returns_df = _load_etf_returns(conn, etf_codes)

    merged = pd.merge(returns_df, proxy_df, on="trade_date", how="inner")
    merged = merged.sort_values("trade_date").reset_index(drop=True)
    merged = merged[merged["trade_date"].notna()].reset_index(drop=True)
    merged = merged[(merged["trade_date"] >= pd.to_datetime(start_date).date())]
    merged = merged[(merged["trade_date"] <= pd.to_datetime(end_date).date())]
    merged = merged.reset_index(drop=True)

    if len(merged) == 0:
        raise ValueError("No data in date range")

    merged["weekday"] = pd.to_datetime(merged["trade_date"]).dt.weekday
    if rebalance_freq == "weekly":
        merged["is_rebalance_time"] = (merged["weekday"] == 0)
    elif rebalance_freq == "monthly":
        merged["month"] = pd.to_datetime(merged["trade_date"]).dt.month
        merged["year"] = pd.to_datetime(merged["trade_date"]).dt.year
        merged["is_rebalance_time"] = (merged["month"] != merged["month"].shift(1))
        merged.loc[0, "is_rebalance_time"] = True
    else:
        merged["is_rebalance_time"] = True

    for code in etf_codes:
        merged[f"{code}_ret"] = merged[code].fillna(0.0)

    confirmed_series = compute_confirmed_regime_series(
        merged["regime"],
        confirm_days=int(regime_confirm_days),
    )
    current_weights = PORTFOLIO_WEIGHTS[MarketRegime.BALANCED].copy()
    confirmed_regime = MarketRegime.BALANCED

    portfolio_returns = []
    rebalance_count = 0
    actual_rebalance_count = 0
    turnover_total = 0.0
    step_adjustments = []
    regime_switch_count = 0
    regime_history = []

    for i, row in merged.iterrows():
        if i == 0:
            portfolio_returns.append(0.0)
            regime_history.append({
                "trade_date": row["trade_date"],
                "regime": row["regime"],
                "confirmed_regime": confirmed_regime,
                "weights": current_weights.copy(),
            })
            continue
        current_raw_regime = row["regime"] if isinstance(row["regime"], MarketRegime) else MarketRegime(str(row["regime"]))
        confirmed_now = confirmed_series.iloc[i]
        confirmed_now = confirmed_now if isinstance(confirmed_now, MarketRegime) else MarketRegime(str(confirmed_now))
        if confirmed_now != confirmed_regime:
            confirmed_regime = confirmed_now
            regime_switch_count += 1

        need_rebalance = False
        target_weights = {}
        if row["is_rebalance_time"]:
            weight_range = get_target_weight_range(confirmed_regime)
            a500_min, a500_max = weight_range["159361"]
            current_a500 = current_weights["159361"]

            if not (a500_min <= current_a500 <= a500_max):
                new_a500 = adjust_position(current_a500, a500_min, a500_max, max_step)
                a500_delta = new_a500 - current_a500
                target_weights = {
                    "159361": new_a500,
                    "159209": 0.30,
                    "515080": 1.0 - new_a500 - 0.30,
                }
                step_adjustments.append(abs(a500_delta))
                need_rebalance = True

        if need_rebalance:
            turnover = sum(
                abs(target_weights[code] - current_weights.get(code, 0.0))
                for code in etf_codes
            )
            if turnover > 0.001:
                turnover_total += turnover
                current_weights = target_weights
                actual_rebalance_count += 1
            rebalance_count += 1

        daily_ret = sum(
            current_weights[code] * (row[f"{code}_ret"] or 0.0)
            for code in etf_codes
        )
        portfolio_returns.append(daily_ret)
        regime_history.append({
            "trade_date": row["trade_date"],
            "regime": row["regime"],
            "confirmed_regime": confirmed_regime,
            "weights": current_weights.copy(),
        })

    merged["portfolio_return"] = portfolio_returns
    merged["portfolio_cum"] = (1 + merged["portfolio_return"]).cumprod()

    baseline_equal = merged[etf_codes].fillna(0.0).mean(axis=1)
    baseline_balanced = (
        merged["159361"].fillna(0.0) * 0.35
        + merged["159209"].fillna(0.0) * 0.30
        + merged["515080"].fillna(0.0) * 0.35
    )

    total_return = merged["portfolio_cum"].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(merged)) - 1

    def max_drawdown(series):
        cum = (1 + series).cumprod()
        running_max = cum.expanding().max()
        dd = (cum - running_max) / running_max
        return dd.min()

    portfolio_max_dd = max_drawdown(merged["portfolio_return"].iloc[1:])

    risk_free = 0.03
    excess = merged["portfolio_return"].iloc[1:] - risk_free / 252
    sharpe = excess.mean() / excess.std() * math.sqrt(252) if excess.std() > 0 else 0.0

    baseline_equal_cum = (1 + baseline_equal.iloc[1:]).cumprod()
    baseline_equal_total = baseline_equal_cum.iloc[-1] - 1
    baseline_equal_annual = (1 + baseline_equal_total) ** (252 / len(baseline_equal.iloc[1:])) - 1
    baseline_equal_dd = max_drawdown(baseline_equal.iloc[1:])

    baseline_balanced_cum = (1 + baseline_balanced.iloc[1:]).cumprod()
    baseline_balanced_total = baseline_balanced_cum.iloc[-1] - 1
    baseline_balanced_annual = (1 + baseline_balanced_total) ** (252 / len(baseline_balanced.iloc[1:])) - 1
    baseline_balanced_dd = max_drawdown(baseline_balanced.iloc[1:])

    regime_counts = confirmed_series.value_counts().to_dict()

    td_non_null = merged["trade_date"].dropna()
    if td_non_null.empty:
        raise ValueError("No valid trade_date after merge/filter")
    start_dt = td_non_null.iloc[0]
    end_dt = td_non_null.iloc[-1]
    if hasattr(start_dt, "isoformat"):
        start_s = start_dt.isoformat()
    else:
        start_s = str(start_dt)
    if hasattr(end_dt, "isoformat"):
        end_s = end_dt.isoformat()
    else:
        end_s = str(end_dt)

    return {
        "portfolio": {
            "total_return": total_return,
            "annualized_return": annual_return,
            "max_drawdown": portfolio_max_dd,
            "sharpe_ratio": sharpe,
            "rebalance_count": rebalance_count,
            "actual_rebalance_count": actual_rebalance_count,
            "regime_switch_count": regime_switch_count,
            "turnover_total": turnover_total,
            "average_turnover": turnover_total / actual_rebalance_count if actual_rebalance_count > 0 else 0,
            "average_step": np.mean(step_adjustments) if len(step_adjustments) > 0 else 0,
        },
        "baseline_equal": {
            "total_return": baseline_equal_total,
            "annualized_return": baseline_equal_annual,
            "max_drawdown": baseline_equal_dd,
        },
        "baseline_balanced": {
            "total_return": baseline_balanced_total,
            "annualized_return": baseline_balanced_annual,
            "max_drawdown": baseline_balanced_dd,
        },
        "regime_history": regime_history,
        "regime_counts": regime_counts,
        "start_date": start_s,
        "end_date": end_s,
        "n_days": len(merged),
    }


@dataclass
class PortfolioAction:
    trade_date: str
    market_regime: MarketRegime
    raw_regime: MarketRegime
    current_position: Dict[str, float]
    target_range: Dict[str, Tuple[float, float]]
    action: str
    adjustment: Dict[str, float]
    after_position: Dict[str, float]
    reason: str


def compute_portfolio_action(
    conn: sqlite3.Connection,
    current_a500: float,
    current_dividend: float,
    current_dividend_growth: float,
    regime_confirm_days: int = 5,
    max_step: float = 0.1,
) -> PortfolioAction:
    current_position = {
        "159361": current_a500,
        "515080": current_dividend,
        "159209": current_dividend_growth,
    }

    proxy_df = _load_market_proxy(conn)
    proxy_df = detect_regime_history(proxy_df)
    proxy_df = proxy_df.sort_values("trade_date").reset_index(drop=True)

    confirmed_series = compute_confirmed_regime_series(
        proxy_df["regime"],
        confirm_days=int(regime_confirm_days),
    )
    confirmed_regime = confirmed_series.iloc[-1] if len(confirmed_series) else MarketRegime.BALANCED
    confirmed_regime = confirmed_regime if isinstance(confirmed_regime, MarketRegime) else MarketRegime(str(confirmed_regime))
    latest_row = proxy_df.iloc[-1]
    trade_date = str(latest_row["trade_date"])
    raw_regime = latest_row["regime"] if isinstance(latest_row["regime"], MarketRegime) else MarketRegime(str(latest_row["regime"]))

    weight_range = get_target_weight_range(confirmed_regime)
    target_range = {
        "159361": weight_range["159361"],
        "159209": weight_range["159209"],
    }

    a500_min, a500_max = weight_range["159361"]
    current_a500_val = current_position["159361"]

    action = "HOLD"
    adjustment = {"159361": 0.0, "515080": 0.0, "159209": 0.0}
    after_position = current_position.copy()
    reason = ""

    if a500_min <= current_a500_val <= a500_max:
        reason = f"当前持仓已在 {confirmed_regime.value} 目标区间内，无需调整"
    else:
        new_a500 = adjust_position(current_a500_val, a500_min, a500_max, max_step)
        a500_delta = new_a500 - current_a500_val

        adjustment = {
            "159361": a500_delta,
            "515080": -a500_delta,
            "159209": 0.0,
        }

        after_position = {
            "159361": current_a500_val + a500_delta,
            "515080": current_position["515080"] - a500_delta,
            "159209": current_position["159209"],
        }

        action = "REBALANCE"
        reason = f"当前 A500 权重 {current_a500_val:.2%} 不在 {confirmed_regime.value} 目标区间 [{a500_min:.0%}, {a500_max:.0%}] 内"

    return PortfolioAction(
        trade_date=trade_date,
        market_regime=confirmed_regime,
        raw_regime=raw_regime,
        current_position=current_position,
        target_range=target_range,
        action=action,
        adjustment=adjustment,
        after_position=after_position,
        reason=reason,
    )
