"""
Single-ETF portfolio simulation: non-overlapping trades, fixed hold, next-day open entry.

One engine supports:

* **Hierarchical tiers** — integer ``signal_tier`` on the frame + ``weights_by_tier``.
* **Research mask** — boolean ``entry_mask`` + scalar ``entry_weight`` (NEG / NEG_LOW / NEG_LOW_HIGH).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .exit_rules import ExitContext, ExitRuleSpec, should_exit_at_open, time_hold_spec
from .signal_engine import research_tier_mask


@dataclass
class Trade:
    entry_date: object
    exit_date: object
    entry_idx: int
    exit_idx: int
    signal_tier: int
    weight: float
    entry_price: float
    exit_price: float
    stock_return: float
    portfolio_return: float
    holding_days: int


@dataclass
class BacktestResult:
    equity: pd.Series
    daily_returns: pd.Series
    exposure: pd.Series
    trades: list[Trade] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    research_tier_label: str | None = None


def _performance_metrics(
    equity: pd.Series,
    daily_returns: pd.Series,
    *,
    trading_days_per_year: int = 252,
    risk_free_rate_annual: float = 0.0,
) -> dict[str, float]:
    eq = equity.astype(float)
    if eq.empty or not np.isfinite(eq.iloc[0]) or eq.iloc[0] == 0:
        return {
            "total_return": float("nan"),
            "annualized_return": float("nan"),
            "max_drawdown": float("nan"),
            "sharpe_ratio": float("nan"),
            "calmar_ratio": float("nan"),
            "n_days": float(len(eq)),
        }

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    n = len(eq)
    years = n / trading_days_per_year
    if years > 0 and total_return > -1:
        annualized_return = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0)
    else:
        annualized_return = float("nan")

    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_drawdown = float(dd.min())

    rf_daily = risk_free_rate_annual / trading_days_per_year
    excess = daily_returns.dropna() - rf_daily
    std = float(excess.std(ddof=1))
    if std > 1e-12 and len(excess) > 1:
        sharpe = float(np.sqrt(trading_days_per_year) * excess.mean() / std)
    else:
        sharpe = float("nan")

    mdd_abs = abs(max_drawdown) if max_drawdown < 0 else float("nan")
    if np.isfinite(annualized_return) and mdd_abs > 1e-12:
        calmar = float(annualized_return / mdd_abs)
    else:
        calmar = float("nan")

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "n_days": float(n),
    }


def _research_tier_code(tier: str) -> int:
    return {"NEG": 1, "NEG_LOW": 2, "NEG_LOW_HIGH": 3}.get(tier, 0)


def _research_tier_weight(tier: str, pr: Any) -> float:
    if tier == "NEG":
        return min(float(pr.weight_neg), float(pr.max_single_symbol))
    if tier == "NEG_LOW":
        return min(float(pr.weight_neg_low), float(pr.max_single_symbol))
    if tier == "NEG_LOW_HIGH":
        return min(float(pr.weight_neg_low_high), float(pr.max_single_symbol))
    raise ValueError(tier)


def run_portfolio_backtest(
    df: pd.DataFrame,
    weights_by_tier: dict[int, float] | None = None,
    *,
    hold_days: int = 120,
    exit_rule: ExitRuleSpec | None = None,
    exit_context: ExitContext | None = None,
    tier_col: str = "signal_tier",
    entry_mask: pd.Series | None = None,
    entry_weight: float | None = None,
    research_signal_tier: int = 0,
    research_tier_label: str | None = None,
    trading_days_per_year: int = 252,
) -> BacktestResult:
    """
    Signal at EOD *t*; enter at open *t+1*; exit via ``exit_rule`` or fixed ``hold_days``.

    Provide either ``weights_by_tier`` (hierarchical integer tiers on the frame) **or**
    ``entry_mask`` + ``entry_weight`` (research string-tier backtest).

    If ``exit_rule`` is None, uses time-hold only with ``hold_days``.
    Non-``time_hold`` rules require ``exit_context`` aligned with sorted ``df``.
    """
    hier = weights_by_tier is not None
    mask_mode = entry_mask is not None
    if hier == mask_mode:
        raise ValueError("Specify exactly one of: weights_by_tier (dict) or entry_mask (Series)")
    if mask_mode:
        if entry_weight is None or float(entry_weight) <= 0:
            raise ValueError("entry_mask mode requires entry_weight > 0")

    spec = exit_rule if exit_rule is not None else time_hold_spec(int(hold_days))
    if spec.kind != "time_hold" and exit_context is None:
        raise ValueError("exit_context is required when exit_rule is not time_hold")

    d = df.sort_values("date").reset_index(drop=True)
    if "open" not in d.columns:
        d = d.copy()
        d["open"] = d["close"]
    o = d["open"].to_numpy(dtype=float)
    c = d["close"].to_numpy(dtype=float)
    o = np.where(np.isfinite(o) & (o > 0), o, c)
    n = len(d)
    dates = d["date"]

    active = np.zeros(n, dtype=bool)
    w_arr = np.zeros(n, dtype=float)
    st_arr = np.zeros(n, dtype=np.int32)

    if hier:
        assert weights_by_tier is not None
        tier = d[tier_col].to_numpy(dtype=int)
        for i in range(n):
            st = int(tier[i])
            if st >= 1:
                w = float(weights_by_tier.get(st, 0.0))
                if w > 0:
                    active[i] = True
                    w_arr[i] = w
                    st_arr[i] = st
    else:
        em = entry_mask.reindex(d.index).fillna(False).reset_index(drop=True)
        active = em.to_numpy(dtype=bool)
        w_arr[:] = float(entry_weight)
        st_arr[:] = int(research_signal_tier)

    cash = 1.0
    shares = 0.0
    equity = np.zeros(n)
    exposure = np.zeros(n)

    pending: tuple[int, float, int] | None = None
    trades: list[Trade] = []
    open_trade: dict | None = None

    def _exit_at_open(i_bar: int) -> None:
        nonlocal cash, shares, open_trade
        exit_px = o[i_bar]
        cash += shares * exit_px
        if open_trade is not None:
            ep = open_trade["entry_px"]
            eq0 = open_trade["eq_before"]
            st = int(open_trade["tier"])
            w = float(open_trade["weight"])
            stk = exit_px / ep - 1.0 if ep > 0 else float("nan")
            prt = cash / eq0 - 1.0 if eq0 > 0 else float("nan")
            trades.append(
                Trade(
                    entry_date=open_trade["entry_date"],
                    exit_date=dates.iloc[i_bar],
                    entry_idx=int(open_trade["entry_idx"]),
                    exit_idx=i_bar,
                    signal_tier=st,
                    weight=w,
                    entry_price=float(ep),
                    exit_price=float(exit_px),
                    stock_return=float(stk),
                    portfolio_return=float(prt),
                    holding_days=i_bar - int(open_trade["entry_idx"]),
                )
            )
            open_trade = None
        shares = 0.0

    for i in range(n):
        if shares > 0 and open_trade is not None:
            ei = int(open_trade["entry_idx"])
            if should_exit_at_open(i, ei, spec, exit_context):
                _exit_at_open(i)

        if pending is not None and i == pending[0]:
            _, w, st = pending
            nav = cash + shares * o[i]
            invest = nav * w
            if invest > 0 and o[i] > 0:
                shares = invest / o[i]
                cash = nav - invest
                eq_before = float(equity[i - 1]) if i > 0 else 1.0
                open_trade = {
                    "entry_idx": i,
                    "entry_date": dates.iloc[i],
                    "entry_px": o[i],
                    "tier": st,
                    "weight": w,
                    "eq_before": eq_before,
                }
            pending = None

        equity[i] = cash + shares * c[i]
        denom = equity[i]
        exposure[i] = (shares * c[i] / denom) if denom > 1e-15 else 0.0

        flat = shares == 0 and pending is None and open_trade is None
        if flat and i + 1 < n and active[i]:
            pending = (i + 1, float(w_arr[i]), int(st_arr[i]))

    if shares > 0:
        last_i = n - 1
        exit_px = c[last_i]
        cash += shares * exit_px
        shares = 0.0
        equity[last_i] = cash
        exposure[last_i] = 0.0
        if open_trade is not None:
            ep = open_trade["entry_px"]
            eq0 = open_trade["eq_before"]
            st = int(open_trade["tier"])
            w = float(open_trade["weight"])
            stk = exit_px / ep - 1.0 if ep > 0 else float("nan")
            prt = cash / eq0 - 1.0 if eq0 > 0 else float("nan")
            trades.append(
                Trade(
                    entry_date=open_trade["entry_date"],
                    exit_date=dates.iloc[last_i],
                    entry_idx=int(open_trade["entry_idx"]),
                    exit_idx=last_i,
                    signal_tier=st,
                    weight=w,
                    entry_price=float(ep),
                    exit_price=float(exit_px),
                    stock_return=float(stk),
                    portfolio_return=float(prt),
                    holding_days=last_i - int(open_trade["entry_idx"]),
                )
            )

    idx = pd.DatetimeIndex(pd.to_datetime(d["date"]))
    eq_s = pd.Series(equity, index=idx, name="equity")
    dret = eq_s.pct_change().dropna()
    exp_s = pd.Series(exposure, index=idx, name="exposure")
    metrics = _performance_metrics(eq_s, dret)
    return BacktestResult(
        equity=eq_s,
        daily_returns=dret,
        exposure=exp_s,
        trades=trades,
        metrics=metrics,
        research_tier_label=research_tier_label,
    )


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def run_research_portfolio_backtest(
    df: pd.DataFrame,
    cfg: Any,
    tier: str,
    *,
    trading_days_per_year: int = 252,
) -> BacktestResult:
    """
    Research framework: features + NEG/LOW/HIGH, then mask/weight via unified engine.
    """
    from .indicators import compute_research_features
    from .signal_engine import assign_research_signal_flags

    cfg.validate()
    hold = int(cfg.position_rule.holding_days)
    w = _research_tier_weight(tier, cfg.position_rule)

    d = compute_research_features(
        df,
        momentum_window=int(cfg.momentum_window),
        bias_ma_window=int(cfg.bias_ma_window),
        volume_ma_window=int(cfg.volume_ma_window),
        use_precomputed_bias=bool(getattr(cfg, "use_precomputed_bias", False)),
        recompute_bias=bool(getattr(cfg, "recompute_bias", True)),
        precomputed_bias_col=str(getattr(cfg, "precomputed_bias_col", "bias_rate")),
    )
    d = assign_research_signal_flags(
        d,
        signal_mode=cfg.signal_mode,
        signal_rolling_window=int(cfg.signal_rolling_window),
        quantile_low=float(cfg.quantile_low),
        quantile_high=float(cfg.quantile_high),
    )
    if getattr(cfg, "volume_rule", "quantile_top_third") != "quantile_top_third":
        raise NotImplementedError(f"volume_rule {cfg.volume_rule}")

    d = d.sort_values("date").reset_index(drop=True)
    signal = research_tier_mask(d, tier)

    return run_portfolio_backtest(
        d,
        None,
        hold_days=hold,
        entry_mask=signal,
        entry_weight=w,
        research_signal_tier=_research_tier_code(tier),
        research_tier_label=tier,
        trading_days_per_year=trading_days_per_year,
    )


def research_portfolio_result_to_dict(res: BacktestResult) -> dict[str, Any]:
    """JSON shape used by ``research.experiment_runner`` (research-tier backtests)."""
    tier = res.research_tier_label or ""
    s = res.metrics
    summary = {
        "total_return": s.get("total_return"),
        "annualized_return": s.get("annualized_return"),
        "max_drawdown": s.get("max_drawdown"),
        "sharpe_ratio": s.get("sharpe_ratio"),
        "n_days": int(s.get("n_days", 0)) if s.get("n_days") == s.get("n_days") else 0,
    }
    return {
        "kind": "portfolio_backtest",
        "tier": tier,
        "summary": summary,
        "n_trades": len(res.trades),
        "trades": [
            {
                "entry_date": str(t.entry_date),
                "exit_date": str(t.exit_date),
                "weight": t.weight,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stock_return": t.stock_return,
                "portfolio_return_over_trade": t.portfolio_return,
            }
            for t in res.trades
        ],
    }
