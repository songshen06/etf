"""
Two-layer recommendation: (1) framework fit for the NEG/LOW/HIGH playbook, (2) best signal setup in a fixed grid.

Does not alter signal or event-study math — only ranks candidates and suggests defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .event_study import tier_event_studies
from .exit_rules import (
    MIN_TRADES_EXIT_EVAL,
    ExitContext,
    ExitRuleEvalRow,
    ExitRuleSpec,
    build_exit_context,
    exit_rule_candidate_specs,
    exit_rule_display_name,
    exit_rule_plain_explanation,
    score_exit_metrics,
)
from .indicators import add_indicators
from .signal_dimensions import (
    BIAS_MA_WINDOWS,
    MOMENTUM_WINDOWS,
    VOLUME_MA_WINDOWS,
    bias_column,
    momentum_column,
    volume_ratio_column,
)
from .position_rules import weights_by_strategy_profile
from .signal_engine import SignalMode, apply_signals

FitLevel = Literal["high", "medium", "low"]

SIGNAL_TIERS = ("NEG", "NEG_LOW", "NEG_LOW_HIGH")


def recommended_position_profile_for(fit: FitLevel, default_signal: str) -> str:
    """Rule-based仓位画像：与网格最优信号、框架适配度联动（非回测引擎逻辑）。"""
    if fit == "low":
        return "defensive"
    if fit == "high" and default_signal == "NEG_LOW_HIGH":
        return "aggressive"
    return "balanced"


SIGNAL_TIERS = ("NEG", "NEG_LOW", "NEG_LOW_HIGH")
SEARCH_MODES = ("full_sample", "rolling")
SEARCH_BIAS = (60, 120, 250)
SEARCH_HORIZONS = (20, 60, 120)


@dataclass
class SignalSetupCandidate:
    rank: int
    signal: str
    mode: str
    bias_ma: int
    horizon_focus: int
    recommendation_score: float
    mean_return_60: float | None = None
    mean_return_120: float | None = None
    win_rate_60: float | None = None
    n_60: int | None = None
    std_60: float | None = None
    monotonicity_bonus: float = 0.0
    score_raw: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("score_raw", None)
        return d


@dataclass
class RecommendationResult:
    code: str
    fit_level: FitLevel
    framework_fit_note: str
    best_signal_setup: SignalSetupCandidate
    top_candidates: list[SignalSetupCandidate]
    notes: list[str] = field(default_factory=list)
    # Mirrors for auto-defaults and legacy callers
    default_signal: str = ""
    recommended_bias_ma: int = 120
    recommended_mode: str = "rolling"
    recommended_horizon_focus: int = 60
    recommended_signals: list[str] = field(default_factory=list)
    recommended_position_profile: str = "balanced"
    # Exit rule sweep (fixed entry = best grid setup; no joint entry/exit search)
    best_exit_rule: ExitRuleEvalRow | None = None
    exit_rule_candidates: list[ExitRuleEvalRow] = field(default_factory=list)
    exit_rule_explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "fit_level": self.fit_level,
            "framework_fit_note": self.framework_fit_note,
            "best_signal_setup": self.best_signal_setup.to_dict(),
            "top_candidates": [c.to_dict() for c in self.top_candidates],
            "notes": list(self.notes),
            "default_signal": self.default_signal,
            "recommended_bias_ma": self.recommended_bias_ma,
            "recommended_mode": self.recommended_mode,
            "recommended_horizon_focus": self.recommended_horizon_focus,
            "recommended_signals": list(self.recommended_signals),
            "recommended_position_profile": self.recommended_position_profile,
            "best_exit_rule": self.best_exit_rule.to_dict() if self.best_exit_rule else None,
            "exit_rule_candidates": [c.to_dict() for c in self.exit_rule_candidates],
            "exit_rule_explanation": self.exit_rule_explanation,
        }


def _annualized_vol(daily_simple_rets: pd.Series) -> float:
    r = daily_simple_rets.dropna()
    if len(r) < 20:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(252.0))


def _max_drawdown(close: pd.Series) -> float:
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) < 2:
        return float("nan")
    peak = c.cummax()
    dd = c / peak - 1.0
    return float(dd.min())


def _cagr(close: pd.Series, trading_days_per_year: float = 252.0) -> float:
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) < 2:
        return float("nan")
    total = float(c.iloc[-1] / c.iloc[0] - 1.0)
    years = len(c) / trading_days_per_year
    if years <= 0 or total <= -1.0:
        return float("nan")
    return float((1.0 + total) ** (1.0 / years) - 1.0)


def _row_at_horizon(tbl: pd.DataFrame, h: int) -> dict[str, Any] | None:
    row = tbl.loc[tbl["horizon"] == int(h)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def _framework_fit_level(
    *,
    ann_vol: float,
    cagr: float,
    mdd: float,
) -> FitLevel:
    high_vol = np.isfinite(ann_vol) and ann_vol > 0.28
    low_vol = np.isfinite(ann_vol) and ann_vol < 0.18
    if high_vol and not low_vol:
        return "low"
    if low_vol and np.isfinite(cagr) and cagr > 0:
        return "high"
    return "medium"


def rank_exit_rules_on_frame(
    d: pd.DataFrame,
    *,
    weights_by_tier: dict[int, float],
    bundle_hold_days: int,
    signal_mode: str,
    bias_ma: int,
    momentum_window: int,
    volume_ma_window: int,
    rolling_window: int,
    quantile_low: float,
    quantile_high: float,
    entry_signal_tier: str | None = None,
) -> tuple[list[ExitRuleEvalRow], str, ExitRuleEvalRow | None]:
    """固定入场 DataFrame，对 :func:`~exit_rules.exit_rule_candidate_specs` 列表各跑一次回测，按 ``score_exit_metrics`` 降序排名（与 optimize-exit / compare-exit-rules 一致）。"""
    from .portfolio_backtest import run_portfolio_backtest
    from .signal_engine import research_tier_mask

    def _run_exit_backtest(sp: ExitRuleSpec, ctx: ExitContext | None):
        if entry_signal_tier is None:
            return run_portfolio_backtest(d, weights_by_tier, exit_rule=sp, exit_context=ctx)
        code = {"NEG": 1, "NEG_LOW": 2, "NEG_LOW_HIGH": 3}[entry_signal_tier]
        w = float(weights_by_tier[code])
        mask = research_tier_mask(d, entry_signal_tier)
        return run_portfolio_backtest(
            d,
            None,
            exit_rule=sp,
            exit_context=ctx,
            entry_mask=mask,
            entry_weight=w,
            research_signal_tier=int(code),
        )

    scored: list[tuple[ExitRuleSpec, dict[str, float], int, float, bool]] = []
    for sp in exit_rule_candidate_specs(int(bundle_hold_days)):
        ctx: ExitContext | None = (
            None
            if sp.kind == "time_hold"
            else build_exit_context(
                d,
                signal_mode=signal_mode,
                bias_ma_window=bias_ma,
                momentum_window=momentum_window,
                volume_ma_window=volume_ma_window,
                rolling_window=rolling_window,
                quantile_low=quantile_low,
                quantile_high=quantile_high,
                spec=sp,
            )
        )
        res = _run_exit_backtest(sp, ctx)
        ntr = len(res.trades)
        met = {**res.metrics}
        avg_exp = float(res.exposure.mean()) if len(res.exposure) else float("nan")
        met["average_exposure"] = avg_exp
        if res.trades:
            prs = [float(t.portfolio_return) for t in res.trades]
            hds = [int(t.holding_days) for t in res.trades]
            met["avg_trade_return"] = float(np.mean(prs))
            met["avg_holding_days"] = float(np.mean(hds))
            met["median_holding_days"] = float(np.median(hds))
            met["win_rate"] = float(sum(1.0 for x in prs if x > 0.0) / len(prs))
        else:
            met["avg_trade_return"] = float("nan")
            met["avg_holding_days"] = float("nan")
            met["median_holding_days"] = float("nan")
            met["win_rate"] = float("nan")
        eligible = ntr >= MIN_TRADES_EXIT_EVAL
        sc = score_exit_metrics(met) if eligible else -1e12
        scored.append((sp, met, ntr, sc, eligible))

    scored.sort(key=lambda x: x[3], reverse=True)
    out: list[ExitRuleEvalRow] = []
    for rank, (sp, met, ntr, sc, el) in enumerate(scored, start=1):
        clean_met: dict[str, float | int | None] = {}
        for k, v in met.items():
            if isinstance(v, float) and v != v:
                clean_met[k] = None
            else:
                clean_met[k] = v
        dn = exit_rule_display_name(sp.rule_id, sp)
        pe = exit_rule_plain_explanation(sp.rule_id, sp)
        out.append(
            ExitRuleEvalRow(
                rank=rank,
                spec=sp,
                label_zh=dn,
                display_name=dn,
                plain_explanation=pe,
                metrics=clean_met,
                score=float(sc),
                n_trades=ntr,
                eligible=el,
            )
        )
    best = next((r for r in out if r.eligible), None)
    if entry_signal_tier:
        entry_note = (
            f"在固定入场层 {entry_signal_tier.replace('_', '+')}、固定 mode/乖离下对各退出规则独立回测"
        )
    else:
        entry_note = "在固定推荐入场（信号×mode×乖离，分层仓位）下对各退出规则独立回测"
    expl = (
        f"{entry_note}；候选含 hold_fixed（{int(bundle_hold_days)} 日）与默认动态/时间规则。"
        f"排序键为 score_exit_metrics（年化、Sharpe、回撤加权）；成交<{MIN_TRADES_EXIT_EVAL}笔的候选标为 ineligible 且不参与优选。"
        "未与入场联合搜索，结论为样本内排序。"
    )
    if best:
        expl += f" 当前优选：{best.display_name}（id={best.spec.rule_id}）。"
    else:
        expl += " 无满足最小样本的退出规则。"
    return out, expl, best


def _build_framework_note(
    fit: FitLevel,
    *,
    ann_vol: float,
    cagr: float,
    mdd: float,
) -> str:
    parts: list[str] = []
    if fit == "high":
        parts.append("该基金波动相对温和、长期价格行为与本框架（弱势区布局）较为匹配。")
    elif fit == "low":
        parts.append(
            "该基金波动或回撤特征偏激进，作为本框架的「核心持仓」匹配度偏低；"
            "但仍可在全网格中选出样本内统计上相对更优的信号参数。"
        )
    else:
        parts.append("框架匹配度中等：建议同时对照分层事件表与下方推荐参数，避免默认等同其他 ETF。")
    if np.isfinite(cagr):
        parts.append(f"长期年化约 {cagr * 100:.1f}%。")
    if np.isfinite(ann_vol):
        parts.append(f"年化波动约 {ann_vol * 100:.1f}%。")
    if np.isfinite(mdd):
        parts.append(f"历史最大回撤约 {mdd * 100:.2f}%。")
    return " ".join(parts)


def recommend_strategy_setup(
    df: pd.DataFrame,
    code: str,
    *,
    available_bias_windows: tuple[int, ...] = SEARCH_BIAS,
    available_modes: tuple[str, ...] = SEARCH_MODES,
    momentum_window: int = 10,
    volume_ma_window: int = 20,
    rolling_window: int = 252,
    quantile_low: float = 0.33,
    quantile_high: float = 0.67,
    eval_horizon: int = 60,
    top_k: int = 12,
    include_exit_rules: bool = False,
) -> RecommendationResult:
    """
    Layer 1: framework fit (drift / vol / drawdown / style heuristic).

    Layer 2: rank all (signal × mode × bias_ma × horizon_focus) using rule-based score
    (mean_return at 60/120, win_rate at 60, sample and volatility penalties, monotonicity bonus).
    Low framework fit does **not** force NEG — the best grid cell is still reported.
    """
    notes: list[str] = []
    bw = tuple(int(w) for w in available_bias_windows if int(w) in BIAS_MA_WINDOWS)
    if not bw:
        bw = tuple(x for x in SEARCH_BIAS if x in BIAS_MA_WINDOWS)
    modes: list[SignalMode] = []
    for m in available_modes:
        if m == "full_sample" or m == "rolling":
            modes.append(m)  # type: ignore[arg-type]
    if not modes:
        modes = ["rolling", "full_sample"]  # type: ignore[list-item]

    if momentum_window not in MOMENTUM_WINDOWS:
        momentum_window = 10
    if volume_ma_window not in VOLUME_MA_WINDOWS:
        volume_ma_window = 20

    close = pd.to_numeric(df["close"], errors="coerce")
    rets = close.pct_change()
    ann_vol = _annualized_vol(rets)
    mdd = _max_drawdown(close)
    cagr = _cagr(close)

    fit = _framework_fit_level(ann_vol=ann_vol, cagr=cagr, mdd=mdd)
    framework_fit_note = _build_framework_note(fit, ann_vol=ann_vol, cagr=cagr, mdd=mdd)

    if np.isfinite(cagr) and cagr < 0:
        notes.append("样本内长期复合收益为负，事件研究均值需谨慎解读。")
    if np.isfinite(ann_vol) and ann_vol > 0.28:
        notes.append("高波动品种：分层信号噪音更大，推荐以样本量与稳定性字段为主。")

    bias_grid = tuple(x for x in bw if x in BIAS_MA_WINDOWS)
    panel = add_indicators(
        df.copy(),
        momentum_windows=(int(momentum_window),),
        volume_ma_windows=(int(volume_ma_window),),
        bias_windows=bias_grid,
    )

    # (mode_str, bias_ma) -> tier -> event study table (horizons 20,60,120)
    study_cache: dict[tuple[str, int], dict[str, pd.DataFrame]] = {}
    frame_cache: dict[tuple[str, int], pd.DataFrame] = {}
    mono_at_60: dict[tuple[str, int], tuple[bool, float]] = {}

    eps = 0.002
    for mode in modes:
        for bmw in bias_grid:
            if bias_column(bmw) not in panel.columns:
                continue
            key = (str(mode), int(bmw))
            try:
                d = apply_signals(
                    panel.copy(),
                    mode=mode,
                    bias_ma_window=bmw,
                    momentum_col=momentum_column(momentum_window),
                    volume_col=volume_ratio_column(volume_ma_window),
                    rolling_window=rolling_window,
                    quantile_low=quantile_low,
                    quantile_high=quantile_high,
                )
                studies = tier_event_studies(d, horizons=SEARCH_HORIZONS)
            except Exception:
                continue
            frame_cache[key] = d
            study_cache[key] = studies
            means: dict[str, float] = {}
            for tier in SIGNAL_TIERS:
                r = _row_at_horizon(studies[tier], 60)
                if not r or r.get("mean_return") is None:
                    means[tier] = float("nan")
                else:
                    v = float(r["mean_return"])
                    means[tier] = v if np.isfinite(v) else float("nan")
            mn, ml, mh = means["NEG"], means["NEG_LOW"], means["NEG_LOW_HIGH"]
            mono = (
                np.isfinite(mn)
                and np.isfinite(ml)
                and np.isfinite(mh)
                and mn + eps <= ml
                and ml + eps <= mh
            )
            bonus = 0.25 if mono else 0.0
            mono_at_60[key] = (mono, bonus)

    if not study_cache:
        notes.append("无法在网格上完成信号评估，退回 rolling / MA120 / NEG / horizon 60。")
        fb = SignalSetupCandidate(
            rank=1,
            signal="NEG",
            mode="rolling",
            bias_ma=120,
            horizon_focus=60,
            recommendation_score=0.0,
            mean_return_60=None,
            mean_return_120=None,
            win_rate_60=None,
            n_60=None,
            std_60=None,
            monotonicity_bonus=0.0,
            score_raw=0.0,
        )
        return RecommendationResult(
            code=str(code),
            fit_level="medium",
            framework_fit_note=framework_fit_note,
            best_signal_setup=fb,
            top_candidates=[fb],
            notes=notes,
            default_signal=fb.signal,
            recommended_bias_ma=fb.bias_ma,
            recommended_mode=fb.mode,
            recommended_horizon_focus=fb.horizon_focus,
            recommended_signals=["NEG", "NEG_LOW", "NEG_LOW_HIGH"],
            recommended_position_profile=recommended_position_profile_for("medium", fb.signal),
            best_exit_rule=None,
            exit_rule_candidates=[],
            exit_rule_explanation="",
        )

    notes.append(
        f"候选空间：信号 {len(SIGNAL_TIERS)} × 模式 {len(modes)} × 乖离窗 {len(bias_grid)} × 持有期 {len(SEARCH_HORIZONS)}；"
        "评分主用 h=60、120 的 mean_return 与 h=60 的 win_rate，并含样本量/波动惩罚、（mode,bias）下三层单调性加成；"
        "深跌 ETF 另加轻度全局回撤惩罚；无机器学习。"
    )
    if int(eval_horizon) not in SEARCH_HORIZONS:
        notes.append(
            f"API 传入 eval_horizon={eval_horizon}；当前网格持有期固定为 {list(SEARCH_HORIZONS)}，未使用其它 horizon。"
        )

    raw_rows: list[dict[str, Any]] = []

    for (mode_str, bmw), studies in study_cache.items():
        _, mono_bonus = mono_at_60.get((mode_str, bmw), (False, 0.0))
        for tier in SIGNAL_TIERS:
            tbl = studies[tier]
            for h_focus in SEARCH_HORIZONS:
                r60 = _row_at_horizon(tbl, 60)
                r120 = _row_at_horizon(tbl, 120)
                rf = _row_at_horizon(tbl, h_focus)
                if r60 is None:
                    continue
                mr60 = r60.get("mean_return")
                mr120 = r120.get("mean_return") if r120 else None
                wr60 = r60.get("win_rate")
                n_raw = r60.get("n")
                try:
                    n60 = int(n_raw) if n_raw is not None and float(n_raw) == float(n_raw) else 0
                except (TypeError, ValueError):
                    n60 = 0
                std60 = r60.get("std")
                mr60f = float(mr60) if mr60 == mr60 and np.isfinite(float(mr60)) else float("nan")
                mr120f = float(mr120) if mr120 and mr120 == mr120 and np.isfinite(float(mr120)) else float("nan")
                wr60f = float(wr60) if wr60 == wr60 and np.isfinite(float(wr60)) else float("nan")
                std60f = float(std60) if std60 == std60 and np.isfinite(float(std60)) else float("nan")

                score_raw = 0.0
                if np.isfinite(mr60f):
                    score_raw += mr60f * 100.0
                if np.isfinite(mr120f):
                    score_raw += mr120f * 55.0
                if np.isfinite(wr60f):
                    score_raw += (wr60f - 0.5) * 35.0
                score_raw += float(np.log(max(n60, 1))) * 0.12
                if n60 < 25:
                    score_raw -= (25 - n60) * 0.04
                if n60 < 8:
                    score_raw -= 1.5
                if np.isfinite(std60f) and std60f > 0:
                    score_raw -= min(std60f * 22.0, 1.8)
                score_raw += mono_bonus
                if np.isfinite(mdd) and mdd < -0.45:
                    score_raw -= 0.15
                if np.isfinite(rf.get("mean_return")):
                    mrf = float(rf["mean_return"])
                    if np.isfinite(mrf):
                        score_raw += mrf * 15.0

                raw_rows.append(
                    {
                        "signal": tier,
                        "mode": mode_str,
                        "bias_ma": int(bmw),
                        "horizon_focus": int(h_focus),
                        "mean_return_60": mr60f if np.isfinite(mr60f) else None,
                        "mean_return_120": mr120f if np.isfinite(mr120f) else None,
                        "win_rate_60": wr60f if np.isfinite(wr60f) else None,
                        "n_60": n60 if n60 > 0 else None,
                        "std_60": std60f if np.isfinite(std60f) else None,
                        "monotonicity_bonus": mono_bonus,
                        "score_raw": score_raw,
                    }
                )

    if not raw_rows:
        notes.append("无有效候选行，使用保守默认。")
        fb = SignalSetupCandidate(
            rank=1,
            signal="NEG",
            mode="rolling",
            bias_ma=120,
            horizon_focus=60,
            recommendation_score=0.0,
            monotonicity_bonus=0.0,
            score_raw=0.0,
        )
        return RecommendationResult(
            code=str(code),
            fit_level=fit,
            framework_fit_note=framework_fit_note,
            best_signal_setup=fb,
            top_candidates=[fb],
            notes=notes,
            default_signal=fb.signal,
            recommended_bias_ma=fb.bias_ma,
            recommended_mode=fb.mode,
            recommended_horizon_focus=fb.horizon_focus,
            recommended_signals=list(SIGNAL_TIERS),
            recommended_position_profile=recommended_position_profile_for(fit, fb.signal),
            best_exit_rule=None,
            exit_rule_candidates=[],
            exit_rule_explanation="",
        )

    raw_scores = np.array([r["score_raw"] for r in raw_rows], dtype=float)
    mu = float(np.nanmean(raw_scores))
    sig = float(np.nanstd(raw_scores))
    if not np.isfinite(sig) or sig < 1e-9:
        z = np.zeros_like(raw_scores)
    else:
        z = (raw_scores - mu) / sig

    for i, r in enumerate(raw_rows):
        r["recommendation_score"] = float(z[i])

    raw_rows.sort(key=lambda x: x["recommendation_score"], reverse=True)
    top_n = raw_rows[: max(1, int(top_k))]

    candidates: list[SignalSetupCandidate] = []
    for rank, r in enumerate(top_n, start=1):
        candidates.append(
            SignalSetupCandidate(
                rank=rank,
                signal=str(r["signal"]),
                mode=str(r["mode"]),
                bias_ma=int(r["bias_ma"]),
                horizon_focus=int(r["horizon_focus"]),
                recommendation_score=float(r["recommendation_score"]),
                mean_return_60=r.get("mean_return_60"),
                mean_return_120=r.get("mean_return_120"),
                win_rate_60=r.get("win_rate_60"),
                n_60=r.get("n_60"),
                std_60=r.get("std_60"),
                monotonicity_bonus=float(r.get("monotonicity_bonus", 0.0)),
                score_raw=float(r["score_raw"]),
            )
        )

    best = candidates[0]
    explore: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c.signal not in seen:
            seen.add(c.signal)
            explore.append(c.signal)
    for t in SIGNAL_TIERS:
        if t not in seen:
            explore.append(t)

    exit_best: ExitRuleEvalRow | None = None
    exit_cands: list[ExitRuleEvalRow] = []
    exit_expl = ""
    if include_exit_rules:
        d_best = frame_cache.get((str(best.mode), int(best.bias_ma)))
        if d_best is None:
            try:
                d_best = apply_signals(
                    panel.copy(),
                    mode=str(best.mode),  # type: ignore[arg-type]
                    bias_ma_window=int(best.bias_ma),
                    momentum_col=momentum_column(momentum_window),
                    volume_col=volume_ratio_column(volume_ma_window),
                    rolling_window=rolling_window,
                    quantile_low=quantile_low,
                    quantile_high=quantile_high,
                )
            except Exception:
                d_best = None
        if d_best is not None:
            wmap = weights_by_strategy_profile(recommended_position_profile_for(fit, best.signal))
            exit_cands, exit_expl, exit_best = rank_exit_rules_on_frame(
                d_best,
                weights_by_tier=wmap,
                bundle_hold_days=int(best.horizon_focus),
                signal_mode=str(best.mode),
                bias_ma=int(best.bias_ma),
                momentum_window=int(momentum_window),
                volume_ma_window=int(volume_ma_window),
                rolling_window=int(rolling_window),
                quantile_low=float(quantile_low),
                quantile_high=float(quantile_high),
            )

    return RecommendationResult(
        code=str(code),
        fit_level=fit,
        framework_fit_note=framework_fit_note,
        best_signal_setup=best,
        top_candidates=candidates,
        notes=notes,
        default_signal=best.signal,
        recommended_bias_ma=best.bias_ma,
        recommended_mode=best.mode,
        recommended_horizon_focus=best.horizon_focus,
        recommended_signals=explore,
        recommended_position_profile=recommended_position_profile_for(fit, best.signal),
        best_exit_rule=exit_best,
        exit_rule_candidates=exit_cands,
        exit_rule_explanation=exit_expl,
    )
