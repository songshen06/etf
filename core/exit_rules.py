"""
Pluggable exit rules for portfolio backtests (tactical exits + fixed horizons).

Used by :func:`portfolio_backtest.run_portfolio_backtest` and by the recommendation
layer to rank exit styles **holding entry setup fixed** (no joint entry/exit search).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .signal_dimensions import bias_column, momentum_column, volume_ratio_column
from .state_quality import add_forward_returns, assign_ternary_states, rank_states_by_horizon

ExitKind = Literal[
    "time_hold",
    "state_bottom_k",
    "state_not_top_k",
    "momentum_positive",
    "bias_positive",
    "below_ma20",
    "below_ma60",
]

# Human copy: rule_id -> 展示名 + 白话（与引擎 id 解耦，供 UI/CLI/JSON）
EXIT_RULE_COPY: dict[str, dict[str, str]] = {
    "time_20": {
        "display_name": "固定持有 20 天退出",
        "plain_explanation": "不看市场状态，到达 20 个交易日后在开盘强制平仓。",
    },
    "time_60": {
        "display_name": "固定持有 60 天退出",
        "plain_explanation": "不看市场状态，到达 60 个交易日后强制平仓。",
    },
    "time_120": {
        "display_name": "固定持有 120 天退出",
        "plain_explanation": "不看市场状态，到达 120 个交易日后强制平仓。",
    },
    "state_exit_bottom5": {
        "display_name": "跌入历史弱势状态后退出",
        "plain_explanation": "当三维状态落入样本内胜率/收益较差的一组状态时提前退出，避免持续承压；另设最长持仓上限。",
    },
    "state_exit_not_top5": {
        "display_name": "状态脱离优区后退出",
        "plain_explanation": "当当前市场状态不再属于历史高胜率（Top）区间时退出，以锁定相对有利阶段；另设最长持仓上限。",
    },
    "momentum_flip_pos": {
        "display_name": "动量转正后退出",
        "plain_explanation": "当短期动量由弱转强（前一日动量特征为正）时视为反弹展开，优先平仓兑现。",
    },
    "bias_flip_pos": {
        "display_name": "低位修复完成后退出",
        "plain_explanation": "当价格相对均线的乖离率从负值修复到正值，视为本轮低位修复基本完成，优先退出。",
    },
    "trend_below_ma20": {
        "display_name": "跌破短期均线后退出",
        "plain_explanation": "前一日收盘跌破 20 日均线时退出，作为短期趋势转弱的信号。",
    },
    "trend_below_ma60": {
        "display_name": "跌破中期均线后退出",
        "plain_explanation": "前一日收盘跌破 60 日均线时退出，作为中期趋势转弱的信号。",
    },
    "hold_fixed": {
        "display_name": "固定持有（bundle 持有天数）",
        "plain_explanation": "不使用动态退出规则；持有满当前侧栏/推荐设定的交易日后在开盘平仓。",
    },
}


@dataclass(frozen=True)
class ExitRuleSpec:
    """Serializable exit policy (one candidate = one spec)."""

    rule_id: str
    kind: ExitKind
    hold_days: int | None = None
    max_hold_days: int = 252
    state_rank_horizon: int = 20
    state_bottom_k: int = 5
    state_top_k: int = 5
    state_min_n: int = 5
    momentum_window: int = 10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExitRuleSpec:
        return cls(
            rule_id=str(d["rule_id"]),
            kind=d["kind"],  # type: ignore[arg-type]
            hold_days=(int(d["hold_days"]) if d.get("hold_days") is not None else None),
            max_hold_days=int(d.get("max_hold_days", 252)),
            state_rank_horizon=int(d.get("state_rank_horizon", 20)),
            state_bottom_k=int(d.get("state_bottom_k", 5)),
            state_top_k=int(d.get("state_top_k", 5)),
            state_min_n=int(d.get("state_min_n", 5)),
            momentum_window=int(d.get("momentum_window", 10)),
        )


def _technical_exit_short_label(spec: ExitRuleSpec) -> str:
    """未在 EXIT_RULE_COPY 注册时的短说明（避免与 human_exit_label 循环）。"""
    if spec.kind == "time_hold":
        return f"固定持有 {spec.hold_days} 交易日"
    if spec.kind == "state_bottom_k":
        return f"状态落入历史最差约 {spec.state_bottom_k} 类则退出（h={spec.state_rank_horizon}）"
    if spec.kind == "state_not_top_k":
        return f"状态不在历史最优约 {spec.state_top_k} 类则退出（h={spec.state_rank_horizon}）"
    if spec.kind == "momentum_positive":
        return "动量转正则退出（前一日收益>0）"
    if spec.kind == "bias_positive":
        return "乖离转正则退出（前一日 bias>0）"
    if spec.kind == "below_ma20":
        return "前一日收盘 < MA20 则退出"
    if spec.kind == "below_ma60":
        return "前一日收盘 < MA60 则退出"
    return spec.rule_id


def _plain_explanation_fallback(spec: ExitRuleSpec) -> str:
    if spec.kind == "state_bottom_k":
        return (
            f"当状态落入样本内较差约 {spec.state_bottom_k} 类时退出（前向 horizon={spec.state_rank_horizon}）；"
            f"最长持有不超过 {spec.max_hold_days} 个交易日。"
        )
    if spec.kind == "state_not_top_k":
        return (
            "当状态不再属于样本内高胜率 Top 区间时退出；"
            f"最长持有不超过 {spec.max_hold_days} 个交易日。"
        )
    if spec.kind == "momentum_positive":
        return "前一日动量指标转正时退出；另受最长持仓天数限制。"
    if spec.kind == "bias_positive":
        return "前一日乖离率转正时退出；另受最长持仓天数限制。"
    if spec.kind == "below_ma20":
        return "前一日收盘低于 20 日均线时退出；另受最长持仓天数限制。"
    if spec.kind == "below_ma60":
        return "前一日收盘低于 60 日均线时退出；另受最长持仓天数限制。"
    return spec.rule_id


def exit_rule_display_name(rule_id: str, spec: ExitRuleSpec | None = None) -> str:
    if rule_id in EXIT_RULE_COPY:
        return EXIT_RULE_COPY[rule_id]["display_name"]
    if spec is not None and spec.kind == "time_hold" and spec.hold_days is not None:
        h = int(spec.hold_days)
        return f"固定持有 {h} 天退出"
    if spec is not None:
        return _technical_exit_short_label(spec)
    return rule_id


def exit_rule_plain_explanation(rule_id: str, spec: ExitRuleSpec | None = None) -> str:
    if rule_id in EXIT_RULE_COPY:
        return EXIT_RULE_COPY[rule_id]["plain_explanation"]
    if spec is not None and spec.kind == "time_hold" and spec.hold_days is not None:
        h = int(spec.hold_days)
        return f"不看市场状态，到达 {h} 个交易日后强制平仓。"
    if spec is not None:
        return _plain_explanation_fallback(spec)
    return ""


def time_hold_spec(hold_days: int, *, rule_id: str | None = None) -> ExitRuleSpec:
    h = int(hold_days)
    return ExitRuleSpec(
        rule_id=rule_id or f"time_{h}",
        kind="time_hold",
        hold_days=h,
        max_hold_days=max(h, 252),
    )


def list_cli_exit_rule_ids() -> list[str]:
    """CLI --exit-rule 可选 id（含 hold_fixed）。"""
    return ["hold_fixed"] + [s.rule_id for s in default_exit_rule_candidates()]


def resolve_cli_exit_rule(rule_id: str) -> ExitRuleSpec | None:
    """
    解析 CLI 退出规则 id。

    Returns
    -------
    None
        ``hold_fixed``：引擎内仅用 ``hold_days``，无 ExitRuleSpec。
    ExitRuleSpec
        与 :func:`default_exit_rule_candidates` 中 id 一致的一条。
    """
    r = str(rule_id).strip()
    if r == "hold_fixed":
        return None
    for sp in default_exit_rule_candidates():
        if sp.rule_id == r:
            return sp
    raise ValueError(
        f"Unknown exit rule id {rule_id!r}. Choose one of: {', '.join(list_cli_exit_rule_ids())}"
    )


def default_exit_rule_candidates() -> list[ExitRuleSpec]:
    """Grid of exit rules evaluated against a fixed entry frame (recommended entry)."""
    return [
        time_hold_spec(20),
        time_hold_spec(60),
        time_hold_spec(120),
        ExitRuleSpec(
            rule_id="state_exit_bottom5",
            kind="state_bottom_k",
            max_hold_days=252,
            state_rank_horizon=20,
            state_bottom_k=5,
            state_top_k=5,
        ),
        ExitRuleSpec(
            rule_id="state_exit_not_top5",
            kind="state_not_top_k",
            max_hold_days=252,
            state_rank_horizon=20,
            state_bottom_k=5,
            state_top_k=5,
        ),
        ExitRuleSpec(rule_id="momentum_flip_pos", kind="momentum_positive", max_hold_days=252, momentum_window=10),
        ExitRuleSpec(rule_id="bias_flip_pos", kind="bias_positive", max_hold_days=252),
        ExitRuleSpec(rule_id="trend_below_ma20", kind="below_ma20", max_hold_days=252),
        ExitRuleSpec(rule_id="trend_below_ma60", kind="below_ma60", max_hold_days=252),
    ]


def exit_rule_candidate_specs(bundle_hold_days: int) -> list[ExitRuleSpec]:
    """
    横评 / evaluate-exit / optimize-exit / compare-exit-rules 共用的候选列表。

    - 首项 ``hold_fixed``：固定持有 ``bundle_hold_days`` 个交易日；
    - 其余来自 :func:`default_exit_rule_candidates`；与 bundle 同天的 ``time_hold`` 跳过以免与 hold_fixed 重复。
    """
    h = max(1, int(bundle_hold_days))
    out: list[ExitRuleSpec] = [time_hold_spec(h, rule_id="hold_fixed")]
    for sp in default_exit_rule_candidates():
        if sp.kind == "time_hold" and sp.hold_days is not None and int(sp.hold_days) == h:
            continue
        out.append(sp)
    return out


@dataclass
class ExitContext:
    """Precomputed arrays aligned with ``df.reset_index(drop=True)`` (length n)."""

    mom: np.ndarray
    bias: np.ndarray
    close: np.ndarray
    ma20: np.ndarray
    ma60: np.ndarray
    state: np.ndarray
    bottom_states: frozenset[str]
    top_states: frozenset[str]


def human_exit_label(spec: ExitRuleSpec) -> str:
    """短标签：优先用 EXIT_RULE_COPY.display_name，与 UI 主标题一致。"""
    return exit_rule_display_name(spec.rule_id, spec)


def build_exit_context(
    df: pd.DataFrame,
    *,
    signal_mode: str,
    bias_ma_window: int,
    momentum_window: int,
    volume_ma_window: int,
    rolling_window: int,
    quantile_low: float,
    quantile_high: float,
    spec: ExitRuleSpec,
) -> ExitContext:
    """Build numpy context for exit checks + state top/bottom sets (in-sample, same as state-quality)."""
    d = df.sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(d["close"], errors="coerce").to_numpy(dtype=float)
    mom_col = momentum_column(int(momentum_window))
    bcol = bias_column(int(bias_ma_window))
    vcol = volume_ratio_column(int(volume_ma_window))
    mom = pd.to_numeric(d[mom_col], errors="coerce").to_numpy(dtype=float)
    bias = pd.to_numeric(d[bcol], errors="coerce").to_numpy(dtype=float)
    s = pd.Series(close)
    ma20 = s.rolling(20, min_periods=20).mean().to_numpy(dtype=float)
    ma60 = s.rolling(60, min_periods=60).mean().to_numpy(dtype=float)

    bottom: frozenset[str] = frozenset()
    top: frozenset[str] = frozenset()
    state_arr = np.array(["MISSING"] * len(d), dtype=object)

    if spec.kind in ("state_bottom_k", "state_not_top_k"):
        h = int(spec.state_rank_horizon)
        tmp = add_forward_returns(d.copy(), (h,), close_col="close", date_col="date")
        tmp = assign_ternary_states(
            tmp,
            momentum_col=mom_col,
            bias_col=bcol,
            volume_col=vcol,
            mode="rolling" if signal_mode == "rolling" else "full_sample",
            rolling_window=int(rolling_window),
            ternary_q1=float(quantile_low),
            ternary_q2=float(quantile_high),
        )
        top_l, bottom_l, _n = rank_states_by_horizon(
            tmp,
            horizon=h,
            min_n=int(spec.state_min_n),
            top_k=int(spec.state_top_k),
            bottom_k=int(spec.state_bottom_k),
        )
        bottom = frozenset(str(x["state"]) for x in bottom_l)
        top = frozenset(str(x["state"]) for x in top_l)
        state_arr = tmp["state"].astype(str).to_numpy()

    return ExitContext(
        mom=mom,
        bias=bias,
        close=close,
        ma20=ma20,
        ma60=ma60,
        state=state_arr,
        bottom_states=bottom,
        top_states=top,
    )


def should_exit_at_open(i: int, entry_open_idx: int, spec: ExitRuleSpec, ctx: ExitContext) -> bool:
    """
    At bar open index ``i``, decide whether to exit a position entered at open ``entry_open_idx``.

    Uses information available **before** today's open: row ``i-1`` close-based features.
    """
    if i <= entry_open_idx:
        return False
    if i >= entry_open_idx + int(spec.max_hold_days):
        return True

    if spec.kind == "time_hold":
        hd = int(spec.hold_days or 0)
        return i >= entry_open_idx + hd

    j = i - 1
    if j < 0:
        return False

    if spec.kind == "momentum_positive":
        v = float(ctx.mom[j])
        return np.isfinite(v) and v > 0.0

    if spec.kind == "bias_positive":
        v = float(ctx.bias[j])
        return np.isfinite(v) and v > 0.0

    if spec.kind == "below_ma20":
        c, m = float(ctx.close[j]), float(ctx.ma20[j])
        return np.isfinite(c) and np.isfinite(m) and c < m

    if spec.kind == "below_ma60":
        c, m = float(ctx.close[j]), float(ctx.ma60[j])
        return np.isfinite(c) and np.isfinite(m) and c < m

    if spec.kind == "state_bottom_k":
        st = str(ctx.state[j])
        if st == "MISSING" or "MISSING" in st:
            return False
        return st in ctx.bottom_states

    if spec.kind == "state_not_top_k":
        st = str(ctx.state[j])
        if st == "MISSING" or "MISSING" in st:
            return False
        return st not in ctx.top_states

    return False


def score_exit_metrics(metrics: dict[str, float]) -> float:
    """
    Higher is better. Penalizes deep drawdowns; rewards ann return and Sharpe.
    (Rule layer — not optimized for live trading.)
    """
    ann = float(metrics.get("annualized_return", float("nan")))
    sh = float(metrics.get("sharpe_ratio", float("nan")))
    mdd = float(metrics.get("max_drawdown", float("nan")))
    if not np.isfinite(ann):
        ann = 0.0
    if not np.isfinite(sh):
        sh = 0.0
    if not np.isfinite(mdd):
        mdd = 0.0
    return ann * 100.0 + sh * 2.5 + mdd * 35.0


MIN_TRADES_EXIT_EVAL = 8


@dataclass
class ExitRuleEvalRow:
    rank: int
    spec: ExitRuleSpec
    label_zh: str
    display_name: str
    plain_explanation: str
    metrics: dict[str, float | int | None]
    score: float
    n_trades: int
    eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "rule_id": self.spec.rule_id,
            "kind": self.spec.kind,
            "label_zh": self.label_zh,
            "display_name": self.display_name,
            "plain_explanation": self.plain_explanation,
            "params": self.spec.to_dict(),
            "metrics": self.metrics,
            "score": self.score,
            "n_trades": self.n_trades,
            "eligible": self.eligible,
        }
