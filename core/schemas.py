"""
Shared request/response models for CLI, Streamlit, and agents.

All business execution goes through `core.runner` using these schemas.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .signal_dimensions import (
    BIAS_MA_WINDOWS,
    MOMENTUM_WINDOWS,
    QUANTILE_HIGH_CHOICES,
    QUANTILE_LOW_CHOICES,
    VOLUME_MA_WINDOWS,
)


class SignalModeEnum(str, Enum):
    full_sample = "full_sample"
    rolling = "rolling"


StrategyMode = Literal["hold", "timing"]


class StrategyProfileEnum(str, Enum):
    """仓位画像（UI 展示名）；与 core.position_rules.PROFILE_WEIGHTS 一致。"""

    aggressive = "aggressive"
    balanced = "balanced"
    defensive = "defensive"
    full = "full"


# 兼容旧 CLI/JSON 名称
PositionRuleEnum = StrategyProfileEnum


class BacktestParamSourceEnum(str, Enum):
    """回测是否一键套用推荐（信号 mode/bias、持有期、仓位画像）。"""

    manual = "manual"
    recommended = "recommended"


class BiasSourceEnum(str, Enum):
    """LOW 乖离：用库内 ``bias_rate`` 或自收盘价重算对应 ``bias_ma`` 列。"""

    recompute = "recompute"
    db = "db"


class SignalParamSourceEnum(str, Enum):
    """信号分位模式与乖离窗口：手动侧栏/CLI，或由推荐层自动套用。"""

    manual = "manual"
    auto = "auto"


class BestSignalSetupSnapshot(BaseModel):
    """单层候选：信号 × 模式 × 乖离窗 × 关注持有期，及规则评分与关键统计。"""

    model_config = ConfigDict(extra="forbid")

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


class ExitRuleEvalSnapshot(BaseModel):
    """单条退出规则回测评分（固定入场后的样本内排序）。"""

    model_config = ConfigDict(extra="forbid")

    rank: int
    rule_id: str
    kind: str
    label_zh: str
    display_name: str = Field(default="", description="人类可读短标题（与 label_zh 通常一致）")
    plain_explanation: str = Field(default="", description="白话说明退出逻辑")
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    score: float
    n_trades: int
    eligible: bool

    @model_validator(mode="before")
    @classmethod
    def _fill_exit_copy(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        from .exit_rules import ExitRuleSpec, exit_rule_display_name, exit_rule_plain_explanation

        rule_id = str(d.get("rule_id", ""))
        params = d.get("params") or {}
        spec: ExitRuleSpec | None = None
        if params:
            try:
                spec = ExitRuleSpec.from_dict(dict(params))
            except Exception:
                spec = None
        lz = str(d.get("label_zh") or "").strip()
        if not str(d.get("display_name") or "").strip():
            d["display_name"] = lz if lz else exit_rule_display_name(rule_id, spec)
        if not str(d.get("plain_explanation") or "").strip():
            d["plain_explanation"] = exit_rule_plain_explanation(rule_id, spec)
        if not lz and str(d.get("display_name") or "").strip():
            d["label_zh"] = str(d["display_name"])
        return d


class ExitRuleComparisonRow(BaseModel):
    """退出规则横评表一行（展示用指标列）。"""

    model_config = ConfigDict(extra="forbid")

    rank: int
    rule_id: str
    display_name: str
    score: float
    n_trades: int
    eligible: bool
    total_return: float | None = None
    annualized_return: float | None = None
    max_drawdown: float | None = None
    sharpe_ratio: float | None = None
    calmar_ratio: float | None = None


class MultiObjectiveInterpretationBlock(BaseModel):
    """多目标退出决策层可读摘要。"""

    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    style_bias: str = ""
    confidence: Literal["low", "medium", "high"] = "medium"


class ExitMultiObjectiveMetricsSnapshot(BaseModel):
    """与 rank_exit 横评一致的原始指标（JSON 键名与 CLI 消费对齐）。"""

    model_config = ConfigDict(extra="forbid")

    total_return: float | None = None
    annualized_return: float | None = None
    max_drawdown: float | None = None
    sharpe: float | None = Field(None, description="来自 metrics.sharpe_ratio")
    calmar: float | None = Field(None, description="来自 metrics.calmar_ratio")
    trade_count: int | None = Field(None, description="来自 ExitRuleEvalRow.n_trades")
    average_exposure: float | None = None
    avg_trade_return: float | None = None
    avg_holding_days: float | None = None
    win_rate: float | None = None


class ExitMultiObjectiveCandidateRow(BaseModel):
    """单条退出候选的多目标视角（不改动 score_exit_metrics）。"""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    display_name: str = ""
    rank: int = 0
    eligible: bool = False
    style_tag: str = ""
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="return_first / risk_first / efficiency_first / robustness_first",
    )
    pareto_member: bool = False
    metrics: ExitMultiObjectiveMetricsSnapshot = Field(
        default_factory=ExitMultiObjectiveMetricsSnapshot,
        description="rank_exit_rules_on_frame 已有 metrics 的只读副本",
    )


class MultiObjectiveDecisionBlock(BaseModel):
    """挂在 BacktestResponse 上的多目标退出层（与 optimize-exit 并存）。"""

    model_config = ConfigDict(extra="forbid")

    pareto_set: list[str] = Field(default_factory=list)
    objective_winners: dict[str, str] = Field(
        default_factory=dict,
        description="各目标下 eligible 最高分规则 id",
    )
    default_objective: str = Field(
        "risk_first",
        description="用于 default_recommendation 的视角；CLI --objective 可覆盖",
    )
    default_recommendation: str = Field(
        "",
        description="默认视角下的最优规则 id（默认同 risk_first 赢家）",
    )
    interpretation: MultiObjectiveInterpretationBlock = Field(
        default_factory=MultiObjectiveInterpretationBlock
    )
    candidates: list[ExitMultiObjectiveCandidateRow] = Field(default_factory=list)


class ExitRuleOptimizationDiagnosticRow(BaseModel):
    """单次回测中退出优选/横评的诊断行（与 optimize / compare 同源）。"""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    display_name: str = Field("", description="与横评表 display_name 一致")
    rank: int
    score: float
    eligible: bool
    n_trades: int
    ranking_key: str = Field(
        default="score_exit_metrics",
        description="排序函数：与 rank_exit_rules_on_frame / optimize-exit 一致",
    )
    included_in_optimize_pool: bool = Field(
        False,
        description="是否参与了本次 optimize-exit 的候选池扫描",
    )
    selected_by_optimize: bool = Field(
        False,
        description="是否为 optimize-exit 最终选中的规则",
    )


class RecommendationSnapshot(BaseModel):
    """两层：框架适配度 + 网格内最优信号参数（不改变底层信号数学）。"""

    model_config = ConfigDict(extra="forbid")

    code: str
    fit_level: str
    framework_fit_note: str
    best_signal_setup: BestSignalSetupSnapshot
    top_candidates: list[BestSignalSetupSnapshot] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    # 自动默认与兼容字段（由 best_signal_setup 与探索序填充）
    default_signal: str
    recommended_bias_ma: int
    recommended_mode: str
    recommended_horizon_focus: int = Field(60, description="推荐优先对照的事件研究持有期（交易日）")
    recommended_signals: list[str] = Field(default_factory=list)
    recommended_position_profile: str = Field(
        default="balanced",
        description="推荐仓位画像 aggressive / balanced / defensive / full",
    )
    best_exit_rule: ExitRuleEvalSnapshot | None = None
    exit_rule_candidates: list[ExitRuleEvalSnapshot] = Field(default_factory=list)
    exit_rule_explanation: str = Field(
        default="",
        description="退出规则横评说明（与入场未联合搜索）",
    )
    strategy_mode: StrategyMode | None = Field(
        None,
        description="来自 entry_map：hold=更适合简单持有；timing=样本内可识别的入场结构。未提供快照或未命中标的时为 null",
    )


def _clean_float(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    return v


class EventStudyRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizon: int
    n: int
    win_rate: float | None = None
    mean_return: float | None = None
    median_return: float | None = None
    std: float | None = None

    @field_validator("win_rate", "mean_return", "median_return", "std", mode="before")
    @classmethod
    def _nan(cls, v: Any) -> Any:
        return _clean_float(v)


class StateRankRow(BaseModel):
    """单状态在某一 horizon 上的前向收益统计（全历史或滚动分桶）。"""

    model_config = ConfigDict(extra="forbid")

    state: str
    n: int
    win_rate: float | None = None
    mean_return: float | None = None
    median_return: float | None = None
    std: float | None = None

    @field_validator("win_rate", "mean_return", "median_return", "std", mode="before")
    @classmethod
    def _nan_sr(cls, v: Any) -> Any:
        return _clean_float(v)


class SignalDimensionsSnapshot(BaseModel):
    """当前生效的三维度 + 分位与模式（CLI/UI/JSON 一致）。"""

    model_config = ConfigDict(extra="forbid")

    neg_momentum_window: int = Field(description="NEG：动量回看天数")
    low_bias_ma: int = Field(description="LOW：乖离所用均线周期")
    high_volume_ma_window: int = Field(description="HIGH：量比的分母均量窗口")
    quantile_low: float = Field(description="NEG/LOW：弱势分位阈值")
    quantile_high: float = Field(description="HIGH：放量分位阈值")
    signal_mode: str
    rolling_window: int
    bias_source: str = Field("recompute", description="recompute=由收盘重算乖离；db=使用库内 bias_rate 覆盖对应 bias_ma 列")


class ValidationIssueRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: str
    code: str
    message: str
    count: int | None = None


class ChartSpec(BaseModel):
    """Plotly figure as JSON-serializable dict (Figure.to_plotly_json() compatible)."""

    model_config = ConfigDict(extra="forbid")

    chart_id: str
    title: str
    plotly_json: dict[str, Any]


class TimeSeriesPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    value: float | None


class TradeRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_date: str
    exit_date: str
    signal_tier: int
    weight: float
    entry_price: float | None = None
    exit_price: float | None = None
    stock_return: float | None = None
    portfolio_return: float | None = None
    holding_days: int | None = None

    @field_validator(
        "weight",
        "entry_price",
        "exit_price",
        "stock_return",
        "portfolio_return",
        mode="before",
    )
    @classmethod
    def _nanf(cls, v: Any) -> Any:
        return _clean_float(v)


# --- Requests ---


class DbEtfRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    etf_code: str = Field(..., min_length=1)
    db_path: str | None = Field(None, description="SQLite path; default from core.paths.resolve_db_path")


class HealthRequest(DbEtfRequest):
    """Data quality for one ETF (raw + cleaned)."""

    invalid_row_limit: int = Field(500, ge=0, description="Max raw invalid rows to include in response")


class SignalParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_param_source: SignalParamSourceEnum = Field(
        SignalParamSourceEnum.manual,
        description="manual=使用本请求中的 mode/bias_ma；auto=先跑 recommend_strategy_setup 再套用推荐",
    )
    signal_mode: SignalModeEnum = SignalModeEnum.rolling
    bias_source: BiasSourceEnum = Field(
        BiasSourceEnum.recompute,
        description="recompute：由收盘价计算乖离；db：使用 SQLite bias_rate 列（需存在）",
    )
    bias_ma: int = Field(120, ge=1, description="LOW：乖离均线周期")
    momentum_window: int = Field(10, description="NEG：动量回看天数")
    volume_ma_window: int = Field(20, description="HIGH：量比均量窗口")
    rolling_window: int = Field(252, ge=20, le=5000)
    quantile_low: float = Field(0.33, description="NEG/LOW 分位")
    quantile_high: float = Field(0.67, description="HIGH 分位")

    @field_validator("bias_ma")
    @classmethod
    def _bias_allowed(cls, v: int) -> int:
        if v not in BIAS_MA_WINDOWS:
            raise ValueError(f"bias_ma must be one of {BIAS_MA_WINDOWS}")
        return v

    @field_validator("momentum_window")
    @classmethod
    def _mom_allowed(cls, v: int) -> int:
        if v not in MOMENTUM_WINDOWS:
            raise ValueError(f"momentum_window must be one of {MOMENTUM_WINDOWS}")
        return v

    @field_validator("volume_ma_window")
    @classmethod
    def _vol_allowed(cls, v: int) -> int:
        if v not in VOLUME_MA_WINDOWS:
            raise ValueError(f"volume_ma_window must be one of {VOLUME_MA_WINDOWS}")
        return v

    @field_validator("quantile_low")
    @classmethod
    def _ql(cls, v: float) -> float:
        if v not in QUANTILE_LOW_CHOICES:
            raise ValueError(f"quantile_low must be one of {QUANTILE_LOW_CHOICES}")
        return v

    @field_validator("quantile_high")
    @classmethod
    def _qh(cls, v: float) -> float:
        if v not in QUANTILE_HIGH_CHOICES:
            raise ValueError(f"quantile_high must be one of {QUANTILE_HIGH_CHOICES}")
        return v

    @model_validator(mode="after")
    def _quantiles_ordered(self) -> SignalParams:
        if self.quantile_low >= self.quantile_high:
            raise ValueError("quantile_low must be strictly less than quantile_high")
        return self


class RecommendationRequest(DbEtfRequest):
    """仅生成推荐，不跑完整事件研究。"""

    model_config = ConfigDict(extra="forbid")

    momentum_window: int = Field(10, description="与推荐网格一致")
    volume_ma_window: int = Field(20)
    rolling_window: int = Field(252, ge=20, le=5000)
    quantile_low: float = Field(0.33)
    quantile_high: float = Field(0.67)
    eval_horizon: int = Field(
        60,
        ge=5,
        le=500,
        description="若不在 {20,60,120} 内，仅在备注中提示；搜索网格持有期固定为 20/60/120",
    )
    top_k: int = Field(12, ge=1, le=54, description="返回排名前 K 个候选（全网格最多 54 个）")
    include_exit_rules: bool = Field(
        False,
        description="在固定推荐入场下横评默认退出规则并写入 best_exit_rule / exit_rule_candidates",
    )
    entry_map_json_path: str | None = Field(
        None,
        description="可选：discover-entry-map 生成的 JSON；合并 strategy_mode 到推荐结果",
    )

    @field_validator("momentum_window")
    @classmethod
    def _rm(cls, v: int) -> int:
        if v not in MOMENTUM_WINDOWS:
            raise ValueError(f"momentum_window must be one of {MOMENTUM_WINDOWS}")
        return v

    @field_validator("volume_ma_window")
    @classmethod
    def _rv(cls, v: int) -> int:
        if v not in VOLUME_MA_WINDOWS:
            raise ValueError(f"volume_ma_window must be one of {VOLUME_MA_WINDOWS}")
        return v

    @field_validator("quantile_low")
    @classmethod
    def _rql(cls, v: float) -> float:
        if v not in QUANTILE_LOW_CHOICES:
            raise ValueError(f"quantile_low must be one of {QUANTILE_LOW_CHOICES}")
        return v

    @field_validator("quantile_high")
    @classmethod
    def _rqh(cls, v: float) -> float:
        if v not in QUANTILE_HIGH_CHOICES:
            raise ValueError(f"quantile_high must be one of {QUANTILE_HIGH_CHOICES}")
        return v

    @model_validator(mode="after")
    def _rq_ord(self) -> RecommendationRequest:
        if self.quantile_low >= self.quantile_high:
            raise ValueError("quantile_low must be strictly less than quantile_high")
        return self


class SignalResearchRequest(DbEtfRequest, SignalParams):
    event_horizons: tuple[int, ...] = Field(
        (20, 60, 120),
        description="Trading-day horizons for overlapping event study",
    )

    @field_validator("event_horizons", mode="before")
    @classmethod
    def _horizons(cls, v: Any) -> tuple[int, ...]:
        if v is None:
            return (20, 60, 120)
        if isinstance(v, (list, tuple)):
            return tuple(int(x) for x in v)
        raise TypeError("event_horizons must be a sequence of int")


class BacktestRequest(DbEtfRequest, SignalParams):
    backtest_param_source: BacktestParamSourceEnum = Field(
        BacktestParamSourceEnum.manual,
        description="manual=侧栏/CLI 的 hold_days 与 strategy_profile；recommended=套用推荐持有期与仓位画像并强制推荐信号参数",
    )
    strategy_profile: StrategyProfileEnum = Field(
        StrategyProfileEnum.balanced,
        description="aggressive / balanced / defensive / full（legacy layered→balanced, conservative→defensive）",
    )
    hold_days: int = Field(120, ge=1, le=2000)
    compare_profiles: bool = Field(
        False,
        description="同信号参数下对比三种仓位画像净值（不改变单次回测数学，仅多跑几次并汇总）",
    )
    compare_manual_vs_recommended: bool = Field(
        False,
        description="对比当前手动参数 vs 一键推荐参数的两条净值",
    )
    evaluate_exit_rules: bool = Field(
        False,
        description="在推荐结果中附带退出规则横评（不改变主回测，除非同时 optimize_exit）",
    )
    optimize_exit: bool = Field(
        False,
        description="主回测使用样本内优选退出规则（需先算推荐横评；与 compare_* 不宜混用）",
    )
    entry_signal_tier: str | None = Field(
        None,
        description="实验：锁定入场层 NEG / NEG_LOW / NEG_LOW_HIGH；None=分层引擎（与网格推荐解耦）",
    )
    explicit_exit_rule_id: str | None = Field(
        None,
        description="实验：指定退出规则 id（如 hold_fixed、time_60、bias_flip_pos）；与 optimize_exit 互斥",
    )
    compare_exit_rules: bool = Field(
        False,
        description="在当前入场设定下横评 hold_fixed + 全部默认退出规则（研究用表，独立于推荐打分文案）",
    )
    multi_objective_exit: bool = Field(
        False,
        description="在横评结果上计算多目标 Pareto / 分视角最优（不替代 score_exit_metrics）",
    )
    exit_objective: str | None = Field(
        None,
        description="覆盖默认推荐视角：return_first | risk_first | efficiency_first | robustness_first",
    )
    export_trades_path: str | None = Field(
        None,
        description="若设置，将主回测成交明细写入该 CSV 路径",
    )
    entry_diagnostics: bool = Field(
        False,
        description="输出原始入场信号诊断（与成交/退出无关；复用 apply_signals + tier 逻辑）",
    )
    entry_diagnostics_dates: bool = Field(
        False,
        description="为 True 时在 JSON 中列出 raw_entry_dates 全表（可能较大）",
    )
    entry_exit_matching: bool = Field(
        False,
        description="入场 regime vs 各退出规则持仓对比诊断；隐含启用 entry 诊断与退出横评数据",
    )
    entry_exit_top: int | None = Field(
        None,
        ge=1,
        le=500,
        description="CLI 非 JSON 时 ENTRY/EXIT MATCHING 表最多展示行数；None=全部",
    )
    bias_quantile_range: str | None = Field(
        None,
        description="可选：乖离分位桶过滤 Q1..Q5（如 Q1、Q1-Q2）；仅压制 signal_tier>0 的入场候选",
    )
    entry_map_json_path: str | None = Field(
        None,
        description="可选：entry_map 快照 JSON；合并 strategy_mode 到 recommendation（不改变回测数学）",
    )

    @field_validator("bias_quantile_range", mode="before")
    @classmethod
    def _bt_bias_q(cls, v: Any) -> str | None:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        s = str(v).strip()
        from quantlab.filters.quantile_filter import parse_quantile_range

        parse_quantile_range(s)
        return s

    @model_validator(mode="after")
    def _exit_vs_compare(self) -> BacktestRequest:
        if self.optimize_exit and (self.compare_profiles or self.compare_manual_vs_recommended):
            raise ValueError("optimize_exit 不能与 compare_profiles 或 compare_manual_vs_recommended 同时开启")
        if self.explicit_exit_rule_id and self.optimize_exit:
            raise ValueError("不能同时使用 explicit_exit_rule_id（--exit-rule）与 optimize_exit（--optimize-exit）")
        if self.entry_signal_tier is not None:
            allowed = {"NEG", "NEG_LOW", "NEG_LOW_HIGH"}
            if self.entry_signal_tier not in allowed:
                raise ValueError(f"entry_signal_tier must be one of {allowed}")
        if self.exit_objective is not None:
            mo = {
                "return_first",
                "risk_first",
                "efficiency_first",
                "robustness_first",
            }
            if self.exit_objective not in mo:
                raise ValueError(f"exit_objective must be one of {sorted(mo)}")
        return self

    @field_validator("strategy_profile", mode="before")
    @classmethod
    def _legacy_profile(cls, v: Any) -> Any:
        if v is None:
            return StrategyProfileEnum.balanced
        s = v.value if hasattr(v, "value") else v
        if s == "layered":
            return StrategyProfileEnum.balanced
        if s == "conservative":
            return StrategyProfileEnum.defensive
        return v


class StateRankingRequest(DbEtfRequest):
    """
    三维三分位状态扫描（对齐 signal-quality-analyzer bucketing），与分层 NEG/LOW/HIGH 信号独立。
    """

    model_config = ConfigDict(extra="forbid")

    signal_mode: SignalModeEnum = Field(
        SignalModeEnum.rolling,
        description="full_sample=全样本分位划桶；rolling=滚动窗口分位划桶",
    )
    momentum_window: int = Field(10, description="动量列 momentum_{n}")
    bias_ma: int = Field(120, description="乖离列 bias_ma{n}")
    volume_ma_window: int = Field(20, description="量比列 volume_ratio_{n}")
    rolling_window: int = Field(252, ge=20, le=5000)
    horizon: int = Field(20, ge=1, le=500, description="前向持有交易日")
    min_n: int = Field(5, ge=1, description="参与排名的最小样本数")
    top_k: int = Field(5, ge=1, le=50)
    bottom_k: int = Field(5, ge=1, le=50)
    ternary_q1: float = Field(0.33, gt=0, lt=1, description="下分位（≤q1 为弱势侧）")
    ternary_q2: float = Field(0.67, gt=0, lt=1, description="上分位（>q2 为强势侧）")

    @field_validator("bias_ma")
    @classmethod
    def _sr_bias(cls, v: int) -> int:
        if v not in BIAS_MA_WINDOWS:
            raise ValueError(f"bias_ma must be one of {BIAS_MA_WINDOWS}")
        return v

    @field_validator("momentum_window")
    @classmethod
    def _sr_mom(cls, v: int) -> int:
        if v not in MOMENTUM_WINDOWS:
            raise ValueError(f"momentum_window must be one of {MOMENTUM_WINDOWS}")
        return v

    @field_validator("volume_ma_window")
    @classmethod
    def _sr_vol(cls, v: int) -> int:
        if v not in VOLUME_MA_WINDOWS:
            raise ValueError(f"volume_ma_window must be one of {VOLUME_MA_WINDOWS}")
        return v

    @model_validator(mode="after")
    def _ternary_ordered(self) -> StateRankingRequest:
        if self.ternary_q1 >= self.ternary_q2:
            raise ValueError("ternary_q1 must be strictly less than ternary_q2")
        return self


class StateTransitionRequest(DbEtfRequest):
    """
    From-state → future-state transitions on the research-frame daily sequence
    (``prepare_research_frame`` + ``assign_ternary_states``); no trades.
    """

    model_config = ConfigDict(extra="forbid")

    signal_mode: SignalModeEnum = Field(
        SignalModeEnum.rolling,
        description="full_sample=全样本分位划桶；rolling=滚动窗口分位划桶",
    )
    bias_source: BiasSourceEnum = Field(
        BiasSourceEnum.recompute,
        description="与回测一致：recompute 或 db bias_rate",
    )
    bias_ma: int = Field(120, description="乖离列 bias_ma{n}")
    momentum_window: int = Field(10, description="动量列 momentum_{n}")
    volume_ma_window: int = Field(20, description="量比列 volume_ratio_{n}")
    rolling_window: int = Field(252, ge=20, le=5000)
    quantile_low: float = Field(0.33, description="与 signal 分层一致；传入 assign_ternary_states 的 ternary 分位")
    quantile_high: float = Field(0.67)
    from_state: str = Field(..., min_length=1, description="精确匹配或前缀：如 NEG_LOW 匹配 NEG_LOW_HIGH 等")
    horizons: tuple[int, ...] = Field(
        (5, 10, 20, 60),
        description="前向交易日 horizon 列表",
    )
    transition_top_k: int | None = Field(
        None,
        ge=1,
        le=200,
        description="若设置，每个 horizon 只保留按 count 降序的前 K 个 to_state",
    )

    @field_validator("bias_ma")
    @classmethod
    def _st_bias(cls, v: int) -> int:
        if v not in BIAS_MA_WINDOWS:
            raise ValueError(f"bias_ma must be one of {BIAS_MA_WINDOWS}")
        return v

    @field_validator("momentum_window")
    @classmethod
    def _st_mom(cls, v: int) -> int:
        if v not in MOMENTUM_WINDOWS:
            raise ValueError(f"momentum_window must be one of {MOMENTUM_WINDOWS}")
        return v

    @field_validator("volume_ma_window")
    @classmethod
    def _st_vol(cls, v: int) -> int:
        if v not in VOLUME_MA_WINDOWS:
            raise ValueError(f"volume_ma_window must be one of {VOLUME_MA_WINDOWS}")
        return v

    @field_validator("quantile_low")
    @classmethod
    def _st_ql(cls, v: float) -> float:
        if v not in QUANTILE_LOW_CHOICES:
            raise ValueError(f"quantile_low must be one of {QUANTILE_LOW_CHOICES}")
        return v

    @field_validator("quantile_high")
    @classmethod
    def _st_qh(cls, v: float) -> float:
        if v not in QUANTILE_HIGH_CHOICES:
            raise ValueError(f"quantile_high must be one of {QUANTILE_HIGH_CHOICES}")
        return v

    @field_validator("horizons", mode="before")
    @classmethod
    def _st_horizons(cls, v: Any) -> tuple[int, ...]:
        if v is None:
            return (5, 10, 20, 60)
        if isinstance(v, (list, tuple)):
            hs = tuple(int(x) for x in v)
        else:
            raise TypeError("horizons must be a sequence of int")
        if not hs:
            raise ValueError("horizons must be non-empty")
        for h in hs:
            if h < 1:
                raise ValueError("each horizon must be >= 1")
        return hs

    @model_validator(mode="after")
    def _st_quantiles_ordered(self) -> StateTransitionRequest:
        if self.quantile_low >= self.quantile_high:
            raise ValueError("quantile_low must be strictly less than quantile_high")
        return self


class PathQualityTargetModeEnum(str, Enum):
    ever = "ever"
    final = "final"


class PathQualityRequest(DbEtfRequest):
    """
    Research-only: among ``from_state`` days, hit if ``target_state`` is reached (ever/final)
    within ``horizon``; feature quantile buckets vs hit / forward return to t+H.
    """

    model_config = ConfigDict(extra="forbid")

    signal_mode: SignalModeEnum = Field(SignalModeEnum.rolling)
    bias_source: BiasSourceEnum = Field(BiasSourceEnum.recompute)
    bias_ma: int = Field(120)
    momentum_window: int = Field(10)
    volume_ma_window: int = Field(20)
    rolling_window: int = Field(252, ge=20, le=5000)
    quantile_low: float = Field(0.33)
    quantile_high: float = Field(0.67)
    from_state: str = Field(..., min_length=1)
    target_state: str = Field(..., min_length=1)
    horizon: int = Field(..., ge=1, le=500, description="交易日窗口长度；前向收益为 close[t+H]/close[t]-1")
    target_mode: PathQualityTargetModeEnum = Field(
        PathQualityTargetModeEnum.ever,
        description="ever=窗口内任一日达目标态；final=仅 t+H 当日",
    )
    bucket_features: tuple[str, ...] = Field(
        ("bias_rate", "momentum", "volume_ratio"),
        description="逗号分隔逻辑名：bias_rate, momentum, volume_ratio, daily_change",
    )
    bucket_n: int = Field(
        5,
        ge=2,
        le=20,
        description="保留字段；路径质量 breakdown 使用全样本预计算的五分位 *_bucket 列，不再在子样本上 qcut",
    )
    bias_quantile_range: str | None = Field(
        None,
        description="可选：全样本乖离分位桶 Q1..Q5（与 backtest 一致）；先筛 origin 再算 breakdown",
    )

    @field_validator("bias_ma")
    @classmethod
    def _pq_bias(cls, v: int) -> int:
        if v not in BIAS_MA_WINDOWS:
            raise ValueError(f"bias_ma must be one of {BIAS_MA_WINDOWS}")
        return v

    @field_validator("momentum_window")
    @classmethod
    def _pq_mom(cls, v: int) -> int:
        if v not in MOMENTUM_WINDOWS:
            raise ValueError(f"momentum_window must be one of {MOMENTUM_WINDOWS}")
        return v

    @field_validator("volume_ma_window")
    @classmethod
    def _pq_vol(cls, v: int) -> int:
        if v not in VOLUME_MA_WINDOWS:
            raise ValueError(f"volume_ma_window must be one of {VOLUME_MA_WINDOWS}")
        return v

    @field_validator("quantile_low")
    @classmethod
    def _pq_ql(cls, v: float) -> float:
        if v not in QUANTILE_LOW_CHOICES:
            raise ValueError(f"quantile_low must be one of {QUANTILE_LOW_CHOICES}")
        return v

    @field_validator("quantile_high")
    @classmethod
    def _pq_qh(cls, v: float) -> float:
        if v not in QUANTILE_HIGH_CHOICES:
            raise ValueError(f"quantile_high must be one of {QUANTILE_HIGH_CHOICES}")
        return v

    @field_validator("bias_quantile_range", mode="before")
    @classmethod
    def _pq_bias_q(cls, v: Any) -> str | None:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        s = str(v).strip()
        from quantlab.filters.quantile_filter import parse_quantile_range

        parse_quantile_range(s)
        return s

    @field_validator("bucket_features", mode="before")
    @classmethod
    def _pq_bucket_features(cls, v: Any) -> tuple[str, ...]:
        if v is None or v == "":
            return ("bias_rate", "momentum", "volume_ratio")
        if isinstance(v, str):
            return tuple(
                x.strip().lower().replace("-", "_") for x in v.split(",") if x.strip()
            )
        if isinstance(v, (list, tuple)):
            return tuple(str(x).strip().lower().replace("-", "_") for x in v)
        raise TypeError("bucket_features must be str, list, or tuple")

    @model_validator(mode="after")
    def _pq_quantiles_and_features(self) -> PathQualityRequest:
        if self.quantile_low >= self.quantile_high:
            raise ValueError("quantile_low must be strictly less than quantile_high")
        allowed = {"bias_rate", "momentum", "volume_ratio", "daily_change"}
        for f in self.bucket_features:
            if f not in allowed:
                raise ValueError(
                    f"unknown bucket feature {f!r}; allowed: {', '.join(sorted(allowed))}"
                )
        return self


class PathRuleMiningRequest(DbEtfRequest):
    """
    Contiguous quantile path rules on from_state samples (aligns with path-quality labeling).
    """

    model_config = ConfigDict(extra="forbid")

    signal_mode: SignalModeEnum = Field(SignalModeEnum.rolling)
    bias_source: BiasSourceEnum = Field(BiasSourceEnum.recompute)
    bias_ma: int = Field(120)
    momentum_window: int = Field(10)
    volume_ma_window: int = Field(20)
    rolling_window: int = Field(252, ge=20, le=5000)
    quantile_low: float = Field(0.33)
    quantile_high: float = Field(0.67)
    from_state: str = Field(..., min_length=1)
    target_state: str = Field(..., min_length=1)
    horizon: int = Field(..., ge=1, le=500)
    target_mode: PathQualityTargetModeEnum = Field(PathQualityTargetModeEnum.ever)
    features: tuple[str, ...] = Field(
        ("bias_rate", "volume_ratio"),
        description="Comma-separated logical names; rules use global *_bucket columns",
    )
    bucket_n: int = Field(
        5,
        ge=2,
        le=20,
        description="保留字段；规则使用全样本预计算 Q1–Q5 *_bucket 列",
    )
    max_combinations: int = Field(2, ge=1, le=2, description="1=single-factor only; 2=add pairs")
    min_count: int = Field(5, ge=1)
    top_k: int = Field(20, ge=0, le=5000, description="Max rules after ranking; 0 = no limit")
    rules_above_baseline_only: bool = Field(
        False,
        description="If True, keep rules with hit_rate >= baseline hit_rate",
    )
    bias_quantile_range: str | None = Field(
        None,
        description="可选：全样本乖离分位桶过滤后再挖矿（与 path-quality / backtest 一致）",
    )

    @field_validator("bias_ma")
    @classmethod
    def _prm_bias(cls, v: int) -> int:
        if v not in BIAS_MA_WINDOWS:
            raise ValueError(f"bias_ma must be one of {BIAS_MA_WINDOWS}")
        return v

    @field_validator("momentum_window")
    @classmethod
    def _prm_mom(cls, v: int) -> int:
        if v not in MOMENTUM_WINDOWS:
            raise ValueError(f"momentum_window must be one of {MOMENTUM_WINDOWS}")
        return v

    @field_validator("volume_ma_window")
    @classmethod
    def _prm_vol(cls, v: int) -> int:
        if v not in VOLUME_MA_WINDOWS:
            raise ValueError(f"volume_ma_window must be one of {VOLUME_MA_WINDOWS}")
        return v

    @field_validator("quantile_low")
    @classmethod
    def _prm_ql(cls, v: float) -> float:
        if v not in QUANTILE_LOW_CHOICES:
            raise ValueError(f"quantile_low must be one of {QUANTILE_LOW_CHOICES}")
        return v

    @field_validator("quantile_high")
    @classmethod
    def _prm_qh(cls, v: float) -> float:
        if v not in QUANTILE_HIGH_CHOICES:
            raise ValueError(f"quantile_high must be one of {QUANTILE_HIGH_CHOICES}")
        return v

    @field_validator("bias_quantile_range", mode="before")
    @classmethod
    def _prm_bias_q(cls, v: Any) -> str | None:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        s = str(v).strip()
        from quantlab.filters.quantile_filter import parse_quantile_range

        parse_quantile_range(s)
        return s

    @field_validator("features", mode="before")
    @classmethod
    def _prm_features(cls, v: Any) -> tuple[str, ...]:
        if v is None or v == "":
            return ("bias_rate", "volume_ratio")
        if isinstance(v, str):
            return tuple(
                x.strip().lower().replace("-", "_") for x in v.split(",") if x.strip()
            )
        if isinstance(v, (list, tuple)):
            return tuple(str(x).strip().lower().replace("-", "_") for x in v)
        raise TypeError("features must be str, list, or tuple")

    @model_validator(mode="after")
    def _prm_quantiles_and_features(self) -> PathRuleMiningRequest:
        if self.quantile_low >= self.quantile_high:
            raise ValueError("quantile_low must be strictly less than quantile_high")
        allowed = {"bias_rate", "momentum", "volume_ratio", "daily_change"}
        if not self.features:
            raise ValueError("features must be non-empty")
        if len(set(self.features)) != len(self.features):
            raise ValueError("features must be unique")
        for f in self.features:
            if f not in allowed:
                raise ValueError(
                    f"unknown feature {f!r}; allowed: {', '.join(sorted(allowed))}"
                )
        return self


class ReportRequest(BacktestRequest):
    output_dir: Path
    write_json: bool = True
    write_csv: bool = True
    write_charts_html: bool = True
    event_horizons: tuple[int, ...] = Field((20, 60, 120))

    @field_validator("event_horizons", mode="before")
    @classmethod
    def _horizons_r(cls, v: Any) -> tuple[int, ...]:
        if v is None:
            return (20, 60, 120)
        if isinstance(v, (list, tuple)):
            return tuple(int(x) for x in v)
        raise TypeError("event_horizons must be a sequence of int")


# --- Responses ---


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    etf_code: str
    etf_name: str
    db_path: str
    rows_in: int
    rows_out: int
    open_fallback_rows: int
    issues: list[ValidationIssueRow]
    invalid_row_count: int
    invalid_rows_sample: list[dict[str, Any]]
    summary_stats: dict[str, dict[str, float | int | None]]
    charts: list[ChartSpec] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    etf_code: str
    etf_name: str
    db_path: str
    recommendation: RecommendationSnapshot


class SignalResearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    etf_code: str
    etf_name: str
    db_path: str
    signal_param_source: str = Field(description="manual 或 auto")
    applied_recommendation_defaults: bool = Field(
        False,
        description="为 True 时，本次事件研究的 mode/bias_ma 来自推荐层",
    )
    recommendation: RecommendationSnapshot
    default_chart_tier: str = Field(description="建议默认展开的信号层（NEG / NEG_LOW / NEG_LOW_HIGH）")
    signal_mode: str
    bias_source: str
    bias_ma: int
    momentum_window: int
    volume_ma_window: int
    rolling_window: int
    quantile_low: float
    quantile_high: float
    signal_dimensions: SignalDimensionsSnapshot
    event_horizons: tuple[int, ...]
    event_studies: dict[str, list[EventStudyRow]]
    charts: list[ChartSpec] = Field(default_factory=list)


class EntrySignalRegimeRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: str
    end_date: str
    duration_days: int


class EntryPersistenceSummaryBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regime_count: int
    avg_duration_days: float | None = None
    max_duration_days: int = 0
    median_duration_days: float | None = None


class EntrySignalDiagnosticsBlock(BaseModel):
    """EOD 原始入场条件（与引擎 active[] 一致），与成交/退出无关。"""

    model_config = ConfigDict(extra="forbid")

    definition_note: str = Field(
        "",
        description="固定层或分层下 raw active 的定义说明",
    )
    raw_entry_days_count: int = Field(0, description="入场条件为 True 的交易日数")
    raw_entry_dates: list[str] | None = Field(
        None,
        description="全部 raw 入场日（ISO 日期）；未请求时为 null",
    )
    entry_regimes: list[EntrySignalRegimeRow] = Field(
        default_factory=list,
        description="连续 True 区间压缩后的 regime",
    )
    entry_persistence_summary: EntryPersistenceSummaryBlock = Field(
        default_factory=lambda: EntryPersistenceSummaryBlock(regime_count=0, max_duration_days=0)
    )


class EntryExitMatchingEntrySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_entry_days_count: int = 0
    regime_count: int = 0
    avg_regime_duration_days: float | None = None
    median_regime_duration_days: float | None = None
    max_regime_duration_days: int = 0


class EntryExitMatchingPerExitRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    display_name: str = ""
    eligible: bool = False
    trade_count: int = 0
    avg_holding_days: float | None = None
    median_holding_days: float | None = None
    holding_vs_entry_avg_ratio: float | None = None
    holding_vs_entry_median_ratio: float | None = None
    trades_per_entry_regime: float | None = None
    alignment_label: str = ""
    notes: str = ""


class EntryExitMatchingDiagnosticsBlock(BaseModel):
    """入场 regime 持久度 vs 各退出规则持仓长度（诊断；不改变回测）。"""

    model_config = ConfigDict(extra="forbid")

    entry_summary: EntryExitMatchingEntrySummary
    per_exit: list[EntryExitMatchingPerExitRow] = Field(default_factory=list)


class BacktestComparisonRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant_label: str
    strategy_profile: str
    strategy_profile_zh: str
    weights_by_tier: dict[str, float]
    metrics: dict[str, float | int | None]
    hold_days: int


class LatestSignalBarSnapshot(BaseModel):
    """最后一根已收盘日 K 上的 NEG/LOW/HIGH 与分层；与回测侧栏有效参数一致。"""

    model_config = ConfigDict(extra="forbid")

    etf_code: str
    etf_name: str
    as_of_date: str = Field(description="库中最后一根 bar 的日期（YYYY-MM-DD）")
    neg: bool
    low: bool
    high: bool
    bias_bucket: str | None = Field(default=None, description="全样本等频五分位：bias_bucket (Q1..Q5)")
    momentum_bucket: str | None = Field(default=None, description="全样本等频五分位：momentum_bucket (Q1..Q5)")
    volume_ratio_bucket: str | None = Field(default=None, description="全样本等频五分位：volume_ratio_bucket (Q1..Q5)")
    signal_tier_raw: int = Field(description="apply_signals 的 signal_tier（0–3），未经 bias 分位过滤")
    signal_tier_effective: int = Field(
        description="分层模式下经 --bias-q 过滤后的 tier；锁层模式下与 raw 相同",
    )
    implied_layer_zh: str = Field(description="人类可读：当前分层含义或锁层是否满足")
    readout_zh: str = Field(
        default="",
        description="终端/UI 展示用：为何 tier 与三 bool 如此、下一日是否可能入场",
    )
    backtest_entry_active: bool = Field(
        description="与引擎一致：最后一根 bar 收盘后是否视为可触发（下一交易日开盘）入场",
    )
    execution_entry_mode: str = Field(description="HIERARCHICAL 或 NEG / NEG_LOW / NEG_LOW_HIGH")
    signal_mode: str
    bias_ma_effective: int
    strategy_profile_effective: str
    strategy_profile_zh: str
    applied_recommendation_defaults: bool
    applied_backtest_recommendation: bool


class BacktestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    etf_code: str
    etf_name: str
    db_path: str
    signal_param_source: str = Field(description="manual 或 auto")
    applied_recommendation_defaults: bool = Field(
        False,
        description="为 True 时，回测的 mode/bias_ma 来自推荐层",
    )
    backtest_param_source: str = Field("manual", description="manual 或 recommended")
    applied_backtest_recommendation: bool = Field(
        False,
        description="为 True 时，持有期与仓位画像也来自推荐层",
    )
    recommendation: RecommendationSnapshot = Field(description="与本次回测同源的规则推荐快照（便于对照）")
    fit_level: str = Field("", description="框架适配度 high/medium/low")
    signal_mode: str
    bias_source: str
    bias_ma: int
    momentum_window: int
    volume_ma_window: int
    rolling_window: int
    quantile_low: float
    quantile_high: float
    signal_dimensions: SignalDimensionsSnapshot
    strategy_profile: str = Field(description="实际执行的仓位画像 aggressive/balanced/defensive/full")
    strategy_profile_zh: str = Field("", description="中文展示名")
    position_rule: str = Field(
        "",
        description="与 strategy_profile 同值，兼容旧字段",
    )
    hold_days: int
    weights_by_tier: dict[str, float]
    metrics: dict[str, float | int | None]
    summary_cards: dict[str, float | int | None] = Field(
        default_factory=dict,
        description="总收益、年化、最大回撤、Calmar、平均暴露、成交笔数等展示用",
    )
    recommended_setup: dict[str, Any] = Field(default_factory=dict, description="推荐 bundle：信号/mode/bias/持有/画像")
    executed_setup: dict[str, Any] = Field(default_factory=dict, description="本次实际执行参数摘要")
    interpretation_notes: list[str] = Field(default_factory=list, description="可读解读要点")
    trades: list[TradeRow]
    equity_curve: list[TimeSeriesPoint]
    drawdown_curve: list[TimeSeriesPoint]
    charts: list[ChartSpec] = Field(default_factory=list)
    applied_exit_optimization: bool = Field(
        False,
        description="为 True 时主回测净值按样本内优选退出规则重算",
    )
    optimized_exit_rule_id: str | None = Field(None, description="生效的退出规则 id")
    optimized_exit_label_zh: str | None = Field(None, description="生效退出规则中文说明")
    optimized_exit_plain_zh: str | None = Field(
        None,
        description="生效退出规则白话说明（与 optimized_exit_label_zh 配套）",
    )
    optimized_exit_rule: str | None = Field(
        None,
        description="同 optimized_exit_rule_id，便于 JSON 消费",
    )
    optimized_exit_score: float | None = Field(None, description="选中规则 score_exit_metrics 得分")
    optimized_exit_display_name: str | None = Field(None, description="选中规则展示名")
    optimized_exit_eligible: bool = Field(
        False,
        description="选中规则是否满足最小成交（优选仅选 eligible）",
    )
    optimized_exit_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="选中规则完整 metrics（含 average_exposure 等）",
    )
    exit_optimization_diagnostics: list[ExitRuleOptimizationDiagnosticRow] = Field(
        default_factory=list,
        description="各候选规则 rank/score/eligible 及是否被 optimize 选中",
    )
    multi_objective_decision: MultiObjectiveDecisionBlock | None = Field(
        None,
        description="多目标退出层：Pareto、分视角赢家、默认推荐；为 None 表示未计算",
    )
    entry_signal_diagnostics: EntrySignalDiagnosticsBlock | None = Field(
        None,
        description="原始入场 EOD 序列诊断；未请求 entry_diagnostics 时为 null",
    )
    entry_exit_matching_diagnostics: EntryExitMatchingDiagnosticsBlock | None = Field(
        None,
        description="入场 regime vs 退出持仓对齐诊断；仅 entry_exit_matching 且存在横评数据时有值",
    )
    executed_strategy_narrative: list[str] = Field(
        default_factory=list,
        description="当前执行策略要点（图表前展示，与主回测一致）",
    )
    exit_rule_comparison_rows: list[ExitRuleComparisonRow] = Field(
        default_factory=list,
        description="退出规则横评全表（含收益/回撤/Sharpe/Calmar 等，供 UI/CLI 对照）",
    )
    exit_sweep_under_entry: list[ExitRuleComparisonRow] = Field(
        default_factory=list,
        description="compare_exit_rules：与 evaluate/optimize 相同候选集，按 score_exit_metrics 排序",
    )
    signal_tier: str = Field(
        "",
        description="JSON/CLI 锚点：实际入场层 NEG / NEG_LOW / NEG_LOW_HIGH / HIERARCHICAL",
    )
    exit_rule: str | None = Field(
        None,
        description="JSON/CLI 锚点：主回测实际退出规则 id（如 hold_fixed、bias_flip_pos）",
    )
    exit_selection_mode: str = Field(
        "",
        description="bundle_hold | optimized | explicit",
    )
    recommended_bundle: dict[str, Any] = Field(
        default_factory=dict,
        description="与 recommended_setup 相同内容，便于 CLI JSON 消费",
    )
    trade_count: int = Field(0, description="成交笔数（冗余 summary_cards.trade_count）")
    trades_export_path: str | None = Field(None, description="若写出成交 CSV，则为绝对或用户路径")
    latest_signal_bar: LatestSignalBarSnapshot | None = Field(
        None,
        description="最后一根 bar 的 NEG/LOW/HIGH 与 signal_tier（与本次回测有效信号参数一致）",
    )
    comparison_rows: list[BacktestComparisonRow] | None = Field(
        None,
        description="对比模式下的各分支指标（无对比时为 None）",
    )


class StateRankingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    etf_code: str
    etf_name: str
    db_path: str
    horizon: int
    bucket_mode: str
    momentum_window: int
    bias_ma: int
    volume_ma_window: int
    rolling_window: int
    ternary_q1: float
    ternary_q2: float
    min_n: int
    top_k: int
    bottom_k: int
    states_ranked: int
    state_pattern: str = Field(
        "MOM_BIAS_VOL",
        description="状态字符串：动量(NEG/NEU/POS)_乖离(LOW/MID/HIGH)_量比(LOW/NORMAL/HIGH)",
    )
    top_best: list[StateRankRow]
    bottom_worst: list[StateRankRow]


class StateTransitionRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    to_state: str
    prob: float
    count: int
    mean_return: float | None = None
    win_rate: float | None = None


class StateTransitionHorizonBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_valid: int = Field(..., description="该 horizon 下有效 t→t+h 样本数（有收盘价）")
    entropy_nats: float = Field(
        ...,
        description="to_state 分布的 Shannon 熵（nat）；在截断 top_k 前对全体 to_state 计算",
    )
    transitions: list[StateTransitionRow]


class StateTransitionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    etf_code: str
    etf_name: str
    db_path: str
    from_state: str
    signal_mode: str
    bias_source: str
    bias_ma: int
    momentum_window: int
    volume_ma_window: int
    rolling_window: int
    quantile_low: float
    quantile_high: float
    state_pattern: str = Field(
        "MOM_BIAS_VOL",
        description="由数据推断：MOM_BIAS_VOL 或 MOM_BIAS",
    )
    total_samples: int = Field(
        ...,
        description="满足 from_state 且 state≠MISSING 的 origin 行数",
    )
    horizons: dict[str, StateTransitionHorizonBlock]


class PathQualityBucketRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bucket: str
    count: int
    hit_count: int
    hit_rate: float
    mean_forward_return: float
    win_rate_forward: float


class PathQualityFeatureBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    buckets: list[PathQualityBucketRow]


class PathQualityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    etf_code: str
    etf_name: str
    db_path: str
    from_state: str
    target_state: str
    horizon: int
    target_mode: str
    signal_mode: str
    bias_source: str
    bias_ma: int
    momentum_window: int
    volume_ma_window: int
    rolling_window: int
    quantile_low: float
    quantile_high: float
    state_pattern: str = Field(
        "MOM_BIAS_VOL",
        description="与 assign_ternary_states 一致",
    )
    total_samples: int
    hit_count: int
    hit_rate: float
    mean_forward_return: float = Field(
        ...,
        description="全体 origin 样本上前向收益 close[t+H]/close[t]-1 的均值（与 hit 标签同源）",
    )
    feature_breakdowns: list[PathQualityFeatureBreakdown]


class PathRuleMiningBaselineBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    hit_count: int
    hit_rate: float
    mean_forward_return: float


class PathRuleFeatureCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    bucket_range: str


class PathRuleMiningRuleRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: str
    feature_conditions: list[PathRuleFeatureCondition]
    count: int
    hit_count: int
    hit_rate: float
    hit_rate_lift: float
    mean_forward_return: float
    return_lift: float
    win_rate_forward: float


class PathRuleMiningResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    etf_code: str
    etf_name: str
    db_path: str
    from_state: str
    target_state: str
    horizon: int
    target_mode: str
    signal_mode: str
    bias_source: str
    bias_ma: int
    momentum_window: int
    volume_ma_window: int
    rolling_window: int
    quantile_low: float
    quantile_high: float
    state_pattern: str = "MOM_BIAS_VOL"
    bucket_n: int
    max_combinations: int
    min_count: int
    top_k: int
    rules_above_baseline_only: bool
    baseline: PathRuleMiningBaselineBlock
    features: list[str]
    rules: list[PathRuleMiningRuleRow]


class ReportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    etf_code: str
    output_dir: str
    artifact_paths: dict[str, str]
    health: HealthResponse
    signal_research: SignalResearchResponse
    backtest: BacktestResponse
