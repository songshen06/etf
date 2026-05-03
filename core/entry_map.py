"""
ETF entry-map discovery (research-only).

Builds ``ETF → research_mode → BEST_STATE → DRIVER → ENTRY_ARCHETYPE`` using **only**
``run_path_quality`` and ``run_path_rule_mining`` — the same stack as
``quantlab analyze-path-quality`` / ``analyze-path-rules``.

This module does **not** run backtests, alter signal tiers, or replace CLI
commands. Snapshots are a *cached view*; ``discover_entry_map`` remains the
canonical generator.

**Cooperation (routing layer, not execution):**

- For research CLIs, ``default_from_state_for_code(snapshot, code)`` can
  suggest a default ``--from-state`` (path-quality / path-rules **from-state**
  is a research origin mask; it is **related but not identical** to backtest
  ``--signal-tier``).
- A plausible *future* translation for advisory layers: NEG → ``NEG`` tier,
  NEG_LOW → ``NEG_LOW``, NEG_LOW_HIGH → ``NEG_LOW_HIGH`` — still not
  interchangeable semantically; document when wiring backtest.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, model_validator

from core.runner import run_path_quality, run_path_rule_mining
from core.schemas import (
    BiasSourceEnum,
    PathQualityRequest,
    PathQualityTargetModeEnum,
    PathRuleMiningRequest,
    PathRuleMiningResponse,
    PathRuleMiningRuleRow,
    SignalModeEnum,
    StrategyMode,
)

CandidateState = Literal["NEG", "NEG_LOW", "NEG_LOW_HIGH"]
Driver = Literal["bias-led", "momentum-led", "volume-assisted", "unstable"]
EntryArchetype = Literal[
    "mean_reversion",
    "trend_follow",
    "reversal_confirmation",
    "conditional_breakout",
    "no_edge",
]
Confidence = Literal["LOW", "MEDIUM", "HIGH"]
ResearchMode = Literal["rolling", "full_sample"]
DataSufficiency = Literal["ok", "insufficient_for_rolling", "insufficient_overall"]

DEFAULT_CANDIDATE_STATES: tuple[CandidateState, ...] = ("NEG", "NEG_LOW", "NEG_LOW_HIGH")

DEFAULT_ETF_UNIVERSE: list[dict[str, str]] = [
    {"code": "159209", "name": "红利质量"},
    {"code": "159361", "name": "A500 ETF"},
    {"code": "510300", "name": "沪深300"},
    {"code": "510500", "name": "中证500"},
    {"code": "159531", "name": "南方中证2000ETF"},
    {"code": "510150", "name": "消费ETF"},
    {"code": "159992", "name": "医疗ETF"},
    {"code": "510410", "name": "资源ETF"},
    {"code": "588000", "name": "科创50"},
    {"code": "513050", "name": "中概互联"},
    {"code": "512880", "name": "证券ETF"},
    {"code": "515080", "name": "中证红利ETF"},
    {"code": "518880", "name": "黄金ETF"},
    {"code": "515880", "name": "通信ETF"},
]

# Top-rule bias bucket: "higher" = weak-side recovery starting from Q3+ territory
_TREND_BIAS_MIN_Q = 3
_BIAS_RANGE_RE = re.compile(r"^Q([1-5])(?:-Q([1-5]))?$", re.IGNORECASE)


class EntryMapNotes(BaseModel):
    model_config = {"extra": "forbid"}

    target_state: str
    horizon: int
    candidate_states: list[str]
    min_samples: int
    weak_hit_rate_floor: float
    auto_mode: bool = Field(
        True,
        description="If True, fall back to full_sample when rolling yields no eligible best_state",
    )
    signal_mode: str
    bias_ma: int
    path_quality_target_mode: str
    path_rule_features: list[str]
    path_rule_max_combinations: int
    path_rule_min_count: int
    path_rule_top_k: int


class StatePathQualityRow(BaseModel):
    model_config = {"extra": "forbid"}

    state: str
    total_samples: int
    hit_rate: float
    mean_forward_return: float


class EntryMapEtfRow(BaseModel):
    model_config = {"extra": "forbid"}

    code: str
    name: str
    research_mode: ResearchMode = Field(
        default="rolling",
        description="Path-quality / path-rules signal_mode used for this ETF (single mode, no mixing)",
    )
    data_sufficiency: DataSufficiency = Field(
        default="ok",
        description="ok=rolling sufficient; insufficient_for_rolling=used full_sample fallback; insufficient_overall=no eligible state in either mode",
    )
    best_state: str | None = Field(
        description="Path-quality origin state with highest hit_rate among eligible rows; null if none meet min_samples",
    )
    best_state_samples: int = Field(0, description="total_samples for best_state; 0 if best_state is null")
    best_state_hit_rate: float | None = None
    best_state_mean_forward_return: float | None = None
    driver: Driver
    top_rule: str | None = None
    top_rule_count: int | None = None
    top_rule_hit_rate: float | None = None
    top_rule_mean_forward_return: float | None = None
    entry_archetype: EntryArchetype
    confidence: Confidence
    weak_path_quality: bool = Field(
        ...,
        description="True if no eligible state, or best hit_rate < weak_hit_rate_floor",
    )
    state_metrics: list[StatePathQualityRow]
    strategy_mode: StrategyMode = Field(
        ...,
        description="hold=样本内未形成可交易入场结构；timing=存在可研究的 timing 结构（与 entry_archetype 一致导出）",
    )

    @model_validator(mode="before")
    @classmethod
    def _backfill_strategy_mode(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "strategy_mode" not in d and "entry_archetype" in d:
            d["strategy_mode"] = infer_strategy_mode_from_entry_archetype(d["entry_archetype"])  # type: ignore[arg-type]
        return d


class EntryMapSnapshotV1(BaseModel):
    model_config = {"extra": "forbid"}

    version: Literal["v1"] = "v1"
    notes: EntryMapNotes
    etfs: list[EntryMapEtfRow]
    generated_at: str | None = None


@dataclass(frozen=True)
class EntryMapDiscoveryConfig:
    """Shared research parameters for path-quality + path-rules (aligned with CLI defaults)."""

    db_path: str | None = None
    signal_mode: SignalModeEnum = SignalModeEnum.rolling
    bias_source: BiasSourceEnum = BiasSourceEnum.recompute
    bias_ma: int = 120
    momentum_window: int = 10
    volume_ma_window: int = 20
    rolling_window: int = 252
    quantile_low: float = 0.33
    quantile_high: float = 0.67
    target_state: str = "POS_HIGH_HIGH"
    horizon: int = 60
    path_quality_target_mode: PathQualityTargetModeEnum = PathQualityTargetModeEnum.ever
    min_samples: int = 5
    weak_hit_rate_floor: float = 0.40
    path_rule_features: tuple[str, ...] = ("bias_rate", "momentum", "volume_ratio")
    path_rule_max_combinations: int = 2
    path_rule_min_count: int = 5
    path_rule_top_k: int = 5
    bias_quantile_range: str | None = None


def _confidence_from_n(n: int, research_mode: ResearchMode) -> Confidence:
    """Sample-count bands; ``full_sample`` caps at MEDIUM (less robust than rolling)."""
    if n >= 20:
        c: Confidence = "HIGH"
    elif n >= 10:
        c = "MEDIUM"
    else:
        c = "LOW"
    if research_mode == "full_sample" and c == "HIGH":
        return "MEDIUM"
    return c


def _path_quality_request(
    etf_code: str,
    from_state: str,
    cfg: EntryMapDiscoveryConfig,
) -> PathQualityRequest:
    return PathQualityRequest(
        etf_code=etf_code,
        db_path=cfg.db_path,
        signal_mode=cfg.signal_mode,
        bias_source=cfg.bias_source,
        bias_ma=cfg.bias_ma,
        momentum_window=cfg.momentum_window,
        volume_ma_window=cfg.volume_ma_window,
        rolling_window=cfg.rolling_window,
        quantile_low=cfg.quantile_low,
        quantile_high=cfg.quantile_high,
        from_state=from_state,
        target_state=cfg.target_state,
        horizon=cfg.horizon,
        target_mode=cfg.path_quality_target_mode,
        bucket_features=("bias_rate", "momentum", "volume_ratio"),
        bucket_n=5,
        bias_quantile_range=cfg.bias_quantile_range,
    )


def _path_rule_request(etf_code: str, from_state: str, cfg: EntryMapDiscoveryConfig) -> PathRuleMiningRequest:
    return PathRuleMiningRequest(
        etf_code=etf_code,
        db_path=cfg.db_path,
        signal_mode=cfg.signal_mode,
        bias_source=cfg.bias_source,
        bias_ma=cfg.bias_ma,
        momentum_window=cfg.momentum_window,
        volume_ma_window=cfg.volume_ma_window,
        rolling_window=cfg.rolling_window,
        quantile_low=cfg.quantile_low,
        quantile_high=cfg.quantile_high,
        from_state=from_state,
        target_state=cfg.target_state,
        horizon=cfg.horizon,
        target_mode=cfg.path_quality_target_mode,
        features=cfg.path_rule_features,
        bucket_n=5,
        max_combinations=cfg.path_rule_max_combinations,
        min_count=cfg.path_rule_min_count,
        top_k=cfg.path_rule_top_k,
        rules_above_baseline_only=False,
        bias_quantile_range=cfg.bias_quantile_range,
    )


def discover_best_state(
    etf_code: str,
    cfg: EntryMapDiscoveryConfig,
    *,
    candidate_states: Sequence[CandidateState] | None = None,
) -> tuple[CandidateState | None, list[StatePathQualityRow], bool]:
    """
    Run path-quality once per candidate ``from_state``; pick the state with max
    ``hit_rate``, tie-break by larger ``total_samples``. States with
    ``total_samples < min_samples`` are ignored for selection but kept in
    ``state_metrics``.

    Returns:
        (best_state | None, all state rows, weak_path_quality)
        ``weak_path_quality`` is True if there is no eligible pick, or the
        chosen state's hit_rate is below ``weak_hit_rate_floor``.
    """
    states = tuple(candidate_states) if candidate_states is not None else DEFAULT_CANDIDATE_STATES
    rows: list[StatePathQualityRow] = []
    for st in states:
        pq = run_path_quality(_path_quality_request(etf_code, st, cfg))
        rows.append(
            StatePathQualityRow(
                state=st,
                total_samples=pq.total_samples,
                hit_rate=pq.hit_rate,
                mean_forward_return=pq.mean_forward_return,
            )
        )

    eligible = [r for r in rows if r.total_samples >= cfg.min_samples]
    if not eligible:
        return None, rows, True

    best = max(eligible, key=lambda r: (r.hit_rate, r.total_samples))
    weak = best.hit_rate < cfg.weak_hit_rate_floor
    # mypy: best.state is one of candidate_states
    return best.state, rows, weak  # type: ignore[return-value]


def run_path_quality_for_mode(
    etf_code: str,
    cfg: EntryMapDiscoveryConfig,
    mode: ResearchMode,
    *,
    candidate_states: Sequence[CandidateState] | None = None,
) -> tuple[CandidateState | None, list[StatePathQualityRow], bool]:
    """Thin wrapper: ``discover_best_state`` with ``signal_mode`` set to rolling or full_sample."""
    sm = SignalModeEnum.rolling if mode == "rolling" else SignalModeEnum.full_sample
    return discover_best_state(etf_code, replace(cfg, signal_mode=sm), candidate_states=candidate_states)


def select_research_mode(
    etf_code: str,
    cfg: EntryMapDiscoveryConfig,
    *,
    auto_mode: bool = True,
    candidate_states: Sequence[CandidateState] | None = None,
) -> tuple[
    ResearchMode,
    DataSufficiency,
    EntryMapDiscoveryConfig,
    CandidateState | None,
    list[StatePathQualityRow],
    bool,
]:
    """
    Try **rolling** first (default). If no eligible ``best_state`` (all states
    ``< min_samples`` or zero), optionally fall back to **full_sample** — never
    mix modes within one ETF.

    Returns:
        (research_mode, data_sufficiency, cfg_active, best_state, state_metrics, weak_path_quality)
    """
    cfg_r = replace(cfg, signal_mode=SignalModeEnum.rolling)
    best_r, rows_r, weak_r = discover_best_state(etf_code, cfg_r, candidate_states=candidate_states)
    if best_r is not None:
        return "rolling", "ok", cfg_r, best_r, rows_r, weak_r

    if not auto_mode:
        return "rolling", "insufficient_overall", cfg_r, None, rows_r, weak_r

    cfg_f = replace(cfg, signal_mode=SignalModeEnum.full_sample)
    best_f, rows_f, weak_f = discover_best_state(etf_code, cfg_f, candidate_states=candidate_states)
    if best_f is not None:
        return "full_sample", "insufficient_for_rolling", cfg_f, best_f, rows_f, weak_f

    # Both modes exhausted; last metrics are full_sample (what we would use if any sample existed)
    return "full_sample", "insufficient_overall", cfg_f, None, rows_f, weak_f


def _bias_range_low_q(bucket_range: str) -> int | None:
    m = _BIAS_RANGE_RE.match(str(bucket_range).strip())
    if not m:
        return None
    return int(m.group(1))


def top_rule_bias_suggests_trend_follow(rule: PathRuleMiningRuleRow) -> bool:
    """
    ``trend_follow`` requires bias-led rules whose bias condition sits in Q3+
    (less weak / higher bucket rank on the global 5-bucket scale).
    """
    for c in rule.feature_conditions:
        if c.feature == "bias_rate":
            lo = _bias_range_low_q(c.bucket_range)
            if lo is not None and lo >= _TREND_BIAS_MIN_Q:
                return True
    return False


def infer_driver(rules: list[PathRuleMiningRuleRow], baseline_hit_rate: float) -> Driver:
    """
    Explicit heuristics (interpretable, no ML):

    - **bias-led**: single-factor top rule on ``bias_rate`` only.
    - **momentum-led**: single-factor ``momentum``, or a bias+momentum pair
      (confirmation structure).
    - **volume-assisted**: two-factor rule where ``volume_ratio`` appears
      together with ``bias_rate`` or ``momentum`` (never volume alone).
    - **unstable**: no rules, top rule weaker than baseline, contradictory
      lifts among top rules, volume-only, or unrecognized pattern.
    """
    if not rules:
        return "unstable"
    top = rules[0]
    if top.hit_rate + 1e-12 < baseline_hit_rate:
        return "unstable"
    if len(rules) >= 2:
        a, b = rules[0].hit_rate_lift, rules[1].hit_rate_lift
        if a > 0.03 and b < -0.03:
            return "unstable"

    feats = [c.feature for c in top.feature_conditions]
    fs = set(feats)
    if len(feats) == 1:
        if feats[0] == "bias_rate":
            return "bias-led"
        if feats[0] == "momentum":
            return "momentum-led"
        if feats[0] == "volume_ratio":
            return "unstable"
        return "unstable"

    if len(feats) == 2:
        if "volume_ratio" in fs and ("bias_rate" in fs or "momentum" in fs):
            return "volume-assisted"
        if "bias_rate" in fs and "momentum" in fs:
            return "momentum-led"
    return "unstable"


def discover_driver(
    etf_code: str,
    best_state: str,
    cfg: EntryMapDiscoveryConfig,
) -> tuple[Driver, PathRuleMiningResponse]:
    """Run path-rules on ``best_state`` only; return driver label + full response (for auditing)."""
    resp = run_path_rule_mining(_path_rule_request(etf_code, best_state, cfg))
    d = infer_driver(resp.rules, resp.baseline.hit_rate)
    return d, resp


def infer_strategy_mode_from_entry_archetype(arch: EntryArchetype) -> StrategyMode:
    """no_edge → 简单持有优先；其余 archetype → 可考虑 timing 策略研究。"""
    if arch == "no_edge":
        return "hold"
    return "timing"


def classify_entry_archetype(
    best_state: CandidateState | None,
    driver: Driver,
    top_rule: PathRuleMiningRuleRow | None,
    *,
    weak_path_quality: bool,
    n_samples: int,
    min_samples: int,
) -> EntryArchetype:
    """
    Small fixed taxonomy. Order matters (e.g. NEG_LOW + momentum before NEG_LOW + bias).
    """
    if best_state is None or n_samples < min_samples:
        return "no_edge"
    if weak_path_quality or driver == "unstable":
        return "no_edge"

    if best_state == "NEG_LOW" and driver in ("momentum-led", "volume-assisted"):
        return "conditional_breakout"
    if best_state == "NEG_LOW" and driver == "bias-led":
        return "mean_reversion"
    if best_state == "NEG" and driver == "bias-led" and top_rule is not None:
        if top_rule_bias_suggests_trend_follow(top_rule):
            return "trend_follow"
        return "no_edge"
    if best_state == "NEG_LOW_HIGH" and driver in ("momentum-led", "volume-assisted"):
        return "reversal_confirmation"
    return "no_edge"


def discover_entry_map(
    universe: Sequence[dict[str, str]],
    cfg: EntryMapDiscoveryConfig,
    *,
    candidate_states: Sequence[CandidateState] | None = None,
    include_generated_at: bool = True,
    auto_mode: bool = True,
) -> EntryMapSnapshotV1:
    """Full workflow over a list of ``{"code","name"}`` dicts.

    When ``auto_mode`` is True (default), path-quality / path-rules use **rolling**
    first; if no eligible ``best_state``, falls back to **full_sample** for that
    ETF only. No cross-mode mixing within an ETF.
    """
    states = list(candidate_states) if candidate_states is not None else list(DEFAULT_CANDIDATE_STATES)
    etf_rows: list[EntryMapEtfRow] = []

    for u in universe:
        code = str(u["code"]).strip()
        name = str(u.get("name", code)).strip()

        rmode, suff, cfg_active, best, state_metrics, weak = select_research_mode(
            code,
            cfg,
            auto_mode=auto_mode,
            candidate_states=candidate_states,
        )
        driver: Driver = "unstable"
        top_rule: PathRuleMiningRuleRow | None = None
        tr_count = tr_hr = tr_mfr = None
        top_rule_s: str | None = None

        if best is not None:
            driver, pr_resp = discover_driver(code, best, cfg_active)
            if pr_resp.rules:
                top_rule = pr_resp.rules[0]
                top_rule_s = top_rule.rule
                tr_count = top_rule.count
                tr_hr = top_rule.hit_rate
                tr_mfr = top_rule.mean_forward_return

        st_row = next((r for r in state_metrics if r.state == best), None)
        n_best = int(st_row.total_samples) if st_row is not None else 0
        bhr = float(st_row.hit_rate) if st_row is not None else None
        bmfr = float(st_row.mean_forward_return) if st_row is not None else None

        arch = classify_entry_archetype(
            best,
            driver,
            top_rule,
            weak_path_quality=weak,
            n_samples=n_best,
            min_samples=cfg.min_samples,
        )
        conf = _confidence_from_n(n_best, rmode)
        strategy_mode = infer_strategy_mode_from_entry_archetype(arch)

        etf_rows.append(
            EntryMapEtfRow(
                code=code,
                name=name,
                research_mode=rmode,
                data_sufficiency=suff,
                best_state=best,
                best_state_samples=n_best,
                best_state_hit_rate=bhr,
                best_state_mean_forward_return=bmfr,
                driver=driver,
                top_rule=top_rule_s,
                top_rule_count=tr_count,
                top_rule_hit_rate=tr_hr,
                top_rule_mean_forward_return=tr_mfr,
                entry_archetype=arch,
                confidence=conf,
                weak_path_quality=weak,
                state_metrics=state_metrics,
                strategy_mode=strategy_mode,
            )
        )

    notes = EntryMapNotes(
        target_state=cfg.target_state,
        horizon=cfg.horizon,
        candidate_states=states,
        min_samples=cfg.min_samples,
        weak_hit_rate_floor=cfg.weak_hit_rate_floor,
        auto_mode=auto_mode,
        signal_mode=str(cfg.signal_mode.value),
        bias_ma=cfg.bias_ma,
        path_quality_target_mode=str(cfg.path_quality_target_mode.value),
        path_rule_features=list(cfg.path_rule_features),
        path_rule_max_combinations=cfg.path_rule_max_combinations,
        path_rule_min_count=cfg.path_rule_min_count,
        path_rule_top_k=cfg.path_rule_top_k,
    )
    ts = datetime.now(timezone.utc).isoformat() if include_generated_at else None
    return EntryMapSnapshotV1(notes=notes, etfs=etf_rows, generated_at=ts)


def save_entry_map_snapshot(path: str | Path, snapshot: EntryMapSnapshotV1) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")


def load_entry_map_snapshot(path: str | Path) -> EntryMapSnapshotV1:
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if data.get("version") != "v1":
        raise ValueError(f"unsupported entry map version: {data.get('version')!r}")
    return EntryMapSnapshotV1.model_validate(data)


def load_strategy_mode_from_entry_map_file(path: str | Path, etf_code: str) -> StrategyMode | None:
    """供 runner 合并 recommendation.strategy_mode；文件缺失或代码不在宇宙时返回 None。"""
    try:
        snap = load_entry_map_snapshot(path)
    except Exception:
        return None
    row = archetype_row_for_code(snap, etf_code)
    if row is None:
        return None
    return row.strategy_mode


def default_from_state_for_code(snapshot: EntryMapSnapshotV1, etf_code: str) -> str | None:
    """Routing helper: suggested path-quality / path-rules ``from-state`` for an ETF code."""
    c = str(etf_code).strip()
    for row in snapshot.etfs:
        if row.code == c:
            return row.best_state
    return None


def archetype_row_for_code(snapshot: EntryMapSnapshotV1, etf_code: str) -> EntryMapEtfRow | None:
    c = str(etf_code).strip()
    for row in snapshot.etfs:
        if row.code == c:
            return row
    return None
