"""
Single entry points for health, signal research, backtest, and full report.

CLI and Streamlit call these functions only — no duplicated quant logic.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .charts import (
    buy_hold_normalized_from_ohlcv,
    figure_backtest_dashboard,
    figure_close_history,
    figure_equity_curves_compare,
    figure_equity_and_drawdown,
    figure_event_study_vs_horizon,
    figure_to_json,
)
from .data_loader import etf_name_map, load_etf_sqlite
from .entry_exit_matching import build_entry_exit_matching_diagnostics
from .entry_signal_diagnostics import build_entry_signal_diagnostics
from .exit_rules import (
    ExitRuleEvalRow,
    ExitRuleSpec,
    build_exit_context,
    exit_rule_display_name,
    exit_rule_plain_explanation,
    resolve_cli_exit_rule,
)
from .data_validation import collect_invalid_rows, issues_to_records, validate_ohlcv_panel
from .event_study import tier_event_studies
from .jsonutil import dataframe_records_json_safe, json_safe_value
from .paths import resolve_db_path
from .pipeline import prepare_indicator_panel, prepare_research_frame
from .portfolio_backtest import drawdown_series, run_portfolio_backtest
from .position_rules import (
    profile_label_zh,
    profile_weight_percent_triple,
    tier_weight_labels,
    weights_by_strategy_profile,
)
from .multi_objective_exit import ObjectiveName, build_multi_objective_decision
from .recommendation import rank_exit_rules_on_frame, recommend_strategy_setup
from .schemas import (
    BacktestComparisonRow,
    BacktestParamSourceEnum,
    BacktestRequest,
    BacktestResponse,
    ExitRuleComparisonRow,
    ExitRuleEvalSnapshot,
    ExitRuleOptimizationDiagnosticRow,
    MultiObjectiveDecisionBlock,
    EntryExitMatchingDiagnosticsBlock,
    EntrySignalDiagnosticsBlock,
    ChartSpec,
    EventStudyRow,
    HealthRequest,
    HealthResponse,
    LatestSignalBarSnapshot,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationSnapshot,
    ReportRequest,
    ReportResponse,
    SignalDimensionsSnapshot,
    SignalModeEnum,
    SignalParamSourceEnum,
    SignalResearchRequest,
    SignalResearchResponse,
    StateRankingRequest,
    StateRankingResponse,
    StateRankRow,
    PathQualityFeatureBreakdown,
    PathQualityRequest,
    PathQualityResponse,
    PathQualityBucketRow,
    PathRuleFeatureCondition,
    PathRuleMiningBaselineBlock,
    PathRuleMiningRequest,
    PathRuleMiningResponse,
    PathRuleMiningRuleRow,
    StateTransitionHorizonBlock,
    StateTransitionRequest,
    StateTransitionResponse,
    StateTransitionRow,
    TimeSeriesPoint,
    TradeRow,
    ValidationIssueRow,
)


ENTRY_TIER_HIERARCHICAL = "HIERARCHICAL"


def _db_path(req_db: str | None) -> str:
    return str(resolve_db_path(req_db))


def _merge_rec_dict_with_entry_map_strategy_mode(
    req: RecommendationRequest | BacktestRequest | SignalResearchRequest,
    rec_dict: dict[str, Any],
) -> dict[str, Any]:
    """合并 strategy_mode（显式路径优先；否则尝试仓库默认 artifacts/entry_map_v1.json）。"""
    path = getattr(req, "entry_map_json_path", None)
    path_s = str(path).strip() if path is not None else ""
    if not path_s:
        path_s = str((Path(__file__).resolve().parents[1] / "artifacts" / "entry_map_v1.json"))
    from core.entry_map import load_strategy_mode_from_entry_map_file

    sm = load_strategy_mode_from_entry_map_file(path_s, str(req.etf_code))
    if sm is None:
        return rec_dict
    merged = dict(rec_dict)
    merged["strategy_mode"] = sm
    return merged


def _mode_str(mode: Any) -> str:
    return mode.value if hasattr(mode, "value") else str(mode)


def _signal_dims_snap(
    req: SignalResearchRequest | BacktestRequest,
    *,
    bias_ma_override: int | None = None,
    signal_mode_override: Any | None = None,
) -> SignalDimensionsSnapshot:
    bm = int(bias_ma_override if bias_ma_override is not None else req.bias_ma)
    sm = _mode_str(signal_mode_override if signal_mode_override is not None else req.signal_mode)
    return SignalDimensionsSnapshot(
        neg_momentum_window=int(req.momentum_window),
        low_bias_ma=bm,
        high_volume_ma_window=int(req.volume_ma_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        signal_mode=sm,
        rolling_window=int(req.rolling_window),
        bias_source=str(req.bias_source.value),
    )


def _event_study_rows(df: pd.DataFrame) -> list[EventStudyRow]:
    rows: list[EventStudyRow] = []
    for rec in df.to_dict(orient="records"):
        rows.append(EventStudyRow.model_validate(rec))
    return rows


def _apply_bias_quantile_to_frame_for_hierarchical_backtest(
    df: pd.DataFrame,
    *,
    bias_ma_window: int,
    bias_quantile_range_str: str | None,
) -> pd.DataFrame:
    """Copy-on-write: zero ``signal_tier`` where tier>0 and bias quintile outside range."""
    bias_ok = _bias_quantile_entry_ok_mask(
        df,
        bias_ma_window=bias_ma_window,
        bias_quantile_range_str=bias_quantile_range_str,
    )
    if bias_ok is None:
        return df
    out = df.copy()
    t = out["signal_tier"].to_numpy(dtype=int)
    ok_np = bias_ok.to_numpy(dtype=bool)
    t = np.where((t > 0) & ~ok_np, 0, t)
    out["signal_tier"] = t
    return out


def _bias_quantile_entry_ok_mask(
    df: pd.DataFrame,
    *,
    bias_ma_window: int,
    bias_quantile_range_str: str | None,
) -> pd.Series | None:
    """
    Boolean Series aligned to ``df``: True where full-sample bias quintile lies in the
    requested range. ``None`` if no filter.
    """
    from quantlab.filters.quantile_filter import (
        mask_in_quantile_range,
        parse_quantile_range,
    )

    qr = parse_quantile_range(bias_quantile_range_str)
    if qr is None:
        return None
    from .quantile_buckets import BIAS_BUCKET_COL

    if BIAS_BUCKET_COL not in df.columns:
        raise KeyError(
            f"missing {BIAS_BUCKET_COL!r}; prepare_research_frame must assign global buckets before backtest --bias-q"
        )
    labs = df[BIAS_BUCKET_COL]
    ok = mask_in_quantile_range(labs, qr)
    return pd.Series(ok, index=df.index)


def build_latest_signal_bar_snapshot(
    df: pd.DataFrame,
    req: BacktestRequest,
    eff: dict[str, Any],
    *,
    code: str,
    name: str,
    wmap: dict[int, float],
    entry_signal_tier: str | None,
    executed_signal_tier_str: str,
) -> LatestSignalBarSnapshot:
    """Last row flags + tiers; matches portfolio entry logic (see run_portfolio_backtest)."""
    from .signal_engine import research_tier_mask

    if df.empty or "NEG" not in df.columns:
        raise ValueError("build_latest_signal_bar_snapshot requires prepare_research_frame output")
    last = df.iloc[-1]
    as_of = pd.Timestamp(last["date"]).strftime("%Y-%m-%d")
    neg = bool(last["NEG"])
    low = bool(last["LOW"])
    high = bool(last["HIGH"])
    st_raw = int(last["signal_tier"])
    bias_eff = int(eff["bias_ma_eff"])
    profile_str = str(eff["profile_str"])

    df_adj = (
        _apply_bias_quantile_to_frame_for_hierarchical_backtest(
            df,
            bias_ma_window=bias_eff,
            bias_quantile_range_str=req.bias_quantile_range,
        )
        if entry_signal_tier is None
        else df
    )
    st_eff = int(df_adj["signal_tier"].iloc[-1])

    if entry_signal_tier is None:
        backtest_entry_active = st_eff >= 1 and float(wmap.get(st_eff, 0.0)) > 0
        implied = {0: "无层", 1: "NEG", 2: "NEG+LOW", 3: "NEG+LOW+HIGH"}.get(st_eff, f"tier={st_eff}")
    else:
        mask = research_tier_mask(df, entry_signal_tier)
        bias_ok = _bias_quantile_entry_ok_mask(
            df,
            bias_ma_window=bias_eff,
            bias_quantile_range_str=req.bias_quantile_range,
        )
        if bias_ok is not None:
            mask = mask & bias_ok
        backtest_entry_active = bool(mask.iloc[-1])
        layer = entry_signal_tier.replace("_", "+")
        implied = f"{layer}（条件满足）" if backtest_entry_active else f"未满足 {layer}"

    mode_zh = "分层入场（NEG→NEG+LOW→NEG+LOW+HIGH）" if executed_signal_tier_str == ENTRY_TIER_HIERARCHICAL else f"锁定单层 {executed_signal_tier_str.replace('_', '+')} 入场"
    readout_parts: list[str] = [
        "【三维度】NEG=动量相对弱势；LOW=乖离相对低位；HIGH=量比相对放量（均相对 rolling/full_sample 分位）。",
    ]
    if entry_signal_tier is None:
        if not neg:
            readout_parts.append(
                "【分层 signal_tier】引擎规则：**只有 NEG=true 时** 才会把 tier 设为 1/2/3；"
                "NEG=false 时 **tier 固定为 0**（你看到的「无层」），**LOW/HIGH 单独为真也不会抬高 tier**。"
            )
            if low or high:
                readout_parts.append(
                    f"【你这条数据】LOW={low}、HIGH={high} 表示乖离/量比各自满足弱势或放量线，"
                    "但因 **动量不满足 NEG**，仍 **不算** 进入 NEG+LOW / NEG+LOW+HIGH 分层，故不会按分层信号开仓。"
                )
            else:
                readout_parts.append("【你这条数据】动量、乖离、量比均未同时走到「分层所需」组合，故 tier=0。")
        else:
            readout_parts.append(
                f"【分层】NEG 已成立：原始 tier={st_raw}；"
                f"经 bias 分位过滤后 effective={st_eff}（与回测主图一致时即此档仓位映射）。"
            )
            if st_raw != st_eff:
                readout_parts.append("（raw 与 effective 不同：说明启用了 --bias-q 等过滤，把部分 bar 的 tier 置 0。）")
    else:
        readout_parts.append(
            f"【锁层】只判断「{entry_signal_tier.replace('_', '+')}」组合是否成立；"
            "与上面 signal_tier 数值关系见引擎 research_tier_mask。"
        )
    nxt = "会" if backtest_entry_active else "不会"
    readout_parts.append(
        f"【下一交易日开盘】在 **{mode_zh}** 与当前画像下，引擎视为 **{nxt}** 因本 bar 触发新开仓（与回测逻辑一致）。"
    )
    readout_zh = "\n".join(readout_parts)

    from .quantile_buckets import BIAS_BUCKET_COL, MOMENTUM_BUCKET_COL, VOLUME_RATIO_BUCKET_COL
    from quantlab.filters.quantile_filter import normalize_bucket_label

    last = df.iloc[-1] if not df.empty else None
    bias_bucket = None
    momentum_bucket = None
    volume_bucket = None
    if last is not None:
        if BIAS_BUCKET_COL in df.columns:
            bias_bucket = normalize_bucket_label(last.get(BIAS_BUCKET_COL))
        if MOMENTUM_BUCKET_COL in df.columns:
            momentum_bucket = normalize_bucket_label(last.get(MOMENTUM_BUCKET_COL))
        if VOLUME_RATIO_BUCKET_COL in df.columns:
            volume_bucket = normalize_bucket_label(last.get(VOLUME_RATIO_BUCKET_COL))

    return LatestSignalBarSnapshot(
        etf_code=code,
        etf_name=name,
        as_of_date=as_of,
        neg=neg,
        low=low,
        high=high,
        bias_bucket=bias_bucket,
        momentum_bucket=momentum_bucket,
        volume_ratio_bucket=volume_bucket,
        signal_tier_raw=st_raw,
        signal_tier_effective=st_eff,
        implied_layer_zh=implied,
        readout_zh=readout_zh,
        backtest_entry_active=backtest_entry_active,
        execution_entry_mode=executed_signal_tier_str,
        signal_mode=_mode_str(eff["signal_mode_eff"]),
        bias_ma_effective=bias_eff,
        strategy_profile_effective=profile_str,
        strategy_profile_zh=profile_label_zh(profile_str),
        applied_recommendation_defaults=bool(eff["applied_signal"]),
        applied_backtest_recommendation=bool(eff["applied_bt_rec"]),
    )


def run_latest_signal_state(req: BacktestRequest) -> LatestSignalBarSnapshot:
    """Standalone: same effective signal params as backtest, no portfolio run."""
    req = req.model_copy(update={"compare_profiles": False, "compare_manual_vs_recommended": False})
    db = _db_path(req.db_path)
    code = str(req.etf_code)
    name = str(etf_name_map(db).get(code, code))
    df_load, _ = load_etf_sqlite(db, code)
    rec = recommend_strategy_setup(
        df_load,
        code,
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        include_exit_rules=False,
    )
    eff = _backtest_effective_params(req, rec)
    mode = _mode_str(eff["signal_mode_eff"])
    df, _ = prepare_research_frame(
        db,
        code,
        signal_mode=mode,  # type: ignore[arg-type]
        bias_ma_window=int(eff["bias_ma_eff"]),
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        bias_source=str(req.bias_source.value),  # type: ignore[arg-type]
    )
    wmap = weights_by_strategy_profile(str(eff["profile_str"]))
    ent = req.entry_signal_tier
    executed = ent if ent else ENTRY_TIER_HIERARCHICAL
    snap = build_latest_signal_bar_snapshot(
        df,
        req,
        eff,
        code=code,
        name=name,
        wmap=wmap,
        entry_signal_tier=ent,
        executed_signal_tier_str=executed,
    )
    return snap


def _metrics_clean(m: dict[str, Any]) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {}
    for k, v in m.items():
        if isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            x = float(v)
            out[k] = x if np.isfinite(x) else None
        else:
            out[k] = None
    return out


def _describe_to_nested(df: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    desc = df.describe().T
    out: dict[str, dict[str, float | int | None]] = {}
    for idx, row in desc.iterrows():
        inner: dict[str, float | int | None] = {}
        for k, v in row.items():
            inner[str(k)] = json_safe_value(v)  # type: ignore[assignment]
        out[str(idx)] = inner
    return out


def run_health(req: HealthRequest) -> HealthResponse:
    db = _db_path(req.db_path)
    code = str(req.etf_code)
    name = str(etf_name_map(db).get(code, code))

    conn = sqlite3.connect(str(db))
    try:
        raw = pd.read_sql_query(
            """
            SELECT trade_date AS date, price AS close, volume, prev_close
            FROM etf_daily_metrics
            WHERE etf_code = ?
            ORDER BY trade_date
            """,
            conn,
            params=(code,),
        )
    finally:
        conn.close()

    if raw.empty:
        raise ValueError(f"No rows for etf_code={code!r} in {db}")

    pre = pd.DataFrame()
    pre["date"] = pd.to_datetime(raw["date"])
    pre["close"] = pd.to_numeric(raw["close"], errors="coerce")
    pre["open"] = pd.to_numeric(raw["prev_close"], errors="coerce")
    pre["volume"] = pd.to_numeric(raw["volume"], errors="coerce")

    invalid_full = collect_invalid_rows(pre)
    lim = int(req.invalid_row_limit)
    invalid_sample = invalid_full.head(lim) if lim > 0 else invalid_full.iloc[:0]

    cleaned, vres = validate_ohlcv_panel(pre)
    issues = [ValidationIssueRow.model_validate(x) for x in issues_to_records(vres.issues)]
    summary = _describe_to_nested(cleaned[["close", "open", "volume"]])

    hist = figure_close_history(cleaned["date"], cleaned["close"], title=f"{code} {name} — close (cleaned)")
    charts = [
        ChartSpec(
            chart_id="health.close_history",
            title=hist.layout.title.text if hist.layout and hist.layout.title else "Close",
            plotly_json=figure_to_json(hist),
        )
    ]

    return HealthResponse(
        etf_code=code,
        etf_name=name,
        db_path=db,
        rows_in=vres.rows_in,
        rows_out=vres.rows_out,
        open_fallback_rows=vres.open_fallback_rows,
        issues=issues,
        invalid_row_count=int(len(invalid_full)),
        invalid_rows_sample=dataframe_records_json_safe(invalid_sample),
        summary_stats=summary,
        charts=charts,
    )


def run_recommendation(req: RecommendationRequest) -> RecommendationResponse:
    db = _db_path(req.db_path)
    code = str(req.etf_code)
    name = str(etf_name_map(db).get(code, code))
    df_load, _ = load_etf_sqlite(db, code)
    rec = recommend_strategy_setup(
        df_load,
        code,
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        eval_horizon=int(req.eval_horizon),
        top_k=int(req.top_k),
        include_exit_rules=bool(req.include_exit_rules),
    )
    snap = RecommendationSnapshot.model_validate(
        _merge_rec_dict_with_entry_map_strategy_mode(req, rec.to_dict())
    )
    return RecommendationResponse(etf_code=code, etf_name=name, db_path=db, recommendation=snap)


def run_signal_research(req: SignalResearchRequest) -> SignalResearchResponse:
    db = _db_path(req.db_path)
    code = str(req.etf_code)
    name = str(etf_name_map(db).get(code, code))

    df_load, _ = load_etf_sqlite(db, code)
    rec = recommend_strategy_setup(
        df_load,
        code,
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
    )
    rec_snap = RecommendationSnapshot.model_validate(
        _merge_rec_dict_with_entry_map_strategy_mode(req, rec.to_dict())
    )

    signal_mode_eff: SignalModeEnum = req.signal_mode
    bias_ma_eff = int(req.bias_ma)
    applied = False
    if req.signal_param_source == SignalParamSourceEnum.auto:
        applied = True
        try:
            signal_mode_eff = SignalModeEnum(str(rec.recommended_mode))
        except ValueError:
            signal_mode_eff = SignalModeEnum.rolling
        bias_ma_eff = int(rec.recommended_bias_ma)

    mode = _mode_str(signal_mode_eff)
    df, _vres = prepare_research_frame(
        db,
        code,
        signal_mode=mode,  # type: ignore[arg-type]
        bias_ma_window=bias_ma_eff,
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        bias_source=str(req.bias_source.value),  # type: ignore[arg-type]
    )

    horizons = tuple(int(x) for x in req.event_horizons)
    studies = tier_event_studies(df, horizons=horizons)
    dims = _signal_dims_snap(req, bias_ma_override=bias_ma_eff, signal_mode_override=signal_mode_eff)
    event_studies = {tier: _event_study_rows(tdf) for tier, tdf in studies.items()}

    charts: list[ChartSpec] = []
    for tier, tdf in studies.items():
        f1 = figure_event_study_vs_horizon(
            tdf,
            y_col="win_rate",
            title=f"Win rate vs horizon — {tier}",
            y_tick_pct_decimals=1,
        )
        charts.append(
            ChartSpec(
                chart_id=f"signal.{tier}.win_rate",
                title=str(f1.layout.title.text) if f1.layout and f1.layout.title else tier,
                plotly_json=figure_to_json(f1),
            )
        )
        f2 = figure_event_study_vs_horizon(
            tdf,
            y_col="mean_return",
            title=f"Mean return vs horizon — {tier}",
            y_tick_pct_decimals=2,
        )
        charts.append(
            ChartSpec(
                chart_id=f"signal.{tier}.mean_return",
                title=str(f2.layout.title.text) if f2.layout and f2.layout.title else tier,
                plotly_json=figure_to_json(f2),
            )
        )

    return SignalResearchResponse(
        etf_code=code,
        etf_name=name,
        db_path=db,
        signal_param_source=str(req.signal_param_source.value),
        applied_recommendation_defaults=applied,
        recommendation=rec_snap,
        default_chart_tier=str(rec.default_signal),
        signal_mode=mode,
        bias_source=str(req.bias_source.value),
        bias_ma=bias_ma_eff,
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        signal_dimensions=dims,
        event_horizons=horizons,
        event_studies=event_studies,
        charts=charts,
    )


def _backtest_effective_params(req: BacktestRequest, rec: Any) -> dict[str, Any]:
    full_rec = req.backtest_param_source == BacktestParamSourceEnum.recommended
    if full_rec:
        try:
            sm = SignalModeEnum(str(rec.recommended_mode))
        except ValueError:
            sm = SignalModeEnum.rolling
        bm = int(rec.recommended_bias_ma)
        hd = max(1, min(2000, int(rec.recommended_horizon_focus)))
        prof = str(rec.recommended_position_profile)
        return {
            "signal_mode_eff": sm,
            "bias_ma_eff": bm,
            "hold_eff": hd,
            "profile_str": prof,
            "applied_signal": True,
            "applied_bt_rec": True,
        }
    sm = req.signal_mode
    bm = int(req.bias_ma)
    applied_signal = False
    if req.signal_param_source == SignalParamSourceEnum.auto:
        try:
            sm = SignalModeEnum(str(rec.recommended_mode))
        except ValueError:
            sm = SignalModeEnum.rolling
        bm = int(rec.recommended_bias_ma)
        applied_signal = True
    hd = int(req.hold_days)
    prof = _mode_str(req.strategy_profile)
    return {
        "signal_mode_eff": sm,
        "bias_ma_eff": bm,
        "hold_eff": hd,
        "profile_str": prof,
        "applied_signal": applied_signal,
        "applied_bt_rec": False,
    }


def _interpretation_notes_backtest(
    *,
    fit_level: str,
    applied_bt_rec: bool,
    applied_signal: bool,
    best_signal: str,
    entry_signal_tier_fixed: str | None = None,
) -> list[str]:
    notes: list[str] = []
    if fit_level == "high":
        notes.append("框架适配度较高：更适合作为「核心框架 ETF」做结构性跟踪；回测仍属样本内战术检验。")
    elif fit_level == "low":
        notes.append("框架适配度偏低：宜视为战术/实验性配置；推荐信号为网格内样本内最优，而非保守默认 NEG。")
    else:
        notes.append("中等适配：建议结合信号研究表与推荐参数交叉验证；回测为战术层验证。")
    if applied_bt_rec:
        notes.append("已一键套用推荐：信号 mode/乖离窗、持有期、仓位画像均来自规则推荐层。")
    elif applied_signal:
        notes.append("仅信号 mode/乖离窗来自推荐；持有期与仓位画像为手动选择。")
    else:
        notes.append("全手动参数；可与推荐 bundle 对照解读。")
    if entry_signal_tier_fixed:
        notes.append(
            f"入场信号层已 **锁定为 {entry_signal_tier_fixed.replace('_', '+')}**（实验参数），"
            "未使用分层引擎在多层之间自动切换触发。"
        )
    notes.append(f"网格最优信号层（样本内）：{best_signal.replace('_', '+')}。")
    return notes


def _tier_display_bt(sig: str) -> str:
    return str(sig or "").replace("_", "+")


def _weights_readable_pct(weights: dict[str, float]) -> str:
    """Human-readable state→weight from actual keys (ordered tier keys first, then any extras)."""
    order = ("NEG", "NEG_LOW", "NEG_LOW_HIGH")
    labels = {"NEG": "NEG", "NEG_LOW": "NEG+LOW", "NEG_LOW_HIGH": "NEG+LOW+HIGH"}
    parts: list[str] = []
    seen: set[str] = set()
    for k in order:
        if k not in weights:
            continue
        seen.add(k)
        disp = labels.get(k, k)
        parts.append(f"{disp} {float(weights[k]) * 100:.0f}%")
    for k in sorted(weights.keys(), key=str):
        if k in seen:
            continue
        parts.append(f"{labels.get(k, k)} {float(weights[k]) * 100:.0f}%")
    return " / ".join(parts) if parts else "—"


def _metric_float_from_map(m: dict[str, Any], key: str) -> float | None:
    v = m.get(key)
    if v is None:
        return None
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _exit_comparison_rows_from_snapshots(cands: list[Any]) -> list[ExitRuleComparisonRow]:
    rows: list[ExitRuleComparisonRow] = []
    for c in sorted(cands, key=lambda x: int(x.rank)):
        m = c.metrics or {}
        dn = (c.display_name or c.label_zh or c.rule_id).strip()
        rows.append(
            ExitRuleComparisonRow(
                rank=int(c.rank),
                rule_id=str(c.rule_id),
                display_name=dn,
                score=float(c.score),
                n_trades=int(c.n_trades),
                eligible=bool(c.eligible),
                total_return=_metric_float_from_map(m, "total_return"),
                annualized_return=_metric_float_from_map(m, "annualized_return"),
                max_drawdown=_metric_float_from_map(m, "max_drawdown"),
                sharpe_ratio=_metric_float_from_map(m, "sharpe_ratio"),
                calmar_ratio=_metric_float_from_map(m, "calmar_ratio"),
            )
        )
    return rows


def _exit_eval_rows_to_comparison_rows(rows: list[ExitRuleEvalRow]) -> list[ExitRuleComparisonRow]:
    out: list[ExitRuleComparisonRow] = []
    for er in rows:
        met = er.metrics or {}
        out.append(
            ExitRuleComparisonRow(
                rank=int(er.rank),
                rule_id=str(er.spec.rule_id),
                display_name=str(er.display_name).strip(),
                score=float(er.score),
                n_trades=int(er.n_trades),
                eligible=bool(er.eligible),
                total_return=_metric_float_from_map(met, "total_return"),
                annualized_return=_metric_float_from_map(met, "annualized_return"),
                max_drawdown=_metric_float_from_map(met, "max_drawdown"),
                sharpe_ratio=_metric_float_from_map(met, "sharpe_ratio"),
                calmar_ratio=_metric_float_from_map(met, "calmar_ratio"),
            )
        )
    return out


def _exit_optimization_diagnostics(
    rows: list[ExitRuleEvalRow],
    *,
    optimize_ran: bool,
    selected_rule_id: str | None,
) -> list[ExitRuleOptimizationDiagnosticRow]:
    out: list[ExitRuleOptimizationDiagnosticRow] = []
    for er in rows:
        rid = str(er.spec.rule_id)
        out.append(
            ExitRuleOptimizationDiagnosticRow(
                rule_id=rid,
                display_name=str(er.display_name).strip(),
                rank=int(er.rank),
                score=float(er.score),
                eligible=bool(er.eligible),
                n_trades=int(er.n_trades),
                included_in_optimize_pool=bool(optimize_ran),
                selected_by_optimize=bool(optimize_ran and selected_rule_id == rid),
            )
        )
    return out


def _executed_strategy_narrative_lines(
    *,
    code: str,
    name: str,
    fit_level: str,
    executed_signal_tier: str,
    signal_mode: str,
    bias_ma: int,
    profile_str: str,
    weights_by_tier: dict[str, float],
    exit_spec: ExitRuleSpec | None,
    exit_selection_mode: str,
    reference_hold_days: int,
    exit_rule_id_for_display: str | None,
) -> list[str]:
    wtext = _weights_readable_pct(weights_by_tier)
    prof_zh = profile_label_zh(profile_str)
    lines: list[str] = [
        f"标的：{code} {name}",
        f"框架适配度：{fit_level}",
    ]
    if executed_signal_tier == ENTRY_TIER_HIERARCHICAL:
        lines.append(
            f"Entry：分层入场（NEG / NEG+LOW / NEG+LOW+HIGH；{signal_mode} / MA{bias_ma}）；"
            "信号落在哪一档，下一交易日按该档对应仓位开仓（状态切换仓位）。"
        )
    else:
        lines.append(
            f"Entry：锁定单层 {_tier_display_bt(executed_signal_tier)}（{signal_mode} / MA{bias_ma}）；"
            "仓位仍按画像对已实现分层映射。"
        )

    is_optimized = exit_selection_mode == "optimized"
    is_explicit = exit_selection_mode == "explicit"
    is_bundle = exit_selection_mode == "bundle_hold"

    if exit_spec is not None:
        dn = exit_rule_display_name(exit_spec.rule_id, exit_spec)
        rid = exit_spec.rule_id
        if exit_spec.kind != "time_hold":
            lines.append(
                f"Exit：{dn}（`{rid}`）；最长持有兜底 {int(exit_spec.max_hold_days)} 个交易日（未触发规则或达上限时平仓）。"
            )
        else:
            lines.append(f"Exit：{dn}（`{rid}`）。")
    else:
        eid = exit_rule_id_for_display or "hold_fixed"
        dn = exit_rule_display_name(eid, None)
        lines.append(
            f"Exit：{dn}（`{eid}`）— 固定持有 {reference_hold_days} 个交易日；未使用动态退出规则。"
        )

    lines.append(
        f"仓位策略：**{prof_zh}**（`{profile_str}`）；**按状态切换仓位**；状态映射：{wtext}"
    )

    if exit_spec is not None and exit_spec.kind != "time_hold":
        pe = exit_rule_plain_explanation(exit_spec.rule_id, exit_spec)
        if is_optimized:
            lines.append(
                "说明：当前回测采用 **样本内优选动态退出**，主图净值与回撤为该规则下的 **真实结果**，"
                f"**不等同于** 固定持有 {reference_hold_days} 个交易日。"
            )
        elif is_explicit:
            lines.append(
                "说明：当前为 **指定动态退出** 回测，主图净值与回撤为该规则下的 **真实结果**，"
                f"**不等同于** 固定持有 {reference_hold_days} 个交易日。"
            )
        lines.append("入场信号仍为本页实际执行的信号层与参数（可与「推荐策略配置」中的网格最优对照）。")
        lines.append(f"退出逻辑（白话）：{pe}")
        lines.append(
            "当规则条件满足时会提前平仓；若长期未触发，则在最长持仓上限处强制平仓，避免无限持有。"
        )
    elif exit_spec is not None and exit_spec.kind == "time_hold" and (is_optimized or is_explicit):
        hd = int(exit_spec.hold_days or reference_hold_days)
        tag = "样本内优选" if is_optimized else "指定"
        lines.append(
            f"说明：当前为 **{tag} 固定时间退出** 回测，约 {hd} 个交易日强制平仓；主图净值与该规则一致。"
        )
    elif is_bundle:
        lines.append(
            f"说明：主图按 **固定持有 {reference_hold_days} 个交易日**（bundle_hold）生成净值；"
            "未启用动态退出或 CLI 指定退出规则。"
        )
    return lines


def _main_dashboard_title_zh(
    code: str,
    name: str,
    *,
    executed_signal_tier_str: str,
    exit_spec_applied: ExitRuleSpec | None,
    exit_selection_mode: str,
    hold_eff: int,
    applied_exit_optimization_flag: bool,
) -> str:
    """主图标题：明确当前执行入场 + 退出语义，避免被误认为「抽象最佳策略」。"""
    if executed_signal_tier_str == ENTRY_TIER_HIERARCHICAL:
        entry_zh = "分层入场"
    else:
        entry_zh = f"锁定 {executed_signal_tier_str.replace('_', '+')}"

    if exit_spec_applied is None:
        exit_zh = f"固定持有 {hold_eff} 个交易日"
    elif exit_spec_applied.kind == "time_hold":
        hd = int(exit_spec_applied.hold_days) if exit_spec_applied.hold_days is not None else hold_eff
        rid = exit_spec_applied.rule_id
        if applied_exit_optimization_flag and exit_selection_mode == "optimized":
            exit_zh = f"优化时间退出（{rid}，{hd} 个交易日）"
        elif exit_selection_mode == "explicit":
            exit_zh = f"指定时间退出（{rid}，{hd} 个交易日）"
        else:
            exit_zh = f"时间型退出（{rid}，{hd} 个交易日）"
    else:
        rid = exit_spec_applied.rule_id
        if applied_exit_optimization_flag and exit_selection_mode == "optimized":
            exit_zh = f"优化动态退出（{rid}）"
        elif exit_selection_mode == "explicit":
            exit_zh = f"指定动态退出（{rid}）"
        else:
            exit_zh = f"动态退出（{rid}）"

    return f"策略回测 — {code} {name}｜当前执行：{entry_zh}；{exit_zh}"


def _exit_top3_equity_charts(
    df: pd.DataFrame,
    wmap: dict[int, float],
    *,
    mode: str,
    bias_ma_eff: int,
    req: BacktestRequest,
    snapshot_candidates: list[Any],
    code: str,
    name: str,
) -> list[ChartSpec]:
    ordered = sorted(snapshot_candidates, key=lambda x: int(x.rank))
    top3 = [c for c in ordered if c.eligible][:3]
    if len(top3) < 2:
        return []
    series_by_label: dict[str, pd.Series] = {}
    trades_by_label: dict[str, list[Any]] = {}
    first_equity_index: pd.Index | None = None
    for c in top3:
        sp = ExitRuleSpec.from_dict(dict(c.params))
        ctx = None
        if sp.kind != "time_hold":
            ctx = build_exit_context(
                df,
                signal_mode=mode,
                bias_ma_window=bias_ma_eff,
                momentum_window=int(req.momentum_window),
                volume_ma_window=int(req.volume_ma_window),
                rolling_window=int(req.rolling_window),
                quantile_low=float(req.quantile_low),
                quantile_high=float(req.quantile_high),
                spec=sp,
            )
        res_bt = run_portfolio_backtest(df, wmap, exit_rule=sp, exit_context=ctx)
        if first_equity_index is None:
            first_equity_index = res_bt.equity.index
        dn = (c.display_name or c.label_zh or c.rule_id).strip()
        label = f"{dn} ({c.rule_id})"
        series_by_label[label] = res_bt.equity
        trades_by_label[label] = list(res_bt.trades)
    bh_top3 = buy_hold_normalized_from_ohlcv(
        df, pd.DatetimeIndex(pd.to_datetime(first_equity_index))
    )
    entry_sets: list[set[Any]] = []
    for _lb, trs in trades_by_label.items():
        entry_sets.append({pd.Timestamp(t.entry_date).normalize() for t in trs})
    shared_entries: list[Any] = []
    if entry_sets:
        common = set.intersection(*entry_sets)
        if len(common) >= 2:
            shared_entries = sorted(common)
        else:
            first_lb = next(iter(trades_by_label))
            shared_entries = sorted({pd.Timestamp(t.entry_date).normalize() for t in trades_by_label[first_lb]})
    exit_dates_by_label = {k: [t.exit_date for t in v] for k, v in trades_by_label.items()}
    fig = figure_equity_curves_compare(
        series_by_label,
        title=f"Top3 退出规则对比（仅用于分析）— {code} {name}",
        benchmark=bh_top3,
        benchmark_label="买入持有基准（对照）",
        shared_entry_dates=shared_entries,
        exit_dates_by_label=exit_dates_by_label,
    )
    return [
        ChartSpec(
            chart_id="backtest.exit_compare_top3",
            title="Top3 退出规则对比（仅用于分析）",
            plotly_json=figure_to_json(fig),
        )
    ]


def _run_portfolio_for_request(
    df: pd.DataFrame,
    wmap: dict[int, float],
    req: BacktestRequest,
    mode: str,
    bias_ma_eff: int,
    *,
    exit_spec: ExitRuleSpec | None,
    bundle_hold_days: int,
    entry_signal_tier: str | None,
):
    ctx = None
    if exit_spec is not None and exit_spec.kind != "time_hold":
        ctx = build_exit_context(
            df,
            signal_mode=mode,
            bias_ma_window=bias_ma_eff,
            momentum_window=int(req.momentum_window),
            volume_ma_window=int(req.volume_ma_window),
            rolling_window=int(req.rolling_window),
            quantile_low=float(req.quantile_low),
            quantile_high=float(req.quantile_high),
            spec=exit_spec,
        )
    if entry_signal_tier is None:
        d_use = _apply_bias_quantile_to_frame_for_hierarchical_backtest(
            df,
            bias_ma_window=int(bias_ma_eff),
            bias_quantile_range_str=req.bias_quantile_range,
        )
        return run_portfolio_backtest(
            d_use,
            wmap,
            hold_days=int(bundle_hold_days),
            exit_rule=exit_spec,
            exit_context=ctx,
        )
    from .signal_engine import research_tier_mask

    code = {"NEG": 1, "NEG_LOW": 2, "NEG_LOW_HIGH": 3}[entry_signal_tier]
    w = float(wmap[code])
    mask = research_tier_mask(df, entry_signal_tier)
    bias_ok = _bias_quantile_entry_ok_mask(
        df,
        bias_ma_window=int(bias_ma_eff),
        bias_quantile_range_str=req.bias_quantile_range,
    )
    if bias_ok is not None:
        mask = mask & bias_ok
    return run_portfolio_backtest(
        df,
        None,
        hold_days=int(bundle_hold_days),
        exit_rule=exit_spec,
        exit_context=ctx,
        entry_mask=mask,
        entry_weight=w,
        research_signal_tier=code,
    )


def _build_backtest_response_from_run(
    *,
    req: BacktestRequest,
    db: str,
    code: str,
    name: str,
    rec: Any,
    df: pd.DataFrame,
    eff: dict[str, Any],
    res: Any,
    rec_snap: RecommendationSnapshot,
    exit_spec_applied: ExitRuleSpec | None = None,
    hold_eff_override: int | None = None,
    extra_charts_after_dashboard: list[ChartSpec] | None = None,
    exit_rule_comparison_rows: list[ExitRuleComparisonRow] | None = None,
    executed_signal_tier_str: str,
    exit_rule_id_for_json: str | None,
    exit_selection_mode: str,
    applied_exit_optimization_flag: bool,
    exit_sweep_under_entry: list[ExitRuleComparisonRow] | None = None,
    entry_signal_tier_for_interp: str | None = None,
    exit_optimization_diagnostics: list[ExitRuleOptimizationDiagnosticRow] | None = None,
    optimized_chosen_eval_row: ExitRuleEvalRow | None = None,
    multi_objective_decision: MultiObjectiveDecisionBlock | None = None,
    entry_signal_diagnostics: EntrySignalDiagnosticsBlock | None = None,
    entry_exit_matching_diagnostics: EntryExitMatchingDiagnosticsBlock | None = None,
) -> BacktestResponse:
    mode = _mode_str(eff["signal_mode_eff"])
    bias_ma_eff = int(eff["bias_ma_eff"])
    hold_eff = int(hold_eff_override) if hold_eff_override is not None else int(eff["hold_eff"])
    profile_str = str(eff["profile_str"])
    dims = _signal_dims_snap(req, bias_ma_override=bias_ma_eff, signal_mode_override=eff["signal_mode_eff"])
    dd = drawdown_series(res.equity)
    wmap = weights_by_strategy_profile(profile_str)
    weights_str = tier_weight_labels(wmap)
    mdd = float(dd.min()) if len(dd) else float("nan")
    mdd_trough = dd.idxmin() if len(dd) else None
    avg_exp = float(res.exposure.mean()) if len(res.exposure) else float("nan")
    bh = buy_hold_normalized_from_ohlcv(df, pd.DatetimeIndex(pd.to_datetime(res.equity.index)))

    trades = [
        TradeRow(
            entry_date=str(t.entry_date),
            exit_date=str(t.exit_date),
            signal_tier=int(t.signal_tier),
            weight=float(t.weight),
            entry_price=json_safe_value(t.entry_price),
            exit_price=json_safe_value(t.exit_price),
            stock_return=json_safe_value(t.stock_return),
            portfolio_return=json_safe_value(t.portfolio_return),
            holding_days=int(t.holding_days),
        )
        for t in res.trades
    ]
    equity_curve = [
        TimeSeriesPoint(
            date=idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
            value=float(v) if np.isfinite(v) else None,
        )
        for idx, v in res.equity.items()
    ]
    drawdown_curve = [
        TimeSeriesPoint(
            date=idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
            value=float(v) if np.isfinite(v) else None,
        )
        for idx, v in dd.items()
    ]
    met = _metrics_clean(res.metrics)
    summary_cards: dict[str, float | int | None] = {
        "total_return": met.get("total_return"),
        "annualized_return": met.get("annualized_return"),
        "max_drawdown": met.get("max_drawdown"),
        "calmar_ratio": met.get("calmar_ratio"),
        "average_exposure": avg_exp if np.isfinite(avg_exp) else None,
        "trade_count": int(len(trades)),
    }
    dash_title = _main_dashboard_title_zh(
        code,
        name,
        executed_signal_tier_str=executed_signal_tier_str,
        exit_spec_applied=exit_spec_applied,
        exit_selection_mode=exit_selection_mode,
        hold_eff=hold_eff,
        applied_exit_optimization_flag=applied_exit_optimization_flag,
    )
    dash = figure_backtest_dashboard(
        res.equity,
        dd,
        res.exposure,
        bh,
        res.trades,
        title=dash_title,
        max_dd_pct=mdd if np.isfinite(mdd) else None,
        mdd_trough_date=pd.Timestamp(mdd_trough) if mdd_trough is not None else None,
        avg_exposure=avg_exp if np.isfinite(avg_exp) else None,
    )
    legacy = figure_equity_and_drawdown(
        res.equity, dd, title=f"Equity & drawdown — {code} {name}（同主图执行）"
    )
    dash_chart = ChartSpec(
        chart_id="backtest.equity_dashboard",
        title=str(dash.layout.title.text) if dash.layout and dash.layout.title else "Backtest",
        plotly_json=figure_to_json(dash),
    )
    legacy_chart = ChartSpec(
        chart_id="backtest.equity_drawdown",
        title="Equity & drawdown (legacy)",
        plotly_json=figure_to_json(legacy),
    )
    mid = list(extra_charts_after_dashboard or [])
    charts = [dash_chart, *mid, legacy_chart]
    best = rec_snap.best_signal_setup.signal
    interp = _interpretation_notes_backtest(
        fit_level=rec_snap.fit_level,
        applied_bt_rec=bool(eff["applied_bt_rec"]),
        applied_signal=bool(eff["applied_signal"]),
        best_signal=best,
        entry_signal_tier_fixed=entry_signal_tier_for_interp,
    )
    er_json = exit_rule_id_for_json or "hold_fixed"
    if exit_spec_applied is not None:
        dn = exit_rule_display_name(exit_spec_applied.rule_id, exit_spec_applied)
        if exit_selection_mode == "optimized":
            interp = list(interp) + [f"主回测已按 **样本内优选退出** 执行：{dn}（id=`{exit_spec_applied.rule_id}`）。"]
        elif exit_selection_mode == "explicit":
            interp = list(interp) + [f"主回测已按 **指定退出规则** 执行：{dn}（id=`{exit_spec_applied.rule_id}`）。"]
        else:
            interp = list(interp) + [f"主回测退出：{dn}（id=`{exit_spec_applied.rule_id}`）。"]
    elif exit_selection_mode == "bundle_hold":
        interp = list(interp) + [
            f"主回测为 **固定持有 {hold_eff} 个交易日** 平仓（exit=`{er_json}`，bundle_hold）。",
        ]
    elif exit_selection_mode == "explicit" and exit_spec_applied is None:
        interp = list(interp) + [
            f"主回测为 **CLI 指定固定持有**（`hold_fixed`），{hold_eff} 个交易日平仓。",
        ]
    rec_setup = {
        "signal_tier": rec_snap.default_signal,
        "signal_mode": rec_snap.recommended_mode,
        "bias_ma": rec_snap.recommended_bias_ma,
        "hold_days": rec_snap.recommended_horizon_focus,
        "strategy_profile": rec_snap.recommended_position_profile,
    }
    if exit_spec_applied is None:
        exit_semantics = "fixed_sidebar_hold"
    elif exit_spec_applied.kind == "time_hold":
        exit_semantics = "time_hold"
    else:
        exit_semantics = "dynamic"
    exe_exit_rid = exit_spec_applied.rule_id if exit_spec_applied else er_json
    exe_exit_dn = (
        exit_rule_display_name(exit_spec_applied.rule_id, exit_spec_applied)
        if exit_spec_applied
        else exit_rule_display_name(er_json, None)
    )
    exe_exit_pe = (
        exit_rule_plain_explanation(exit_spec_applied.rule_id, exit_spec_applied)
        if exit_spec_applied
        else exit_rule_plain_explanation(er_json, None)
    )
    exe_setup = {
        "signal_tier": executed_signal_tier_str,
        "grid_best_signal_tier": best,
        "signal_mode": mode,
        "bias_ma": bias_ma_eff,
        "hold_days": hold_eff,
        "strategy_profile": profile_str,
        "weights_by_tier": weights_str,
        "exit_rule_id": exe_exit_rid,
        "exit_rule_kind": exit_spec_applied.kind if exit_spec_applied else None,
        "exit_display_name": exe_exit_dn,
        "exit_plain_explanation": exe_exit_pe,
        "exit_semantics": exit_semantics,
        "exit_selection_mode": exit_selection_mode,
        "max_hold_days_cap": (
            int(exit_spec_applied.max_hold_days)
            if exit_spec_applied and exit_spec_applied.kind != "time_hold"
            else None
        ),
    }
    narrative = _executed_strategy_narrative_lines(
        code=code,
        name=name,
        fit_level=str(rec_snap.fit_level),
        executed_signal_tier=executed_signal_tier_str,
        signal_mode=mode,
        bias_ma=bias_ma_eff,
        profile_str=profile_str,
        weights_by_tier=weights_str,
        exit_spec=exit_spec_applied,
        exit_selection_mode=exit_selection_mode,
        reference_hold_days=hold_eff,
        exit_rule_id_for_display=exit_rule_id_for_json,
    )
    opt_plain = (
        exit_rule_plain_explanation(exit_spec_applied.rule_id, exit_spec_applied)
        if applied_exit_optimization_flag and exit_spec_applied
        else None
    )
    diag_rows = list(exit_optimization_diagnostics or [])
    opt_rule_id_json: str | None = None
    opt_score: float | None = None
    opt_dn: str | None = None
    opt_elig = False
    opt_met: dict[str, Any] = {}
    if applied_exit_optimization_flag and optimized_chosen_eval_row is not None:
        opt_rule_id_json = str(optimized_chosen_eval_row.spec.rule_id)
        opt_score = float(optimized_chosen_eval_row.score)
        opt_dn = str(optimized_chosen_eval_row.display_name).strip()
        opt_elig = bool(optimized_chosen_eval_row.eligible)
        opt_met = dict(optimized_chosen_eval_row.metrics or {})
    elif applied_exit_optimization_flag and exit_spec_applied is not None:
        opt_rule_id_json = str(exit_spec_applied.rule_id)
        opt_dn = exit_rule_display_name(exit_spec_applied.rule_id, exit_spec_applied)
    opt_display_final: str | None = None
    if applied_exit_optimization_flag:
        opt_display_final = (opt_dn or "").strip() or (
            exit_rule_display_name(exit_spec_applied.rule_id, exit_spec_applied)
            if exit_spec_applied
            else None
        )
    cmp_rows = exit_rule_comparison_rows if exit_rule_comparison_rows is not None else []
    sweep_rows = list(exit_sweep_under_entry or [])
    trades_export_path_out: str | None = None
    if req.export_trades_path:
        outp = Path(req.export_trades_path).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([t.model_dump() for t in trades]).to_csv(outp, index=False)
        trades_export_path_out = str(outp)
    latest_signal_bar = build_latest_signal_bar_snapshot(
        df,
        req,
        eff,
        code=code,
        name=name,
        wmap=wmap,
        entry_signal_tier=entry_signal_tier_for_interp,
        executed_signal_tier_str=executed_signal_tier_str,
    )
    return BacktestResponse(
        etf_code=code,
        etf_name=name,
        db_path=db,
        signal_param_source=str(req.signal_param_source.value),
        applied_recommendation_defaults=bool(eff["applied_signal"]),
        backtest_param_source=str(req.backtest_param_source.value),
        applied_backtest_recommendation=bool(eff["applied_bt_rec"]),
        recommendation=rec_snap,
        fit_level=rec_snap.fit_level,
        signal_mode=mode,
        bias_source=str(req.bias_source.value),
        bias_ma=bias_ma_eff,
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        signal_dimensions=dims,
        strategy_profile=profile_str,
        strategy_profile_zh=profile_label_zh(profile_str),
        position_rule=profile_str,
        hold_days=hold_eff,
        weights_by_tier=weights_str,
        metrics=met,
        summary_cards=summary_cards,
        recommended_setup=rec_setup,
        executed_setup=exe_setup,
        interpretation_notes=interp,
        trades=trades,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        charts=charts,
        latest_signal_bar=latest_signal_bar,
        comparison_rows=None,
        applied_exit_optimization=applied_exit_optimization_flag,
        optimized_exit_rule_id=opt_rule_id_json
        if opt_rule_id_json
        else (exit_spec_applied.rule_id if (applied_exit_optimization_flag and exit_spec_applied) else None),
        optimized_exit_label_zh=opt_display_final,
        optimized_exit_plain_zh=opt_plain,
        optimized_exit_rule=opt_rule_id_json
        if opt_rule_id_json
        else (exit_spec_applied.rule_id if (applied_exit_optimization_flag and exit_spec_applied) else None),
        optimized_exit_score=opt_score,
        optimized_exit_display_name=opt_display_final,
        optimized_exit_eligible=opt_elig if applied_exit_optimization_flag else False,
        optimized_exit_metrics=opt_met if applied_exit_optimization_flag else {},
        exit_optimization_diagnostics=diag_rows,
        multi_objective_decision=multi_objective_decision,
        entry_signal_diagnostics=entry_signal_diagnostics,
        entry_exit_matching_diagnostics=entry_exit_matching_diagnostics,
        executed_strategy_narrative=narrative,
        exit_rule_comparison_rows=cmp_rows,
        exit_sweep_under_entry=sweep_rows,
        signal_tier=executed_signal_tier_str,
        exit_rule=exit_rule_id_for_json,
        exit_selection_mode=exit_selection_mode,
        recommended_bundle=dict(rec_setup),
        trade_count=int(len(trades)),
        trades_export_path=trades_export_path_out,
    )


def _run_backtest_single(req: BacktestRequest) -> BacktestResponse:
    db = _db_path(req.db_path)
    code = str(req.etf_code)
    name = str(etf_name_map(db).get(code, code))
    df_load, _ = load_etf_sqlite(db, code)
    want_multi = bool(req.multi_objective_exit or req.exit_objective is not None)
    want_entry_exit_matching = bool(req.entry_exit_matching)
    need_exit_eval = bool(
        req.evaluate_exit_rules
        or req.optimize_exit
        or req.compare_exit_rules
        or want_multi
        or want_entry_exit_matching
    )
    rec = recommend_strategy_setup(
        df_load,
        code,
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        include_exit_rules=False,
    )
    eff = _backtest_effective_params(req, rec)
    mode = _mode_str(eff["signal_mode_eff"])
    df, _vres = prepare_research_frame(
        db,
        code,
        signal_mode=mode,  # type: ignore[arg-type]
        bias_ma_window=int(eff["bias_ma_eff"]),
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        bias_source=str(req.bias_source.value),  # type: ignore[arg-type]
    )
    wmap = weights_by_strategy_profile(eff["profile_str"])
    entry_tier = req.entry_signal_tier
    executed_signal_tier_str = entry_tier if entry_tier else ENTRY_TIER_HIERARCHICAL

    exit_eval_rows: list[ExitRuleEvalRow] = []
    exit_eval_expl = ""
    exit_eval_best: ExitRuleEvalRow | None = None
    if need_exit_eval:
        exit_eval_rows, exit_eval_expl, exit_eval_best = rank_exit_rules_on_frame(
            df,
            weights_by_tier=wmap,
            bundle_hold_days=int(eff["hold_eff"]),
            signal_mode=mode,
            bias_ma=int(eff["bias_ma_eff"]),
            momentum_window=int(req.momentum_window),
            volume_ma_window=int(req.volume_ma_window),
            rolling_window=int(req.rolling_window),
            quantile_low=float(req.quantile_low),
            quantile_high=float(req.quantile_high),
            entry_signal_tier=entry_tier,
        )

    rec_snap = RecommendationSnapshot.model_validate(
        _merge_rec_dict_with_entry_map_strategy_mode(req, rec.to_dict())
    )
    if need_exit_eval and exit_eval_rows:
        rec_snap = rec_snap.model_copy(
            update={
                "exit_rule_candidates": [
                    ExitRuleEvalSnapshot.model_validate(x.to_dict()) for x in exit_eval_rows
                ],
                "best_exit_rule": (
                    ExitRuleEvalSnapshot.model_validate(exit_eval_best.to_dict())
                    if exit_eval_best
                    else None
                ),
                "exit_rule_explanation": exit_eval_expl,
            }
        )

    cmp_rows = _exit_eval_rows_to_comparison_rows(exit_eval_rows) if exit_eval_rows else []
    sweep_rows = cmp_rows if (req.compare_exit_rules and exit_eval_rows) else []

    best_eval = exit_eval_best if req.optimize_exit else None

    exit_spec_applied: ExitRuleSpec | None = None
    hold_eff_override: int | None = None
    exit_selection_mode = "bundle_hold"
    exit_rule_id_for_json: str | None = "hold_fixed"
    applied_exit_optimization_flag = False

    if req.explicit_exit_rule_id:
        spec = resolve_cli_exit_rule(req.explicit_exit_rule_id)
        exit_spec_applied = spec
        exit_selection_mode = "explicit"
        if spec is None:
            exit_rule_id_for_json = "hold_fixed"
            hold_eff_override = None
        else:
            exit_rule_id_for_json = spec.rule_id
            if spec.kind == "time_hold" and spec.hold_days is not None:
                hold_eff_override = int(spec.hold_days)
        res = _run_portfolio_for_request(
            df,
            wmap,
            req,
            mode,
            int(eff["bias_ma_eff"]),
            exit_spec=exit_spec_applied,
            bundle_hold_days=int(eff["hold_eff"]),
            entry_signal_tier=entry_tier,
        )
    elif req.optimize_exit and best_eval is not None and best_eval.eligible:
        exit_spec_applied = ExitRuleSpec.from_dict(best_eval.spec.to_dict())
        exit_selection_mode = "optimized"
        applied_exit_optimization_flag = True
        exit_rule_id_for_json = exit_spec_applied.rule_id
        res = _run_portfolio_for_request(
            df,
            wmap,
            req,
            mode,
            int(eff["bias_ma_eff"]),
            exit_spec=exit_spec_applied,
            bundle_hold_days=int(eff["hold_eff"]),
            entry_signal_tier=entry_tier,
        )
        if exit_spec_applied.kind == "time_hold" and exit_spec_applied.hold_days is not None:
            hold_eff_override = int(exit_spec_applied.hold_days)
    else:
        res = _run_portfolio_for_request(
            df,
            wmap,
            req,
            mode,
            int(eff["bias_ma_eff"]),
            exit_spec=None,
            bundle_hold_days=int(eff["hold_eff"]),
            entry_signal_tier=entry_tier,
        )
        exit_spec_applied = None
        exit_rule_id_for_json = "hold_fixed"
        exit_selection_mode = "bundle_hold"
        hold_eff_override = None

    optimized_chosen: ExitRuleEvalRow | None = None
    if applied_exit_optimization_flag and exit_spec_applied is not None:
        rid = str(exit_spec_applied.rule_id)
        for er in exit_eval_rows:
            if str(er.spec.rule_id) == rid:
                optimized_chosen = er
                break

    exit_diag = _exit_optimization_diagnostics(
        exit_eval_rows,
        optimize_ran=bool(req.optimize_exit),
        selected_rule_id=(
            str(exit_spec_applied.rule_id)
            if (applied_exit_optimization_flag and exit_spec_applied)
            else None
        ),
    )

    mo_block: MultiObjectiveDecisionBlock | None = None
    if want_multi and exit_eval_rows:
        mo_block = MultiObjectiveDecisionBlock.model_validate(
            build_multi_objective_decision(
                exit_eval_rows,
                objective_override=cast(ObjectiveName | None, req.exit_objective),
            )
        )

    entry_diag_block: EntrySignalDiagnosticsBlock | None = None
    if req.entry_diagnostics or want_entry_exit_matching:
        entry_diag_block = EntrySignalDiagnosticsBlock.model_validate(
            build_entry_signal_diagnostics(
                df,
                weights_by_tier=wmap,
                entry_signal_tier=entry_tier,
                include_raw_dates=bool(req.entry_diagnostics_dates),
                signal_mode=mode,
                bias_ma=int(eff["bias_ma_eff"]),
                strategy_profile=str(eff["profile_str"]),
            )
        )

    entry_exit_match_block: EntryExitMatchingDiagnosticsBlock | None = None
    if (
        want_entry_exit_matching
        and entry_diag_block is not None
        and exit_eval_rows
    ):
        entry_exit_match_block = EntryExitMatchingDiagnosticsBlock.model_validate(
            build_entry_exit_matching_diagnostics(entry_diag_block, exit_eval_rows)
        )

    extra_exit_charts = (
        _exit_top3_equity_charts(
            df,
            wmap,
            mode=mode,
            bias_ma_eff=int(eff["bias_ma_eff"]),
            req=req,
            snapshot_candidates=list(rec_snap.exit_rule_candidates),
            code=code,
            name=name,
        )
        if need_exit_eval and rec_snap.exit_rule_candidates
        else []
    )
    return _build_backtest_response_from_run(
        req=req,
        db=db,
        code=code,
        name=name,
        rec=rec,
        df=df,
        eff=eff,
        res=res,
        rec_snap=rec_snap,
        exit_spec_applied=exit_spec_applied,
        hold_eff_override=hold_eff_override,
        extra_charts_after_dashboard=extra_exit_charts,
        exit_rule_comparison_rows=cmp_rows,
        executed_signal_tier_str=executed_signal_tier_str,
        exit_rule_id_for_json=exit_rule_id_for_json,
        exit_selection_mode=exit_selection_mode,
        applied_exit_optimization_flag=applied_exit_optimization_flag,
        exit_sweep_under_entry=sweep_rows,
        entry_signal_tier_for_interp=entry_tier,
        exit_optimization_diagnostics=exit_diag,
        optimized_chosen_eval_row=optimized_chosen,
        multi_objective_decision=mo_block,
        entry_signal_diagnostics=entry_diag_block,
        entry_exit_matching_diagnostics=entry_exit_match_block,
    )


def _run_backtest_compare_profiles(req: BacktestRequest) -> BacktestResponse:
    req_base = req.model_copy(
        update={
            "compare_profiles": False,
            "compare_manual_vs_recommended": False,
            "compare_exit_rules": False,
        }
    )
    base = _run_backtest_single(req_base)
    db = _db_path(req.db_path)
    code = str(req.etf_code)
    name = str(etf_name_map(db).get(code, code))
    df_load, _ = load_etf_sqlite(db, code)
    rec = recommend_strategy_setup(
        df_load,
        code,
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
    )
    eff = _backtest_effective_params(req_base, rec)
    mode = _mode_str(eff["signal_mode_eff"])
    df, _ = prepare_research_frame(
        db,
        code,
        signal_mode=mode,  # type: ignore[arg-type]
        bias_ma_window=int(eff["bias_ma_eff"]),
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        bias_source=str(req.bias_source.value),  # type: ignore[arg-type]
    )
    hold_eff = int(eff["hold_eff"])
    df_bt = _apply_bias_quantile_to_frame_for_hierarchical_backtest(
        df,
        bias_ma_window=int(eff["bias_ma_eff"]),
        bias_quantile_range_str=req.bias_quantile_range,
    )
    curves: dict[str, pd.Series] = {}
    rows: list[BacktestComparisonRow] = []
    for p in ("aggressive", "balanced", "defensive"):
        wm = weights_by_strategy_profile(p)
        r = run_portfolio_backtest(df_bt, wm, hold_days=hold_eff)
        label = f"{profile_label_zh(p)} · {profile_weight_percent_triple(p)}"
        curves[label] = r.equity
        rows.append(
            BacktestComparisonRow(
                variant_label=label,
                strategy_profile=p,
                strategy_profile_zh=profile_label_zh(p),
                weights_by_tier=tier_weight_labels(wm),
                metrics=_metrics_clean(r.metrics),
                hold_days=hold_eff,
            )
        )
    cmp_fig = figure_equity_curves_compare(
        curves,
        title=f"仓位画像对比 — {code} {name}（持有 {hold_eff} 日）",
    )
    base.charts.append(
        ChartSpec(
            chart_id="backtest.equity_compare_profiles",
            title="仓位画像对比",
            plotly_json=figure_to_json(cmp_fig),
        )
    )
    base.comparison_rows = rows
    base.interpretation_notes = list(base.interpretation_notes) + [
        "已叠加激进/均衡/防御三条净值（同一信号参数与持有期）。",
    ]
    return base


def _run_backtest_compare_manual_rec(req: BacktestRequest) -> BacktestResponse:
    req_m = req.model_copy(
        update={
            "backtest_param_source": BacktestParamSourceEnum.manual,
            "signal_param_source": req.signal_param_source,
            "compare_profiles": False,
            "compare_manual_vs_recommended": False,
        }
    )
    req_r = req.model_copy(
        update={
            "backtest_param_source": BacktestParamSourceEnum.recommended,
            "signal_param_source": SignalParamSourceEnum.auto,
            "compare_profiles": False,
            "compare_manual_vs_recommended": False,
        }
    )
    manual_resp = _run_backtest_single(req_m)
    _ = _run_backtest_single(req_r)
    db = manual_resp.db_path
    code = manual_resp.etf_code
    name = manual_resp.etf_name
    df_load, _ = load_etf_sqlite(db, code)
    rec = recommend_strategy_setup(
        df_load,
        code,
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
    )
    eff_m = _backtest_effective_params(req_m, rec)
    eff_r = _backtest_effective_params(req_r, rec)
    mode_m = _mode_str(eff_m["signal_mode_eff"])
    mode_r = _mode_str(eff_r["signal_mode_eff"])
    df_m, _ = prepare_research_frame(
        db,
        code,
        signal_mode=mode_m,  # type: ignore[arg-type]
        bias_ma_window=int(eff_m["bias_ma_eff"]),
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        bias_source=str(req.bias_source.value),  # type: ignore[arg-type]
    )
    df_r, _ = prepare_research_frame(
        db,
        code,
        signal_mode=mode_r,  # type: ignore[arg-type]
        bias_ma_window=int(eff_r["bias_ma_eff"]),
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        bias_source=str(req.bias_source.value),  # type: ignore[arg-type]
    )
    df_m = _apply_bias_quantile_to_frame_for_hierarchical_backtest(
        df_m,
        bias_ma_window=int(eff_m["bias_ma_eff"]),
        bias_quantile_range_str=req.bias_quantile_range,
    )
    df_r = _apply_bias_quantile_to_frame_for_hierarchical_backtest(
        df_r,
        bias_ma_window=int(eff_r["bias_ma_eff"]),
        bias_quantile_range_str=req.bias_quantile_range,
    )
    wm_m = weights_by_strategy_profile(eff_m["profile_str"])
    wm_r = weights_by_strategy_profile(eff_r["profile_str"])
    res_m = run_portfolio_backtest(df_m, wm_m, hold_days=int(eff_m["hold_eff"]))
    res_r = run_portfolio_backtest(df_r, wm_r, hold_days=int(eff_r["hold_eff"]))
    pm = profile_label_zh(str(eff_m["profile_str"]))
    pr = profile_label_zh(str(eff_r["profile_str"]))
    curves = {
        f"手动 · {pm} {profile_weight_percent_triple(str(eff_m['profile_str']))} · 持有{eff_m['hold_eff']}日": res_m.equity,
        f"推荐 · {pr} {profile_weight_percent_triple(str(eff_r['profile_str']))} · 持有{eff_r['hold_eff']}日": res_r.equity,
    }
    cmp_fig = figure_equity_curves_compare(curves, title=f"手动 vs 推荐 — {code} {name}")
    manual_resp.charts.append(
        ChartSpec(
            chart_id="backtest.equity_compare_manual_recommended",
            title="手动参数 vs 一键推荐",
            plotly_json=figure_to_json(cmp_fig),
        )
    )
    manual_resp.comparison_rows = [
        BacktestComparisonRow(
            variant_label="手动",
            strategy_profile=str(eff_m["profile_str"]),
            strategy_profile_zh=profile_label_zh(str(eff_m["profile_str"])),
            weights_by_tier=tier_weight_labels(wm_m),
            metrics=_metrics_clean(res_m.metrics),
            hold_days=int(eff_m["hold_eff"]),
        ),
        BacktestComparisonRow(
            variant_label="一键推荐",
            strategy_profile=str(eff_r["profile_str"]),
            strategy_profile_zh=profile_label_zh(str(eff_r["profile_str"])),
            weights_by_tier=tier_weight_labels(wm_r),
            metrics=_metrics_clean(res_r.metrics),
            hold_days=int(eff_r["hold_eff"]),
        ),
    ]
    manual_resp.interpretation_notes = list(manual_resp.interpretation_notes) + [
        "上图主回测为「手动」分支；叠加曲线为「一键推荐」bundle。",
    ]
    return manual_resp


def run_backtest(req: BacktestRequest) -> BacktestResponse:
    if req.compare_profiles and req.compare_manual_vs_recommended:
        raise ValueError("不能同时开启 compare_profiles 与 compare_manual_vs_recommended")
    if req.compare_profiles:
        return _run_backtest_compare_profiles(req)
    if req.compare_manual_vs_recommended:
        return _run_backtest_compare_manual_rec(req)
    return _run_backtest_single(req)


def run_state_ranking(req: StateRankingRequest) -> StateRankingResponse:
    from .signal_dimensions import bias_column, momentum_column, volume_ratio_column
    from .state_quality import run_state_quality_scan

    db = _db_path(req.db_path)
    code = str(req.etf_code)
    name = str(etf_name_map(db).get(code, code))
    mode = _mode_str(req.signal_mode)

    df, _ = prepare_indicator_panel(db, code)
    _, top, bottom, n_st = run_state_quality_scan(
        df,
        momentum_col=momentum_column(req.momentum_window),
        bias_col=bias_column(req.bias_ma),
        volume_col=volume_ratio_column(req.volume_ma_window),
        mode=mode,  # type: ignore[arg-type]
        rolling_window=int(req.rolling_window),
        horizon=int(req.horizon),
        ternary_q1=float(req.ternary_q1),
        ternary_q2=float(req.ternary_q2),
        min_n=int(req.min_n),
        top_k=int(req.top_k),
        bottom_k=int(req.bottom_k),
    )

    return StateRankingResponse(
        etf_code=code,
        etf_name=name,
        db_path=db,
        horizon=int(req.horizon),
        bucket_mode=mode,
        momentum_window=int(req.momentum_window),
        bias_ma=int(req.bias_ma),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        ternary_q1=float(req.ternary_q1),
        ternary_q2=float(req.ternary_q2),
        min_n=int(req.min_n),
        top_k=int(req.top_k),
        bottom_k=int(req.bottom_k),
        states_ranked=n_st,
        top_best=[StateRankRow.model_validate(x) for x in top],
        bottom_worst=[StateRankRow.model_validate(x) for x in bottom],
    )


def run_state_transition(req: StateTransitionRequest) -> StateTransitionResponse:
    from .signal_dimensions import bias_column, momentum_column, volume_ratio_column
    from .state_quality import assign_ternary_states
    from .state_transition import compute_state_transitions

    db = _db_path(req.db_path)
    code = str(req.etf_code)
    name = str(etf_name_map(db).get(code, code))
    mode = _mode_str(req.signal_mode)

    df, _ = prepare_research_frame(
        db,
        code,
        signal_mode=mode,  # type: ignore[arg-type]
        bias_ma_window=int(req.bias_ma),
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        bias_source=str(req.bias_source.value),  # type: ignore[arg-type]
    )
    mom_col = momentum_column(req.momentum_window)
    bcol = bias_column(req.bias_ma)
    vcol = volume_ratio_column(req.volume_ma_window)
    df = assign_ternary_states(
        df,
        momentum_col=mom_col,
        bias_col=bcol,
        volume_col=vcol,
        mode="rolling" if mode == "rolling" else "full_sample",
        rolling_window=int(req.rolling_window),
        ternary_q1=float(req.quantile_low),
        ternary_q2=float(req.quantile_high),
    )

    raw = compute_state_transitions(
        df,
        from_state=str(req.from_state),
        horizons=tuple(int(x) for x in req.horizons),
        top_k=req.transition_top_k,
    )

    hz: dict[str, StateTransitionHorizonBlock] = {}
    for hk, block in raw["horizons"].items():
        hz[str(hk)] = StateTransitionHorizonBlock(
            n_valid=int(block["n_valid"]),
            entropy_nats=float(block["entropy_nats"]),
            transitions=[StateTransitionRow.model_validate(r) for r in block["transitions"]],
        )

    return StateTransitionResponse(
        etf_code=code,
        etf_name=name,
        db_path=db,
        from_state=str(req.from_state),
        signal_mode=mode,
        bias_source=str(req.bias_source.value),
        bias_ma=int(req.bias_ma),
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        state_pattern=str(raw["state_pattern"]),
        total_samples=int(raw["total_samples"]),
        horizons=hz,
    )


def run_path_quality(req: PathQualityRequest) -> PathQualityResponse:
    from quantlab.filters.quantile_filter import parse_quantile_range

    from .path_quality import compute_path_quality
    from .signal_dimensions import bias_column, momentum_column, volume_ratio_column
    from .state_quality import assign_ternary_states
    from .state_transition import infer_state_pattern

    db = _db_path(req.db_path)
    code = str(req.etf_code)
    name = str(etf_name_map(db).get(code, code))
    mode = _mode_str(req.signal_mode)

    df, _ = prepare_research_frame(
        db,
        code,
        signal_mode=mode,  # type: ignore[arg-type]
        bias_ma_window=int(req.bias_ma),
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        bias_source=str(req.bias_source.value),  # type: ignore[arg-type]
    )
    mom_col = momentum_column(req.momentum_window)
    bcol = bias_column(req.bias_ma)
    vcol = volume_ratio_column(req.volume_ma_window)
    df = assign_ternary_states(
        df,
        momentum_col=mom_col,
        bias_col=bcol,
        volume_col=vcol,
        mode="rolling" if mode == "rolling" else "full_sample",
        rolling_window=int(req.rolling_window),
        ternary_q1=float(req.quantile_low),
        ternary_q2=float(req.quantile_high),
    )

    raw = compute_path_quality(
        df,
        from_state=str(req.from_state),
        target_state=str(req.target_state),
        horizon=int(req.horizon),
        target_mode=str(req.target_mode.value),  # type: ignore[arg-type]
        feature_names=tuple(str(x) for x in req.bucket_features),
        bucket_n=int(req.bucket_n),
        bias_ma=int(req.bias_ma),
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        bias_quantile_range=parse_quantile_range(req.bias_quantile_range),
    )

    fbs: list[PathQualityFeatureBreakdown] = []
    for block in raw["feature_breakdowns"]:
        fbs.append(
            PathQualityFeatureBreakdown(
                feature=str(block["feature"]),
                buckets=[PathQualityBucketRow.model_validate(r) for r in block["buckets"]],
            )
        )

    return PathQualityResponse(
        etf_code=code,
        etf_name=name,
        db_path=db,
        from_state=str(req.from_state),
        target_state=str(req.target_state),
        horizon=int(req.horizon),
        target_mode=str(req.target_mode.value),
        signal_mode=mode,
        bias_source=str(req.bias_source.value),
        bias_ma=int(req.bias_ma),
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        state_pattern=str(infer_state_pattern(df["state"])),
        total_samples=int(raw["total_samples"]),
        hit_count=int(raw["hit_count"]),
        hit_rate=float(raw["hit_rate"]),
        mean_forward_return=float(raw["mean_forward_return"]),
        feature_breakdowns=fbs,
    )


def run_path_rule_mining(req: PathRuleMiningRequest) -> PathRuleMiningResponse:
    from quantlab.filters.quantile_filter import parse_quantile_range

    from .path_rule_mining import compute_path_rule_mining
    from .signal_dimensions import bias_column, momentum_column, volume_ratio_column
    from .state_quality import assign_ternary_states
    from .state_transition import infer_state_pattern

    db = _db_path(req.db_path)
    code = str(req.etf_code)
    name = str(etf_name_map(db).get(code, code))
    mode = _mode_str(req.signal_mode)

    df, _ = prepare_research_frame(
        db,
        code,
        signal_mode=mode,  # type: ignore[arg-type]
        bias_ma_window=int(req.bias_ma),
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        bias_source=str(req.bias_source.value),  # type: ignore[arg-type]
    )
    mom_col = momentum_column(req.momentum_window)
    bcol = bias_column(req.bias_ma)
    vcol = volume_ratio_column(req.volume_ma_window)
    df = assign_ternary_states(
        df,
        momentum_col=mom_col,
        bias_col=bcol,
        volume_col=vcol,
        mode="rolling" if mode == "rolling" else "full_sample",
        rolling_window=int(req.rolling_window),
        ternary_q1=float(req.quantile_low),
        ternary_q2=float(req.quantile_high),
    )

    raw = compute_path_rule_mining(
        df,
        from_state=str(req.from_state),
        target_state=str(req.target_state),
        horizon=int(req.horizon),
        target_mode=str(req.target_mode.value),  # type: ignore[arg-type]
        feature_names=tuple(str(x) for x in req.features),
        bucket_n=int(req.bucket_n),
        max_combinations=int(req.max_combinations),
        min_count=int(req.min_count),
        top_k=int(req.top_k),
        rules_above_baseline_only=bool(req.rules_above_baseline_only),
        bias_ma=int(req.bias_ma),
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        bias_quantile_range=parse_quantile_range(req.bias_quantile_range),
    )

    rules: list[PathRuleMiningRuleRow] = []
    for r in raw["rules"]:
        conds = [PathRuleFeatureCondition.model_validate(c) for c in r["feature_conditions"]]
        rules.append(
            PathRuleMiningRuleRow(
                rule=str(r["rule"]),
                feature_conditions=conds,
                count=int(r["count"]),
                hit_count=int(r["hit_count"]),
                hit_rate=float(r["hit_rate"]),
                hit_rate_lift=float(r["hit_rate_lift"]),
                mean_forward_return=float(r["mean_forward_return"]),
                return_lift=float(r["return_lift"]),
                win_rate_forward=float(r["win_rate_forward"]),
            )
        )

    return PathRuleMiningResponse(
        etf_code=code,
        etf_name=name,
        db_path=db,
        from_state=str(req.from_state),
        target_state=str(req.target_state),
        horizon=int(req.horizon),
        target_mode=str(req.target_mode.value),
        signal_mode=mode,
        bias_source=str(req.bias_source.value),
        bias_ma=int(req.bias_ma),
        momentum_window=int(req.momentum_window),
        volume_ma_window=int(req.volume_ma_window),
        rolling_window=int(req.rolling_window),
        quantile_low=float(req.quantile_low),
        quantile_high=float(req.quantile_high),
        state_pattern=str(infer_state_pattern(df["state"])),
        bucket_n=int(req.bucket_n),
        max_combinations=int(req.max_combinations),
        min_count=int(req.min_count),
        top_k=int(req.top_k),
        rules_above_baseline_only=bool(req.rules_above_baseline_only),
        baseline=PathRuleMiningBaselineBlock.model_validate(raw["baseline"]),
        features=list(raw["features"]),
        rules=rules,
    )


def _write_chart_html(spec: ChartSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(spec.plotly_json)
    fig.write_html(str(path))


def run_report(req: ReportRequest) -> ReportResponse:
    out = Path(req.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, str] = {}

    hreq = HealthRequest(etf_code=req.etf_code, db_path=req.db_path, invalid_row_limit=500)
    health = run_health(hreq)

    sreq = SignalResearchRequest(
        etf_code=req.etf_code,
        db_path=req.db_path,
        signal_param_source=req.signal_param_source,
        signal_mode=req.signal_mode,
        bias_source=req.bias_source,
        bias_ma=req.bias_ma,
        momentum_window=req.momentum_window,
        volume_ma_window=req.volume_ma_window,
        rolling_window=req.rolling_window,
        quantile_low=req.quantile_low,
        quantile_high=req.quantile_high,
        event_horizons=req.event_horizons,
    )
    signal_research = run_signal_research(sreq)

    breq = BacktestRequest(
        etf_code=req.etf_code,
        db_path=req.db_path,
        signal_param_source=req.signal_param_source,
        backtest_param_source=req.backtest_param_source,
        signal_mode=SignalModeEnum(str(signal_research.signal_mode)),
        bias_source=req.bias_source,
        bias_ma=int(signal_research.bias_ma),
        momentum_window=req.momentum_window,
        volume_ma_window=req.volume_ma_window,
        rolling_window=req.rolling_window,
        quantile_low=req.quantile_low,
        quantile_high=req.quantile_high,
        strategy_profile=req.strategy_profile,
        hold_days=req.hold_days,
        compare_profiles=False,
        compare_manual_vs_recommended=False,
        evaluate_exit_rules=req.evaluate_exit_rules,
        optimize_exit=req.optimize_exit,
        entry_diagnostics=req.entry_diagnostics,
        entry_diagnostics_dates=req.entry_diagnostics_dates,
        entry_exit_matching=req.entry_exit_matching,
        entry_exit_top=req.entry_exit_top,
    )
    backtest = run_backtest(breq)

    if req.write_json:
        p = out / "health.json"
        p.write_text(health.model_dump_json(indent=2), encoding="utf-8")
        artifact_paths["health_json"] = str(p)
        p = out / "signal_research.json"
        p.write_text(signal_research.model_dump_json(indent=2), encoding="utf-8")
        artifact_paths["signal_research_json"] = str(p)
        p = out / "backtest.json"
        p.write_text(backtest.model_dump_json(indent=2), encoding="utf-8")
        artifact_paths["backtest_json"] = str(p)
        combined = {
            "etf_code": req.etf_code,
            "health": json.loads(health.model_dump_json()),
            "signal_research": json.loads(signal_research.model_dump_json()),
            "backtest": json.loads(backtest.model_dump_json()),
        }
        p = out / "report_bundle.json"
        p.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        artifact_paths["report_bundle_json"] = str(p)

    if req.write_csv:
        from .tables import event_study_as_percent_df

        for tier, rows in signal_research.event_studies.items():
            tdf = pd.DataFrame([r.model_dump() for r in rows])
            tdf.to_csv(out / f"event_study_{tier}.csv", index=False)
            tdfp = event_study_as_percent_df(
                pd.DataFrame([r.model_dump() for r in rows]),
            )
            tdfp.to_csv(out / f"event_study_{tier}_percent.csv", index=False)
            artifact_paths[f"event_study_csv_{tier}"] = str(out / f"event_study_{tier}.csv")

        pd.DataFrame([t.model_dump() for t in backtest.trades]).to_csv(out / "trades.csv", index=False)
        artifact_paths["trades_csv"] = str(out / "trades.csv")

        pd.DataFrame([p.model_dump() for p in backtest.equity_curve]).to_csv(out / "equity_curve.csv", index=False)
        artifact_paths["equity_curve_csv"] = str(out / "equity_curve.csv")

        pd.DataFrame(health.invalid_rows_sample).to_csv(out / "invalid_rows_sample.csv", index=False)
        artifact_paths["invalid_rows_sample_csv"] = str(out / "invalid_rows_sample.csv")

    if req.write_charts_html:
        charts_dir = out / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        for ch in health.charts + signal_research.charts + backtest.charts:
            safe = ch.chart_id.replace(".", "_").replace("/", "_")
            fp = charts_dir / f"{safe}.html"
            _write_chart_html(ch, fp)
            artifact_paths[f"chart_{safe}"] = str(fp)

    summary_md = out / "SUMMARY.md"
    sd = signal_research.signal_dimensions
    rec = signal_research.recommendation
    src_note = (
        "自动推荐（事件研究与回测的 mode / bias MA 已按推荐套用）"
        if signal_research.applied_recommendation_defaults
        else "手动侧栏/CLI 参数（未套用推荐覆盖 mode / bias MA）"
    )
    lines = [
        f"# Quantlab report — {req.etf_code}",
        "",
        f"- DB: `{health.db_path}`",
        f"- Name: **{health.etf_name}**",
        f"- Rows: {health.rows_in} → {health.rows_out}",
        "",
        "## Strategy fit / recommendation",
        "",
        f"- Parameter source: **{signal_research.signal_param_source}** — {src_note}",
        "### Framework fit",
        "",
        f"- **Fit level:** {rec.fit_level}",
        f"- **Summary:** {rec.framework_fit_note}",
        "",
        "### Best signal setup (grid winner)",
        "",
        f"- **Signal:** {rec.best_signal_setup.signal} · **mode:** {rec.best_signal_setup.mode} · "
        f"**bias MA:** {rec.best_signal_setup.bias_ma} · **horizon focus:** {rec.best_signal_setup.horizon_focus} d",
        f"- **Score (z vs grid):** {rec.best_signal_setup.recommendation_score:.4f}",
        f"- **Auto-applied event-study mode / bias MA:** **{signal_research.signal_mode}** / **{signal_research.bias_ma}**",
        f"- **Exploration order (signals):** {', '.join(rec.recommended_signals)}",
        "",
        "### Detail notes",
        "",
    ]
    for note in rec.notes:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Signal dimensions (effective)",
            "",
            f"- NEG momentum window: **{sd.neg_momentum_window}** d",
            f"- LOW bias MA: **{sd.low_bias_ma}**",
            f"- HIGH volume MA window: **{sd.high_volume_ma_window}** d",
            f"- Quantile low (NEG/LOW): **{sd.quantile_low}** | high (HIGH): **{sd.quantile_high}**",
            f"- Mode: **{sd.signal_mode}** | rolling window: **{sd.rolling_window}**",
            f"- Bias source: **{sd.bias_source}** (recompute=from close; db=SQLite bias_rate)",
            "",
            "## Backtest configuration (executed)",
            "",
            f"- Backtest preset: **{backtest.backtest_param_source}**"
            f"{'（一键推荐 bundle）' if backtest.applied_backtest_recommendation else ''}",
            f"- Strategy profile: **{backtest.strategy_profile}** ({backtest.strategy_profile_zh})",
            f"- Weights by tier: `{json.dumps(backtest.weights_by_tier, ensure_ascii=False)}`",
            f"- Hold days: **{backtest.hold_days}** | signal mode / bias MA: **{backtest.signal_mode}** / **{backtest.bias_ma}**",
            f"- Fit level: **{backtest.fit_level}**",
        ]
    )
    if backtest.applied_exit_optimization:
        lines.extend(
            [
                f"- **Exit optimization:** 已启用 — `{backtest.optimized_exit_rule_id}` "
                f"（{backtest.optimized_exit_label_zh or ''}）",
            ]
        )
    br = backtest.recommendation
    if br.exit_rule_candidates:
        lines.extend(
            [
                "",
                "### Exit rule sweep (fixed entry, in-sample)",
                "",
                br.exit_rule_explanation,
                "",
            ]
        )
        for er in br.exit_rule_candidates[:8]:
            dn = (er.display_name or er.label_zh or er.rule_id).strip()
            lines.append(
                f"- **{er.rank}** {dn} (`{er.rule_id}`) score={er.score:.4f} trades={er.n_trades} "
                f"eligible={er.eligible}"
            )
    lines.extend(
        [
            "",
            "### Executed vs recommended bundle",
            "",
            f"- Recommended (rules): `{json.dumps(backtest.recommended_setup, ensure_ascii=False)}`",
            f"- Executed: `{json.dumps(backtest.executed_setup, ensure_ascii=False)}`",
            "",
            "### Interpretation",
            "",
        ]
    )
    for ln in backtest.interpretation_notes:
        lines.append(f"- {ln}")
    lines.extend(
        [
            "",
            "## Backtest metrics",
            "",
            "```json",
            json.dumps(backtest.metrics, indent=2),
            "```",
            "",
            "## Artifacts",
            "",
        ]
    )
    for k, v in sorted(artifact_paths.items()):
        lines.append(f"- `{k}`: {v}")
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    artifact_paths["summary_md"] = str(summary_md)

    return ReportResponse(
        etf_code=str(req.etf_code),
        output_dir=str(out),
        artifact_paths=artifact_paths,
        health=health,
        signal_research=signal_research,
        backtest=backtest,
    )
