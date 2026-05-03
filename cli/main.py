"""
Unified CLI: `quantlab <subcommand>` (install with `pip install -e .` or run `python -m cli.main`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


if str(_repo_root()) not in sys.path:
    sys.path.insert(0, str(_repo_root()))


def _parse_horizons(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(x) for x in parts) if parts else (20, 60, 120)


def _print_entry_exit_matching_cli(block: Any, top_n: int | None) -> None:
    """Compact human table from ``EntryExitMatchingDiagnosticsBlock``."""
    es = block.entry_summary
    print("\nENTRY/EXIT MATCHING")
    av, md = es.avg_regime_duration_days, es.median_regime_duration_days
    avs = f"{av:.2f}" if av is not None else "—"
    mds = f"{md:.2f}" if md is not None else "—"
    print(
        f"entry raw days: {es.raw_entry_days_count} | regimes: {es.regime_count} | "
        f"avg duration: {avs}d | median: {mds}d | max: {es.max_regime_duration_days}d"
    )
    rows = list(block.per_exit)
    if top_n is not None:
        rows = rows[: int(top_n)]
    print(
        f"{'rule_id':<22} {'trades':>6} {'avg_hold':>8} {'hold/ent':>9} "
        f"{'t/regime':>10} {'alignment':<24}"
    )
    for r in rows:
        rid = r.rule_id if len(r.rule_id) <= 22 else r.rule_id[:19] + "..."
        ratio = (
            f"{r.holding_vs_entry_avg_ratio:.2f}x"
            if r.holding_vs_entry_avg_ratio is not None
            else "—"
        )
        tpr = (
            f"{r.trades_per_entry_regime:.2f}"
            if r.trades_per_entry_regime is not None
            else "—"
        )
        ah = f"{r.avg_holding_days:.1f}" if r.avg_holding_days is not None else "—"
        al = (r.alignment_label or "")[:24]
        print(f"{rid:<22} {r.trade_count:>6} {ah:>8} {ratio:>9} {tpr:>10} {al:<24}")


def _add_etf_db(p: argparse.ArgumentParser) -> None:
    p.add_argument("--code", "--etf-code", dest="etf_code", required=True, help="ETF code, e.g. 515080")
    p.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="SQLite path (default: ETF_DB_PATH env, else repo root etf_data.db, else db/etf_data.db)",
    )


def _add_entry_map_json_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--entry-map-json",
        type=str,
        default=None,
        dest="entry_map_json",
        help="可选：discover-entry-map 快照 JSON；合并 strategy_mode (hold/timing) 到推荐/回测结果",
    )


def _add_signal_dimensions(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--momentum-window",
        type=int,
        default=10,
        choices=[5, 10, 20, 60],
        help="NEG: momentum lookback (days)",
    )
    p.add_argument(
        "--volume-ma-window",
        type=int,
        default=20,
        choices=[5, 10, 20, 60],
        help="HIGH: volume / MA(volume) denominator window",
    )
    p.add_argument(
        "--quantile-low",
        type=float,
        default=0.33,
        choices=[0.25, 0.30, 0.33, 0.40],
        help="NEG/LOW: weakness quantile threshold",
    )
    p.add_argument(
        "--quantile-high",
        type=float,
        default=0.67,
        choices=[0.60, 0.67, 0.70, 0.75],
        help="HIGH: surge quantile threshold",
    )


def _add_bias_source(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--bias-source",
        choices=["recompute", "db"],
        default="recompute",
        dest="bias_source",
        help="LOW bias: recompute from close vs MA, or use SQLite bias_rate column (must exist)",
    )


def add_bias_quantile_filter_arg(p: argparse.ArgumentParser) -> None:
    """Shared optional full-sample bias quintile gate (Q1..Q5 contiguous only)."""
    p.add_argument(
        "--bias-q",
        dest="bias_quantile_range",
        default=None,
        metavar="RANGE",
        help="Optional bias quantile filter: Q1, Q1-Q2, Q2-Q4 (contiguous Q1–Q5 only)",
    )


def _add_exit_rule_cli(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--evaluate-exit",
        action="store_true",
        dest="evaluate_exit",
        help="横评退出规则并写入 JSON（分层入场时与网格最优信号对齐；若配合 --signal-tier 则与锁定层对齐）",
    )
    p.add_argument(
        "--optimize-exit",
        action="store_true",
        dest="optimize_exit",
        help="主回测按横评 **最高分且满足最小成交** 的退出规则重算（与 --exit-rule 互斥；勿与 --compare-profiles 同开）",
    )
    p.add_argument(
        "--multi-objective",
        action="store_true",
        dest="multi_objective_exit",
        help="在横评结果上计算多目标 Pareto / 分视角最优（需跑退出横评；不替代 score_exit_metrics 与 --optimize-exit）",
    )
    p.add_argument(
        "--objective",
        choices=[
            "return_first",
            "risk_first",
            "efficiency_first",
            "robustness_first",
        ],
        default=None,
        dest="exit_objective",
        help="多目标默认推荐视角（默认 risk_first）；隐含启用横评数据用于多目标层",
    )


def _add_backtest_experiment_cli(p: argparse.ArgumentParser) -> None:
    from core.exit_rules import list_cli_exit_rule_ids

    rule_ids = list_cli_exit_rule_ids()
    p.add_argument(
        "--signal-tier",
        choices=["NEG", "NEG_LOW", "NEG_LOW_HIGH"],
        default=None,
        dest="entry_signal_tier",
        help="实验：锁定入场层（仅该层布尔条件触发）；不设则分层引擎按画像对 1/2/3 层加权",
    )
    p.add_argument(
        "--exit-rule",
        choices=rule_ids,
        default=None,
        dest="explicit_exit_rule_id",
        metavar="RULE_ID",
        help=f"实验：主回测强制使用该退出规则（与 --optimize-exit 互斥）。可选: {', '.join(rule_ids)}",
    )
    p.add_argument(
        "--export-trades",
        type=Path,
        default=None,
        dest="export_trades",
        help="将主回测成交明细写入 CSV（绝对路径或相对路径）",
    )
    p.add_argument(
        "--compare-exit-rules",
        action="store_true",
        dest="compare_exit_rules",
        help="在当前入场设定下横评 hold_fixed + 全部默认退出；结果在 JSON 的 exit_sweep_under_entry",
    )
    p.add_argument(
        "--entry-diagnostics",
        action="store_true",
        dest="entry_diagnostics",
        help="输出原始入场 EOD 条件诊断（entry_signal_diagnostics：regime/持久度等，非成交反推）",
    )
    p.add_argument(
        "--entry-diagnostics-dates",
        action="store_true",
        dest="entry_diagnostics_dates",
        help="与 --entry-diagnostics 合用：在 JSON 中列出全部 raw_entry_dates（可能很长）",
    )
    p.add_argument(
        "--entry-exit-matching",
        action="store_true",
        dest="entry_exit_matching",
        help="入场 regime vs 各退出持仓对齐诊断（JSON: entry_exit_matching_diagnostics；隐含 entry 诊断与退出横评）",
    )
    p.add_argument(
        "--entry-exit-top",
        type=int,
        default=None,
        metavar="N",
        dest="entry_exit_top",
        help="非 --json 时 ENTRY/EXIT MATCHING 表只打印前 N 行（默认全部）",
    )


def _add_backtest_strategy_cli(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--strategy-profile",
        dest="strategy_profile",
        default="balanced",
        choices=["aggressive", "balanced", "defensive", "full", "layered", "conservative"],
        help="仓位画像（激进/均衡/防御/满仓；layered≈均衡、conservative≈防御）",
    )
    p.add_argument(
        "--backtest-preset",
        dest="backtest_preset",
        choices=["manual", "recommended"],
        default="manual",
        help="recommended：套用推荐 signal mode/乖离/持有期/仓位画像（全 bundle）",
    )
    p.add_argument(
        "--apply-recommendation-bundle",
        action="store_true",
        dest="apply_recommendation_bundle",
        help="等同 --backtest-preset recommended",
    )
    p.add_argument(
        "--compare-profiles",
        action="store_true",
        help="同一信号参数下对比 aggressive/balanced/defensive 三条净值",
    )
    p.add_argument(
        "--compare-manual-vs-recommended",
        action="store_true",
        dest="compare_manual_vs_recommended",
        help="对比当前 CLI 手动参数 vs 一键推荐 bundle 两条净值",
    )


def _add_signal_preset_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--signal-preset",
        choices=["manual", "auto"],
        default="manual",
        dest="signal_preset",
        help="manual=use --mode and --bias-ma; auto=recommend then apply mode/bias_ma for signal research (and report backtest)",
    )
    p.add_argument(
        "--use-recommendation",
        action="store_true",
        help="Same as --signal-preset auto",
    )


def _add_signal_params(p: argparse.ArgumentParser) -> None:
    _add_etf_db(p)
    p.add_argument(
        "--mode",
        choices=["full_sample", "rolling"],
        default="rolling",
        help="Signal quantile mode",
    )
    _add_bias_source(p)
    p.add_argument("--bias-ma", type=int, default=120, dest="bias_ma", help="LOW: bias MA window (60/120/250)")
    _add_signal_dimensions(p)
    p.add_argument("--rolling-window", type=int, default=252, dest="rolling_window")
    p.add_argument(
        "--horizons",
        type=str,
        default="20,60,120",
        help="Comma-separated event-study horizons (trading days)",
    )
    p.add_argument("--json", action="store_true", help="Print full JSON response to stdout")
    p.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Also write response JSON to this file",
    )
    _add_signal_preset_flags(p)


def cmd_health(ns: argparse.Namespace) -> int:
    from core.runner import run_health
    from core.schemas import HealthRequest

    req = HealthRequest(etf_code=ns.etf_code, db_path=ns.db_path, invalid_row_limit=ns.invalid_limit)
    resp = run_health(req)
    if ns.json:
        print(resp.model_dump_json(indent=2))
    else:
        print(f"ETF {resp.etf_code} ({resp.etf_name})")
        print(f"DB: {resp.db_path}")
        print(f"Rows: {resp.rows_in} -> {resp.rows_out} (open->close fixes: {resp.open_fallback_rows})")
        print(f"Invalid rows (pre-clean): {resp.invalid_row_count}")
        for iss in resp.issues:
            c = f" n={iss.count}" if iss.count is not None else ""
            print(f"  [{iss.severity}] {iss.code}: {iss.message}{c}")
    if ns.save_json:
        ns.save_json.write_text(resp.model_dump_json(indent=2), encoding="utf-8")
        if not ns.json:
            print(f"Wrote {ns.save_json}", file=sys.stderr)
    return 0


def cmd_signal_research(ns: argparse.Namespace) -> int:
    from core.runner import run_signal_research
    from core.schemas import (
        BiasSourceEnum,
        SignalModeEnum,
        SignalParamSourceEnum,
        SignalResearchRequest,
    )

    horizons = _parse_horizons(ns.horizons)
    preset = "auto" if ns.use_recommendation else ns.signal_preset
    sp_src = SignalParamSourceEnum.auto if preset == "auto" else SignalParamSourceEnum.manual
    req = SignalResearchRequest(
        etf_code=ns.etf_code,
        db_path=ns.db_path,
        signal_param_source=sp_src,
        signal_mode=SignalModeEnum(ns.mode),
        bias_source=BiasSourceEnum(ns.bias_source),
        bias_ma=ns.bias_ma,
        momentum_window=ns.momentum_window,
        volume_ma_window=ns.volume_ma_window,
        rolling_window=ns.rolling_window,
        quantile_low=ns.quantile_low,
        quantile_high=ns.quantile_high,
        event_horizons=horizons,
    )
    resp = run_signal_research(req)
    if ns.json:
        print(resp.model_dump_json(indent=2))
    else:
        d = resp.signal_dimensions
        src = resp.signal_param_source
        applied = resp.applied_recommendation_defaults
        src_human = (
            f"param_source={src} (auto-applied mode/bias_ma)"
            if applied
            else f"param_source={src} (effective mode/bias_ma from CLI)"
        )
        print(
            f"ETF {resp.etf_code} ({resp.etf_name}) | {src_human} | "
            f"effective mode={resp.signal_mode} | bias_source={resp.bias_source} | "
            f"NEG={d.neg_momentum_window}d LOW=MA{d.low_bias_ma} HIGH=volMA{d.high_volume_ma_window}d | "
            f"q_low={d.quantile_low} q_high={d.quantile_high}"
        )
        rec = resp.recommendation
        b = rec.best_signal_setup
        print(
            f"Recommendation: fit={rec.fit_level} | best_grid: {b.signal} {b.mode} MA{b.bias_ma} h={b.horizon_focus} "
            f"(score_z={b.recommendation_score:.3f}) | mirrors: bias_ma={rec.recommended_bias_ma} mode={rec.recommended_mode}"
        )
        for note in rec.notes:
            print(f"  note: {note}")
        for tier, rows in resp.event_studies.items():
            print(f"\n=== {tier} ===")
            for r in rows:
                print(
                    f"  h={r.horizon} n={r.n} win%={r.win_rate} mean={r.mean_return} "
                    f"median={r.median_return} std={r.std}"
                )
        print(f"\nCharts: {len(resp.charts)} (use --json for plotly_json)")
    if ns.save_json:
        ns.save_json.write_text(resp.model_dump_json(indent=2), encoding="utf-8")
        if not ns.json:
            print(f"Wrote {ns.save_json}", file=sys.stderr)
    return 0


def namespace_to_backtest_request(ns: argparse.Namespace) -> Any:
    """Shared CLI → BacktestRequest (backtest / latest-state)."""
    from core.schemas import (
        BacktestParamSourceEnum,
        BacktestRequest,
        BiasSourceEnum,
        SignalModeEnum,
        SignalParamSourceEnum,
    )

    preset = "auto" if ns.use_recommendation else ns.signal_preset
    sp_src = SignalParamSourceEnum.auto if preset == "auto" else SignalParamSourceEnum.manual
    bt_rec = ns.backtest_preset == "recommended" or bool(getattr(ns, "apply_recommendation_bundle", False))
    bps = BacktestParamSourceEnum.recommended if bt_rec else BacktestParamSourceEnum.manual
    return BacktestRequest(
        etf_code=ns.etf_code,
        db_path=ns.db_path,
        signal_param_source=sp_src,
        backtest_param_source=bps,
        signal_mode=SignalModeEnum(ns.mode),
        bias_source=BiasSourceEnum(ns.bias_source),
        bias_ma=ns.bias_ma,
        momentum_window=ns.momentum_window,
        volume_ma_window=ns.volume_ma_window,
        rolling_window=ns.rolling_window,
        quantile_low=ns.quantile_low,
        quantile_high=ns.quantile_high,
        strategy_profile=ns.strategy_profile,
        hold_days=ns.hold_days,
        compare_profiles=bool(getattr(ns, "compare_profiles", False)),
        compare_manual_vs_recommended=bool(getattr(ns, "compare_manual_vs_recommended", False)),
        evaluate_exit_rules=bool(getattr(ns, "evaluate_exit", False)),
        optimize_exit=bool(getattr(ns, "optimize_exit", False)),
        entry_signal_tier=getattr(ns, "entry_signal_tier", None),
        explicit_exit_rule_id=getattr(ns, "explicit_exit_rule_id", None),
        compare_exit_rules=bool(getattr(ns, "compare_exit_rules", False)),
        export_trades_path=str(ns.export_trades) if getattr(ns, "export_trades", None) else None,
        multi_objective_exit=bool(getattr(ns, "multi_objective_exit", False)),
        exit_objective=getattr(ns, "exit_objective", None),
        entry_diagnostics=bool(getattr(ns, "entry_diagnostics", False)),
        entry_diagnostics_dates=bool(getattr(ns, "entry_diagnostics_dates", False)),
        entry_exit_matching=bool(getattr(ns, "entry_exit_matching", False)),
        entry_exit_top=getattr(ns, "entry_exit_top", None),
        bias_quantile_range=getattr(ns, "bias_quantile_range", None),
        entry_map_json_path=getattr(ns, "entry_map_json", None),
    )


def cmd_latest_state(ns: argparse.Namespace) -> int:
    from core.runner import run_latest_signal_state

    req = namespace_to_backtest_request(ns)
    snap = run_latest_signal_state(req)
    if ns.json:
        print(snap.model_dump_json(indent=2))
    else:
        def _yn(b: bool) -> str:
            return "是" if b else "否"

        print(f"=== {snap.etf_code} {snap.etf_name} · 数据截至 {snap.as_of_date} ===\n")
        print("— 三维度（各自独立布尔）—")
        print(f"  弱势动量 NEG ……… {_yn(snap.neg)}")
        print(f"  弱势乖离 LOW ……… {_yn(snap.low)}")
        print(f"  放量 HIGH ………… {_yn(snap.high)}")
        print("\n— 分位（全样本等频 Q1–Q5；用于解释“相对位置”）—")
        print(f"  bias_bucket（乖离分位）……… {snap.bias_bucket or 'NA'}")
        print(f"  momentum_bucket（动量分位）… {snap.momentum_bucket or 'NA'}")
        print(f"  volume_bucket（量比分位）…… {snap.volume_ratio_bucket or 'NA'}")
        print("\n— 分层 signal_tier（仅在与回测相同的分层规则下有意义）—")
        print(f"  raw（未做 bias 分位调 tier）…… {snap.signal_tier_raw}")
        print(f"  effective（回测主图口径）…… {snap.signal_tier_effective}")
        print(f"  摘要标签 …………………… {snap.implied_layer_zh}")
        print("\n— 下一交易日开盘会否因「本 bar」触发新开仓（回测引擎同一套规则）—")
        print(f"  → {'会' if snap.backtest_entry_active else '不会'}")
        print("\n— 参数快照 —")
        print(
            f"  入场模式: {snap.execution_entry_mode}  |  signal_mode={snap.signal_mode}  "
            f"MA{snap.bias_ma_effective}  |  画像 {snap.strategy_profile_zh}"
        )
        sig = "信号: 已套用推荐 mode/乖离" if snap.applied_recommendation_defaults else "信号: CLI/侧栏手动"
        bt = "持有+画像: 一键推荐 bundle" if snap.applied_backtest_recommendation else "持有+画像: 手动"
        print(f"  {sig}  |  {bt}")
        print("\n── 怎么读（必读）──")
        print(snap.readout_zh)
        print()
    if ns.save_json:
        ns.save_json.write_text(snap.model_dump_json(indent=2), encoding="utf-8")
        if not ns.json:
            print(f"Wrote {ns.save_json}", file=sys.stderr)
    return 0


def cmd_backtest(ns: argparse.Namespace) -> int:
    import core.position_rules as _pr

    _sw = getattr(_pr, "state_weights_readable_line", None)
    if _sw is None:

        def _sw(weights_by_tier, *, markdown_bold_pct=False):
            order = ("NEG", "NEG_LOW", "NEG_LOW_HIGH")
            parts: list[str] = []
            for k in order:
                if k not in weights_by_tier:
                    continue
                try:
                    pct_s = f"{float(weights_by_tier[k]) * 100:.0f}%"
                    parts.append(f"{k} {pct_s}")
                except (TypeError, ValueError):
                    parts.append(f"{k} {weights_by_tier[k]!r}")
            if not parts:
                for k, v in sorted(weights_by_tier.items()):
                    try:
                        parts.append(f"{k} {float(v) * 100:.0f}%")
                    except (TypeError, ValueError):
                        parts.append(f"{k} {v!r}")
            return " / ".join(parts) if parts else "—"

    from core.runner import run_backtest

    req = namespace_to_backtest_request(ns)
    resp = run_backtest(req)
    if ns.json:
        print(resp.model_dump_json(indent=2))
    else:
        sig_src = "signal: auto mode/bias" if resp.applied_recommendation_defaults else "signal: manual CLI"
        bt_src = "backtest: recommended bundle" if resp.applied_backtest_recommendation else "backtest: manual hold/profile"
        print(
            f"ETF {resp.etf_code} ({resp.etf_name}) | {sig_src} | {bt_src}\n"
            f"  executed: profile={resp.strategy_profile} ({resp.strategy_profile_zh}) hold={resp.hold_days}d "
            f"mode={resp.signal_mode} bias_ma={resp.bias_ma}\n"
            f"  signal_tier (actual): {resp.signal_tier!r} | exit_rule (actual): {resp.exit_rule!r} "
            f"| exit_selection_mode: {resp.exit_selection_mode!r}\n"
            f"  weights: {resp.weights_by_tier}\n"
            f"  state→position: {_sw(resp.weights_by_tier or {})}\n"
            f"  仓位策略: 按分层信号状态切换仓位（同一画像下 NEG / NEG+LOW / NEG+LOW+HIGH 对应不同权重，非固定满仓）\n"
            f"  recommended bundle: {json.dumps(resp.recommended_setup, ensure_ascii=False)}"
        )
        if resp.recommendation.strategy_mode is not None:
            print(f"  strategy_mode (entry_map): {resp.recommendation.strategy_mode}")
        print("  当前执行策略（与主图一致）:")
        for line in resp.executed_strategy_narrative:
            print(f"    · {line}")
        if resp.applied_exit_optimization:
            print(
                f"  exit optimization: `{resp.optimized_exit_rule_id}` — {resp.optimized_exit_label_zh}"
            )
            if resp.optimized_exit_plain_zh:
                print(f"    白话: {resp.optimized_exit_plain_zh}")
        rc = resp.recommendation.exit_rule_candidates
        if rc:
            print("  exit rule sweep (top 3, 人类可读):")
            for er in rc[:3]:
                dn = (er.display_name or er.label_zh or er.rule_id).strip()
                print(
                    f"    #{er.rank} {dn}  (`{er.rule_id}`)  score={er.score:.4f}  trades={er.n_trades}  "
                    f"eligible={er.eligible}"
                )
                if (er.plain_explanation or "").strip():
                    print(f"      → {er.plain_explanation}")
        if resp.exit_rule_comparison_rows:
            print("  exit comparison (full table in JSON; preview columns):")
            for row in resp.exit_rule_comparison_rows[:5]:
                print(
                    f"    #{row.rank} {row.display_name}  ret={row.total_return}  ann={row.annualized_return}  "
                    f"mdd={row.max_drawdown}  sh={row.sharpe_ratio}  calmar={row.calmar_ratio}"
                )
        if resp.exit_sweep_under_entry:
            print("  exit sweep under entry (compare-exit-rules; sorted by score_exit_metrics):")
            for row in resp.exit_sweep_under_entry[:12]:
                print(
                    f"    #{row.rank} `{row.rule_id}` {row.display_name}  score={row.score:.4f}  eligible={row.eligible}  "
                    f"ret={row.total_return}  ann={row.annualized_return}  mdd={row.max_drawdown}  trades={row.n_trades}"
                )
        mo = resp.multi_objective_decision
        if resp.entry_signal_diagnostics is not None:
            ed = resp.entry_signal_diagnostics
            ps = ed.entry_persistence_summary
            print(
                f"  entry diagnostics: raw_days={ed.raw_entry_days_count} regimes={ps.regime_count} "
                f"avg_run={ps.avg_duration_days} max_run={ps.max_duration_days}"
            )
        if resp.entry_exit_matching_diagnostics is not None:
            _print_entry_exit_matching_cli(
                resp.entry_exit_matching_diagnostics,
                getattr(ns, "entry_exit_top", None),
            )
        if mo is not None:
            print("  multi-objective exit (Pareto + per-objective winners; legacy optimize unchanged):")
            print(f"    Pareto set (eligible): {mo.pareto_set}")
            for k, v in sorted(mo.objective_winners.items()):
                print(f"    best under {k}: {v}")
            print(
                f"    default_recommendation ({mo.default_objective}): {mo.default_recommendation or '—'}"
            )
            ib = mo.interpretation
            print(f"    confidence: {ib.confidence} | {ib.style_bias}")
        if resp.trades_export_path:
            print(f"  trades CSV: {resp.trades_export_path}")
        br = resp.recommendation.best_signal_setup
        print(
            f"Grid best: {br.signal} {br.mode} MA{br.bias_ma} h={br.horizon_focus} | fit={resp.recommendation.fit_level}"
        )
        print("Interpretation:")
        for line in resp.interpretation_notes:
            print(f"  - {line}")
        if resp.comparison_rows:
            print("Comparison:")
            for row in resp.comparison_rows:
                print(f"  - {row.variant_label}: profile={row.strategy_profile} ret={row.metrics.get('total_return')}")
        print("Summary cards:", json.dumps(resp.summary_cards, indent=2))
        print("Metrics:", json.dumps(resp.metrics, indent=2))
        print(f"Trades: {resp.trade_count}")
    if ns.save_json:
        ns.save_json.write_text(resp.model_dump_json(indent=2), encoding="utf-8")
        if not ns.json:
            print(f"Wrote {ns.save_json}", file=sys.stderr)
    return 0


def cmd_analyze_path_rules(ns: argparse.Namespace) -> int:
    from core.runner import run_path_rule_mining
    from core.schemas import (
        BiasSourceEnum,
        PathQualityTargetModeEnum,
        PathRuleMiningRequest,
        SignalModeEnum,
    )

    req = PathRuleMiningRequest(
        etf_code=ns.etf_code,
        db_path=ns.db_path,
        signal_mode=SignalModeEnum(ns.mode),
        bias_source=BiasSourceEnum(ns.bias_source),
        bias_ma=ns.bias_ma,
        momentum_window=ns.momentum_window,
        volume_ma_window=ns.volume_ma_window,
        rolling_window=ns.rolling_window,
        quantile_low=ns.quantile_low,
        quantile_high=ns.quantile_high,
        from_state=ns.from_state,
        target_state=ns.target_state,
        horizon=ns.horizon,
        target_mode=PathQualityTargetModeEnum(ns.target_mode),
        features=ns.features,
        bucket_n=ns.bucket_n,
        max_combinations=ns.max_combinations,
        min_count=ns.min_count,
        top_k=ns.top_k,
        rules_above_baseline_only=bool(getattr(ns, "rules_above_baseline_only", False)),
        bias_quantile_range=getattr(ns, "bias_quantile_range", None),
    )
    resp = run_path_rule_mining(req)
    if ns.json:
        print(resp.model_dump_json(indent=2))
    else:
        b = resp.baseline
        print("PATH RULE MINING")
        print(
            f"from_state={resp.from_state} target_state={resp.target_state} "
            f"horizon={resp.horizon} target_mode={resp.target_mode}"
        )
        print(
            f"baseline: count={b.count} hit_rate={b.hit_rate:.3f} "
            f"mean_forward_return={b.mean_forward_return:.4f}"
        )
        print("\ntop rules:")
        for i, r in enumerate(resp.rules, 1):
            lift = r.hit_rate_lift
            print(f"{i}. {r.rule}")
            print(
                f"   count={r.count} hit_rate={r.hit_rate:.3f} ({lift:+.3f}) "
                f"mean_fwd_ret={r.mean_forward_return:.4f}"
            )
    if ns.save_json:
        ns.save_json.write_text(resp.model_dump_json(indent=2), encoding="utf-8")
        if not ns.json:
            print(f"Wrote {ns.save_json}", file=sys.stderr)
    return 0


def cmd_analyze_path_quality(ns: argparse.Namespace) -> int:
    from core.runner import run_path_quality
    from core.schemas import (
        BiasSourceEnum,
        PathQualityRequest,
        PathQualityTargetModeEnum,
        SignalModeEnum,
    )

    req = PathQualityRequest(
        etf_code=ns.etf_code,
        db_path=ns.db_path,
        signal_mode=SignalModeEnum(ns.mode),
        bias_source=BiasSourceEnum(ns.bias_source),
        bias_ma=ns.bias_ma,
        momentum_window=ns.momentum_window,
        volume_ma_window=ns.volume_ma_window,
        rolling_window=ns.rolling_window,
        quantile_low=ns.quantile_low,
        quantile_high=ns.quantile_high,
        from_state=ns.from_state,
        target_state=ns.target_state,
        horizon=ns.horizon,
        target_mode=PathQualityTargetModeEnum(ns.target_mode),
        bucket_features=ns.bucket_features,
        bucket_n=ns.bucket_n,
        bias_quantile_range=getattr(ns, "bias_quantile_range", None),
    )
    resp = run_path_quality(req)
    if ns.json:
        print(resp.model_dump_json(indent=2))
    else:
        print(
            f"{resp.etf_code} ({resp.etf_name}) | from={resp.from_state!r} → target={resp.target_state!r} | "
            f"H={resp.horizon} mode={resp.target_mode} | n={resp.total_samples} hits={resp.hit_count} "
            f"hit_rate={resp.hit_rate:.4f}"
        )
        for fb in resp.feature_breakdowns:
            print(f"\n--- {fb.feature} ---")
            for b in fb.buckets:
                print(
                    f"  {b.bucket:<6} n={b.count:4d}  hit={b.hit_count:4d} ({b.hit_rate:.3f})  "
                    f"mean_fwd={b.mean_forward_return:+.4f}  win_fwd={b.win_rate_forward:.3f}"
                )
    if ns.save_json:
        ns.save_json.write_text(resp.model_dump_json(indent=2), encoding="utf-8")
        if not ns.json:
            print(f"Wrote {ns.save_json}", file=sys.stderr)
    return 0


def cmd_discover_entry_map(ns: argparse.Namespace) -> int:
    from core.data_loader import etf_universe_from_db
    from core.entry_map import (
        DEFAULT_ETF_UNIVERSE,
        EntryMapDiscoveryConfig,
        discover_entry_map,
        save_entry_map_snapshot,
    )
    from core.paths import resolve_db_path
    from core.schemas import BiasSourceEnum, PathQualityTargetModeEnum, SignalModeEnum

    db_resolved = resolve_db_path(ns.db_path)
    if ns.etf_json is not None:
        raw = json.loads(Path(ns.etf_json).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("--etf-json must contain a JSON array of objects with code/name")
        universe: list[dict[str, str]] = [dict(x) for x in raw]
    else:
        universe = etf_universe_from_db(db_resolved)
        if not universe:
            universe = list(DEFAULT_ETF_UNIVERSE)
            if not ns.json:
                print(
                    "discover-entry-map: 数据库中未读到 ETF，已回退为内置 DEFAULT_ETF_UNIVERSE。",
                    file=sys.stderr,
                )

    cfg = EntryMapDiscoveryConfig(
        db_path=str(db_resolved),
        signal_mode=SignalModeEnum(ns.mode),
        bias_source=BiasSourceEnum(ns.bias_source),
        bias_ma=ns.bias_ma,
        momentum_window=ns.momentum_window,
        volume_ma_window=ns.volume_ma_window,
        rolling_window=ns.rolling_window,
        quantile_low=ns.quantile_low,
        quantile_high=ns.quantile_high,
        target_state=str(ns.target_state).strip(),
        horizon=int(ns.horizon),
        path_quality_target_mode=PathQualityTargetModeEnum(ns.target_mode),
        min_samples=int(ns.min_samples),
        weak_hit_rate_floor=float(ns.weak_hit_rate_floor),
        bias_quantile_range=getattr(ns, "bias_quantile_range", None),
    )

    snap = discover_entry_map(universe, cfg, auto_mode=bool(getattr(ns, "auto_mode", True)))
    if ns.json:
        print(snap.model_dump_json(indent=2))
    else:
        print(
            f"discover-entry-map: target={snap.notes.target_state} H={snap.notes.horizon} "
            f"states={snap.notes.candidate_states} auto_mode={snap.notes.auto_mode} "
            f"n_etfs={len(snap.etfs)}"
        )
        for e in snap.etfs:
            bs = e.best_state or "—"
            print(
                f"  {e.code} {e.name}: mode={e.research_mode} suff={e.data_sufficiency} best={bs} "
                f"driver={e.driver} arch={e.entry_archetype} conf={e.confidence} weak={e.weak_path_quality}"
            )
    if ns.save_json:
        save_entry_map_snapshot(ns.save_json, snap)
        if not ns.json:
            print(f"Wrote {ns.save_json}", file=sys.stderr)
    return 0


def cmd_analyze_transition(ns: argparse.Namespace) -> int:
    from core.runner import run_state_transition
    from core.schemas import BiasSourceEnum, SignalModeEnum, StateTransitionRequest

    horizons = _parse_horizons(ns.horizons)
    req = StateTransitionRequest(
        etf_code=ns.etf_code,
        db_path=ns.db_path,
        signal_mode=SignalModeEnum(ns.mode),
        bias_source=BiasSourceEnum(ns.bias_source),
        bias_ma=ns.bias_ma,
        momentum_window=ns.momentum_window,
        volume_ma_window=ns.volume_ma_window,
        rolling_window=ns.rolling_window,
        quantile_low=ns.quantile_low,
        quantile_high=ns.quantile_high,
        from_state=ns.from_state,
        horizons=horizons,
        transition_top_k=ns.transition_top_k,
    )
    resp = run_state_transition(req)
    if ns.json:
        print(resp.model_dump_json(indent=2))
    else:
        print(
            f"{resp.etf_code} ({resp.etf_name}) | from_state={resp.from_state!r} | "
            f"total_origin_days={resp.total_samples} | pattern={resp.state_pattern} | "
            f"mode={resp.signal_mode} bias=MA{resp.bias_ma} mom={resp.momentum_window}d volMA={resp.volume_ma_window}d"
        )
        for hk in sorted(resp.horizons.keys(), key=lambda x: int(x)):
            blk = resp.horizons[hk]
            print(f"\n--- horizon {hk}d (n_valid={blk.n_valid}, entropy={blk.entropy_nats:.4f} nats) ---")
            for r in blk.transitions[:12]:
                mr = f"{r.mean_return:.4f}" if r.mean_return is not None else "—"
                wr = f"{r.win_rate:.4f}" if r.win_rate is not None else "—"
                print(
                    f"  {r.to_state:<24}  p={r.prob:.3f}  n={r.count:4d}  "
                    f"mean_ret={mr}  win={wr}"
                )
            if len(blk.transitions) > 12:
                print(f"  ... ({len(blk.transitions) - 12} more rows)")
    if ns.save_json:
        ns.save_json.write_text(resp.model_dump_json(indent=2), encoding="utf-8")
        if not ns.json:
            print(f"Wrote {ns.save_json}", file=sys.stderr)
    return 0


def cmd_state_rank(ns: argparse.Namespace) -> int:
    from core.runner import run_state_ranking
    from core.schemas import SignalModeEnum, StateRankingRequest

    req = StateRankingRequest(
        etf_code=ns.etf_code,
        db_path=ns.db_path,
        signal_mode=SignalModeEnum(ns.mode),
        momentum_window=ns.momentum_window,
        bias_ma=ns.bias_ma,
        volume_ma_window=ns.volume_ma_window,
        rolling_window=ns.rolling_window,
        horizon=ns.horizon,
        min_n=ns.min_n,
        top_k=ns.top_k,
        bottom_k=ns.bottom_k,
        ternary_q1=ns.ternary_q1,
        ternary_q2=ns.ternary_q2,
    )
    resp = run_state_ranking(req)
    if ns.json:
        print(resp.model_dump_json(indent=2))
    else:
        print(
            f"{resp.etf_code} ({resp.etf_name}) | horizon={resp.horizon}d | bucket={resp.bucket_mode} | "
            f"mom={resp.momentum_window}d bias=MA{resp.bias_ma} volMA={resp.volume_ma_window}d | "
            f"ternary q1={resp.ternary_q1} q2={resp.ternary_q2} | states_ranked={resp.states_ranked}"
        )
        print("\n--- Top best (by win rate) ---")
        for i, r in enumerate(resp.top_best, 1):
            print(
                f"  {i}. {r.state}  n={r.n}  win={r.win_rate:.4f}  mean={r.mean_return:.4f}  "
                f"median={r.median_return:.4f}  std={r.std:.4f}"
            )
        print("\n--- Bottom worst ---")
        for i, r in enumerate(resp.bottom_worst, 1):
            print(
                f"  {i}. {r.state}  n={r.n}  win={r.win_rate:.4f}  mean={r.mean_return:.4f}  "
                f"median={r.median_return:.4f}  std={r.std:.4f}"
            )
    if ns.save_json:
        ns.save_json.write_text(resp.model_dump_json(indent=2), encoding="utf-8")
        if not ns.json:
            print(f"Wrote {ns.save_json}", file=sys.stderr)
    return 0


def cmd_report(ns: argparse.Namespace) -> int:
    from core.runner import run_report
    from core.schemas import (
        BacktestParamSourceEnum,
        BiasSourceEnum,
        ReportRequest,
        SignalModeEnum,
        SignalParamSourceEnum,
    )

    horizons = _parse_horizons(ns.horizons)
    preset = "auto" if ns.use_recommendation else ns.signal_preset
    sp_src = SignalParamSourceEnum.auto if preset == "auto" else SignalParamSourceEnum.manual
    bt_rec = ns.backtest_preset == "recommended" or bool(getattr(ns, "apply_recommendation_bundle", False))
    bps = BacktestParamSourceEnum.recommended if bt_rec else BacktestParamSourceEnum.manual
    req = ReportRequest(
        etf_code=ns.etf_code,
        db_path=ns.db_path,
        signal_param_source=sp_src,
        backtest_param_source=bps,
        signal_mode=SignalModeEnum(ns.mode),
        bias_source=BiasSourceEnum(ns.bias_source),
        bias_ma=ns.bias_ma,
        momentum_window=ns.momentum_window,
        volume_ma_window=ns.volume_ma_window,
        rolling_window=ns.rolling_window,
        quantile_low=ns.quantile_low,
        quantile_high=ns.quantile_high,
        strategy_profile=ns.strategy_profile,
        hold_days=ns.hold_days,
        compare_profiles=bool(ns.compare_profiles),
        compare_manual_vs_recommended=bool(ns.compare_manual_vs_recommended),
        evaluate_exit_rules=bool(getattr(ns, "evaluate_exit", False)),
        optimize_exit=bool(getattr(ns, "optimize_exit", False)),
        entry_signal_tier=getattr(ns, "entry_signal_tier", None),
        explicit_exit_rule_id=getattr(ns, "explicit_exit_rule_id", None),
        compare_exit_rules=bool(getattr(ns, "compare_exit_rules", False)),
        export_trades_path=str(ns.export_trades) if getattr(ns, "export_trades", None) else None,
        multi_objective_exit=bool(getattr(ns, "multi_objective_exit", False)),
        exit_objective=getattr(ns, "exit_objective", None),
        entry_diagnostics=bool(getattr(ns, "entry_diagnostics", False)),
        entry_diagnostics_dates=bool(getattr(ns, "entry_diagnostics_dates", False)),
        entry_exit_matching=bool(getattr(ns, "entry_exit_matching", False)),
        entry_exit_top=getattr(ns, "entry_exit_top", None),
        entry_map_json_path=getattr(ns, "entry_map_json", None),
        event_horizons=horizons,
        output_dir=Path(ns.output),
        write_json=not ns.no_json,
        write_csv=not ns.no_csv,
        write_charts_html=not ns.no_charts,
    )
    resp = run_report(req)
    if ns.print_json:
        print(resp.model_dump_json(indent=2))
    else:
        print(f"Report written under: {resp.output_dir}")
        for k, v in sorted(resp.artifact_paths.items()):
            print(f"  {k}: {v}")
    return 0


def _recommend_flat_payload(resp: Any) -> dict[str, Any]:
    """Merge ETF meta + recommendation for CLI JSON (fit_level, framework_fit_note, best_signal_setup, …)."""
    inner = resp.recommendation.model_dump(mode="json")
    return {
        "etf_code": resp.etf_code,
        "etf_name": resp.etf_name,
        "db_path": resp.db_path,
        **inner,
    }


def cmd_recommend(ns: argparse.Namespace) -> int:
    from core.runner import run_recommendation
    from core.schemas import RecommendationRequest

    req = RecommendationRequest(
        etf_code=ns.etf_code,
        db_path=ns.db_path,
        momentum_window=ns.momentum_window,
        volume_ma_window=ns.volume_ma_window,
        rolling_window=ns.rolling_window,
        quantile_low=ns.quantile_low,
        quantile_high=ns.quantile_high,
        eval_horizon=ns.eval_horizon,
        top_k=ns.top_k,
        include_exit_rules=bool(getattr(ns, "include_exit", False)),
        entry_map_json_path=getattr(ns, "entry_map_json", None),
    )
    resp = run_recommendation(req)
    flat = _recommend_flat_payload(resp)
    if ns.json:
        print(json.dumps(flat, indent=2, ensure_ascii=False))
    else:
        r = resp.recommendation
        b = r.best_signal_setup
        print(f"ETF {resp.etf_code} ({resp.etf_name})")
        print()
        print("--- Framework fit ---")
        print(f"fit_level: {r.fit_level}")
        print(f"framework_fit_note: {r.framework_fit_note}")
        if r.strategy_mode is not None:
            print(f"strategy_mode (entry_map): {r.strategy_mode}")
        print()
        print("--- Best signal setup (grid) ---")
        print(f"signal: {b.signal}  mode: {b.mode}  bias_ma: {b.bias_ma}  horizon_focus: {b.horizon_focus}")
        print(f"recommendation_score (z): {b.recommendation_score:.4f}")
        print(
            f"mean_return_60: {b.mean_return_60}  mean_return_120: {b.mean_return_120}  "
            f"win_rate_60: {b.win_rate_60}  n_60: {b.n_60}"
        )
        print(f"mirrors: default_signal={r.default_signal}  explore: {', '.join(r.recommended_signals)}")
        print()
        print("--- Top candidates (ranked) ---")
        for c in r.top_candidates:
            print(
                f"  #{c.rank}  {c.signal}  {c.mode}  MA{c.bias_ma}  h={c.horizon_focus}  "
                f"score_z={c.recommendation_score:.4f}  mr60={c.mean_return_60}  wr60={c.win_rate_60}  n60={c.n_60}"
            )
        print()
        if r.exit_rule_candidates:
            print()
            print("--- Exit rules (fixed entry = best grid setup) ---")
            print(r.exit_rule_explanation)
            for er in r.exit_rule_candidates[:8]:
                dn = (er.display_name or er.label_zh or er.rule_id).strip()
                print(
                    f"  #{er.rank}  {dn}  (`{er.rule_id}`)  score={er.score:.4f}  trades={er.n_trades}  "
                    f"eligible={er.eligible}"
                )
                if (er.plain_explanation or "").strip():
                    print(f"      {er.plain_explanation}")
            if r.best_exit_rule:
                be = r.best_exit_rule
                bdn = (be.display_name or be.label_zh or be.rule_id).strip()
                print(f"  => best eligible: `{be.rule_id}` — {bdn}")
        print()
        print("notes:")
        for note in r.notes:
            print(f"  - {note}")
    if ns.save_json:
        ns.save_json.write_text(json.dumps(flat, indent=2, ensure_ascii=False), encoding="utf-8")
        if not ns.json:
            print(f"Wrote {ns.save_json}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab", description="ETF quant research CLI (core-backed)")
    sub = parser.add_subparsers(dest="command", required=True)

    ph = sub.add_parser("health", help="Data quality / validation for one ETF")
    _add_etf_db(ph)
    ph.add_argument("--invalid-limit", type=int, default=500, dest="invalid_limit")
    ph.add_argument("--json", action="store_true")
    ph.add_argument("--save-json", type=Path, default=None)
    ph.set_defaults(_run=cmd_health)

    ps = sub.add_parser("signal-research", help="Event-study tables + chart specs")
    _add_signal_params(ps)
    ps.set_defaults(_run=cmd_signal_research)

    prc = sub.add_parser("recommend", help="Rule-based ETF fit + default signal / mode / bias MA (no event-study run)")
    _add_etf_db(prc)
    _add_entry_map_json_arg(prc)
    _add_signal_dimensions(prc)
    prc.add_argument("--rolling-window", type=int, default=252, dest="rolling_window")
    prc.add_argument(
        "--eval-horizon",
        type=int,
        default=60,
        dest="eval_horizon",
        help="Documented in notes; scoring uses horizons 20/60/120 in the search grid",
    )
    prc.add_argument(
        "--top-k",
        type=int,
        default=12,
        dest="top_k",
        help="Number of ranked candidates to return (max 54)",
    )
    prc.add_argument(
        "--include-exit",
        action="store_true",
        dest="include_exit",
        help="在固定推荐入场下横评退出规则并写入 best_exit_rule / exit_rule_candidates",
    )
    prc.add_argument("--json", action="store_true")
    prc.add_argument("--save-json", type=Path, default=None)
    prc.set_defaults(_run=cmd_recommend)

    pst = sub.add_parser(
        "state-rank",
        help="Ternary state scan (NEG/NEU/POS × LOW/MID/HIGH × volume); top/bottom by win rate",
    )
    _add_etf_db(pst)
    pst.add_argument("--mode", choices=["full_sample", "rolling"], default="rolling")
    pst.add_argument("--bias-ma", type=int, default=120, dest="bias_ma")
    pst.add_argument("--momentum-window", type=int, default=10, choices=[5, 10, 20, 60])
    pst.add_argument("--volume-ma-window", type=int, default=20, choices=[5, 10, 20, 60])
    pst.add_argument("--rolling-window", type=int, default=252, dest="rolling_window")
    pst.add_argument("--horizon", type=int, default=20, help="Forward holding period (trading days)")
    pst.add_argument("--min-n", type=int, default=5, dest="min_n")
    pst.add_argument("--top", type=int, default=5, dest="top_k")
    pst.add_argument("--bottom", type=int, default=5, dest="bottom_k")
    pst.add_argument("--ternary-q1", type=float, default=0.33, dest="ternary_q1")
    pst.add_argument("--ternary-q2", type=float, default=0.67, dest="ternary_q2")
    pst.add_argument("--json", action="store_true")
    pst.add_argument("--save-json", type=Path, default=None)
    pst.set_defaults(_run=cmd_state_rank)

    ptt = sub.add_parser(
        "analyze-transition",
        help="Origin state → future states over horizons (research frame; no trades)",
    )
    _add_etf_db(ptt)
    ptt.add_argument("--mode", choices=["full_sample", "rolling"], default="rolling")
    _add_bias_source(ptt)
    ptt.add_argument("--bias-ma", type=int, default=120, dest="bias_ma")
    _add_signal_dimensions(ptt)
    ptt.add_argument("--rolling-window", type=int, default=252, dest="rolling_window")
    ptt.add_argument("--from-state", required=True, dest="from_state", help="e.g. NEG_LOW or POS_MID_HIGH (prefix match for partial codes)")
    ptt.add_argument(
        "--horizons",
        type=str,
        default="5,10,20,60",
        help="Comma-separated forward horizons in trading days",
    )
    ptt.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="transition_top_k",
        metavar="K",
        help="Keep only top K destination states by count per horizon (optional)",
    )
    ptt.add_argument("--json", action="store_true")
    ptt.add_argument("--save-json", type=Path, default=None)
    ptt.set_defaults(_run=cmd_analyze_transition)

    pq = sub.add_parser(
        "analyze-path-quality",
        help="Origin days → target state within horizon; feature quantile breakdown (no trades)",
    )
    _add_etf_db(pq)
    pq.add_argument("--mode", choices=["full_sample", "rolling"], default="rolling")
    _add_bias_source(pq)
    pq.add_argument("--bias-ma", type=int, default=120, dest="bias_ma")
    _add_signal_dimensions(pq)
    pq.add_argument("--rolling-window", type=int, default=252, dest="rolling_window")
    pq.add_argument("--from-state", required=True, dest="from_state")
    pq.add_argument("--target-state", required=True, dest="target_state")
    pq.add_argument("--horizon", type=int, required=True, help="Trading days; forward return to t+H")
    pq.add_argument(
        "--target-mode",
        choices=["ever", "final"],
        default="ever",
        dest="target_mode",
        help="ever=target appears in (t+1..t+H]; final=state at t+H only",
    )
    pq.add_argument(
        "--bucket-features",
        type=str,
        default="bias_rate,momentum,volume_ratio",
        help="Comma-separated: bias_rate, momentum, volume_ratio, daily_change",
    )
    pq.add_argument(
        "--bucket-n",
        type=int,
        default=5,
        dest="bucket_n",
        help="Ignored for path-quality (global Q1–Q5 *_bucket columns from full sample)",
    )
    add_bias_quantile_filter_arg(pq)
    pq.add_argument("--json", action="store_true")
    pq.add_argument("--save-json", type=Path, default=None)
    pq.set_defaults(_run=cmd_analyze_path_quality)

    pem = sub.add_parser(
        "discover-entry-map",
        help="Path-quality + path-rules: ETF → best from-state, driver, entry archetype (research-only)",
    )
    pem.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="SQLite path (default: ETF_DB_PATH env, else repo etf_data.db, else db/etf_data.db)",
    )
    pem.add_argument(
        "--etf-json",
        type=Path,
        default=None,
        dest="etf_json",
        metavar="PATH",
        help="JSON array of {code,name}；不传则默认从当前 --db 的 etf_daily_metrics 动态读取全部 ETF",
    )
    pem.add_argument("--mode", choices=["full_sample", "rolling"], default="rolling")
    _add_bias_source(pem)
    pem.add_argument("--bias-ma", type=int, default=120, dest="bias_ma")
    _add_signal_dimensions(pem)
    pem.add_argument("--rolling-window", type=int, default=252, dest="rolling_window")
    pem.add_argument("--horizon", type=int, default=60, help="Path-quality / path-rules horizon (trading days)")
    pem.add_argument("--target-state", default="POS_HIGH_HIGH", dest="target_state")
    pem.add_argument(
        "--target-mode",
        choices=["ever", "final"],
        default="ever",
        dest="target_mode",
        help="Path-quality / path-rules target reach mode",
    )
    pem.add_argument("--min-samples", type=int, default=5, dest="min_samples")
    pem.add_argument(
        "--weak-hit-floor",
        type=float,
        default=0.40,
        dest="weak_hit_rate_floor",
        help="Mark weak_path_quality when best state's hit_rate is below this",
    )
    pem.add_argument(
        "--auto-mode",
        dest="auto_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable rolling→full_sample fallback when rolling has no eligible best_state (default: on; use --no-auto-mode for rolling-only debug)",
    )
    add_bias_quantile_filter_arg(pem)
    pem.add_argument("--json", action="store_true", help="Print full snapshot JSON")
    pem.add_argument("--save-json", type=Path, default=None, dest="save_json")
    pem.set_defaults(_run=cmd_discover_entry_map)

    prm = sub.add_parser(
        "analyze-path-rules",
        help="Mine contiguous quantile path rules on from_state samples (no trades)",
    )
    _add_etf_db(prm)
    prm.add_argument("--mode", choices=["full_sample", "rolling"], default="rolling")
    _add_bias_source(prm)
    prm.add_argument("--bias-ma", type=int, default=120, dest="bias_ma")
    _add_signal_dimensions(prm)
    prm.add_argument("--rolling-window", type=int, default=252, dest="rolling_window")
    prm.add_argument("--from-state", required=True, dest="from_state")
    prm.add_argument("--target-state", required=True, dest="target_state")
    prm.add_argument("--horizon", type=int, required=True)
    prm.add_argument(
        "--target-mode",
        choices=["ever", "final"],
        default="ever",
        dest="target_mode",
    )
    prm.add_argument(
        "--features",
        type=str,
        default="bias_rate,volume_ratio",
        help="Comma-separated: bias_rate, volume_ratio, momentum, daily_change",
    )
    prm.add_argument(
        "--bucket-n",
        type=int,
        default=5,
        dest="bucket_n",
        help="Ignored (global Q1–Q5 *_bucket columns)",
    )
    prm.add_argument(
        "--max-combinations",
        type=int,
        default=2,
        choices=[1, 2],
        dest="max_combinations",
        help="1=single-factor only; 2=include two-factor AND rules",
    )
    prm.add_argument("--min-count", type=int, default=5, dest="min_count")
    prm.add_argument(
        "--top-k",
        type=int,
        default=20,
        dest="top_k",
        help="Max rules to return after ranking (0 = no limit)",
    )
    prm.add_argument(
        "--rules-above-baseline-only",
        action="store_true",
        dest="rules_above_baseline_only",
        help="Keep only rules with hit_rate >= baseline hit_rate",
    )
    add_bias_quantile_filter_arg(prm)
    prm.add_argument("--json", action="store_true")
    prm.add_argument("--save-json", type=Path, default=None)
    prm.set_defaults(_run=cmd_analyze_path_rules)

    pb = sub.add_parser(
        "backtest",
        help="Non-overlapping hold backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "参数分两类：\n"
            "  【推荐层】--signal-preset auto / --backtest-preset recommended：套用规则层默认 mode、乖离、持有、画像。\n"
            "  【实验层】--signal-tier / --exit-rule / --compare-exit-rules：在明确假设下做可复现回测；"
            "不会用网格最优信号层覆盖你已锁定的 --signal-tier。\n"
            "\n"
            "退出相关：\n"
            "  --optimize-exit  在横评中按 score_exit_metrics 选最优 eligible 规则驱动 **主回测**（样本内）。\n"
            "  --exit-rule      你 **直接指定** 规则驱动主回测；不可与 --optimize-exit 同用。\n"
            "  --evaluate-exit  只算横评表（主回测仍为固定持有，除非同时 optimize）。\n"
            "  --compare-exit-rules  额外输出全规则对照表（exit_sweep_under_entry），不改变 optimize/exit-rule 的主线逻辑。\n"
            "  --multi-objective  在横评结果上输出 Pareto 与各视角最优（JSON: multi_objective_decision）。\n"
            "  --objective  指定默认推荐视角（return_first|risk_first|efficiency_first|robustness_first）；隐含需要横评数据。\n"
            "  --entry-diagnostics  输出原始入场 EOD 信号诊断（JSON: entry_signal_diagnostics）。\n"
            "  --entry-diagnostics-dates  同时列出全部 raw_entry_dates（可能很长）。\n"
            "  --entry-exit-matching  入场 regime vs 退出持仓对齐（JSON: entry_exit_matching_diagnostics）。\n"
            "  --entry-exit-top N  人类可读表只打印前 N 条 per_exit。\n"
        ),
    )
    _add_etf_db(pb)
    _add_entry_map_json_arg(pb)
    pb.add_argument("--mode", choices=["full_sample", "rolling"], default="rolling")
    _add_bias_source(pb)
    add_bias_quantile_filter_arg(pb)
    pb.add_argument("--bias-ma", type=int, default=120, dest="bias_ma")
    _add_signal_dimensions(pb)
    pb.add_argument("--rolling-window", type=int, default=252, dest="rolling_window")
    pb.add_argument("--json", action="store_true")
    pb.add_argument("--save-json", type=Path, default=None)
    pb.add_argument("--hold-days", type=int, default=120, dest="hold_days")
    _add_backtest_strategy_cli(pb)
    _add_exit_rule_cli(pb)
    _add_backtest_experiment_cli(pb)
    _add_signal_preset_flags(pb)
    pb.set_defaults(_run=cmd_backtest)

    pls = sub.add_parser(
        "latest-state",
        help="最后一根已收盘日的 NEG/LOW/HIGH 与 signal_tier（与 backtest 有效信号参数一致，不跑组合回测）",
    )
    _add_etf_db(pls)
    _add_entry_map_json_arg(pls)
    pls.add_argument("--mode", choices=["full_sample", "rolling"], default="rolling")
    _add_bias_source(pls)
    add_bias_quantile_filter_arg(pls)
    pls.add_argument("--bias-ma", type=int, default=120, dest="bias_ma")
    _add_signal_dimensions(pls)
    pls.add_argument("--rolling-window", type=int, default=252, dest="rolling_window")
    pls.add_argument("--hold-days", type=int, default=120, dest="hold_days")
    _add_backtest_strategy_cli(pls)
    _add_backtest_experiment_cli(pls)
    _add_signal_preset_flags(pls)
    pls.add_argument("--json", action="store_true")
    pls.add_argument("--save-json", type=Path, default=None)
    pls.set_defaults(_run=cmd_latest_state)

    pr = sub.add_parser("report", help="Run health + signal + backtest and save artifacts")
    _add_signal_params(pr)
    _add_entry_map_json_arg(pr)
    _add_backtest_strategy_cli(pr)
    _add_exit_rule_cli(pr)
    _add_backtest_experiment_cli(pr)
    pr.add_argument("--hold-days", type=int, default=120, dest="hold_days")
    pr.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    pr.add_argument("--no-json", action="store_true", help="Skip JSON artifacts")
    pr.add_argument("--no-csv", action="store_true", help="Skip CSV artifacts")
    pr.add_argument("--no-charts", action="store_true", help="Skip chart HTML")
    pr.add_argument("--print-json", action="store_true", help="Also print full ReportResponse JSON")
    pr.set_defaults(_run=cmd_report)

    from quantlab.cli.score_bloody_chip import add_parser as add_score_bloody_chip_parser
    from quantlab.cli.rank_bloody_chip import add_parser as add_rank_bloody_chip_parser
    from quantlab.cli.explain_bloody_chip import add_parser as add_explain_bloody_chip_parser
    from quantlab.cli.dividend_signal import add_parser as add_dividend_signal_parser
    from quantlab.cli.recommend_strategy import add_parser as add_recommend_parser
    from quantlab.cli.backtest_strategy_report import add_parser as add_backtest_strategy_report_parser
    from quantlab.cli.debug_trades import add_parser as add_debug_trades_parser
    from quantlab.cli.market_state import add_parser as add_market_state_parser
    from quantlab.cli.portfolio_regime_cli import add_parser as add_portfolio_regime_parser
    
    add_score_bloody_chip_parser(sub)
    add_rank_bloody_chip_parser(sub)
    add_explain_bloody_chip_parser(sub)
    add_dividend_signal_parser(sub)
    add_recommend_parser(sub)
    add_backtest_strategy_report_parser(sub)
    add_debug_trades_parser(sub)
    add_market_state_parser(sub)
    add_portfolio_regime_parser(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    ns = parser.parse_args(argv)
    try:
        return int(ns._run(ns))
    except Exception as e:
        print(f"quantlab: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
