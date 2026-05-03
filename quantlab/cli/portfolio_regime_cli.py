import argparse
import sys
import json
from pathlib import Path
from typing import Any

import sqlite3
from core.paths import resolve_db_path

from quantlab.cli.portfolio_regime import (
    get_latest_regime,
    get_portfolio_weights,
    backtest_portfolio,
    compute_portfolio_action,
    MarketRegime,
    ETF_NAMES,
)


def cmd_detect_market_regime(ns: argparse.Namespace) -> int:
    db = resolve_db_path(ns.db_path)
    try:
        with sqlite3.connect(db) as conn:
            detection = get_latest_regime(conn)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"{'=' * 60}")
    print(f"市场状态识别")
    print(f"{'=' * 60}")
    print()
    print(f"最新交易日: {detection.trade_date}")
    print()
    print(f"market_regime (confirmed): {detection.regime.value}")
    print(f"raw_regime: {detection.raw_regime.value}")
    print()
    print(f"指标详情:")
    print(f"  bias: {detection.bias:.6f}")
    print(f"  bias_percentile: {detection.bias_percentile:.4f}")
    print(f"  momentum: {detection.momentum:.6f}")
    print(f"  breadth: {detection.breadth:.6f}")
    print()
    print(f"判断理由: {detection.reason}")

    if getattr(ns, "json", False):
        result = {
            "trade_date": detection.trade_date,
            "market_regime": detection.regime.value,
            "market_regime_raw": detection.raw_regime.value,
            "bias": detection.bias,
            "bias_percentile": detection.bias_percentile,
            "momentum": detection.momentum,
            "breadth": detection.breadth,
            "reason": detection.reason,
        }
        print()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if getattr(ns, "save_json", None):
        save_path = Path(ns.save_json)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        result = {
            "trade_date": detection.trade_date,
            "market_regime": detection.regime.value,
            "market_regime_raw": detection.raw_regime.value,
            "bias": detection.bias,
            "bias_percentile": detection.bias_percentile,
            "momentum": detection.momentum,
            "breadth": detection.breadth,
            "reason": detection.reason,
        }
        save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if not getattr(ns, "json", False):
            print(f"Wrote {save_path}", file=sys.stderr)

    return 0


def cmd_recommend_portfolio(ns: argparse.Namespace) -> int:
    db = resolve_db_path(ns.db_path)
    try:
        with sqlite3.connect(db) as conn:
            detection = get_latest_regime(conn)
            weights = get_portfolio_weights(detection.regime)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"{'=' * 60}")
    print(f"组合推荐")
    print(f"{'=' * 60}")
    print()
    print(f"市场状态 (confirmed): {detection.regime.value} (raw={detection.raw_regime.value})")
    print()
    print(f"目标权重:")
    for code, w in weights.items():
        name = ETF_NAMES.get(code, code)
        print(f"  {code} ({name}): {w * 100:.1f}%")
    print()

    interpretation = ""
    if detection.regime == MarketRegime.AGGRESSIVE:
        interpretation = "偏进攻，提升 A500 权重"
    elif detection.regime == MarketRegime.DEFENSIVE:
        interpretation = "偏防守，提升中证红利权重"
    else:
        interpretation = "均衡配置"

    print(f"简短解读: {interpretation}")

    if getattr(ns, "json", False):
        result = {
            "market_regime": detection.regime.value,
            "market_regime_raw": detection.raw_regime.value,
            "target_weights": weights,
            "interpretation": interpretation,
            "trade_date": detection.trade_date,
        }
        print()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if getattr(ns, "save_json", None):
        save_path = Path(ns.save_json)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        result = {
            "market_regime": detection.regime.value,
            "market_regime_raw": detection.raw_regime.value,
            "target_weights": weights,
            "interpretation": interpretation,
            "trade_date": detection.trade_date,
        }
        save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if not getattr(ns, "json", False):
            print(f"Wrote {save_path}", file=sys.stderr)

    return 0


def cmd_backtest_portfolio_regime(ns: argparse.Namespace) -> int:
    db = resolve_db_path(ns.db_path)
    start_date = ns.start_date or "2020-02-06"
    end_date = ns.end_date or "2099-12-31"
    rebalance = getattr(ns, "rebalance", "monthly")
    regime_confirm_days = getattr(ns, "regime_confirm_days", 5)
    max_step = getattr(ns, "max_step", 0.1)

    try:
        with sqlite3.connect(db) as conn:
            result = backtest_portfolio(
                conn, start_date, end_date, 
                rebalance_freq=rebalance,
                regime_confirm_days=regime_confirm_days,
                max_step=max_step
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"{'=' * 60}")
    print(f"组合回测 (低频 + 滞后 + 区间)")
    print(f"{'=' * 60}")
    print()
    print(f"回测区间: {result['start_date']} ~ {result['end_date']}")
    print(f"交易天数: {result['n_days']}")
    print(f"调仓频率: {rebalance}")
    print(f"Regime 滞后确认: {regime_confirm_days} 天")
    print(f"单次调仓最大幅度: {max_step * 100:.0f}%")
    print()

    print(f"{'Strategy':<20} {'Total Return':>12} {'Annual':>10} {'Max DD':>10} {'Sharpe':>10}")
    print(f"{'-' * 72}")
    p = result["portfolio"]
    print(f"{'Regime-Based':<20} {p['total_return']:>12.2%} {p['annualized_return']:>10.2%} {p['max_drawdown']:>10.2%} {p['sharpe_ratio']:>10.2f}")
    be = result["baseline_equal"]
    print(f"{'Equal-Weight':<20} {be['total_return']:>12.2%} {be['annualized_return']:>10.2%} {be['max_drawdown']:>10.2%} {'N/A':>10}")
    bb = result["baseline_balanced"]
    print(f"{'Balanced-Fixed':<20} {bb['total_return']:>12.2%} {bb['annualized_return']:>10.2%} {bb['max_drawdown']:>10.2%} {'N/A':>10}")
    print()

    print(f"调仓统计:")
    print(f"  调仓检查次数: {p['rebalance_count']}")
    print(f"  实际调仓次数: {p['actual_rebalance_count']}")
    print(f"  Regime 切换次数: {p['regime_switch_count']}")
    print(f"  累计换手: {p['turnover_total']:.2%}")
    print(f"  平均单次换手: {p['average_turnover']:.2%}")
    print(f"  平均单次调整幅度: {p['average_step']:.2%}")
    print()

    print(f"市场状态分布:")
    total = sum(result["regime_counts"].values())
    for r, cnt in result["regime_counts"].items():
        print(f"  {r}: {cnt} 天 ({cnt / total:.1%})")

    if getattr(ns, "json", False):
        output = {k: v for k, v in result.items() if k != "regime_history"}
        print()
        print(json.dumps(output, ensure_ascii=False, indent=2))

    if getattr(ns, "save_json", None):
        save_path = Path(ns.save_json)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        output = {k: v for k, v in result.items() if k != "regime_history"}
        save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        if not getattr(ns, "json", False):
            print(f"Wrote {save_path}", file=sys.stderr)

    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p1 = subparsers.add_parser("detect-market-regime", help="识别当前市场状态 (AGGRESSIVE/BALANCED/DEFENSIVE)")
    p1.add_argument("--db", dest="db_path", default=None, help="SQLite path")
    p1.add_argument("--json", action="store_true", help="Print JSON")
    p1.add_argument("--save-json", type=Path, default=None, help="Save JSON to file")
    p1.set_defaults(_run=cmd_detect_market_regime)

    p2 = subparsers.add_parser("recommend-portfolio", help="根据市场状态输出三 ETF 组合目标权重")
    p2.add_argument("--db", dest="db_path", default=None, help="SQLite path")
    p2.add_argument("--json", action="store_true", help="Print JSON")
    p2.add_argument("--save-json", type=Path, default=None, help="Save JSON to file")
    p2.set_defaults(_run=cmd_recommend_portfolio)

    p3 = subparsers.add_parser("backtest-portfolio-regime", help="用历史数据回测 regime-based 组合")
    p3.add_argument("--db", dest="db_path", default=None, help="SQLite path")
    p3.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    p3.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    p3.add_argument("--rebalance", choices=["weekly", "monthly", "daily"], default="monthly", help="Rebalance frequency (default: monthly)")
    p3.add_argument("--regime-confirm-days", type=int, default=5, dest="regime_confirm_days", help="Regime 滞后确认天数 (default: 5)")
    p3.add_argument("--max-step", type=float, default=0.1, dest="max_step", help="单次调仓最大幅度 (default: 0.1 = 10%%)")
    p3.add_argument("--json", action="store_true", help="Print JSON")
    p3.add_argument("--save-json", type=Path, default=None, help="Save JSON to file")
    p3.set_defaults(_run=cmd_backtest_portfolio_regime)


def cmd_recommend_portfolio_action(ns: argparse.Namespace) -> int:
    db = resolve_db_path(ns.db_path)
    current_a500 = ns.current_a500
    current_dividend = ns.current_dividend
    current_dividend_growth = ns.current_dividend_growth
    regime_confirm_days = getattr(ns, "regime_confirm_days", 5)
    max_step = getattr(ns, "max_step", 0.1)

    total = current_a500 + current_dividend + current_dividend_growth
    if abs(total - 1.0) > 0.001:
        print(f"Error: 当前仓位总和为 {total:.4f}，应约等于 1.0", file=sys.stderr)
        return 1

    try:
        with sqlite3.connect(db) as conn:
            action = compute_portfolio_action(
                conn,
                current_a500,
                current_dividend,
                current_dividend_growth,
                regime_confirm_days=regime_confirm_days,
                max_step=max_step
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"{'=' * 60}")
    print(f"组合执行建议")
    print(f"{'=' * 60}")
    print()
    print(f"当前市场状态 (confirmed): {action.market_regime.value} (raw={action.raw_regime.value})")
    print(f"最新交易日: {action.trade_date}")
    print()

    print(f"{'=' * 60}")
    print(f"当前持仓")
    print(f"{'=' * 60}")
    print()
    for code, w in action.current_position.items():
        name = ETF_NAMES.get(code, code)
        print(f"  {code} ({name}): {w * 100:.1f}%")
    print()

    print(f"{'=' * 60}")
    print(f"目标区间")
    print(f"{'=' * 60}")
    print()
    tr = action.target_range
    a500_min, a500_max = tr["159361"]
    print(f"  A500 ETF: {a500_min * 100:.0f}% ~ {a500_max * 100:.0f}%")
    print(f"  红利质量 ETF: 固定 30.0%")
    print(f"  中证红利 ETF: 剩余仓位")
    print()

    print(f"{'=' * 60}")
    print(f"本次建议")
    print(f"{'=' * 60}")
    print()
    print(f"Action: {action.action}")
    if action.action == "REBALANCE":
        print()
        print(f"This step:")
        adj = action.adjustment
        for code, delta in adj.items():
            name = ETF_NAMES.get(code, code)
            sign = "+" if delta > 0 else ""
            print(f"  - {name}: {sign}{delta * 100:.1f}%")
    print()
    print(f"Reason: {action.reason}")
    print()

    print(f"{'=' * 60}")
    print(f"执行后持仓")
    print(f"{'=' * 60}")
    print()
    for code, w in action.after_position.items():
        name = ETF_NAMES.get(code, code)
        print(f"  {code} ({name}): {w * 100:.1f}%")
    print()

    if getattr(ns, "json", False):
        result = {
            "trade_date": action.trade_date,
            "market_regime": action.market_regime.value,
            "market_regime_raw": action.raw_regime.value,
            "current_position": action.current_position,
            "target_range": {
                "159361": list(action.target_range["159361"]),
                "159209": list(action.target_range["159209"]),
            },
            "action": action.action,
            "adjustment": action.adjustment,
            "after_position": action.after_position,
            "reason": action.reason,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if getattr(ns, "save_json", None):
        save_path = Path(ns.save_json)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        result = {
            "trade_date": action.trade_date,
            "market_regime": action.market_regime.value,
            "market_regime_raw": action.raw_regime.value,
            "current_position": action.current_position,
            "target_range": {
                "159361": list(action.target_range["159361"]),
                "159209": list(action.target_range["159209"]),
            },
            "action": action.action,
            "adjustment": action.adjustment,
            "after_position": action.after_position,
            "reason": action.reason,
        }
        save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if not getattr(ns, "json", False):
            print(f"Wrote {save_path}", file=sys.stderr)

    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p1 = subparsers.add_parser("detect-market-regime", help="识别当前市场状态 (AGGRESSIVE/BALANCED/DEFENSIVE)")
    p1.add_argument("--db", dest="db_path", default=None, help="SQLite path")
    p1.add_argument("--json", action="store_true", help="Print JSON")
    p1.add_argument("--save-json", type=Path, default=None, help="Save JSON to file")
    p1.set_defaults(_run=cmd_detect_market_regime)

    p2 = subparsers.add_parser("recommend-portfolio", help="根据市场状态输出三 ETF 组合目标权重")
    p2.add_argument("--db", dest="db_path", default=None, help="SQLite path")
    p2.add_argument("--json", action="store_true", help="Print JSON")
    p2.add_argument("--save-json", type=Path, default=None, help="Save JSON to file")
    p2.set_defaults(_run=cmd_recommend_portfolio)

    p3 = subparsers.add_parser("backtest-portfolio-regime", help="用历史数据回测 regime-based 组合")
    p3.add_argument("--db", dest="db_path", default=None, help="SQLite path")
    p3.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    p3.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    p3.add_argument("--rebalance", choices=["weekly", "monthly", "daily"], default="monthly", help="Rebalance frequency (default: monthly)")
    p3.add_argument("--regime-confirm-days", type=int, default=5, dest="regime_confirm_days", help="Regime 滞后确认天数 (default: 5)")
    p3.add_argument("--max-step", type=float, default=0.1, dest="max_step", help="单次调仓最大幅度 (default: 0.1 = 10%%)")
    p3.add_argument("--json", action="store_true", help="Print JSON")
    p3.add_argument("--save-json", type=Path, default=None, help="Save JSON to file")
    p3.set_defaults(_run=cmd_backtest_portfolio_regime)

    p4 = subparsers.add_parser("recommend-portfolio-action", help="根据当前持仓输出实盘调仓建议")
    p4.add_argument("--db", dest="db_path", default=None, help="SQLite path")
    p4.add_argument("--current-a500", type=float, required=True, dest="current_a500", help="当前 A500 ETF (159361) 权重 (0~1)")
    p4.add_argument("--current-dividend", type=float, required=True, dest="current_dividend", help="当前中证红利 ETF (515080) 权重 (0~1)")
    p4.add_argument("--current-dividend-growth", type=float, required=True, dest="current_dividend_growth", help="当前红利质量 ETF (159209) 权重 (0~1)")
    p4.add_argument("--regime-confirm-days", type=int, default=5, dest="regime_confirm_days", help="Regime 滞后确认天数 (default: 5)")
    p4.add_argument("--max-step", type=float, default=0.1, dest="max_step", help="单次调仓最大幅度 (default: 0.1 = 10%%)")
    p4.add_argument("--json", action="store_true", help="Print JSON")
    p4.add_argument("--save-json", type=Path, default=None, help="Save JSON to file")
    p4.set_defaults(_run=cmd_recommend_portfolio_action)
