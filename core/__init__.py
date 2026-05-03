"""Reusable ETF signal research and backtest engine."""

from .data_loader import etf_name_map, list_etf_codes, load_etf_sqlite
from .data_validation import DataValidationResult, collect_invalid_rows, validate_ohlcv_panel
from .event_study import add_forward_returns, run_event_study, tier_event_studies
from .indicators import add_indicators
from .paths import project_root, resolve_db_path
from .portfolio_backtest import (
    BacktestResult,
    Trade,
    drawdown_series,
    research_portfolio_result_to_dict,
    run_portfolio_backtest,
    run_research_portfolio_backtest,
)
from .pipeline import prepare_indicator_panel, prepare_research_frame
from .position_rules import profile_label_zh, weights_by_strategy_profile, weights_by_tier
from .recommendation import RecommendationResult, SignalSetupCandidate, recommend_strategy_setup
from .runner import (
    run_backtest,
    run_health,
    run_recommendation,
    run_report,
    run_signal_research,
    run_state_ranking,
)
from .signal_engine import apply_signals

__all__ = [
    "add_indicators",
    "apply_signals",
    "BacktestResult",
    "RecommendationResult",
    "SignalSetupCandidate",
    "collect_invalid_rows",
    "DataValidationResult",
    "drawdown_series",
    "etf_name_map",
    "add_forward_returns",
    "run_event_study",
    "list_etf_codes",
    "load_etf_sqlite",
    "prepare_indicator_panel",
    "prepare_research_frame",
    "project_root",
    "resolve_db_path",
    "run_backtest",
    "run_health",
    "research_portfolio_result_to_dict",
    "run_portfolio_backtest",
    "run_research_portfolio_backtest",
    "Trade",
    "recommend_strategy_setup",
    "run_recommendation",
    "run_report",
    "run_signal_research",
    "run_state_ranking",
    "tier_event_studies",
    "validate_ohlcv_panel",
    "weights_by_tier",
    "weights_by_strategy_profile",
    "profile_label_zh",
]
