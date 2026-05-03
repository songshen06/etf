import pandas as pd
from typing import Dict, Any, Callable
from .models import BloodyChipConfig, BloodyChipScoreResult, DimensionScore
from .dimensions import (
    score_drawdown_damage,
    score_chip_structure,
    score_reversal_potential,
    score_valuation_compression,
    score_sentiment_extreme,
)
from .evidence_builder import build_summary

DIMENSION_FUNCTIONS: Dict[str, Callable] = {
    "drawdown_damage": score_drawdown_damage,
    "chip_structure": score_chip_structure,
    "reversal_potential": score_reversal_potential,
    "valuation_compression": score_valuation_compression,
    "sentiment_extreme": score_sentiment_extreme,
}

def score_etf(etf_code: str, df: pd.DataFrame, config: BloodyChipConfig) -> BloodyChipScoreResult:
    """
    Score an ETF for 'bloody chip' state using the provided DataFrame and config.
    """
    if df.empty:
        raise ValueError(f"Empty DataFrame provided for {etf_code}")
        
    # Assume df is sorted by date, last row is the current snapshot
    current_row = df.iloc[-1]
    
    # get date
    if "date" in df.columns:
        snapshot_date = str(current_row["date"])
    elif df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
        snapshot_date = str(current_row.name)
    else:
        snapshot_date = "latest"
        
    total_score_raw = 0.0
    dimension_scores = {}
    sum_weights = 0.0

    dims = ["drawdown_damage", "chip_structure", "reversal_potential"]
    for dim_name in dims:
        dim_config = config.dimensions.get(dim_name)
        if dim_config is None:
            continue
        if dim_name not in DIMENSION_FUNCTIONS:
            continue
            
        # check if enabled (default true)
        if not dim_config.rules.get("enabled", True):
            dim_score = DimensionScore(score=0.0, evidence={"note": "disabled"}, reason_codes=[f"{dim_name.upper()}_DISABLED"])
        else:
            func = DIMENSION_FUNCTIONS[dim_name]
            score_val, evidence, reasons = func(df, current_row, dim_config.rules, dim_config.max_score)
            dim_score = DimensionScore(score=score_val, evidence=evidence, reason_codes=reasons)
            
            # Add weighted score
            total_score_raw += score_val * dim_config.weight
            sum_weights += dim_config.weight if dim_config.weight > 0 else 0.0
            
        dimension_scores[dim_name] = dim_score
        
    # Normalize total score to explicit 0-10 scale regardless of weight sum
    normalized_total = total_score_raw / (sum_weights if sum_weights > 0 else 1.0)

    dd_ds = dimension_scores.get("drawdown_damage", DimensionScore(score=0.0, evidence={}, reason_codes=[]))
    chip_ds = dimension_scores.get("chip_structure", DimensionScore(score=0.0, evidence={}, reason_codes=[]))
    rev_ds = dimension_scores.get("reversal_potential", DimensionScore(score=0.0, evidence={}, reason_codes=[]))
    dd_score = float(dd_ds.score or 0.0)
    chip_score = float(chip_ds.score or 0.0)
    rev_score = float(rev_ds.score or 0.0)

    chip_tags = chip_ds.reason_codes or []
    rev_tags = rev_ds.reason_codes or []
    has_high_volume = "REV_HIGH_VOLUME" in rev_tags
    has_recovery = "CHIP_MOMENTUM_RECOVERING" in chip_tags
    early_reversal_signal = has_high_volume or has_recovery or rev_score >= 5.0

    if dd_score < 2.0:
        category = "EARLY_REVERSAL" if early_reversal_signal else "NOT_CANDIDATE"
    elif dd_score < 3.0:
        if normalized_total >= 6.0 and (chip_score >= 5.0 or rev_score >= 5.0):
            category = "WEAK_BLOODY_CHIP"
        elif early_reversal_signal:
            category = "EARLY_REVERSAL"
        else:
            category = "NOT_CANDIDATE"
    else:
        if normalized_total >= 6.5:
            category = "STANDARD_BLOODY_CHIP"
        elif normalized_total >= 5.0:
            category = "WEAK_BLOODY_CHIP"
        elif early_reversal_signal:
            category = "EARLY_REVERSAL"
        else:
            category = "NOT_CANDIDATE"

    summary = build_summary(normalized_total, category, dimension_scores)
    
    return BloodyChipScoreResult(
        total_score=round(normalized_total, 4),
        scale_min=0.0,
        scale_max=10.0,
        snapshot_date=snapshot_date,
        etf_code=etf_code,
        dimension_scores=dimension_scores,
        summary=summary,
        category=category
    )
