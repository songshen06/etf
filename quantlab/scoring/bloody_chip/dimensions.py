import pandas as pd
from typing import Dict, Any, Tuple
import numpy as np

def score_drawdown_damage(df: pd.DataFrame, current_row: pd.Series, rules: Dict[str, Any], max_score: float) -> Tuple[float, Dict[str, Any], list[str]]:
    """
    Score how damaged the ETF is based on maximum drawdown over a lookback period.
    """
    lookback = rules.get("lookback_days", 252)
    min_dd = rules.get("min_drawdown", -0.15)
    max_dd = rules.get("max_drawdown", -0.40)
    
    current_dd = 0.0
    if len(df) > 0:
        recent_df = df.tail(lookback)
        high_col = "high" if "high" in recent_df.columns else ("close" if "close" in recent_df.columns else "收盘")
        highest = recent_df[high_col].max()
        current_price = current_row.get("close", current_row.get("收盘", current_row.get(high_col, 0.0)))
        if pd.notna(highest) and highest > 0 and pd.notna(current_price):
            current_dd = (current_price / highest) - 1.0
            
    evidence = {"current_drawdown": float(current_dd), "lookback_days": lookback}
    reasons = []
    
    if current_dd >= min_dd:
        score = 0.0
        reasons.append("DD_TOO_SHALLOW")
    elif current_dd <= max_dd:
        score = max_score
        reasons.append("DD_MAX_DAMAGE")
    else:
        # Interpolate between min_dd and max_dd
        ratio = (current_dd - min_dd) / (max_dd - min_dd)
        score = float(ratio * max_score)
        reasons.append("DD_PARTIAL_DAMAGE")
        
    return score, evidence, reasons

def _interpolate(val: float, base: float, target: float) -> float:
    """Helper for linear interpolation bounded between 0 and 1."""
    if base == target:
        return 0.0 if val < target else 1.0
    ratio = (val - base) / (target - base)
    return max(0.0, min(1.0, ratio))

def score_chip_structure(df: pd.DataFrame, current_row: pd.Series, rules: Dict[str, Any], max_score: float) -> Tuple[float, Dict[str, Any], list[str]]:
    """
    Score the chip structure based on bias rate depth and momentum recovery.
    Uses linear interpolation for smoother scoring.
    """
    bias_min = rules.get("bias_rate_min", 0.0)
    bias_max = rules.get("bias_rate_max", -0.15)
    mom_min = rules.get("momentum_min", -0.05)
    mom_max = rules.get("momentum_max", 0.05)
    
    bias = current_row.get("bias_rate", current_row.get("bias_ma120", 0.0))
    if pd.isna(bias):
        bias = 0.0
        
    momentum = current_row.get("momentum_10", current_row.get("momentum", 0.0))
    if pd.isna(momentum):
        momentum = 0.0
        
    evidence = {"bias_rate": float(bias), "momentum": float(momentum)}
    reasons = []
    
    # 70% weight to bias depth, 30% to momentum recovery
    bias_score = _interpolate(bias, bias_min, bias_max)
    mom_score = _interpolate(momentum, mom_min, mom_max)
    
    score = max_score * (0.7 * bias_score + 0.3 * mom_score)
    
    if bias_score > 0.5:
        reasons.append("CHIP_OVERSOLD")
    else:
        reasons.append("CHIP_NORMAL_BIAS")
        
    if mom_score > 0.5:
        reasons.append("CHIP_MOMENTUM_RECOVERING")
    else:
        reasons.append("CHIP_MOMENTUM_WEAK")
        
    return min(score, max_score), evidence, reasons

def score_reversal_potential(df: pd.DataFrame, current_row: pd.Series, rules: Dict[str, Any], max_score: float) -> Tuple[float, Dict[str, Any], list[str]]:
    """
    Score the reversal potential based on volume surges and recent short-term returns.
    Uses linear interpolation for smoother scoring.
    """
    vol_min = rules.get("volume_ratio_min", 1.0)
    vol_max = rules.get("volume_ratio_max", 2.0)
    ret_min = rules.get("returns_20d_min", -0.05)
    ret_max = rules.get("returns_20d_max", 0.05)
    
    vol_ratio = current_row.get("volume_ratio", current_row.get("volume_ratio_20", 1.0))
    if pd.isna(vol_ratio):
        vol_ratio = 1.0
        
    ret_20d = current_row.get("momentum_20", 0.0)
    if pd.isna(ret_20d):
        ret_20d = 0.0
        
    evidence = {"volume_ratio": float(vol_ratio), "returns_20d": float(ret_20d)}
    reasons = []
    
    # 50% weight to volume, 50% to momentum
    vol_score = _interpolate(vol_ratio, vol_min, vol_max)
    ret_score = _interpolate(ret_20d, ret_min, ret_max)
    
    score = max_score * (0.5 * vol_score + 0.5 * ret_score)
    
    if vol_score > 0.5:
        reasons.append("REV_HIGH_VOLUME")
    else:
        reasons.append("REV_NORMAL_VOLUME")
        
    if ret_score > 0.5:
        reasons.append("REV_STRONG_MOMENTUM")
    else:
        reasons.append("REV_WEAK_MOMENTUM")
        
    return min(score, max_score), evidence, reasons

def score_valuation_compression(df: pd.DataFrame, current_row: pd.Series, rules: Dict[str, Any], max_score: float) -> Tuple[float, Dict[str, Any], list[str]]:
    """
    MVP: Placeholder for valuation compression.
    """
    return 0.0, {"note": "MVP fallback"}, ["VAL_NO_DATA"]

def score_sentiment_extreme(df: pd.DataFrame, current_row: pd.Series, rules: Dict[str, Any], max_score: float) -> Tuple[float, Dict[str, Any], list[str]]:
    """
    MVP: Placeholder for extreme sentiment.
    """
    return 0.0, {"note": "MVP fallback"}, ["SENT_NO_DATA"]
