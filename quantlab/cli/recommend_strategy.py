import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

DIVIDEND = ["159209", "515080"]
CORE_INDEX = ["510300", "510500", "159361"]
SECTOR_ROTATION = ["510150", "159992", "510410", "515880"]
HIGH_BETA = ["513050", "512880", "588000", "159531", "159740"]
THEMATIC = ["562060", "513500", "159501"]
DEFENSIVE = ["511130", "518880"]

MIN_EXPOSURE = {
    "DIVIDEND": 0.5,
    "CORE_INDEX": 0.4,
}


def classify_etf(code: str) -> str:
    c = str(code)
    if c in DIVIDEND:
        return "DIVIDEND"
    if c in CORE_INDEX:
        return "CORE_INDEX"
    if c in SECTOR_ROTATION:
        return "SECTOR_ROTATION"
    if c in HIGH_BETA:
        return "HIGH_BETA"
    if c in THEMATIC:
        return "THEMATIC"
    if c in DEFENSIVE:
        return "DEFENSIVE"
    raise ValueError(f"ETF code {code!r} not recognized by strict classification")


def _as_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not (v == v):
        return None
    return v


def _price_from_row(row: pd.Series) -> float | None:
    for k in ("close_norm", "close", "price", "收盘"):
        if k in row:
            v = _as_float(row.get(k))
            if v is not None:
                return v
    return None


def rolling_bucket_rank(values: pd.Series, *, window: int, n_buckets: int = 5) -> int | None:
    from quantlab.filters.quantile_filter import assign_equal_frequency_quantile_labels, bucket_label_to_rank

    s = pd.to_numeric(values, errors="coerce")
    tail = s.tail(int(window))
    if len(tail) == 0:
        return None
    labels = assign_equal_frequency_quantile_labels(tail, n_buckets=int(n_buckets))
    lab = labels.iloc[-1] if len(labels) > 0 else None
    return bucket_label_to_rank(lab)


def _ensure_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    req = ["category", "strategy", "signal", "action", "position_after", "position_max", "exit_plan", "reason"]
    for k in req:
        if k not in d or d[k] is None or (isinstance(d[k], str) and not d[k].strip()):
            raise ValueError(f"missing or empty field: {k}")
    if not isinstance(d["exit_plan"], list) or not d["exit_plan"]:
        raise ValueError("exit_plan must be a non-empty list")
    return d


def dividend_strategy(*, bias_q: int | None, momentum_q: int | None, current_position: float) -> Dict[str, Any]:
    category = "DIVIDEND"
    strat = "DIVIDEND"
    base_target = 0.70
    add_step1 = 0.85
    add_step2 = 1.00
    pos_max = 1.00

    pos = float(current_position or 0.0)
    pos = 0.0 if pos < 1e-9 else pos

    low_zone = bias_q in (1, 2) and momentum_q in (1, 2)
    deep_low = bias_q == 1 and (momentum_q in (1, 2))
    extreme_rich = bias_q == 5

    if pos <= 1e-12:
        if low_zone:
            return _ensure_keys(
                {
                    "category": category,
                    "strategy": strat,
                    "signal": "BUILD_BASE",
                    "action": f"BUY to {int(base_target*100)}%",
                    "position_after": float(base_target),
                    "position_max": float(pos_max),
                    "exit_plan": [
                        f"deep low (bias_q==Q1 and momentum_q in Q1/Q2) → add to {int(add_step1*100)}% then {int(add_step2*100)}%",
                        f"extreme rich (bias_q==Q5) → reduce to base {int(base_target*100)}% (no full exit)",
                    ],
                    "reason": "cash → low zone detected; establish dividend base (participation-first)",
                }
            )
        return _ensure_keys(
            {
                "category": category,
                "strategy": strat,
                "signal": "HOLD_CASH",
                "action": "HOLD",
                "position_after": 0.0,
                "position_max": float(pos_max),
                "exit_plan": [
                    f"enter only in low zone (bias_q in Q1/Q2 and momentum_q in Q1/Q2) → build base {int(base_target*100)}%",
                ],
                "reason": "cash → not in low zone; do not chase dividend ETFs",
            }
        )

    if extreme_rich and pos > base_target + 1e-12:
        return _ensure_keys(
            {
                "category": category,
                "strategy": strat,
                "signal": "REDUCE_TO_BASE",
                "action": f"SELL to {int(base_target*100)}%",
                "position_after": float(base_target),
                "position_max": float(pos_max),
                "exit_plan": [f"bias_q==Q5 → trim excess back to base {int(base_target*100)}%"],
                "reason": "bias_q==Q5 → trim only excess; no normal exit",
            }
        )

    if pos < base_target - 1e-12:
        if low_zone:
            return _ensure_keys(
                {
                    "category": category,
                    "strategy": strat,
                    "signal": "BUILD_BASE",
                    "action": f"BUY to {int(base_target*100)}%",
                    "position_after": float(base_target),
                    "position_max": float(pos_max),
                    "exit_plan": [
                        f"deep low (bias_q==Q1 and momentum_q in Q1/Q2) → add to {int(add_step1*100)}% then {int(add_step2*100)}%",
                        f"bias_q==Q5 → trim to base {int(base_target*100)}%",
                    ],
                    "reason": "base incomplete; low zone detected; complete base",
                }
            )
        return _ensure_keys(
            {
                "category": category,
                "strategy": strat,
                "signal": "HOLD_BASE",
                "action": "HOLD",
                "position_after": float(pos),
                "position_max": float(pos_max),
                "exit_plan": [f"wait for low zone to complete base to {int(base_target*100)}%"],
                "reason": "base incomplete but not in low zone; hold",
            }
        )

    if deep_low and pos < add_step1 - 1e-12:
        return _ensure_keys(
            {
                "category": category,
                "strategy": strat,
                "signal": "ADD",
                "action": f"BUY to {int(add_step1*100)}%",
                "position_after": float(add_step1),
                "position_max": float(pos_max),
                "exit_plan": [f"deep low persists → add to {int(add_step2*100)}%"],
                "reason": "deep low zone → slow accumulation step",
            }
        )
    if deep_low and pos < add_step2 - 1e-12 and pos >= add_step1 - 1e-12:
        return _ensure_keys(
            {
                "category": category,
                "strategy": strat,
                "signal": "ADD",
                "action": f"BUY to {int(add_step2*100)}%",
                "position_after": float(add_step2),
                "position_max": float(pos_max),
                "exit_plan": [f"bias_q==Q5 → trim to base {int(base_target*100)}%"],
                "reason": "deep low persists → complete accumulation to 100%",
            }
        )

    return _ensure_keys(
        {
            "category": category,
            "strategy": strat,
            "signal": "HOLD_BASE",
            "action": "HOLD",
            "position_after": float(pos),
            "position_max": float(pos_max),
            "exit_plan": [
                f"deep low (bias_q==Q1 and momentum_q in Q1/Q2) → add to {int(add_step1*100)}% then {int(add_step2*100)}%",
                f"extreme rich (bias_q==Q5) → trim to base {int(base_target*100)}% (no full exit)",
            ],
            "reason": "hold base; very low-frequency actions only in deep low or extreme rich",
        }
    )

def dividend_growth_strategy(
    *,
    bias_q: int | None,
    momentum_q: int | None,
    current_position: float,
    base_target: float | None = None,
    overlay_max: float | None = None,
) -> Dict[str, Any]:
    category = "DIVIDEND"
    strat = "DIVIDEND_GROWTH"
    base_t = 0.60 if base_target is None else float(base_target)
    add_unit = 0.20
    ov_max = (1.0 - base_t) if overlay_max is None else float(overlay_max)
    pos_max = float(min(1.0, max(base_t, base_t + ov_max)))
    base_floor = min(base_t, 0.60)

    pos = float(current_position or 0.0)
    pos = 0.0 if pos < 1e-9 else pos

    if pos <= 1e-12:
        return _ensure_keys(
            {
                "category": category,
                "strategy": strat,
                "signal": "BUILD_BASE",
                "action": f"BUY to {int(base_t*100)}%",
                "position_after": float(base_t),
                "position_max": float(pos_max),
                "exit_plan": [
                    f"low zone add: bias_q==Q1 and momentum_q<=Q2 → add {int(add_unit*100)}% (cap {int(pos_max*100)}%)",
                    f"risk reduce: bias_q>=Q4 and momentum_q<=Q2 → reduce base to {int(base_floor*100)}% (no full exit)",
                    f"rich zone trim: bias_q>=Q4 → reduce to base {int(base_t*100)}% (no full exit)",
                ],
                "reason": "cash → build base early for participation (DIVIDEND_GROWTH)",
            }
        )

    if bias_q is not None and bias_q >= 4 and (momentum_q is not None and momentum_q <= 2) and pos > base_floor + 1e-12:
        return _ensure_keys(
            {
                "category": category,
                "strategy": strat,
                "signal": "BASE_REDUCE",
                "action": f"SELL to {int(base_floor*100)}%",
                "position_after": float(base_floor),
                "position_max": float(pos_max),
                "exit_plan": [f"bias_q>=Q4 and momentum_q<=Q2 → reduce base to {int(base_floor*100)}% (no full exit)"],
                "reason": "risk: bias_q>=Q4 and momentum_q<=Q2 → reduce base exposure (keep floor, no full exit)",
            }
        )

    if pos > base_t + 1e-12 and bias_q is not None and bias_q >= 4:
        return _ensure_keys(
            {
                "category": category,
                "strategy": strat,
                "signal": "REDUCE_TO_BASE",
                "action": f"SELL to {int(base_t*100)}%",
                "position_after": float(base_t),
                "position_max": float(pos_max),
                "exit_plan": [f"bias_q>=Q4 → reduce overlay back to base {int(base_t*100)}%"],
                "reason": "bias_q>=Q4 → reduce overlay only (keep base)",
            }
        )

    if pos >= base_floor - 1e-12 and pos < pos_max - 1e-12:
        if bias_q == 1 and (momentum_q is not None and momentum_q <= 2):
            buy_to = min(pos_max, pos + add_unit)
            return _ensure_keys(
                {
                    "category": category,
                    "strategy": strat,
                    "signal": "ADD_OVERLAY",
                    "action": f"BUY to {int(buy_to*100)}%",
                    "position_after": float(buy_to),
                    "position_max": float(pos_max),
                    "exit_plan": [f"bias_q>=Q4 → reduce to base {int(base_t*100)}%"],
                    "reason": "bias_q==Q1 and momentum_q<=Q2 → add overlay in low zone",
                }
            )

    return _ensure_keys(
        {
            "category": category,
            "strategy": strat,
            "signal": "HOLD",
            "action": "HOLD",
            "position_after": float(pos),
            "position_max": float(pos_max),
            "exit_plan": [
                f"base={int(base_t*100)}% always kept once built",
                f"bias_q==Q1 and momentum_q<=Q2 → add overlay (cap {int(pos_max*100)}%)",
                f"bias_q>=Q4 and momentum_q<=Q2 → reduce base to {int(base_floor*100)}% (no full exit)",
                f"bias_q>=Q4 → reduce to base {int(base_t*100)}%",
            ],
            "reason": "hold: no base-build / low-add / rich-trim condition met",
        }
    )
    return _ensure_keys(
        {
            "category": category,
            "strategy": strat,
            "signal": signal,
            "action": action,
            "position_after": float(pos_after),
            "position_max": float(pos_max),
            "exit_plan": [
                "bias_q == 4 → reduce 20% (if position < 70%, reduce 10%)",
                "bias_q == 5 → reduce 30% (if position < 70%, reduce 15%)",
                "no reduce allowed while position < 50%",
            ],
            "reason": reason,
        }
    )


def mean_reversion_strategy(
    *,
    category: str,
    bias_q: int | None,
    momentum_q: int | None,
    current_position: float,
) -> Dict[str, Any]:
    strat = "MEAN_REVERSION"
    signal = "HOLD"
    action = "HOLD"
    pos_after = current_position
    pos_max = 0.8
    reason = f"bias_q={bias_q}, momentum_q={momentum_q}"
    entry_ok = bias_q == 1 and (momentum_q is not None and momentum_q <= 2)

    if current_position <= 1e-12:
        if entry_ok:
            signal = "BUY"
            action = "BUY 60%"
            pos_after = 0.60
            reason = "entry: bias_q==1 and momentum_q<=2"
    elif abs(current_position - 0.60) < 1e-9 and bias_q == 1:
        signal = "BUY"
        action = "BUY 20%"
        pos_after = 0.80
        reason = "add: current=60% and bias_q==1"

    # HOLD gating for overlay: once in position, maintain MR_HOLD rules
    if current_position > 0 and bias_q is not None:
        if bias_q <= 2:
            # strict HOLD in low zone
            return _ensure_keys(
                {
                    "category": category,
                    "strategy": strat,
                    "signal": "HOLD",
                    "action": "HOLD",
                    "position_after": float(current_position),
                    "position_max": float(pos_max),
                    "exit_plan": ["bias_q >= 4 → exit all", "no reduce allowed while bias_q <= 2"],
                    "reason": "MR_HOLD: bias_q<=2 → HOLD",
                }
            )
        if bias_q == 3:
            # single step decay to half, then let future logic handle next states
            return _ensure_keys(
                {
                    "category": category,
                    "strategy": strat,
                    "signal": "REDUCE",
                    "action": "REDUCE 50%",
                    "position_after": float(max(0.0, current_position * 0.5)),
                    "position_max": float(pos_max),
                    "exit_plan": ["bias_q == 3 → reduce 50%", "bias_q >= 4 → exit all"],
                    "reason": "MR_REDUCE: bias_q==3 → reduce 50%",
                }
            )
        if bias_q >= 4:
            return _ensure_keys(
                {
                    "category": category,
                    "strategy": strat,
                    "signal": "REDUCE",
                    "action": "EXIT ALL",
                    "position_after": 0.0,
                    "position_max": float(pos_max),
                    "exit_plan": ["bias_q >= 4 → exit all"],
                    "reason": "MR_EXIT: bias_q>=4 → exit all",
                }
            )

    min_exp = float(MIN_EXPOSURE.get(category, 0.0))
    if bias_q is not None and bias_q <= 2 and current_position > 0:
        signal = "HOLD"
        action = "HOLD"
        pos_after = current_position
        reason = "reduce blocked: bias_q<=2 (must not reduce in low zone)"
    if current_position >= min_exp:
        if bias_q is not None and bias_q >= 4 and current_position > 0:
            signal = "REDUCE"
            action = "EXIT ALL"
            pos_after = 0.0
            reason = "exit: bias_q>=4 (clear high zone)"
    else:
        if bias_q is not None and bias_q >= 4:
            reason = f"exit blocked: current_position<{min_exp:.1f} (mandatory min exposure guardrail)"
    return _ensure_keys(
        {
            "category": category,
            "strategy": strat,
            "signal": signal,
            "action": action,
            "position_after": float(pos_after),
            "position_max": float(pos_max),
            "exit_plan": ["bias_q >= 4 → exit all", "no reduce allowed while bias_q <= 2"],
            "reason": reason,
        }
    )

def trend_following_strategy(
    *,
    category: str,
    bias_q: int | None,
    momentum_q: int | None,
    current_position: float,
) -> Dict[str, Any]:
    strat = "TREND_FOLLOWING"
    signal = "HOLD"
    action = "HOLD"
    pos_after = current_position
    pos_max = 0.8
    reason = f"bias_q={bias_q}, momentum_q={momentum_q}"
    entry_ok = (momentum_q is not None and momentum_q >= 4) and (bias_q is not None and bias_q <= 3)

    if current_position <= 1e-12:
        if entry_ok:
            signal = "BUY"
            action = "BUY 60%"
            pos_after = 0.60
            reason = "entry: momentum_q>=4 and bias_q<=3 (trend-following strength)"
    elif abs(current_position - 0.60) < 1e-9 and momentum_q == 5:
        signal = "BUY"
        action = "BUY 20%"
        pos_after = 0.80
        reason = "add: current=60% and momentum_q==5 (trend continuation)"

    # HOLD gating for overlay: once in position, maintain TF_HOLD rules
    if current_position > 0 and momentum_q is not None:
        if momentum_q >= 3:
            return _ensure_keys(
                {
                    "category": category,
                    "strategy": strat,
                    "signal": "HOLD",
                    "action": "HOLD",
                    "position_after": float(current_position),
                    "position_max": float(pos_max),
                    "exit_plan": ["momentum_q <= 2 → exit all"],
                    "reason": "TF_HOLD: momentum_q>=3 → HOLD",
                }
            )
        if momentum_q <= 2:
            return _ensure_keys(
                {
                    "category": category,
                    "strategy": strat,
                    "signal": "REDUCE",
                    "action": "EXIT ALL",
                    "position_after": 0.0,
                    "position_max": float(pos_max),
                    "exit_plan": ["momentum_q <= 2 → exit all"],
                    "reason": "TF_EXIT: momentum_q<=2 → exit all",
                }
            )
    return _ensure_keys(
        {
            "category": category,
            "strategy": strat,
            "signal": signal,
            "action": action,
            "position_after": float(pos_after),
            "position_max": float(pos_max),
            "exit_plan": ["momentum_q <= 2 → exit all"],
            "reason": reason,
        }
    )
def bloody_chip_strategy(
    *,
    category: str,
    bias_q: int | None,
    momentum_q: int | None,
    current_position: float,
) -> Dict[str, Any]:
    strat = "BLOODY_CHIP"
    signal = "HOLD"
    action = "HOLD"
    pos_after = current_position
    pos_max = 0.3
    reason = f"bias_q={bias_q}, momentum_q={momentum_q}"
    if bias_q == 1 and momentum_q == 1 and current_position < 0.3:
        signal = "BUY"
        action = "BUY 30%"
        pos_after = 0.3
        reason = "entry: NEG_LOW state (bias_q==1, momentum_q==1)"
    if bias_q is not None and bias_q >= 3 and current_position > 0:
        signal = "REDUCE"
        action = "EXIT ALL"
        pos_after = 0.0
        reason = "bias flipped positive / exit condition"
    return _ensure_keys(
        {
            "category": category,
            "strategy": strat,
            "signal": signal,
            "action": action,
            "position_after": float(pos_after),
            "position_max": float(pos_max),
            "exit_plan": ["time_20", "bias_flip_pos", "state_exit"],
            "reason": reason,
        }
    )

def defensive_strategy(
    *,
    category: str,
    current_position: float,
) -> Dict[str, Any]:
    strat = "DEFENSIVE"
    pos_max = float(MIN_EXPOSURE.get(category, 0.0))
    target = pos_max
    if current_position > target + 1e-12:
        return _ensure_keys(
            {
                "category": category,
                "strategy": strat,
                "signal": "REDUCE",
                "action": f"REDUCE to {int(target * 100)}%",
                "position_after": float(target),
                "position_max": float(pos_max),
                "exit_plan": ["DOWN regime → keep low exposure"],
                "reason": "market_state=DOWN",
            }
        )
    return _ensure_keys(
        {
            "category": category,
            "strategy": strat,
            "signal": "HOLD",
            "action": "HOLD",
            "position_after": float(current_position),
            "position_max": float(pos_max),
            "exit_plan": ["DOWN regime → keep low exposure"],
            "reason": "market_state=DOWN",
        }
    )

def light_trading_strategy(*, bias_q: int | None, momentum_q: int | None, current_position: float) -> Dict[str, Any]:
    category = "THEMATIC"
    strat = "LIGHT_TRADING"
    signal = "HOLD"
    action = "HOLD"
    pos_after = current_position
    pos_max = 0.2
    reason = f"bias_q={bias_q}, momentum_q={momentum_q}"
    if bias_q == 1 and momentum_q == 1 and current_position < 0.2:
        signal = "BUY"
        action = "BUY to 20%"
        pos_after = 0.2
        reason = "entry: NEG_LOW light conviction"
    if bias_q is not None and bias_q >= 3 and current_position > 0:
        signal = "REDUCE"
        action = "EXIT ALL"
        pos_after = 0.0
        reason = "fast exit on valuation normalization"
    return _ensure_keys(
        {
            "category": category,
            "strategy": strat,
            "signal": signal,
            "action": action,
            "position_after": float(pos_after),
            "position_max": float(pos_max),
            "exit_plan": ["bias_q >= 3 → exit all (faster than HIGH_BETA)"],
            "reason": reason,
        }
    )

def core_index_staged_base_decision(
    *,
    current_position: float,
    market_state: str | None,
    bias_q: int | None,
    momentum_q: int | None,
    base_target: float,
    overlay_max: float,
) -> dict[str, Any] | None:
    pos = float(current_position)
    pos = 0.0 if pos < 1e-9 else pos
    st = (market_state or "").upper()

    probe_default = 0.30
    probe_target = min(probe_default, base_target)
    full_target = base_target + overlay_max

    if pos <= 1e-12:
        if st == "DOWN":
            low_zone = bias_q in (1, 2)
            rebound_hint = momentum_q is not None and momentum_q >= 3
            if low_zone and rebound_hint:
                return _ensure_keys(
                    {
                        "category": "CORE_INDEX",
                        "strategy": "STAGED_BASE_ENTRY",
                        "signal": "PROBE",
                        "action": f"BUY {int(probe_target*100)}%",
                        "position_after": probe_target,
                        "position_max": full_target,
                        "exit_plan": [
                            f"if market_state transitions to RANGE/TREND → build base to {int(base_target*100)}%",
                            f"if already at {int(base_target*100)}% and market_state==TREND with valid entry → add overlay to {int(full_target*100)}%",
                            f"if market_state==DOWN and overlay exists → reduce to base ({int(base_target*100)}%)",
                        ],
                        "reason": "DOWN regime; low zone (bias_q in Q1/Q2); rebound hint (momentum_q>=3); start with probe instead of full base",
                        "target_label": "probe only",
                        "constraint_notes": "probe is early base-entry (not overlay)",
                        "suggestion": f"Open a {int(probe_target*100)}% probe position only",
                    }
                )
            return _ensure_keys(
                {
                    "category": "CORE_INDEX",
                    "strategy": "STAGED_BASE_ENTRY",
                    "signal": "HOLD_CASH",
                    "action": "HOLD",
                    "position_after": 0.0,
                    "position_max": full_target,
                    "exit_plan": [
                        f"if market_state becomes RANGE/TREND → build base to {int(base_target*100)}%",
                        f"if DOWN + low zone + rebound hint → probe {int(probe_target*100)}%",
                    ],
                    "reason": "DOWN regime; no valid low-zone rebound opportunity; stay in cash",
                    "target_label": "cash",
                    "constraint_notes": "do not build base in DOWN regime from 0%",
                    "suggestion": "Stay in cash and wait for regime stabilization or a probe setup",
                }
            )

        if st in ("RANGE", "TREND"):
            if bias_q is not None and bias_q >= 4:
                return _ensure_keys(
                    {
                        "category": "CORE_INDEX",
                        "strategy": "STAGED_BASE_ENTRY",
                        "signal": "HOLD_CASH",
                        "action": "HOLD",
                        "position_after": 0.0,
                        "position_max": full_target,
                        "exit_plan": ["avoid initiating base at bias_q>=4 from 0%"],
                        "reason": "RANGE/TREND but bias_q is high (>=4); avoid starting base from 0% at high zone",
                        "target_label": "cash",
                        "constraint_notes": "staged base entry prefers not to initiate in high zone",
                        "suggestion": "Stay in cash; wait for bias_q<=3 to build base",
                    }
                )
            return _ensure_keys(
                {
                    "category": "CORE_INDEX",
                    "strategy": "STAGED_BASE_ENTRY",
                    "signal": "BUILD_BASE",
                    "action": f"BUY to {int(base_target*100)}%",
                    "position_after": base_target,
                    "position_max": full_target,
                    "exit_plan": [
                        "after base is built, overlay decisions depend on regime switching",
                        f"if market_state==DOWN after base built → keep base ({int(base_target*100)}%); exit overlay only",
                    ],
                    "reason": f"{st} regime; suitable to establish base exposure from cash",
                    "target_label": "full base",
                    "constraint_notes": "base is long-term core holding; overlay is separate",
                    "suggestion": f"Build the {int(base_target*100)}% base position",
                }
            )

    if pos > 0 and pos < base_target - 1e-9:
        if pos < probe_target - 1e-9:
            return None
        if st in ("RANGE", "TREND"):
            return _ensure_keys(
                {
                    "category": "CORE_INDEX",
                    "strategy": "STAGED_BASE_ENTRY",
                    "signal": "BUILD_BASE",
                    "action": f"BUY to {int(base_target*100)}%",
                    "position_after": base_target,
                    "position_max": full_target,
                    "exit_plan": [
                        "after base is built, overlay decisions depend on regime switching",
                        f"if market_state==DOWN after base built → keep base ({int(base_target*100)}%); exit overlay only",
                    ],
                    "reason": "probe exists and regime stabilized (not DOWN); complete base construction",
                    "target_label": "full base",
                    "constraint_notes": "probe is part of base entry, not overlay",
                    "suggestion": f"Build from probe to full base ({int(base_target*100)}%)",
                }
            )
        return _ensure_keys(
            {
                "category": "CORE_INDEX",
                "strategy": "STAGED_BASE_ENTRY",
                "signal": "HOLD",
                "action": "HOLD",
                "position_after": pos,
                "position_max": full_target,
                "exit_plan": ["wait for regime != DOWN to build base"],
                "reason": "probe exists but regime is DOWN; do not accelerate base build",
                "target_label": "probe only",
                "constraint_notes": "avoid building full base in DOWN regime",
                "suggestion": "Hold probe only",
            }
        )

    if pos >= base_target - 1e-9 and pos < full_target - 1e-9:
        if st == "DOWN":
            return _ensure_keys(
                {
                    "category": "CORE_INDEX",
                    "strategy": "STAGED_BASE_ENTRY",
                    "signal": "REDUCE_TO_BASE",
                    "action": "HOLD",
                    "position_after": base_target,
                    "position_max": full_target,
                    "exit_plan": ["market_state==DOWN → keep base, exit overlay only"],
                    "reason": f"DOWN regime; keep base ({int(base_target*100)}%) and avoid adding overlay",
                    "target_label": "base only",
                    "constraint_notes": "no overlay in DOWN regime",
                    "suggestion": "Keep base; do not add overlay",
                }
            )
        if st == "TREND":
            entry_ok = (momentum_q is not None and momentum_q >= 4) and (bias_q is not None and bias_q <= 3)
            if entry_ok:
                return _ensure_keys(
                    {
                        "category": "CORE_INDEX",
                        "strategy": "STAGED_BASE_ENTRY",
                        "signal": "ADD_OVERLAY",
                        "action": f"BUY to {int(full_target*100)}%",
                        "position_after": full_target,
                        "position_max": full_target,
                        "exit_plan": ["overlay exit is controlled by regime switching (tactical layer)"],
                        "reason": "TREND regime with valid entry; add tactical overlay on top of base",
                        "target_label": "full allocation",
                        "constraint_notes": "overlay is tactical; base remains core holding",
                        "suggestion": f"Add overlay to reach {int(full_target*100)}% total allocation",
                    }
                )
        return _ensure_keys(
            {
                "category": "CORE_INDEX",
                "strategy": "STAGED_BASE_ENTRY",
                "signal": "HOLD",
                "action": "HOLD",
                "position_after": pos,
                "position_max": full_target,
                "exit_plan": ["overlay add only in TREND with valid entry"],
                "reason": "base established; no overlay add condition met",
                "target_label": "base",
                "constraint_notes": "overlay is optional; keep base as core holding",
                "suggestion": "Hold base; wait for TREND entry to add overlay",
            }
        )

    return None

def hold_only_strategy(*, current_position: float) -> Dict[str, Any]:
    category = "DEFENSIVE"
    strat = "HOLD_ONLY"
    return _ensure_keys(
        {
            "category": category,
            "strategy": strat,
            "signal": "HOLD",
            "action": "HOLD",
            "position_after": float(current_position),
            "position_max": 1.0,
            "exit_plan": ["manual rebalance only"],
            "reason": "defensive bucket: no active trading",
        }
    )

def cmd_recommend(ns: argparse.Namespace) -> int:
    from core.data_loader import load_etf_sqlite
    from core.data_loader import etf_name_map
    from core.indicators import add_analyzer_indicators
    from core.paths import resolve_db_path
    from quantlab.cli.dividend_signal import DEFAULT_STATE_PATH, load_state, DividendPositionState
    from quantlab.cli.date_range import build_calc_df, compute_effective_range, filter_df_by_effective_range, get_db_date_bounds, parse_cli_date
    from quantlab.config.etf_metadata import get_inception_date

    code = str(ns.etf)
    category = classify_etf(code)
    strategy_name = {
        "DIVIDEND": "dividend_strategy",
        "CORE_INDEX": "trend_following_strategy",
        "SECTOR_ROTATION": "mean_reversion_strategy",
        "HIGH_BETA": "bloody_chip_strategy",
        "THEMATIC": "light_trading_strategy",
        "DEFENSIVE": "hold_only_strategy",
    }[category]

    db_path = resolve_db_path(ns.db_path)
    name = str(etf_name_map(db_path).get(code, code))
    df, _ = load_etf_sqlite(db_path, code)
    df = df.sort_values("date") if "date" in df.columns else df
    if df.empty:
        print(f"[{code}] no data", file=sys.stderr)
        return 1
    db_start, db_end = get_db_date_bounds(df)
    if db_start is None or db_end is None:
        print(f"[{code}] date bounds unavailable", file=sys.stderr)
        return 1
    eff = compute_effective_range(
        user_start=parse_cli_date(getattr(ns, "start_date", None)),
        user_end=parse_cli_date(getattr(ns, "end_date", None)),
        inception_date=get_inception_date(code),
        db_start=db_start,
        db_end=db_end,
    )
    df_calc, warmup_start = build_calc_df(df, eff=eff, warmup_days=int(getattr(ns, "warmup_days", 0) or 0))
    df = add_analyzer_indicators(df_calc)
    df = df.sort_values("date") if "date" in df.columns else df
    df_eval = filter_df_by_effective_range(df, eff)
    if df_eval.empty:
        print(f"[{code}] no data in effective range", file=sys.stderr)
        return 1
    row = df_eval.iloc[-1]
    price = _price_from_row(row)
    bias = _as_float(row.get("bias_rate"))
    bias_q = rolling_bucket_rank(df["bias_rate"], window=int(ns.rolling_window), n_buckets=5)
    mom_col = "momentum_10" if "momentum_10" in df.columns else "momentum"
    momentum_q = rolling_bucket_rank(df[mom_col], window=int(ns.rolling_window), n_buckets=5)

    current_position = float(getattr(ns, "current_position", 0.0) or 0.0)
    if category == "DIVIDEND":
        state_path = Path(getattr(ns, "state_path", None) or DEFAULT_STATE_PATH)
        st_map = load_state(state_path)
        st = st_map.get(code, DividendPositionState())
        current_position = float(st.position or 0.0)

    market_state = None
    if category == "CORE_INDEX":
        from quantlab.cli.market_regime import rolling_bucket_rank_series, detect_market_state

        tmp = df.copy()
        tmp["momentum_q"] = rolling_bucket_rank_series(tmp[mom_col], window=int(ns.rolling_window), n_buckets=5)
        ms = detect_market_state(tmp[["momentum_q"]], window=20, min_persist_days=10)
        market_state = str(ms.iloc[-1]["stable_state"]) if len(ms) else None

    base_target_arg = getattr(ns, "base_target", None)
    if base_target_arg is not None:
        if not (0.3 <= float(base_target_arg) <= 0.9):
            print("base-target must be between 0.3 and 0.9", file=sys.stderr)
            return 1
    base_target = float(base_target_arg) if base_target_arg is not None else 0.7
    overlay_max = max(0.0, 1.0 - base_target)

    div_growth_base_used = None
    div_growth_overlay_used = None
    if category == "DIVIDEND":
        if code == "159209":
            bt = getattr(ns, "base_target", None)
            base_t = float(bt) if bt is not None else None
            if base_t is not None and (not (0.3 <= base_t <= 0.9)):
                print("base-target must be between 0.3 and 0.9", file=sys.stderr)
                return 1
            div_growth_base_used = 0.60 if base_t is None else float(base_t)
            div_growth_overlay_used = max(0.0, 1.0 - div_growth_base_used)
            strategy_name = "dividend_growth_strategy"
            res = dividend_growth_strategy(
                bias_q=bias_q,
                momentum_q=momentum_q,
                current_position=current_position,
                base_target=div_growth_base_used if base_t is not None else None,
                overlay_max=div_growth_overlay_used if base_t is not None else None,
            )
        else:
            strategy_name = "dividend_strategy"
            res = dividend_strategy(bias_q=bias_q, momentum_q=momentum_q, current_position=current_position)
    elif category == "CORE_INDEX":
        staged = core_index_staged_base_decision(
            current_position=current_position,
            market_state=market_state,
            bias_q=bias_q,
            momentum_q=momentum_q,
            base_target=base_target,
            overlay_max=overlay_max,
        )
        if staged is not None:
            strategy_name = "staged_base_entry"
            res = staged
        else:
            if market_state == "TREND":
                strategy_name = "trend_following_strategy"
                res = trend_following_strategy(
                    category=category,
                    bias_q=bias_q,
                    momentum_q=momentum_q,
                    current_position=current_position,
                )
            elif market_state == "RANGE":
                strategy_name = "mean_reversion_strategy"
                res = mean_reversion_strategy(
                    category=category,
                    bias_q=bias_q,
                    momentum_q=momentum_q,
                    current_position=current_position,
                )
            elif market_state == "DOWN":
                strategy_name = "defensive_strategy"
                res = defensive_strategy(category=category, current_position=current_position)
            else:
                strategy_name = "mean_reversion_strategy"
                res = mean_reversion_strategy(
                    category=category,
                    bias_q=bias_q,
                    momentum_q=momentum_q,
                    current_position=current_position,
                )
    elif category == "SECTOR_ROTATION":
        res = mean_reversion_strategy(category=category, bias_q=bias_q, momentum_q=momentum_q, current_position=current_position)
    elif category == "HIGH_BETA":
        res = bloody_chip_strategy(category=category, bias_q=bias_q, momentum_q=momentum_q, current_position=current_position)
    elif category == "THEMATIC":
        res = light_trading_strategy(bias_q=bias_q, momentum_q=momentum_q, current_position=current_position)
    else:
        res = hold_only_strategy(current_position=current_position)

    if ns.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    def _h(en: str, zh: str) -> str:
        return f"{en} / {zh}"

    print(f"[{code} {name}]")
    print("")
    print(f"{_h('category', '分类')}: {category}")
    print(f"{_h('strategy', '策略')}: {strategy_name.replace('_', ' ').upper()}")
    print("")
    print(_h("market_state", "市场状态") + ":")
    print(f"  {_h('bias', '乖离')}: {float(bias) * 100.0:.1f}%")
    print((f"  {_h('bias_q', '乖离分位')}: Q{bias_q}") if bias_q is not None else f"  {_h('bias_q', '乖离分位')}: NA")
    print((f"  {_h('momentum_q', '动量分位')}: Q{momentum_q}") if momentum_q is not None else f"  {_h('momentum_q', '动量分位')}: NA")
    if category == "CORE_INDEX" and market_state is not None:
        print(f"  {_h('market_state', '市场状态')}: {market_state}")
    if category == "CORE_INDEX":
        print("")
        print(_h("config", "配置") + ":")
        print(f"  {_h('base_target', '底仓目标')}: {int(base_target*100)}%")
        print(f"  {_h('overlay_max', '战术层上限')}: {int(overlay_max*100)}%")
    if getattr(ns, "start_date", None) is not None or getattr(ns, "end_date", None) is not None:
        rs = getattr(ns, "start_date", None) or "数据库最早"
        re = getattr(ns, "end_date", None) or "数据库最新"
        print("")
        print(_h("range", "日期区间") + ":")
        print(f"  {_h('requested', '用户输入')}: {rs} ~ {re}")
        print(f"  {_h('effective', '实际生效')}: {eff.effective_start.isoformat()} ~ {eff.effective_end.isoformat()}")
        w = int(getattr(ns, "warmup_days", 0) or 0)
        if w > 0:
            cs = warmup_start.isoformat() if warmup_start is not None else eff.effective_start.isoformat()
            print(f"  {_h('calc', '计算区间')}: {cs} ~ {eff.effective_end.isoformat()} (warmup_days={w})")
    print("")
    print(_h("decision", "决策") + ":")
    print(f"  {_h('signal', '信号')}: {res['signal']}")
    print(f"  {_h('action', '动作')}: {res['action']}")
    print("")
    print(_h("position_plan", "仓位计划") + ":")
    print(f"  {_h('current', '当前')}: {current_position * 100:.0f}%")
    print(f"  {_h('after_trade', '交易后')}: {res['position_after'] * 100:.0f}%")
    if category == "DIVIDEND":
        if code == "159209":
            bt = 0.60 if div_growth_base_used is None else float(div_growth_base_used)
            om = max(0.0, 1.0 - bt) if div_growth_overlay_used is None else float(div_growth_overlay_used)
            print(f"  {_h('target', '目标')}: base {int(bt*100)}% + overlay {int(om*100)}% (cap {int((bt+om)*100)}%)")
        else:
            print(f"  {_h('target', '目标')}: base 70% (low-zone add to 85%/100%; trim to base only at Q5)")
    elif category == "CORE_INDEX":
        if res.get("target_label"):
            print(f"  {_h('target', '目标')}: {res.get('target_label')}")
        else:
            print(f"  {_h('target', '目标')}: full base {int(base_target*100)}% (optional overlay to {int((base_target+overlay_max)*100)}%)")
    else:
        print(f"  {_h('max', '上限')}: {res['position_max'] * 100:.0f}%")
    print("")
    if category in ("DIVIDEND", "CORE_INDEX"):
        print(_h("constraint", "约束") + ":")
        if category == "DIVIDEND":
            if code == "159209":
                bt = 0.60 if div_growth_base_used is None else float(div_growth_base_used)
                om = max(0.0, 1.0 - bt) if div_growth_overlay_used is None else float(div_growth_overlay_used)
                print("  dividend_growth: mid-high base + light overlay")
                print(f"  base_target: {int(bt*100)}% (overlay_max: {int(om*100)}%)")
                print("  build base early for participation")
                print("  add overlay only when bias_q==Q1 and momentum_q<=Q2 (step 20%)")
                print("  reduce overlay only when bias_q>=Q4 back to base; no normal full exit")
            else:
                print("  dividend ETF core holding: high base, low-frequency")
                print("  build base only in low zone (bias_q in Q1/Q2 and momentum_q in Q1/Q2)")
                print("  add only in deep low (bias_q==Q1 and momentum_q in Q1/Q2)")
                print("  trim only at extreme rich (bias_q==Q5) back to base; no normal full exit")
        if category == "CORE_INDEX":
            if res.get("constraint_notes"):
                print(f"  {res.get('constraint_notes')}")
            print("  base = long-term core holding; overlay = tactical layer on top of base")
            print("  min holding days = 15 (blocks frequent trades unless momentum exit)")
        print("")
    print(_h("exit_plan", "退出计划") + ":")
    for rule in res["exit_plan"]:
        print(f"  - {rule}")
    print("")
    print(_h("reason", "原因") + ":")
    print(f"  {res['reason']}")
    if category == "CORE_INDEX" and res.get("suggestion"):
        print("")
        print(_h("suggestion", "建议") + ":")
        print(f"  {res.get('suggestion')}")

    print("")
    print(_h("summary", "摘要") + ":")
    print(f"  {_h('recommended_action', '建议动作')}: {res['action']}")
    print(f"  {_h('position_after', '交易后仓位')}: {res['position_after'] * 100:.0f}%")
    if category == "CORE_INDEX" and market_state is not None:
        print(f"  {_h('market_state', '市场状态')}: {market_state}")
    if bias_q is not None and momentum_q is not None:
        print(f"  {_h('context', '关键信号')}: bias_q=Q{bias_q}, momentum_q=Q{momentum_q}")
    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "recommend-strategy",
        help="Production-grade ETF strategy decision system (category → strategy routing)",
    )
    p.add_argument("--etf", dest="etf", required=True, help="ETF code")
    p.add_argument("--db", dest="db_path", default=None, help="SQLite path")
    p.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD), inclusive")
    p.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD), inclusive")
    p.add_argument("--warmup-days", type=int, default=0, help="Warmup trading days before effective start for indicator stability")
    p.add_argument("--rolling-window", type=int, default=252, help="Rolling window for quantiles")
    p.add_argument(
        "--base-target",
        type=float,
        default=None,
        help="target base position for CORE_INDEX (0.0~1.0; default 0.7 if not specified)",
    )
    p.add_argument("--current-position", type=float, default=0.0, help="Your current position (0.0..1.0)")
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.add_argument("--state-path", type=str, default=None, help="Dividend state json (optional override)")
    p.set_defaults(_run=cmd_recommend)
