"""
Multi-objective exit decision layer on top of :func:`~core.recommendation.rank_exit_rules_on_frame`.

Does not replace ``score_exit_metrics`` or legacy ``optimize-exit`` selection.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .exit_rules import ExitRuleEvalRow

ObjectiveName = Literal["return_first", "risk_first", "efficiency_first", "robustness_first"]

OBJECTIVE_KEYS: tuple[str, ...] = (
    "return_first",
    "risk_first",
    "efficiency_first",
    "robustness_first",
)

DEFAULT_OBJECTIVE: ObjectiveName = "risk_first"

_EXP_EPS = 1e-9

# Explicit ids; time_* and hold_fixed handled by prefix / special case
_STYLE_TAG_BY_ID: dict[str, str] = {
    "state_exit_not_top5": "state_decay_exit",
    "state_exit_bottom5": "defensive_exit",
    "bias_flip_pos": "mean_reversion_capture",
    "momentum_flip_pos": "momentum_reversal_exit",
}


def style_tag_for_rule_id(rule_id: str) -> str:
    rid = str(rule_id).strip()
    if rid in _STYLE_TAG_BY_ID:
        return _STYLE_TAG_BY_ID[rid]
    if rid.startswith("time_") or rid == "hold_fixed":
        return "time_trend_hold"
    return "other_exit"


def _optional_json_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def metrics_payload_for_candidate(r: ExitRuleEvalRow) -> dict[str, Any]:
    """
    Raw exit-evaluation fields for multi-objective JSON (from ``r.metrics`` / ``r.n_trades`` only).

    All keys are always present; unknown or non-finite numerics → null.
    """
    m = r.metrics or {}
    tc: int | None
    try:
        tc = int(r.n_trades)
    except (TypeError, ValueError):
        tc = None
    return {
        "total_return": _optional_json_float(m.get("total_return")),
        "annualized_return": _optional_json_float(m.get("annualized_return")),
        "max_drawdown": _optional_json_float(m.get("max_drawdown")),
        "sharpe": _optional_json_float(m.get("sharpe_ratio")),
        "calmar": _optional_json_float(m.get("calmar_ratio")),
        "trade_count": tc,
        "average_exposure": _optional_json_float(m.get("average_exposure")),
        "avg_trade_return": _optional_json_float(m.get("avg_trade_return")),
        "avg_holding_days": _optional_json_float(m.get("avg_holding_days")),
        "win_rate": _optional_json_float(m.get("win_rate")),
    }


def _f(m: dict[str, Any], key: str) -> float:
    v = m.get(key)
    if v is None:
        return float("nan")
    try:
        x = float(v)
        return x if x == x else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _min_max_norm(arr: np.ndarray) -> np.ndarray:
    """Map finite values to [0,1]; non-finite → 0.5; constant → 0.5."""
    out = np.full_like(arr, 0.5, dtype=float)
    fin = np.isfinite(arr)
    if not np.any(fin):
        return out
    lo = float(np.nanmin(arr[fin]))
    hi = float(np.nanmax(arr[fin]))
    if hi - lo < 1e-12:
        out[fin] = 0.5
    else:
        out[fin] = np.clip((arr[fin] - lo) / (hi - lo), 0.0, 1.0)
    return out


def compute_objective_scores(rows: list[ExitRuleEvalRow]) -> dict[str, dict[str, float]]:
    """
    Per-rule scores for each objective (single scalar per objective, not merged across objectives).

    Normalization is min–max across **all** candidates (eligible + ineligible) with finite values.
    """
    n = len(rows)
    if n == 0:
        return {}

    tr = np.array([_f(r.metrics, "total_return") for r in rows], dtype=float)
    ann = np.array([_f(r.metrics, "annualized_return") for r in rows], dtype=float)
    mdd = np.array([_f(r.metrics, "max_drawdown") for r in rows], dtype=float)
    sh = np.array([_f(r.metrics, "sharpe_ratio") for r in rows], dtype=float)
    cal = np.array([_f(r.metrics, "calmar_ratio") for r in rows], dtype=float)
    avg_exp = np.array([_f(r.metrics, "average_exposure") for r in rows], dtype=float)
    atr = np.array([_f(r.metrics, "avg_trade_return") for r in rows], dtype=float)
    ahd = np.array([_f(r.metrics, "avg_holding_days") for r in rows], dtype=float)
    ntr = np.array([float(r.n_trades) for r in rows], dtype=float)

    ntr_safe = np.maximum(ntr, 1.0)
    abs_mdd = np.abs(np.where(np.isfinite(mdd), mdd, np.nan))

    ntr_n = _min_max_norm(ntr)
    tr_n = _min_max_norm(tr)
    ann_n = _min_max_norm(ann)
    atr_n = _min_max_norm(atr)
    abs_mdd_n = _min_max_norm(abs_mdd)
    sh_n = _min_max_norm(sh)
    cal_n = _min_max_norm(cal)
    mdd_higher_better_n = _min_max_norm(mdd)

    eff_ann_exp = ann / np.maximum(avg_exp, _EXP_EPS)
    eff_trade_hold = atr / np.maximum(ahd, 1.0)
    eff1_n = _min_max_norm(eff_ann_exp)
    eff2_n = _min_max_norm(eff_trade_hold)
    eff_combo = (eff1_n + eff2_n) / 2.0

    winr = np.array([_f(r.metrics, "win_rate") for r in rows], dtype=float)
    winr_n = _min_max_norm(winr)

    return_first = (
        (tr_n + ann_n + atr_n) / 3.0 - 0.12 * abs_mdd_n - 0.08 * (1.0 - ntr_n)
    )
    risk_first = (sh_n + cal_n + mdd_higher_better_n) / 3.0 - 0.1 * (1.0 - ntr_n)
    efficiency_first = eff_combo
    robustness_first = 0.4 * ntr_n + 0.35 * sh_n + 0.25 * mdd_higher_better_n + 0.05 * winr_n

    out: dict[str, dict[str, float]] = {}
    for i, r in enumerate(rows):
        rid = str(r.spec.rule_id)
        out[rid] = {
            "return_first": float(return_first[i]),
            "risk_first": float(risk_first[i]),
            "efficiency_first": float(efficiency_first[i]),
            "robustness_first": float(robustness_first[i]),
        }
    return out


def compute_pareto_front(
    rows: list[ExitRuleEvalRow],
    *,
    eps: float = 1e-12,
) -> frozenset[str]:
    """
    Non-dominated set among eligible rules only.

    Dimensions (all maximized): annualized_return, sharpe_ratio, max_drawdown (less negative is better),
    efficiency = annualized_return / max(average_exposure, eps).
    """
    eligible = [r for r in rows if r.eligible]
    if not eligible:
        return frozenset()

    pts: list[tuple[str, np.ndarray]] = []
    for r in eligible:
        m = r.metrics
        ann = _f(m, "annualized_return")
        shv = _f(m, "sharpe_ratio")
        md = _f(m, "max_drawdown")
        ae = _f(m, "average_exposure")
        eff = ann / max(ae, eps) if np.isfinite(ann) and np.isfinite(ae) else float("nan")
        vec = np.array(
            [
                ann if np.isfinite(ann) else -1e18,
                shv if np.isfinite(shv) else -1e18,
                md if np.isfinite(md) else -1e18,
                eff if np.isfinite(eff) else -1e18,
            ],
            dtype=float,
        )
        pts.append((str(r.spec.rule_id), vec))

    pareto: set[str] = set()
    for i, (rid_a, a) in enumerate(pts):
        dominated = False
        for j, (_, b) in enumerate(pts):
            if i == j:
                continue
            if np.all(b >= a - eps) and np.any(b > a + eps):
                dominated = True
                break
        if not dominated:
            pareto.add(rid_a)
    return frozenset(pareto)


def _winner_for_objective(
    scores_by_rule: dict[str, dict[str, float]],
    objective: str,
    eligible_ids: set[str],
) -> str | None:
    cands: list[tuple[str, float]] = []
    for rid, sc in scores_by_rule.items():
        if rid not in eligible_ids:
            continue
        v = float(sc.get(objective, float("nan")))
        if not np.isfinite(v):
            continue
        cands.append((rid, v))
    if not cands:
        return None
    cands.sort(key=lambda x: (-x[1], x[0]))
    return cands[0][0]


def _interpretation_block(
    *,
    pareto_set: list[str],
    objective_winners: dict[str, str],
    default_recommendation: str,
    default_objective_label: str,
    rows_by_id: dict[str, ExitRuleEvalRow],
) -> dict[str, str]:
    tags = [style_tag_for_rule_id(r) for r in pareto_set]
    from collections import Counter

    c = Counter(tags)
    top_tag, top_n = c.most_common(1)[0] if c else ("—", 0)
    style_bias = (
        f"Pareto 风格以「{top_tag}」为主（{top_n}/{len(pareto_set)}）。" if pareto_set else "无 eligible Pareto 成员。"
    )
    summ_parts = [
        "多目标层不替代 score_exit_metrics；legacy optimize-exit 仍按原单一分数选规则。",
        f"Pareto 非支配集（eligible）: {', '.join(pareto_set) if pareto_set else '（空）'}。",
    ]
    ow = ", ".join(f"{k}→{v}" for k, v in sorted(objective_winners.items()))
    summ_parts.append(f"各视角最优: {ow}。" if ow else "各视角最优: （无 eligible 候选）。")
    summ_parts.append(
        f"默认推荐（{default_objective_label}）: {default_recommendation or '—'}。"
    )
    summary = " ".join(summ_parts)

    conf: Literal["low", "medium", "high"] = "medium"
    if default_recommendation and default_recommendation in rows_by_id:
        nt = rows_by_id[default_recommendation].n_trades
        if nt >= 40 and len(pareto_set) >= 2:
            conf = "high"
        elif nt < 12 or len(pareto_set) <= 1:
            conf = "low"
    elif not pareto_set:
        conf = "low"

    return {
        "summary": summary,
        "style_bias": style_bias,
        "confidence": conf,
    }


def build_multi_objective_decision(
    rows: list[ExitRuleEvalRow],
    *,
    objective_override: ObjectiveName | None = None,
) -> dict[str, Any]:
    """
    Build the ``multi_objective_decision`` JSON subtree (plain dict for schema validation).

    ``default_recommendation`` uses ``objective_override`` if set, else ``risk_first`` winner.
    """
    scores = compute_objective_scores(rows)
    pareto = compute_pareto_front(rows)
    eligible_ids = {str(r.spec.rule_id) for r in rows if r.eligible}

    objective_winners: dict[str, str] = {}
    for obj in OBJECTIVE_KEYS:
        w = _winner_for_objective(scores, obj, eligible_ids)
        if w is not None:
            objective_winners[obj] = w

    use_obj: ObjectiveName = objective_override or DEFAULT_OBJECTIVE
    default_recommendation = objective_winners.get(use_obj) or objective_winners.get(
        DEFAULT_OBJECTIVE, ""
    )

    rows_by_id = {str(r.spec.rule_id): r for r in rows}
    pareto_list = sorted(pareto)

    candidates: list[dict[str, Any]] = []
    for r in rows:
        rid = str(r.spec.rule_id)
        candidates.append(
            {
                "rule_id": rid,
                "display_name": str(r.display_name).strip(),
                "rank": int(r.rank),
                "eligible": bool(r.eligible),
                "style_tag": style_tag_for_rule_id(rid),
                "scores": dict(scores.get(rid, {})),
                "pareto_member": bool(r.eligible and rid in pareto),
                "metrics": metrics_payload_for_candidate(r),
            }
        )

    interp = _interpretation_block(
        pareto_set=pareto_list,
        objective_winners=objective_winners,
        default_recommendation=default_recommendation or "",
        default_objective_label=str(use_obj),
        rows_by_id=rows_by_id,
    )

    return {
        "pareto_set": pareto_list,
        "objective_winners": objective_winners,
        "default_objective": use_obj,
        "default_recommendation": default_recommendation or "",
        "interpretation": interp,
        "candidates": candidates,
    }
