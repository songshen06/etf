"""
Interpretable path rules: contiguous quantile ranges on from_state samples (no trades).

Builds on ``path_quality.build_path_labeled_samples`` and **global** ``*_bucket`` columns.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from quantlab.filters.quantile_filter import MAX_QUANTILE_BUCKET, QuantileRange

from .path_quality import (
    TargetMode,
    build_path_labeled_samples,
    filter_labeled_samples_by_bias_quantile,
    ranks_at_indices_for_rules,
    resolve_feature_series,
)
from .quantile_buckets import path_quality_bucket_column


def contiguous_bucket_ranges(k: int) -> list[tuple[int, int, str]]:
    """
    All contiguous 1-indexed ranges over ``Q1``..``Qk``.

    Examples for k=3: Q1, Q2, Q3, Q1-Q2, Q1-Q3, Q2-Q3.
    """
    if k < 1:
        return []
    out: list[tuple[int, int, str]] = []
    for lo in range(1, k + 1):
        for hi in range(lo, k + 1):
            label = f"Q{lo}" if lo == hi else f"Q{lo}-Q{hi}"
            out.append((lo, hi, label))
    return out


def _mask_for_rank_range(ranks: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """``ranks`` are 1..5 or -1 (NA); ``lo``/``hi`` are 1-indexed inclusive Q ranks."""
    return (ranks >= lo) & (ranks <= hi) & (ranks >= 1)


def _rule_stats(
    m: np.ndarray,
    hit_e: np.ndarray,
    fwd_e: np.ndarray,
    *,
    baseline_hit_rate: float,
    baseline_mean_fwd: float,
) -> dict[str, Any] | None:
    cnt = int(m.sum())
    if cnt == 0:
        return None
    h = hit_e[m]
    f = fwd_e[m]
    hc = int(h.sum())
    hr = float(hc / cnt)
    mf = float(np.mean(f))
    wr = float(np.mean(f > 0))
    return {
        "count": cnt,
        "hit_count": hc,
        "hit_rate": hr,
        "hit_rate_lift": float(hr - baseline_hit_rate),
        "mean_forward_return": mf,
        "return_lift": float(mf - baseline_mean_fwd),
        "win_rate_forward": wr,
    }


def compute_path_rule_mining(
    df: Any,
    *,
    from_state: str,
    target_state: str,
    horizon: int,
    target_mode: TargetMode,
    feature_names: tuple[str, ...],
    bucket_n: int,
    max_combinations: int,
    min_count: int,
    top_k: int,
    rules_above_baseline_only: bool,
    bias_ma: int,
    momentum_window: int,
    volume_ma_window: int,
    bias_quantile_range: QuantileRange | None = None,
) -> dict[str, Any]:
    """
    Mine single- and two-factor contiguous quantile rules on labeled from_state samples.

    Uses global ``*_bucket`` columns (``bucket_n`` is ignored; API stability only).
    """
    _ = bucket_n
    d, em_idx, hit_e, fwd_e = build_path_labeled_samples(
        df,
        from_state=from_state,
        target_state=target_state,
        horizon=horizon,
        target_mode=target_mode,
    )
    em_idx, hit_e, fwd_e = filter_labeled_samples_by_bias_quantile(
        d,
        em_idx,
        hit_e,
        fwd_e,
        bias_ma=bias_ma,
        qrange=bias_quantile_range,
    )
    n_s = int(len(hit_e))
    if n_s == 0:
        return {
            "baseline": {
                "count": 0,
                "hit_count": 0,
                "hit_rate": 0.0,
                "mean_forward_return": 0.0,
            },
            "features": list(feature_names),
            "rules": [],
        }

    baseline_hit_count = int(hit_e.sum())
    baseline_hit_rate = float(baseline_hit_count / n_s)
    baseline_mean_fwd = float(np.mean(fwd_e))

    baseline = {
        "count": n_s,
        "hit_count": baseline_hit_count,
        "hit_rate": baseline_hit_rate,
        "mean_forward_return": baseline_mean_fwd,
    }

    k_rules = int(MAX_QUANTILE_BUCKET)
    feat_ranks: list[tuple[str, np.ndarray]] = []
    for raw in feature_names:
        _series, flab = resolve_feature_series(
            d,
            raw,
            bias_ma=bias_ma,
            momentum_window=momentum_window,
            volume_ma_window=volume_ma_window,
        )
        bcol = path_quality_bucket_column(
            raw,
            bias_ma=bias_ma,
            momentum_window=momentum_window,
            volume_ma_window=volume_ma_window,
        )
        if bcol is None:
            continue
        if bcol not in d.columns:
            raise KeyError(
                f"missing column {bcol!r}; run prepare_research_frame before path-rule mining"
            )
        ranks = ranks_at_indices_for_rules(d, bcol, em_idx)
        if int(np.max(ranks)) < 1:
            continue
        feat_ranks.append((flab, ranks))

    rules_out: list[dict[str, Any]] = []
    mc = int(max_combinations)
    if mc < 1:
        mc = 1
    mc = min(mc, 2)

    # Single-factor rules
    if mc >= 1:
        for flab, ranks in feat_ranks:
            for lo, hi, blab in contiguous_bucket_ranges(k_rules):
                m = _mask_for_rank_range(ranks, lo, hi)
                st = _rule_stats(
                    m,
                    hit_e,
                    fwd_e,
                    baseline_hit_rate=baseline_hit_rate,
                    baseline_mean_fwd=baseline_mean_fwd,
                )
                if st is None or st["count"] < int(min_count):
                    continue
                if rules_above_baseline_only and st["hit_rate"] < baseline_hit_rate:
                    continue
                conds = [{"feature": flab, "bucket_range": blab}]
                rule_str = f"{flab} in {blab}"
                rules_out.append(
                    {
                        "rule": rule_str,
                        "feature_conditions": conds,
                        **st,
                    }
                )

    # Two-factor rules (distinct features, CLI order preserved via feat_ranks order)
    if mc >= 2 and len(feat_ranks) >= 2:
        for (fa, ra), (fb, rb) in combinations(feat_ranks, 2):
            for lo_a, hi_a, lab_a in contiguous_bucket_ranges(k_rules):
                m_a = _mask_for_rank_range(ra, lo_a, hi_a)
                for lo_b, hi_b, lab_b in contiguous_bucket_ranges(k_rules):
                    m_b = _mask_for_rank_range(rb, lo_b, hi_b)
                    m = m_a & m_b
                    st = _rule_stats(
                        m,
                        hit_e,
                        fwd_e,
                        baseline_hit_rate=baseline_hit_rate,
                        baseline_mean_fwd=baseline_mean_fwd,
                    )
                    if st is None or st["count"] < int(min_count):
                        continue
                    if rules_above_baseline_only and st["hit_rate"] < baseline_hit_rate:
                        continue
                    conds = [
                        {"feature": fa, "bucket_range": lab_a},
                        {"feature": fb, "bucket_range": lab_b},
                    ]
                    rule_str = f"{fa} in {lab_a} AND {fb} in {lab_b}"
                    rules_out.append(
                        {
                            "rule": rule_str,
                            "feature_conditions": conds,
                            **st,
                        }
                    )

    rules_out.sort(
        key=lambda r: (-r["hit_rate"], -r["count"], -r["mean_forward_return"]),
    )
    tk = max(0, int(top_k))
    rules_out = rules_out[:tk] if tk else rules_out

    return {
        "baseline": baseline,
        "features": list(feature_names),
        "rules": rules_out,
    }
