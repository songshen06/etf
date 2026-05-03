"""
Compare raw entry-regime persistence vs per-exit-rule holding stats (diagnostics only).
"""

from __future__ import annotations

from typing import Any

from .exit_rules import ExitRuleEvalRow
from .schemas import EntrySignalDiagnosticsBlock


def classify_entry_exit_alignment(
    *,
    holding_vs_entry_avg_ratio: float | None,
    trades_per_entry_regime: float | None,
) -> tuple[str, str]:
    """
    Rule-based label + short note.

    Uses avg holding vs avg regime duration; ``trades_per_entry_regime`` for fragmentation.
    """
    if holding_vs_entry_avg_ratio is None or trades_per_entry_regime is None:
        return "mixed", "Insufficient data for alignment classification."
    ar = float(holding_vs_entry_avg_ratio)
    tpr = float(trades_per_entry_regime)
    if ar >= 3.0:
        return (
            "over_hold",
            "Holding period is much longer than entry regime duration.",
        )
    if ar < 0.7:
        return (
            "fast_exit",
            "Average holding is shorter than typical entry regime length.",
        )
    if 0.7 <= ar <= 1.5:
        if tpr <= 1.5:
            return (
                "well_aligned",
                "Holding duration is in line with entry regimes; trades per regime is low.",
            )
        return (
            "aligned_but_fragmented",
            "Holding duration matches entry pulse, but trades per regime is high.",
        )
    return "mixed", "Holding vs entry relationship does not fit a single simple pattern."


def _fmetric(m: dict[str, Any], key: str) -> float | None:
    v = m.get(key)
    if v is None:
        return None
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def build_entry_exit_matching_diagnostics(
    entry_diag: EntrySignalDiagnosticsBlock,
    exit_eval_rows: list[ExitRuleEvalRow],
) -> dict[str, Any]:
    """
    Plain dict for :class:`~core.schemas.EntryExitMatchingDiagnosticsBlock`.

    Requires non-empty ``exit_eval_rows`` and entry persistence (caller ensures).
    """
    ps = entry_diag.entry_persistence_summary
    regime_count = int(ps.regime_count)
    avg_reg = ps.avg_duration_days
    med_reg = ps.median_duration_days
    max_reg = int(ps.max_duration_days)

    entry_summary = {
        "raw_entry_days_count": int(entry_diag.raw_entry_days_count),
        "regime_count": regime_count,
        "avg_regime_duration_days": avg_reg,
        "median_regime_duration_days": med_reg,
        "max_regime_duration_days": max_reg,
    }

    per_exit: list[dict[str, Any]] = []
    for er in exit_eval_rows:
        m = er.metrics or {}
        rule_id = str(er.spec.rule_id)
        tc = int(er.n_trades)
        avg_hold = _fmetric(m, "avg_holding_days")
        med_hold = _fmetric(m, "median_holding_days")

        ratio_avg: float | None = None
        if avg_hold is not None and avg_reg is not None and avg_reg > 1e-12:
            ratio_avg = float(avg_hold) / float(avg_reg)

        ratio_med: float | None = None
        if med_hold is not None and med_reg is not None and med_reg > 1e-12:
            ratio_med = float(med_hold) / float(med_reg)

        tpr: float | None = None
        if regime_count > 0:
            tpr = float(tc) / float(regime_count)

        label, notes = classify_entry_exit_alignment(
            holding_vs_entry_avg_ratio=ratio_avg,
            trades_per_entry_regime=tpr,
        )

        per_exit.append(
            {
                "rule_id": rule_id,
                "display_name": str(er.display_name).strip(),
                "eligible": bool(er.eligible),
                "trade_count": tc,
                "avg_holding_days": avg_hold,
                "median_holding_days": med_hold,
                "holding_vs_entry_avg_ratio": ratio_avg,
                "holding_vs_entry_median_ratio": ratio_med,
                "trades_per_entry_regime": tpr,
                "alignment_label": label,
                "notes": notes,
            }
        )

    return {
        "entry_summary": entry_summary,
        "per_exit": per_exit,
    }
