"""
Parity: research wrappers vs direct ``core`` calls (event study).

Run from repo root::

    python -m core.compat_check

Requires ``etf_data.db`` in the project root (or set ``BIAS_QUANTLAB_DB``).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _deep_diff(a: Any, b: Any, path: str = "") -> list[str]:
    out: list[str] = []
    if type(a) != type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        out.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
        return out
    if isinstance(a, dict):
        ak, bk = set(a), set(b)
        for k in sorted(ak - bk):
            out.append(f"{path}.{k}: missing in b")
        for k in sorted(bk - ak):
            out.append(f"{path}.{k}: missing in a")
        for k in sorted(ak & bk):
            out.extend(_deep_diff(a[k], b[k], f"{path}.{k}"))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: list len {len(a)} vs {len(b)}")
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(_deep_diff(x, y, f"{path}[{i}]"))
        return out
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return out
        if abs(a - b) > 1e-9 * max(1.0, abs(a), abs(b)):
            out.append(f"{path}: {a!r} vs {b!r}")
        return out
    if a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def main() -> int:
    root = _root()
    db = Path(os.environ.get("BIAS_QUANTLAB_DB", str(root / "etf_data.db"))).expanduser()
    if not db.is_file():
        print(f"SKIP: database not found at {db}")
        return 0

    from core.data_loader import load_etf_sqlite
    from core.event_study import run_event_study
    from core.indicators import compute_research_features
    from core.portfolio_backtest import run_research_portfolio_backtest
    from core.signal_engine import assign_research_signal_flags, research_tier_mask

    from research.config import ResearchConfig
    from research.signal_quality import run_event_study as research_run_ev

    code = os.environ.get("BIAS_QUANTLAB_COMPAT_ETF", "515080")
    df, _ = load_etf_sqlite(db, code)
    cfg = ResearchConfig()
    cfg.validate()
    tier = "NEG_LOW"

    ev_r = research_run_ev(df, cfg, tier)
    d = compute_research_features(
        df,
        momentum_window=cfg.momentum_window,
        bias_ma_window=cfg.bias_ma_window,
        volume_ma_window=cfg.volume_ma_window,
        use_precomputed_bias=cfg.use_precomputed_bias,
        recompute_bias=cfg.recompute_bias,
        precomputed_bias_col=cfg.precomputed_bias_col,
    )
    d = assign_research_signal_flags(
        d,
        signal_mode=cfg.signal_mode,
        signal_rolling_window=int(cfg.signal_rolling_window),
        quantile_low=float(cfg.quantile_low),
        quantile_high=float(cfg.quantile_high),
    )
    mask = research_tier_mask(d, tier)
    ev_c = run_event_study(
        d,
        mask,
        tuple(int(x) for x in cfg.event_study_horizons),
        output_format="research",
        tier=tier,
        path_stats_max_days=int(cfg.path_stats_max_days),
    )

    ev_diffs = _deep_diff(ev_r, ev_c)
    print("=== Event study (research vs core.run_event_study) ===")
    if ev_diffs:
        for line in ev_diffs[:50]:
            print(line)
        if len(ev_diffs) > 50:
            print(f"... and {len(ev_diffs) - 50} more")
    else:
        print("OK: no differences")

    pf = run_research_portfolio_backtest(df, cfg, tier)
    print("\n=== Portfolio (unified engine) ===")
    print(f"trades={len(pf.trades)} tier_label={pf.research_tier_label!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
