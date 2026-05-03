"""Strategy position profiles: tier weights (1=NEG, 2=NEG+LOW, 3=NEG+LOW+HIGH)."""

from __future__ import annotations

from typing import Literal

PositionPreset = Literal["aggressive", "balanced", "defensive", "full"]

# User-facing profiles (UI/CLI). Legacy names layered/conservative map to balanced/defensive.
PROFILE_WEIGHTS: dict[str, dict[int, float]] = {
    "aggressive": {1: 0.30, 2: 0.60, 3: 1.00},
    "balanced": {1: 0.15, 2: 0.35, 3: 0.60},
    "defensive": {1: 0.10, 2: 0.25, 3: 0.40},
    "full": {1: 1.0, 2: 1.0, 3: 1.0},
}

PROFILE_LABEL_ZH: dict[str, str] = {
    "aggressive": "激进",
    "balanced": "均衡",
    "defensive": "防御",
    "full": "满仓各层",
}

LEGACY_TO_PROFILE: dict[str, str] = {
    "layered": "balanced",
    "conservative": "defensive",
}


def normalize_strategy_profile(name: str) -> str:
    s = str(name).strip().lower()
    return LEGACY_TO_PROFILE.get(s, s)


def weights_by_strategy_profile(profile: str) -> dict[int, float]:
    key = normalize_strategy_profile(profile)
    if key not in PROFILE_WEIGHTS:
        raise ValueError(f"Unknown strategy profile {profile!r}; choose from {list(PROFILE_WEIGHTS)}")
    return dict(PROFILE_WEIGHTS[key])


def weights_by_tier(preset: str) -> dict[int, float]:
    """Backward-compatible alias: same as :func:`weights_by_strategy_profile`."""
    return weights_by_strategy_profile(preset)


def profile_label_zh(profile: str) -> str:
    key = normalize_strategy_profile(profile)
    return PROFILE_LABEL_ZH.get(key, key)


def tier_weight_labels(wmap: dict[int, float]) -> dict[str, float]:
    """String keys for JSON: NEG / NEG_LOW / NEG_LOW_HIGH."""
    labels = {1: "NEG", 2: "NEG_LOW", 3: "NEG_LOW_HIGH"}
    return {labels.get(k, str(k)): float(v) for k, v in sorted(wmap.items())}


def state_weights_readable_line(
    weights_by_tier: dict[str, float],
    *,
    markdown_bold_pct: bool = False,
) -> str:
    """Human-readable NEG / NEG_LOW / NEG_LOW_HIGH → percent (same keys as tier_weight_labels)."""
    order = ("NEG", "NEG_LOW", "NEG_LOW_HIGH")
    parts: list[str] = []
    for k in order:
        if k not in weights_by_tier:
            continue
        try:
            pct = float(weights_by_tier[k]) * 100
            pct_s = f"{pct:.0f}%"
            if markdown_bold_pct:
                parts.append(f"{k} **{pct_s}**")
            else:
                parts.append(f"{k} {pct_s}")
        except (TypeError, ValueError):
            parts.append(f"{k} {weights_by_tier[k]!r}")
    if not parts:
        for k, v in sorted(weights_by_tier.items()):
            try:
                pct = float(v) * 100
                pct_s = f"{pct:.0f}%"
                if markdown_bold_pct:
                    parts.append(f"{k} **{pct_s}**")
                else:
                    parts.append(f"{k} {pct_s}")
            except (TypeError, ValueError):
                parts.append(f"{k} {v!r}")
    return " / ".join(parts) if parts else "—"


def list_profiles() -> list[str]:
    return list(PROFILE_WEIGHTS)


def profile_weight_percent_triple(profile: str) -> str:
    """User-facing 'NEG / NEG+LOW / NEG+LOW+HIGH' weights as percent string."""
    w = weights_by_strategy_profile(profile)
    return f"{w[1] * 100:.0f}% / {w[2] * 100:.0f}% / {w[3] * 100:.0f}%"
