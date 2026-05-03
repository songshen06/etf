"""
三维三分位状态 + 前向收益，对齐 signal-quality-analyzer 的 bucketing 思路。

- 动量：NEG / NEU / POS（相对分位）
- 乖离：LOW / MID / HIGH
- 量比：LOW / NORMAL / HIGH

组合状态如 ``POS_MID_HIGH``，用于按 horizon 统计胜率/均值并做 Top/Bottom 排名。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

BucketMode = Literal["full_sample", "rolling"]


def _add_forward_returns_one(
    df: pd.DataFrame,
    horizons: tuple[int, ...],
    *,
    close_col: str,
    date_col: str,
) -> pd.DataFrame:
    out = df.sort_values(date_col).reset_index(drop=True).copy()
    close = pd.to_numeric(out[close_col], errors="coerce")
    for h in horizons:
        hh = int(h)
        out[f"forward_return_{hh}"] = close.shift(-hh) / close - 1.0
    return out


def add_forward_returns(
    df: pd.DataFrame,
    horizons: tuple[int, ...],
    *,
    close_col: str = "close",
    date_col: str = "date",
    group_col: str | None = None,
) -> pd.DataFrame:
    """forward_return_h = close[t+h]/close[t] - 1（与 signal-quality-analyzer 一致）。"""
    if group_col and group_col in df.columns:
        parts: list[pd.DataFrame] = []
        for _, g in df.groupby(group_col, sort=False):
            parts.append(_add_forward_returns_one(g, horizons, close_col=close_col, date_col=date_col))
        return pd.concat(parts, ignore_index=True)
    return _add_forward_returns_one(df, horizons, close_col=close_col, date_col=date_col)


def _ternary_labels(
    s: pd.Series,
    t1: float | pd.Series,
    t2: float | pd.Series,
    low_label: str,
    mid_label: str,
    high_label: str,
) -> pd.Series:
    s_np = s.to_numpy(dtype=float)
    if isinstance(t1, pd.Series):
        t1_np = t1.to_numpy(dtype=float)
        t2_np = t2.to_numpy(dtype=float)
    else:
        t1_np = np.full(len(s_np), float(t1), dtype=float)
        t2_np = np.full(len(s_np), float(t2), dtype=float)
    out = np.array(["MISSING"] * len(s_np), dtype=object)
    valid = np.isfinite(s_np) & np.isfinite(t1_np) & np.isfinite(t2_np)
    lo = valid & (s_np <= t1_np)
    mid = valid & ~lo & (s_np <= t2_np)
    hi = valid & ~lo & ~mid
    out[lo] = low_label
    out[mid] = mid_label
    out[hi] = high_label
    return pd.Series(out, index=s.index)


def _quantile_pair(
    override: tuple[float, float] | None, default_q1: float, default_q2: float
) -> tuple[float, float]:
    if override is None:
        q1, q2 = float(default_q1), float(default_q2)
    else:
        q1, q2 = float(override[0]), float(override[1])
    if not (0 < q1 < q2 < 1):
        raise ValueError("require 0 < q_low < q_high < 1 for quantile cutoffs")
    return q1, q2


def assign_ternary_states(
    df: pd.DataFrame,
    *,
    momentum_col: str,
    bias_col: str,
    volume_col: str,
    mode: BucketMode,
    rolling_window: int,
    ternary_q1: float = 0.33,
    ternary_q2: float = 0.67,
    momentum_quants: tuple[float, float] | None = None,
    bias_quants: tuple[float, float] | None = None,
    volume_quants: tuple[float, float] | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    为每一行打上 bucket_momentum / bucket_bias / bucket_volume 与合成 state。

    ``momentum_quants`` / ``bias_quants`` / ``volume_quants`` 若提供，则为各维度独立的
    (q_low, q_high) 分位点；否则三轴共用 ``ternary_q1`` / ``ternary_q2``。
    """
    out = df.sort_values(date_col).reset_index(drop=True).copy()
    m = pd.to_numeric(out[momentum_col], errors="coerce")
    b = pd.to_numeric(out[bias_col], errors="coerce")
    v = pd.to_numeric(out[volume_col], errors="coerce")

    mq1, mq2 = _quantile_pair(momentum_quants, ternary_q1, ternary_q2)
    bq1, bq2 = _quantile_pair(bias_quants, ternary_q1, ternary_q2)
    vq1, vq2 = _quantile_pair(volume_quants, ternary_q1, ternary_q2)

    if mode == "full_sample":
        mt1, mt2 = float(m.quantile(mq1)), float(m.quantile(mq2))
        bt1, bt2 = float(b.quantile(bq1)), float(b.quantile(bq2))
        vt1, vt2 = float(v.quantile(vq1)), float(v.quantile(vq2))
        out["bucket_momentum"] = _ternary_labels(m, mt1, mt2, "NEG", "NEU", "POS")
        out["bucket_bias"] = _ternary_labels(b, bt1, bt2, "LOW", "MID", "HIGH")
        out["bucket_volume"] = _ternary_labels(v, vt1, vt2, "LOW", "NORMAL", "HIGH")
    elif mode == "rolling":
        w = int(rolling_window)
        if w < 20:
            raise ValueError("rolling_window must be >= 20 for rolling terciles")
        mt1 = m.rolling(w, min_periods=w).quantile(mq1)
        mt2 = m.rolling(w, min_periods=w).quantile(mq2)
        bt1 = b.rolling(w, min_periods=w).quantile(bq1)
        bt2 = b.rolling(w, min_periods=w).quantile(bq2)
        vt1 = v.rolling(w, min_periods=w).quantile(vq1)
        vt2 = v.rolling(w, min_periods=w).quantile(vq2)
        out["bucket_momentum"] = _ternary_labels(m, mt1, mt2, "NEG", "NEU", "POS")
        out["bucket_bias"] = _ternary_labels(b, bt1, bt2, "LOW", "MID", "HIGH")
        out["bucket_volume"] = _ternary_labels(v, vt1, vt2, "LOW", "NORMAL", "HIGH")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    has_vol = bool(v.notna().any()) and bool(out["bucket_volume"].ne("MISSING").any())
    if has_vol:
        out["state"] = (
            out["bucket_momentum"].astype(str)
            + "_"
            + out["bucket_bias"].astype(str)
            + "_"
            + out["bucket_volume"].astype(str)
        )
        miss = (
            (out["bucket_momentum"] == "MISSING")
            | (out["bucket_bias"] == "MISSING")
            | (out["bucket_volume"] == "MISSING")
        )
    else:
        out["state"] = out["bucket_momentum"].astype(str) + "_" + out["bucket_bias"].astype(str)
        miss = (out["bucket_momentum"] == "MISSING") | (out["bucket_bias"] == "MISSING")

    out.loc[miss, "state"] = "MISSING"
    return out


def assign_ternary_states_fixed(
    df: pd.DataFrame,
    *,
    momentum_col: str,
    bias_col: str,
    volume_col: str,
    momentum_thresholds: tuple[float, float],
    bias_thresholds: tuple[float, float],
    volume_thresholds: tuple[float, float],
    date_col: str = "date",
) -> pd.DataFrame:
    """
    固定数值阈值三分桶（对齐 signal-quality-analyzer ``bucketing.fixed``）。
    """
    out = df.sort_values(date_col).reset_index(drop=True).copy()
    m = pd.to_numeric(out[momentum_col], errors="coerce")
    b = pd.to_numeric(out[bias_col], errors="coerce")
    v = pd.to_numeric(out[volume_col], errors="coerce")

    mt1, mt2 = float(momentum_thresholds[0]), float(momentum_thresholds[1])
    bt1, bt2 = float(bias_thresholds[0]), float(bias_thresholds[1])
    vt1, vt2 = float(volume_thresholds[0]), float(volume_thresholds[1])

    out["bucket_momentum"] = _ternary_labels(m, mt1, mt2, "NEG", "NEU", "POS")
    out["bucket_bias"] = _ternary_labels(b, bt1, bt2, "LOW", "MID", "HIGH")
    out["bucket_volume"] = _ternary_labels(v, vt1, vt2, "LOW", "NORMAL", "HIGH")

    has_vol = bool(v.notna().any()) and bool(out["bucket_volume"].ne("MISSING").any())
    if has_vol:
        out["state"] = (
            out["bucket_momentum"].astype(str)
            + "_"
            + out["bucket_bias"].astype(str)
            + "_"
            + out["bucket_volume"].astype(str)
        )
        miss = (
            (out["bucket_momentum"] == "MISSING")
            | (out["bucket_bias"] == "MISSING")
            | (out["bucket_volume"] == "MISSING")
        )
    else:
        out["state"] = out["bucket_momentum"].astype(str) + "_" + out["bucket_bias"].astype(str)
        miss = (out["bucket_momentum"] == "MISSING") | (out["bucket_bias"] == "MISSING")

    out.loc[miss, "state"] = "MISSING"
    return out


def rank_states_by_horizon(
    df: pd.DataFrame,
    *,
    horizon: int,
    min_n: int,
    top_k: int,
    bottom_k: int,
) -> tuple[list[dict], list[dict], int]:
    """
    返回 (top_best, bottom_worst, n_states_considered)。

    排序：主键胜率降序，次键均值降序，再次样本量降序。
    """
    col = f"forward_return_{int(horizon)}"
    if col not in df.columns:
        raise KeyError(f"Missing {col}; call add_forward_returns first")

    rows: list[dict] = []
    for state in df["state"].unique():
        st = str(state)
        if st == "MISSING" or "MISSING" in st:
            continue
        r = df.loc[df["state"] == state, col].dropna()
        n = int(len(r))
        if n < min_n:
            continue
        arr = r.to_numpy(dtype=float)
        rows.append(
            {
                "state": st,
                "n": n,
                "win_rate": float((arr > 0).mean()),
                "mean_return": float(np.mean(arr)),
                "median_return": float(np.median(arr)),
                "std": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
            }
        )

    def sort_key(x: dict) -> tuple:
        return (-x["win_rate"], -x["mean_return"], -x["n"])

    rows.sort(key=sort_key)
    top = rows[: max(0, int(top_k))]
    worst_sorted = sorted(rows, key=lambda x: (x["win_rate"], x["mean_return"], x["n"]))
    bottom = worst_sorted[: max(0, int(bottom_k))]
    return top, bottom, len(rows)


def run_state_quality_scan(
    df: pd.DataFrame,
    *,
    momentum_col: str,
    bias_col: str,
    volume_col: str,
    mode: BucketMode,
    rolling_window: int,
    horizon: int,
    ternary_q1: float = 0.33,
    ternary_q2: float = 0.67,
    min_n: int = 5,
    top_k: int = 5,
    bottom_k: int = 5,
) -> tuple[pd.DataFrame, list[dict], list[dict], int]:
    """指标 DataFrame → 打桶 + 前向收益 + 排名。"""
    h = (int(horizon),)
    d = add_forward_returns(df, h)
    d = assign_ternary_states(
        d,
        momentum_col=momentum_col,
        bias_col=bias_col,
        volume_col=volume_col,
        mode=mode,
        rolling_window=rolling_window,
        ternary_q1=ternary_q1,
        ternary_q2=ternary_q2,
    )
    top, bottom, n_st = rank_states_by_horizon(
        d, horizon=int(horizon), min_n=min_n, top_k=top_k, bottom_k=bottom_k
    )
    return d, top, bottom, n_st
