"""Plotly figure builders — used by runner, CLI artifacts, and Streamlit."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def figure_event_study_vs_horizon(
    tbl: pd.DataFrame,
    *,
    y_col: str,
    title: str,
    y_tick_pct_decimals: int,
    x_title: str = "Horizon (trading days)",
    y_title: str | None = None,
) -> go.Figure:
    t = tbl.copy()
    t["horizon"] = pd.to_numeric(t["horizon"], errors="coerce")
    t[y_col] = pd.to_numeric(t[y_col], errors="coerce")
    t = t.dropna(subset=["horizon", y_col])
    fmt = f".{y_tick_pct_decimals}%"
    if y_title is None:
        y_title = "Win rate" if y_col == "win_rate" else "Mean forward return"
    ht = (
        "Horizon=%{x} d<br>Win rate=%{y:.1%}<extra></extra>"
        if y_col == "win_rate"
        else "Horizon=%{x} d<br>Mean return=%{y:.2%}<extra></extra>"
    )
    fig = go.Figure(
        data=[
            go.Scatter(
                x=t["horizon"].astype(float).tolist(),
                y=t[y_col].astype(float).tolist(),
                mode="lines+markers",
                name=y_col,
                line=dict(color="#2563eb", width=3),
                marker=dict(size=16, color="#2563eb", line=dict(width=2, color="white")),
                hovertemplate=ht,
            )
        ]
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        margin=dict(l=48, r=24, t=56, b=48),
        showlegend=False,
    )
    fig.update_xaxes(type="linear")
    fig.update_yaxes(tickformat=fmt)
    return fig


def figure_equity_and_drawdown(
    equity: pd.Series,
    drawdown: pd.Series,
    *,
    title: str,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.62, 0.38],
    )
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity.values, name="Equity"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown", fill="tozeroy"),
        row=2,
        col=1,
    )
    fig.update_layout(height=640, title_text=title, showlegend=True)
    return fig


def peak_to_trough_trading_days(equity: pd.Series, trough_date: pd.Timestamp) -> tuple[pd.Timestamp | None, int]:
    """For max-drawdown trough, return (peak_date, trading days peak→trough inclusive)."""
    t = pd.Timestamp(trough_date)
    sub = equity.sort_index().loc[:t]
    if sub.empty or not np.isfinite(sub.astype(float)).any():
        return None, 0
    peak_date = sub.idxmax()
    mask = (equity.index >= peak_date) & (equity.index <= t)
    return peak_date, int(mask.sum())


def buy_hold_normalized_from_ohlcv(df: pd.DataFrame, equity_index: pd.DatetimeIndex) -> pd.Series:
    """收盘价买入持有，首日归一 1，对齐策略净值日期索引。"""
    d = df.sort_values("date").reset_index(drop=True)
    c = pd.to_numeric(d["close"], errors="coerce")
    idx = pd.DatetimeIndex(pd.to_datetime(d["date"]))
    if len(c) == 0:
        return pd.Series(np.nan, index=equity_index)
    c0 = float(c.iloc[0])
    if not np.isfinite(c0) or c0 <= 0:
        return pd.Series(np.nan, index=equity_index)
    s = pd.Series((c / c0).to_numpy(dtype=float), index=idx)
    return s.reindex(equity_index).ffill()


def _normalize_nav_to_one(equity: pd.Series) -> pd.Series:
    """Scale NAV so the first observation is exactly 1.00 (display semantics)."""
    eq = pd.Series(equity.values.astype(float), index=pd.DatetimeIndex(pd.to_datetime(equity.index)))
    base = float(eq.iloc[0]) if len(eq) else float("nan")
    if not np.isfinite(base) or base <= 1e-15:
        return eq
    return eq / base


def _equity_value_asof(eq: pd.Series, d: Any) -> float | None:
    """Strategy NAV on trade date: use time-aligned as-of on sorted index (same series as plotted)."""
    try:
        ts = pd.Timestamp(d).normalize()
    except Exception:
        return None
    eq = eq.sort_index()
    try:
        v = float(eq.asof(ts))
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def figure_backtest_dashboard(
    equity: pd.Series,
    drawdown: pd.Series,
    exposure: pd.Series,
    buy_hold: pd.Series | None,
    trades: list[Any],
    *,
    title: str,
    max_dd_pct: float | None = None,
    mdd_trough_date: pd.Timestamp | None = None,
    avg_exposure: float | None = None,
) -> go.Figure:
    """策略/买入持有均以 **净值 1.00 起点** 绘制；hover 用预计算文本避免 JSON 后模板失效；上轴净值、下轴回撤 %%。"""
    _ = drawdown  # runner 传入；下图回撤由归一净值重算以保证与曲线一致
    idx = pd.DatetimeIndex(pd.to_datetime(equity.index))
    eq_plot = _normalize_nav_to_one(pd.Series(equity.values.astype(float), index=idx))
    exp = pd.Series(
        exposure.reindex(idx).ffill().fillna(0.0).values.astype(float),
        index=idx,
    )
    dd_lin = eq_plot / eq_plot.cummax() - 1.0
    dd_lin = pd.Series(np.minimum(dd_lin.values.astype(float), 0.0), index=idx)
    dd_pct = dd_lin * 100.0

    bh_plot: pd.Series | None = None
    if buy_hold is not None and buy_hold is not False:
        bh_raw = buy_hold.reindex(idx).ffill().bfill()
        if bh_raw.notna().any():
            fv = bh_raw.dropna()
            if len(fv) and np.isfinite(fv.iloc[0]) and float(fv.iloc[0]) > 1e-15:
                bh_plot = bh_raw.astype(float) / float(fv.iloc[0])

    hover_strategy: list[str] = []
    hover_bh: list[str] = []
    for i, d in enumerate(idx):
        ev = float(eq_plot.iloc[i])
        ex = float(exp.iloc[i])
        ddv = float(dd_pct.iloc[i])
        ds = pd.Timestamp(d).strftime("%Y-%m-%d")
        if bh_plot is not None and np.isfinite(bh_plot.iloc[i]):
            bv = float(bh_plot.iloc[i])
            bh_line = f"买入持有净值: {bv:.4f}"
        else:
            bv = float("nan")
            bh_line = "买入持有净值: —"
        hover_strategy.append(
            f"<b>策略曲线</b><br>"
            f"日期: {ds}<br>"
            f"策略净值: {ev:.4f}<br>"
            f"{bh_line}<br>"
            f"当日仓位: {ex:.1%}<br>"
            f"当前回撤: {ddv:.2f}%"
        )
        if bh_plot is not None:
            hover_bh.append(
                f"<b>买入持有（满仓基准）</b><br>日期: {ds}<br>净值: {bv:.4f}<br>归一起点=1.00"
                if np.isfinite(bv)
                else f"<b>买入持有（满仓基准）</b><br>日期: {ds}<br>净值: —"
            )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35],
        subplot_titles=("净值曲线（策略 vs 买入持有）", "回撤曲线"),
        specs=[[{"secondary_y": True}], [{}]],
    )

    fig.add_trace(
        go.Scatter(
            x=list(idx),
            y=eq_plot.values.tolist(),
            name="策略净值（战术仓位）",
            mode="lines",
            line=dict(color="#1d4ed8", width=3),
            hovertext=hover_strategy,
            hoverinfo="text",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    if bh_plot is not None:
        fig.add_trace(
            go.Scatter(
                x=list(idx),
                y=bh_plot.values.tolist(),
                name="买入持有（满仓归一=1.00）",
                mode="lines",
                line=dict(color="#f97316", width=2.5, dash="dash"),
                hovertext=hover_bh,
                hoverinfo="text",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(
            x=list(idx),
            y=(exp * 100.0).tolist(),
            name="当日仓位（%）",
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(148,163,184,0.28)",
            hoverinfo="skip",
            showlegend=True,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.add_hline(
        y=1.0,
        line=dict(color="rgba(30,41,59,0.45)", width=1, dash="dot"),
        row=1,
        col=1,
        secondary_y=False,
    )

    ex_x: list[pd.Timestamp] = []
    ex_y: list[float] = []
    en_x: list[pd.Timestamp] = []
    en_y: list[float] = []
    en_ht: list[str] = []
    for t in trades:
        try:
            ed = pd.Timestamp(t.entry_date)
            xd = pd.Timestamp(t.exit_date)
        except Exception:
            continue
        ye = _equity_value_asof(eq_plot, ed)
        yx = _equity_value_asof(eq_plot, xd)
        stier = getattr(t, "signal_tier", "")
        wt = getattr(t, "weight", "")
        if ye is not None:
            en_x.append(ed)
            en_y.append(ye)
            ds = ed.strftime("%Y-%m-%d") if hasattr(ed, "strftime") else str(ed)[:10]
            en_ht.append(f"<b>开仓</b><br>日期: {ds}<br>策略净值: {ye:.4f}<br>层 {stier} · 权重 {float(wt):.1%}")
        if yx is not None:
            ex_x.append(xd)
            ex_y.append(yx)
    if en_x:
        fig.add_trace(
            go.Scatter(
                x=en_x,
                y=en_y,
                mode="markers",
                name="开仓",
                marker=dict(symbol="triangle-up", size=12, color="#15803d", line=dict(width=1, color="white")),
                hovertext=en_ht,
                hoverinfo="text",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
    if ex_x:
        ex_ht = []
        for xd, yv in zip(ex_x, ex_y):
            ds = xd.strftime("%Y-%m-%d") if hasattr(xd, "strftime") else str(xd)[:10]
            ex_ht.append(f"<b>平仓</b><br>日期: {ds}<br>策略净值: {yv:.4f}")
        fig.add_trace(
            go.Scatter(
                x=ex_x,
                y=ex_y,
                mode="markers",
                name="平仓",
                marker=dict(symbol="x", size=12, color="#b91c1c", line=dict(width=1, color="white")),
                hovertext=ex_ht,
                hoverinfo="text",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    if len(eq_plot) and np.isfinite(eq_plot.iloc[-1]):
        lv = float(eq_plot.iloc[-1])
        lx = eq_plot.index[-1]
        fig.add_annotation(
            x=lx,
            y=lv,
            text=f"最新净值 {lv:.4f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(size=11, color="#1e3a8a"),
            bgcolor="rgba(255,255,255,0.9)",
            borderpad=4,
            row=1,
            col=1,
            secondary_y=False,
        )

    note_parts: list[str] = []
    if avg_exposure is not None and np.isfinite(avg_exposure) and avg_exposure < 0.25:
        note_parts.append(
            "平均暴露较低：策略大部分时间可能轻仓/空仓；净值上升不代表持续满仓。"
        )
    if avg_exposure is not None and np.isfinite(avg_exposure) and avg_exposure < 0.5:
        note_parts.append(
            "橙色虚线为满仓买入持有基准；蓝色为战术分层仓位净值，二者不可等同解读。"
        )
    if note_parts:
        fig.add_annotation(
            x=0.01,
            y=1.02,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="bottom",
            text="<br>".join(note_parts),
            showarrow=False,
            font=dict(size=11, color="#78350f"),
            bgcolor="rgba(254,249,195,0.95)",
            bordercolor="#eab308",
            borderwidth=1,
            align="left",
        )

    hover_dd = [
        f"日期: {d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10]}<br>回撤: {v:.2f}%"
        for d, v in zip(idx, dd_pct.values)
    ]
    fig.add_trace(
        go.Scatter(
            x=list(idx),
            y=dd_pct.values.tolist(),
            name="回撤（相对历史峰值）",
            mode="lines",
            line=dict(color="#b91c1c", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(220,38,38,0.22)",
            hovertext=hover_dd,
            hoverinfo="text",
        ),
        row=2,
        col=1,
    )

    y2_min = float(np.nanmin(dd_pct.values)) if len(dd_pct) else -1.0
    if not np.isfinite(y2_min):
        y2_min = -1.0
    pad = max(abs(y2_min) * 0.12, 1.0)
    fig.update_yaxes(range=[y2_min - pad, 0.5], row=2, col=1)

    if mdd_trough_date is not None and max_dd_pct is not None and np.isfinite(max_dd_pct):
        mdd_y = float(max_dd_pct) * 100.0
        peak_d, span_days = peak_to_trough_trading_days(eq_plot, pd.Timestamp(mdd_trough_date))
        dur_txt = f"<br>峰值→谷底约 {span_days} 个交易日" if span_days > 0 else ""
        peak_txt = ""
        if peak_d is not None:
            pds = peak_d.strftime("%Y-%m-%d") if hasattr(peak_d, "strftime") else str(peak_d)[:10]
            tds = pd.Timestamp(mdd_trough_date).strftime("%Y-%m-%d")
            peak_txt = f"<br>区间: {pds} → {tds}"
        fig.add_trace(
            go.Scatter(
                x=[pd.Timestamp(mdd_trough_date)],
                y=[mdd_y],
                mode="markers",
                name="最大回撤点",
                marker=dict(size=13, color="#7f1d1d", symbol="circle", line=dict(width=2, color="white")),
                hovertext=[f"最大回撤: {mdd_y:.2f}%{dur_txt}{peak_txt}"],
                hoverinfo="text",
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        fig.add_annotation(
            x=pd.Timestamp(mdd_trough_date),
            y=mdd_y,
            text=f"最大回撤 {mdd_y:.2f}%{(' · ' + str(span_days) + ' 交易日') if span_days > 0 else ''}",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-35,
            font=dict(size=11),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=900,
        title_text=title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0, font=dict(size=10)),
        margin=dict(l=56, r=72, t=88, b=56),
    )
    fig.update_xaxes(
        title_text="日期",
        tickformat="%Y-%m",
        tickangle=-35,
        dtick="M6",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="日期",
        tickformat="%Y-%m",
        tickangle=-35,
        dtick="M6",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="净值（初始=1.00）",
        tickformat=".2f",
        showexponent="none",
        separatethousands=False,
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="仓位（%）",
        range=[0, 105],
        tickformat=".0f",
        showgrid=False,
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.update_yaxes(
        title_text="回撤（%）",
        tickformat=".2f",
        ticksuffix="%",
        showexponent="none",
        row=2,
        col=1,
    )
    return fig


def _subsample_sorted_dates(dates: list[pd.Timestamp], max_n: int) -> list[pd.Timestamp]:
    if not dates or len(dates) <= max_n:
        return dates
    idx = np.linspace(0, len(dates) - 1, num=max_n, dtype=int)
    out: list[pd.Timestamp] = []
    seen: set[int] = set()
    for i in sorted(set(idx.tolist())):
        if i not in seen:
            seen.add(i)
            out.append(dates[i])
    return out


def figure_equity_curves_compare(
    series_by_label: dict[str, pd.Series],
    *,
    title: str,
    benchmark: pd.Series | None = None,
    benchmark_label: str = "买入持有基准（对照）",
    shared_entry_dates: Sequence[Any] | None = None,
    exit_dates_by_label: dict[str, Sequence[Any]] | None = None,
    max_entry_markers: int = 80,
    max_exit_markers_per_series: int = 100,
) -> go.Figure:
    """多条净值曲线对比（每条强制归一到首日为 1.00）；hover 用预计算文本。

    benchmark：与主图一致的收盘价买入持有序列（已对齐策略日期索引时可原样传入；会再归一到首日 1.00）。

    shared_entry_dates：在每条策略净值上叠加绿色三角（入场）；exit_dates_by_label：键与 series_by_label 一致，各线用与曲线同色标记退出日。
    """
    fig = go.Figure()
    colors = ["#1d4ed8", "#ca8a04", "#16a34a", "#9333ea", "#64748b"]
    line_colors: dict[str, str] = {}
    normalized_series: dict[str, pd.Series] = {}
    for i, (label, s) in enumerate(series_by_label.items()):
        c = colors[i % len(colors)]
        line_colors[label] = c
        si = pd.Series(s.values.astype(float), index=pd.DatetimeIndex(pd.to_datetime(s.index)))
        si = _normalize_nav_to_one(si)
        normalized_series[label] = si
        hts = [
            f"<b>{label}</b><br>{pd.Timestamp(d).strftime('%Y-%m-%d')}<br>净值（初始=1.00）: {float(v):.4f}"
            for d, v in zip(si.index, si.values)
        ]
        fig.add_trace(
            go.Scatter(
                x=list(si.index),
                y=si.values.tolist(),
                name=label,
                mode="lines",
                line=dict(width=2.5, color=c),
                hovertext=hts,
                hoverinfo="text",
            )
        )
    if benchmark is not None and len(benchmark):
        bi = pd.Series(benchmark.values.astype(float), index=pd.DatetimeIndex(pd.to_datetime(benchmark.index)))
        bi = _normalize_nav_to_one(bi)
        bhts = [
            f"<b>{benchmark_label}</b><br>{pd.Timestamp(d).strftime('%Y-%m-%d')}<br>净值（初始=1.00）: {float(v):.4f}"
            for d, v in zip(bi.index, bi.values)
        ]
        fig.add_trace(
            go.Scatter(
                x=list(bi.index),
                y=bi.values.tolist(),
                name=benchmark_label,
                mode="lines",
                line=dict(width=2.2, color="#64748b", dash="dash"),
                hovertext=bhts,
                hoverinfo="text",
            )
        )

    if shared_entry_dates:
        raw = [pd.Timestamp(d).normalize() for d in shared_entry_dates]
        raw = sorted({x for x in raw})
        raw = _subsample_sorted_dates(raw, max_entry_markers)
        entry_legend_done = False
        for label, si in normalized_series.items():
            xs_f: list[pd.Timestamp] = []
            ys_f: list[float] = []
            for d in raw:
                yv = _equity_value_asof(si, d)
                if yv is not None:
                    xs_f.append(d)
                    ys_f.append(yv)
            if not xs_f:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[pd.Timestamp(x) for x in xs_f],
                    y=ys_f,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=9,
                        color="#15803d",
                        opacity=0.78,
                        line=dict(width=0),
                    ),
                    name="入场（共用）" if not entry_legend_done else "",
                    legendgroup="entry_shared",
                    showlegend=not entry_legend_done,
                    hovertemplate="<b>入场（共用）</b><br>%{x|%Y-%m-%d}<br>净值: %{y:.4f}<extra></extra>",
                )
            )
            entry_legend_done = True

    if exit_dates_by_label:
        for label, si in normalized_series.items():
            ed = exit_dates_by_label.get(label)
            if not ed:
                continue
            raw_e = sorted({pd.Timestamp(x).normalize() for x in ed})
            raw_e = _subsample_sorted_dates(raw_e, max_exit_markers_per_series)
            xs_f: list[pd.Timestamp] = []
            ys_f: list[float] = []
            for d in raw_e:
                yv = _equity_value_asof(si, d)
                if yv is not None:
                    xs_f.append(d)
                    ys_f.append(yv)
            if not xs_f:
                continue
            if label.rfind("(") >= 0 and label.endswith(")"):
                rid = label[label.rfind("(") + 1 : -1].strip()
            else:
                rid = label[:32]
            leg = f"退出：{rid}"
            col = line_colors.get(label, "#64748b")
            fig.add_trace(
                go.Scatter(
                    x=[pd.Timestamp(x) for x in xs_f],
                    y=ys_f,
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=7,
                        color=col,
                        opacity=0.72,
                        line=dict(width=0),
                    ),
                    name=leg,
                    legendgroup=f"exit_{label}",
                    showlegend=True,
                    hovertemplate=f"<b>{leg}</b><br>%{{x|%Y-%m-%d}}<br>净值: %{{y:.4f}}<extra></extra>",
                )
            )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="日期",
        yaxis_title="净值（初始=1.00）",
        height=560,
        hovermode="x unified",
        legend=dict(orientation="v", yanchor="top", y=0.98, x=1.02, xanchor="left", font=dict(size=10)),
        margin=dict(l=52, r=168, t=56, b=52),
    )
    fig.update_xaxes(tickformat="%Y-%m", tickangle=-35, dtick="M6")
    fig.update_yaxes(tickformat=".2f", showexponent="none")
    return fig


def figure_close_history(dates: pd.Series, close: pd.Series, *, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scatter(
                x=pd.to_datetime(dates),
                y=pd.to_numeric(close, errors="coerce"),
                mode="lines",
                name="Close",
                line=dict(color="#059669", width=2),
            )
        ]
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Close",
        margin=dict(l=48, r=24, t=56, b=48),
        showlegend=False,
    )
    return fig


def figure_to_json(fig: go.Figure) -> dict[str, Any]:
    """JSON-native dict for ChartSpec (avoids numpy leaves from to_dict())."""
    return json.loads(fig.to_json())
