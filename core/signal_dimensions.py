"""
三维度信号的可选网格（NEG=动量 / LOW=乖离 / HIGH=量比）。

UI 与 CLI 仅允许从这些集合中选值；指标层会一次性算齐所需列，避免重复逻辑。
"""

from __future__ import annotations

# 动量：回看周期（收益率 NEG 用 <= 分位）
MOMENTUM_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)

# 乖离：价格相对均线的偏离（LOW 用 <= 分位）
BIAS_MA_WINDOWS: tuple[int, ...] = (60, 120, 250)

# 量比：成交量 / 均量 的均量窗口（HIGH 用 >= 分位）
VOLUME_MA_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)

# NEG / LOW：同时用「弱势」分位阈值（收盘价动量、乖离同时判低）
QUANTILE_LOW_CHOICES: tuple[float, ...] = (0.25, 0.30, 0.33, 0.40)

# HIGH：放量侧分位阈值
QUANTILE_HIGH_CHOICES: tuple[float, ...] = (0.60, 0.67, 0.70, 0.75)


def momentum_column(window: int) -> str:
    return f"momentum_{int(window)}"


def volume_ratio_column(ma_window: int) -> str:
    return f"volume_ratio_{int(ma_window)}"


def bias_column(ma_window: int) -> str:
    return f"bias_ma{int(ma_window)}"
