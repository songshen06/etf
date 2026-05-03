"""Resolve project root and default SQLite path."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_db_path(explicit: str | None = None) -> Path:
    """默认以仓库根目录 ``etf_data.db`` 为准；``db/etf_data.db`` 仅作兼容。

    解析顺序：显式参数 → 环境变量 ``ETF_DB_PATH`` → 根目录 ``etf_data.db``
    → ``db/etf_data.db``。若均不存在，返回根目录路径（供创建/提示）。"""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("ETF_DB_PATH")
    if env:
        return Path(env).expanduser().resolve()
    root = project_root()
    for candidate in (root / "etf_data.db", root / "db" / "etf_data.db"):
        if candidate.is_file():
            return candidate
    return root / "etf_data.db"
