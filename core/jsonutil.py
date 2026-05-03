"""JSON-safe helpers for CLI / agent output."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel


def json_safe_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, (float, np.floating)):
        x = float(v)
        return x if np.isfinite(x) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if pd.isna(v):
        return None
    return v


def dataframe_records_json_safe(df: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        d: dict[str, Any] = {}
        for k, val in row.items():
            d[str(k)] = json_safe_value(val)
        out.append(d)
    return out


def pydantic_dumps(model: BaseModel) -> str:
    """Serialize model to JSON string (NaN/inf handled via schema validators)."""
    return model.model_dump_json(indent=2, exclude_none=False)


def pydantic_dump_obj(model: BaseModel) -> dict[str, Any]:
    return json.loads(model.model_dump_json())
