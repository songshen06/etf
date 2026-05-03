import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DIVIDEND_ETF_LIST = [
    "159209",
    "515080",
]


DEFAULT_STATE_PATH = Path.home() / ".quantlab" / "dividend_state.json"


def _as_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not (v == v):
        return None
    return v


def _price_from_row(row: pd.Series) -> float | None:
    for k in ("close_norm", "close", "price", "收盘"):
        if k in row:
            v = _as_float(row.get(k))
            if v is not None:
                return v
    return None


def _date_from_row(row: pd.Series) -> str:
    if "date" in row:
        return str(row.get("date"))
    return "latest"


def rolling_bucket_rank(values: pd.Series, *, window: int, n_buckets: int = 5) -> int | None:
    from quantlab.filters.quantile_filter import assign_equal_frequency_quantile_labels, bucket_label_to_rank

    s = pd.to_numeric(values, errors="coerce")
    tail = s.tail(int(window))
    labels = assign_equal_frequency_quantile_labels(tail, n_buckets=int(n_buckets))
    lab = labels.iloc[-1] if len(labels) > 0 else None
    return bucket_label_to_rank(lab)


def momentum_label_from_rank(rank: int | None) -> str:
    if rank is None:
        return "unknown"
    if rank == 1:
        return "weak"
    if rank in (2, 3):
        return "neutral"
    return "strong"


def is_dividend_entry(*, bias_q: int | None, momentum_q: int | None) -> bool:
    if bias_q is None or momentum_q is None:
        return False
    return bias_q in (1, 2) and momentum_q in (1, 2)


@dataclass
class DividendPositionState:
    position: float = 0.0
    avg_cost: float | None = None
    first_entry_price: float | None = None
    first_entry_bias: float | None = None
    first_entry_bias_q: int | None = None
    layer: int = 0
    last_date: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DividendPositionState":
        return cls(
            position=float(d.get("position", 0.0) or 0.0),
            avg_cost=_as_float(d.get("avg_cost")),
            first_entry_price=_as_float(d.get("first_entry_price")),
            first_entry_bias=_as_float(d.get("first_entry_bias")),
            first_entry_bias_q=int(d.get("first_entry_bias_q")) if d.get("first_entry_bias_q") is not None else None,
            layer=int(d.get("layer", 0) or 0),
            last_date=str(d.get("last_date")) if d.get("last_date") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": float(self.position),
            "avg_cost": self.avg_cost,
            "first_entry_price": self.first_entry_price,
            "first_entry_bias": self.first_entry_bias,
            "first_entry_bias_q": self.first_entry_bias_q,
            "layer": int(self.layer),
            "last_date": self.last_date,
        }


def load_state(path: Path) -> dict[str, DividendPositionState]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, DividendPositionState] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[str(k)] = DividendPositionState.from_dict(v)
    return out


def save_state(path: Path, data: dict[str, DividendPositionState]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {k: v.to_dict() for k, v in data.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)


def _weighted_avg_cost(old_cost: float | None, old_pos: float, buy_price: float, buy_pos: float) -> float:
    if old_cost is None or old_pos <= 0:
        return float(buy_price)
    total = float(old_pos + buy_pos)
    if total <= 0:
        return float(buy_price)
    return float((old_cost * old_pos + buy_price * buy_pos) / total)


def decide_dividend_action(
    *,
    state: DividendPositionState,
    price: float,
    bias_rate: float,
    bias_q: int | None,
    momentum_q: int | None,
    snapshot_date: str,
) -> tuple[str, str, str, DividendPositionState]:
    base_target = 0.70
    add_step1 = 0.85
    add_step2 = 1.00
    low_zone = bias_q in (1, 2) and momentum_q in (1, 2)
    deep_low = bias_q == 1 and (momentum_q in (1, 2))
    extreme_rich = bias_q == 5

    pos = float(state.position or 0.0)
    pos = 0.0 if pos < 1e-9 else pos

    if pos <= 1e-12:
        if low_zone:
            ns = DividendPositionState(
                position=base_target,
                avg_cost=_weighted_avg_cost(None, 0.0, price, base_target),
                first_entry_price=price,
                first_entry_bias=bias_rate,
                first_entry_bias_q=bias_q,
                layer=1,
                last_date=snapshot_date,
            )
            return (
                "BUILD_BASE",
                f"BUY to {int(base_target*100)}%",
                "dividend ETF in true low zone (bias_q in Q1/Q2 and momentum_q in Q1/Q2); establish long-term base directly",
                ns,
            )
        ns = DividendPositionState(
            position=0.0,
            avg_cost=None,
            first_entry_price=None,
            first_entry_bias=None,
            first_entry_bias_q=None,
            layer=0,
            last_date=snapshot_date,
        )
        return (
            "HOLD_CASH",
            "HOLD",
            "not in low zone; dividend strategy avoids chasing; stay in cash",
            ns,
        )

    if extreme_rich and pos > base_target + 1e-12:
        ns = DividendPositionState(
            position=base_target,
            avg_cost=state.avg_cost,
            first_entry_price=state.first_entry_price,
            first_entry_bias=state.first_entry_bias,
            first_entry_bias_q=state.first_entry_bias_q,
            layer=1,
            last_date=snapshot_date,
        )
        return (
            "REDUCE_TO_BASE",
            f"SELL to {int(base_target*100)}%",
            "extreme richness (bias_q==Q5); trim only excess back to base",
            ns,
        )

    if pos < base_target - 1e-12:
        if low_zone:
            add_pos = base_target - pos
            ns = DividendPositionState(
                position=base_target,
                avg_cost=_weighted_avg_cost(state.avg_cost, pos, price, add_pos),
                first_entry_price=state.first_entry_price or price,
                first_entry_bias=state.first_entry_bias if state.first_entry_bias is not None else bias_rate,
                first_entry_bias_q=state.first_entry_bias_q if state.first_entry_bias_q is not None else bias_q,
                layer=1,
                last_date=snapshot_date,
            )
            return (
                "BUILD_BASE",
                f"BUY to {int(base_target*100)}%",
                "base not completed; low zone detected; complete base construction",
                ns,
            )
        ns = DividendPositionState(
            position=pos,
            avg_cost=state.avg_cost,
            first_entry_price=state.first_entry_price,
            first_entry_bias=state.first_entry_bias,
            first_entry_bias_q=state.first_entry_bias_q,
            layer=state.layer,
            last_date=snapshot_date,
        )
        return ("HOLD_BASE", "HOLD", "base not completed but not in low zone; hold and wait", ns)

    if deep_low and pos < add_step1 - 1e-12:
        buy_to = add_step1
        buy_pos = buy_to - pos
        ns = DividendPositionState(
            position=buy_to,
            avg_cost=_weighted_avg_cost(state.avg_cost, pos, price, buy_pos),
            first_entry_price=state.first_entry_price or price,
            first_entry_bias=state.first_entry_bias if state.first_entry_bias is not None else bias_rate,
            first_entry_bias_q=state.first_entry_bias_q if state.first_entry_bias_q is not None else bias_q,
            layer=max(2, int(state.layer or 0)),
            last_date=snapshot_date,
        )
        return (
            "ADD",
            f"BUY to {int(buy_to*100)}%",
            "deep low zone (bias_q==Q1 and momentum_q in Q1/Q2); slow accumulation step toward 100%",
            ns,
        )

    if deep_low and pos < add_step2 - 1e-12 and pos >= add_step1 - 1e-12:
        buy_to = add_step2
        buy_pos = buy_to - pos
        ns = DividendPositionState(
            position=buy_to,
            avg_cost=_weighted_avg_cost(state.avg_cost, pos, price, buy_pos),
            first_entry_price=state.first_entry_price or price,
            first_entry_bias=state.first_entry_bias if state.first_entry_bias is not None else bias_rate,
            first_entry_bias_q=state.first_entry_bias_q if state.first_entry_bias_q is not None else bias_q,
            layer=max(3, int(state.layer or 0)),
            last_date=snapshot_date,
        )
        return (
            "ADD",
            f"BUY to {int(buy_to*100)}%",
            "deep low persists; complete accumulation to 100%",
            ns,
        )

    ns = DividendPositionState(
        position=pos,
        avg_cost=state.avg_cost,
        first_entry_price=state.first_entry_price,
        first_entry_bias=state.first_entry_bias,
        first_entry_bias_q=state.first_entry_bias_q,
        layer=max(1, int(state.layer or 0)),
        last_date=snapshot_date,
    )
    return ("HOLD_BASE", "HOLD", "hold base; no deep-low add and no extreme trim condition", ns)


def decide_dividend_growth_action(
    *,
    state: DividendPositionState,
    price: float,
    bias_rate: float,
    bias_q: int | None,
    momentum_q: int | None,
    snapshot_date: str,
) -> tuple[str, str, str, DividendPositionState]:
    base_target = 0.60
    pos_max = 1.00
    add_unit = 0.20

    pos = float(state.position or 0.0)
    pos = 0.0 if pos < 1e-9 else pos

    if pos <= 1e-12:
        ns = DividendPositionState(
            position=base_target,
            avg_cost=_weighted_avg_cost(None, 0.0, price, base_target),
            first_entry_price=price,
            first_entry_bias=bias_rate,
            first_entry_bias_q=bias_q,
            layer=1,
            last_date=snapshot_date,
        )
        return ("BUILD_BASE", f"BUY to {int(base_target*100)}%", "DIVIDEND_GROWTH: build base early for participation", ns)

    if pos > base_target + 1e-12 and bias_q is not None and bias_q >= 4:
        ns = DividendPositionState(
            position=base_target,
            avg_cost=state.avg_cost,
            first_entry_price=state.first_entry_price,
            first_entry_bias=state.first_entry_bias,
            first_entry_bias_q=state.first_entry_bias_q,
            layer=max(1, int(state.layer or 0)),
            last_date=snapshot_date,
        )
        return ("REDUCE_TO_BASE", f"SELL to {int(base_target*100)}%", "DIVIDEND_GROWTH: rich zone → reduce overlay to base", ns)

    if pos >= base_target - 1e-12 and pos < pos_max - 1e-12:
        if bias_q == 1 and momentum_q is not None and momentum_q <= 2:
            buy_to = min(pos_max, pos + add_unit)
            buy_pos = buy_to - pos
            ns = DividendPositionState(
                position=buy_to,
                avg_cost=_weighted_avg_cost(state.avg_cost, pos, price, buy_pos),
                first_entry_price=state.first_entry_price or price,
                first_entry_bias=state.first_entry_bias if state.first_entry_bias is not None else bias_rate,
                first_entry_bias_q=state.first_entry_bias_q if state.first_entry_bias_q is not None else bias_q,
                layer=max(2, int(state.layer or 0)),
                last_date=snapshot_date,
            )
            return ("ADD", f"BUY to {int(buy_to*100)}%", "DIVIDEND_GROWTH: deep low → add overlay", ns)

    ns = DividendPositionState(
        position=pos,
        avg_cost=state.avg_cost,
        first_entry_price=state.first_entry_price,
        first_entry_bias=state.first_entry_bias,
        first_entry_bias_q=state.first_entry_bias_q,
        layer=max(1, int(state.layer or 0)),
        last_date=snapshot_date,
    )
    return ("HOLD_BASE", "HOLD", "DIVIDEND_GROWTH: hold base", ns)


def cmd_dividend_signal(ns: argparse.Namespace) -> int:
    from core.data_loader import load_etf_sqlite
    from core.indicators import add_analyzer_indicators
    from core.paths import resolve_db_path

    code = str(ns.etf_code)
    if code not in DIVIDEND_ETF_LIST:
        print(f"[{code}] not in dividend ETF universe: {', '.join(DIVIDEND_ETF_LIST)}", file=sys.stderr)
        return 2

    db_path = resolve_db_path(ns.db_path)
    df, _ = load_etf_sqlite(db_path, code)
    df = add_analyzer_indicators(df)
    if df.empty:
        print(f"[{code}] no data", file=sys.stderr)
        return 1
    df = df.sort_values("date") if "date" in df.columns else df
    row = df.iloc[-1]
    price = _price_from_row(row)
    if price is None:
        print(f"[{code}] missing price", file=sys.stderr)
        return 1

    bias = _as_float(row.get("bias_rate"))
    mom = _as_float(row.get("momentum_10", row.get("momentum")))
    vol_ratio = _as_float(row.get("volume_ratio", row.get("volume_ratio_20")))
    if bias is None or mom is None or vol_ratio is None:
        print(f"[{code}] missing indicators (bias_rate/momentum/volume_ratio)", file=sys.stderr)
        return 1

    window = int(ns.rolling_window)
    bias_q = rolling_bucket_rank(df["bias_rate"], window=window, n_buckets=5)
    mom_series = df["momentum_10"] if "momentum_10" in df.columns else df["momentum"]
    momentum_q = rolling_bucket_rank(mom_series, window=window, n_buckets=5)

    state_path = Path(ns.state_path) if ns.state_path is not None else DEFAULT_STATE_PATH
    state_map = load_state(state_path)
    st = state_map.get(code, DividendPositionState())

    snapshot_date = _date_from_row(row)
    if code == "159209":
        signal, action, reason, new_state = decide_dividend_growth_action(
            state=st,
            price=float(price),
            bias_rate=float(bias),
            bias_q=bias_q,
            momentum_q=momentum_q,
            snapshot_date=snapshot_date,
        )
        strategy_title = "DIVIDEND_GROWTH (CORE+OVERLAY)"
    else:
        signal, action, reason, new_state = decide_dividend_action(
            state=st,
            price=float(price),
            bias_rate=float(bias),
            bias_q=bias_q,
            momentum_q=momentum_q,
            snapshot_date=snapshot_date,
        )
        strategy_title = "DIVIDEND CORE HOLDING"
    state_map[code] = new_state
    if not getattr(ns, "dry_run", False):
        save_state(state_path, state_map)

    bias_pct = float(bias) * 100.0

    print(f"[{code}]")
    print("")
    print(f"strategy: {strategy_title}")
    print("")
    print("market_state:")
    print(f"  bias: {bias_pct:.1f}%")
    print(f"  bias_q: Q{bias_q}" if bias_q is not None else "  bias_q: NA")
    print(f"  momentum_q: Q{momentum_q}" if momentum_q is not None else "  momentum_q: NA")
    print("")
    print("decision:")
    print(f"  signal: {signal}")
    print(f"  action: {action}")
    print("")
    print("position_plan:")
    print(f"  current: {st.position * 100:.0f}%")
    print(f"  after_trade: {new_state.position * 100:.0f}%")
    print("  target: core base (70%) + optional low-zone adds (up to 100%)")
    print("")
    print("reason:")
    print(f"  - {reason}")
    print("")
    print("state:")
    print(f"  position: {new_state.position:.2f}")
    print(f"  layer: {new_state.layer}")
    if new_state.avg_cost is not None:
        pnl = (float(price) / float(new_state.avg_cost) - 1.0) * 100.0
        print(f"  avg_cost: {new_state.avg_cost:.3f}")
        print(f"  unrealized_pnl: {pnl:.1f}%")
    if ns.state_path is not None:
        print(f"state_path: {state_path}")
    return 0


def cmd_dividend_status(ns: argparse.Namespace) -> int:
    from core.data_loader import load_etf_sqlite
    from core.indicators import add_analyzer_indicators
    from core.paths import resolve_db_path

    state_path = Path(ns.state_path) if ns.state_path is not None else DEFAULT_STATE_PATH
    state_map = load_state(state_path)
    db_path = resolve_db_path(ns.db_path)

    codes = DIVIDEND_ETF_LIST if ns.etf_code is None else [str(ns.etf_code)]
    for code in codes:
        st = state_map.get(code, DividendPositionState())
        df, _ = load_etf_sqlite(db_path, code)
        df = add_analyzer_indicators(df)
        df = df.sort_values("date") if "date" in df.columns else df
        row = df.iloc[-1] if not df.empty else None
        price = _price_from_row(row) if row is not None else None

        print(f"[{code}]")
        pos = float(st.position or 0.0)
        if pos < 1e-9:
            stage = "CASH"
        elif pos < 0.70 - 1e-9:
            stage = "PARTIAL_BASE"
        elif pos < 0.85 - 1e-9:
            stage = "BASE"
        elif pos < 1.00 - 1e-9:
            stage = "BASE_PLUS_ADD1"
        else:
            stage = "FULL"
        print(f"position: {pos:.2f}")
        print(f"stage: {stage}")
        print(f"layer: {st.layer}")
        if st.avg_cost is not None:
            print(f"avg_cost: {st.avg_cost:.3f}")
        if price is not None and st.avg_cost is not None and st.position > 0:
            pnl = (float(price) / float(st.avg_cost) - 1.0) * 100.0
            print(f"last_price: {float(price):.3f}")
            print(f"unrealized_pnl: {pnl:.1f}%")
        if st.last_date is not None:
            print(f"last_update: {st.last_date}")
        print("")

    if ns.state_path is not None:
        print(f"state_path: {state_path}")
    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("dividend-signal", help="Dividend ETF accumulation signal (entry + layered sizing + light reduce)")
    p.add_argument("--etf-code", "--code", dest="etf_code", required=True, help="ETF code")
    p.add_argument("--db", dest="db_path", default=None, help="SQLite path")
    p.add_argument("--rolling-window", type=int, default=252, help="Rolling window for quantile buckets")
    p.add_argument("--state-path", type=str, default=None, help="State JSON path (default: ~/.quantlab/dividend_state.json)")
    p.add_argument("--dry-run", action="store_true", help="Do not persist state")
    p.set_defaults(_run=cmd_dividend_signal)

    ps = subparsers.add_parser("dividend-status", help="Show saved dividend accumulation status")
    ps.add_argument("--db", dest="db_path", default=None, help="SQLite path")
    ps.add_argument("--etf-code", "--code", dest="etf_code", default=None, help="ETF code (default: show all in universe)")
    ps.add_argument("--state-path", type=str, default=None, help="State JSON path (default: ~/.quantlab/dividend_state.json)")
    ps.set_defaults(_run=cmd_dividend_status)
