#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 历史数据及每日增量更新脚本 (Skill 内置版)
功能：使用 akshare 获取指定 ETF 列表过去 3 年的日线行情数据，计算 MA20 和乖离率，并批量写入 SQLite 数据库。
支持初始化回填和日常增量更新。
支持：先检查缺失/异常数据（--check-only），或智能增量（--smart：仅更新需要修复或滞后的标的）。
标的列表：默认同内置 ETF_LIST；可将 etf_universe.json 放在数据库同目录或 scripts 目录，或用 --universe / ETF_UNIVERSE_PATH。
"""

import json
import sys
import pandas as pd
import sqlite3
import time
import datetime
from typing import List, Dict, Optional, Any
import os
from data_fetcher import DataFetcher
from etf_index_mapping import ETF_TO_CSINDEX_INDEX_CODE
from valuation_fetcher import fetch_cn10y_eastmoney_kline, fetch_dividend_index_indicators_multi_source, fetch_index_valuation_multi_source

# 默认在 skill 目录下的上一级目录，与 etf_monitor.py 保持一致
DB_PATH = os.environ.get("ETF_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "etf_data.db"))
START_DATE = "19000101"
END_DATE = datetime.datetime.now().strftime("%Y%m%d")

# ETF 市场代理篮子配置
MARKET_PROXY_ETFS = {
    "510300": {"name": "沪深300", "weight": 0.25},
    "510500": {"name": "中证500", "weight": 0.25},
    "159845": {"name": "中证1000", "weight": 0.20},
    "159531": {"name": "中证2000", "weight": 0.15},
    "588000": {"name": "科创50", "weight": 0.15},
}

# 目标 ETF 列表
ETF_LIST = [
    {'code': '159209', 'name': '红利质量'},
    {'code': '159361', 'name': 'A500 ETF'},
    {'code': '510300', 'name': '沪深300'},
    {'code': '510500', 'name': '中证500'},
    {'code': '159845', 'name': '中证1000'},
    {'code': '159531', 'name': '南方中证2000ETF'},
    {'code': '510150', 'name': '消费ETF'},
    {'code': '159992', 'name': '医疗ETF'},
    {'code': '510410', 'name': '资源 ETF'},
    {'code': '588000', 'name': '科创 50'},
    {'code': '513050', 'name': '中概互联'},
    {'code': '512880', 'name': '证券ETF'},
    {'code': '515080', 'name': '中证红利ETF'},
    {'code': '511130', 'name': '30年国债ETF'},
    {'code': '518880', 'name': '黄金ETF'},
    {'code': '562060', 'name': '562060 ETF'},
    {'code': '515880', 'name': '通信ETF'},
    {'code': '513500', 'name': '513500 ETF'},
    {'code': '159501', 'name': '159501 ETF'},
    {'code': '159941', 'name': '纳指ETF'},
    {'code': '516290', 'name': '光伏ETF'},
    {'code': '159740', 'name': '恒生科技ETF'},
    {'code': '159566', 'name': '储能电池'},
    {'code': '159307', 'name': '红利低波动100'}
]

ETF_INCEPTION_DATE = {
    "159209": "2025-03-12",
    "159361": "2024-11-12",
    "510300": "2012-05-04",
    "510500": "2013-02-06",
    "159845": "2020-04-10",
    "159531": "2023-09-07",
    "510150": "2010-12-08",
    "159992": "2020-03-20",
    "510410": "2012-04-10",
    "588000": "2020-09-28",
    "513050": "2017-01-04",
    "512880": "2016-07-26",
    "515080": "2019-11-28",
    "511130": "2024-03-20",
    "518880": "2013-07-18",
    "562060": "2023-12-08",
    "515880": "2019-08-16",
    "513500": "2013-12-05",
    "159501": "2023-05-31",
    "159941": "2015-07-13",
    "516290": "2021-09-17",
    "159740": "2021-05-18",
    "159566": "2025-01-27",
    "159307": "2025-01-27",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def last_expected_eod_date() -> datetime.date:
    """粗略期望库中应已覆盖到的最近交易日（仅剔除周末，不含法定节假日）。"""
    d = datetime.date.today()
    while d.weekday() >= 5:
        d -= datetime.timedelta(days=1)
    return d


def parse_trade_date(value: Any) -> datetime.date:
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    s = str(value).strip()[:10].replace("/", "-")
    if len(s) == 10 and s[4] == "-":
        y, m, dd = int(s[0:4]), int(s[5:7]), int(s[8:10])
        return datetime.date(y, m, dd)
    s8 = str(value).replace("-", "")[:8]
    y, m, dd = int(s8[0:4]), int(s8[4:6]), int(s8[6:8])
    return datetime.date(y, m, dd)

def parse_inception_date(value: Any) -> Optional[datetime.date]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s10 = s[:10].replace("/", "-")
    if len(s10) == 10 and s10[4] == "-" and s10[7] == "-":
        try:
            y, m, dd = int(s10[0:4]), int(s10[5:7]), int(s10[8:10])
            return datetime.date(y, m, dd)
        except Exception:
            return None
    return None


def get_effective_start_date(code: str) -> datetime.date:
    start_floor = datetime.datetime.strptime(START_DATE, "%Y%m%d").date()
    inc = parse_inception_date(ETF_INCEPTION_DATE.get(str(code)))
    if inc is None:
        return start_floor
    return max(start_floor, inc)


def resolve_universe_path(db_path: str) -> Optional[str]:
    env = os.environ.get("ETF_UNIVERSE_PATH")
    if env and os.path.isfile(env):
        return env
    db_dir = os.path.dirname(os.path.abspath(db_path))
    for name in ("etf_universe.json",):
        p = os.path.join(db_dir, name)
        if os.path.isfile(p):
            return p
    p2 = os.path.join(SCRIPT_DIR, "etf_universe.json")
    if os.path.isfile(p2):
        return p2
    return None


def load_etf_list(universe_path: Optional[str]) -> List[Dict]:
    if not universe_path:
        return [dict(x) for x in ETF_LIST]
    with open(universe_path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("etf_list") or data.get("etfs")
    if not items:
        print('❌ 标的文件需包含 "etf_list" 数组', file=sys.stderr)
        sys.exit(1)
    out: List[Dict] = []
    for x in items:
        code = str(x.get("code", "")).strip()
        if not code:
            continue
        out.append({"code": code, "name": str(x.get("name", code)).strip()})
    if not out:
        print("❌ 标的列表为空", file=sys.stderr)
        sys.exit(1)
    return out


def check_etf_data_status(conn: sqlite3.Connection, code: str, name: str) -> Dict[str, Any]:
    """检查单只标的：是否无数据、成交价异常、是否明显滞后于最近交易日。"""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*), MIN(trade_date), MAX(trade_date)
        FROM etf_daily_metrics
        WHERE etf_code = ?
        """,
        (code,),
    )
    row = cursor.fetchone()
    row_count = int(row[0] or 0)
    min_d = row[1]
    max_d = row[2]

    bad_first: Optional[str] = None
    bad_count = 0
    if row_count > 0:
        cursor.execute(
            """
            SELECT COUNT(*), MIN(trade_date)
            FROM etf_daily_metrics
            WHERE etf_code = ?
              AND (price IS NULL OR price <= 0)
            """,
            (code,),
        )
        br = cursor.fetchone()
        bad_count = int(br[0] or 0)
        bad_first = br[1]

    reasons: List[str] = []
    needs = False
    if row_count == 0:
        needs = True
        reasons.append("库中无记录")
    if bad_count > 0:
        needs = True
        reasons.append(f"异常 price（NULL/≤0）共 {bad_count} 条")
    if row_count > 0 and max_d is not None:
        try:
            mx = parse_trade_date(max_d)
            exp = last_expected_eod_date()
            if mx < exp:
                needs = True
                reasons.append(f"数据滞后：最新 {mx} 早于预期交易日 {exp}")
        except Exception:
            needs = True
            reasons.append(f"最新 trade_date 无法解析: {max_d!r}")
    if row_count > 0 and min_d is not None:
        try:
            mn = parse_trade_date(min_d)
            eff = get_effective_start_date(code)
            if mn > eff:
                needs = True
                reasons.append(f"历史缺失：最早 {mn} 晚于有效起点 {eff}")
        except Exception:
            needs = True
            reasons.append(f"最早 trade_date 无法解析: {min_d!r}")

    return {
        "code": code,
        "name": name,
        "row_count": row_count,
        "min_date": min_d,
        "max_date": max_d,
        "bad_price_count": bad_count,
        "bad_price_first_date": bad_first,
        "needs_update": needs,
        "reasons": reasons,
    }


def print_check_report(etf_list: List[Dict], statuses: Dict[str, Dict[str, Any]], db_path: str) -> None:
    exp = last_expected_eod_date()
    print(f"\n📋 数据库体检（期望最近交易日≈ {exp}，不含法定节假日修正）")
    print("-" * 72)
    hdr = f"{'代码':<8} {'名称':<12} {'条数':>6} {'最早':<12} {'最新':<12} {'需更新':^6} 说明"
    print(hdr)
    print("-" * 72)
    for etf in etf_list:
        st = statuses[etf["code"]]
        flag = "是" if st["needs_update"] else "否"
        rsn = "；".join(st["reasons"]) if st["reasons"] else "—"
        mn = st["min_date"] or "—"
        mx = st["max_date"] or "—"
        print(
            f"{st['code']:<8} {st['name'][:10]:<12} {st['row_count']:>6} "
            f"{str(mn):<12} {str(mx):<12} {flag:^6} {rsn}"
        )
    print("-" * 72)

    with sqlite3.connect(db_path) as _conn:
        cur = _conn.cursor()
        cur.execute("SELECT DISTINCT etf_code FROM etf_daily_metrics ORDER BY etf_code")
        in_db = {r[0] for r in cur.fetchall()}
    configured = {e["code"] for e in etf_list}
    orphan = sorted(in_db - configured)
    if orphan:
        print(f"ℹ️  库中存在但当前清单未包含的代码: {', '.join(orphan)}")


def compute_actual_start(code: str, full: bool, status: Dict[str, Any]) -> str:
    """增量起点：新表从 START_DATE；否则 max-30；若有异常 price 则从最早异常日前再推 30 天。"""
    if full:
        return get_effective_start_date(code).strftime("%Y%m%d")
    start_floor = get_effective_start_date(code)

    latest_str = get_latest_date_in_db(code)
    eff_start_str = start_floor.strftime("%Y%m%d")
    if latest_str == eff_start_str:
        base = start_floor
    else:
        latest_dt = datetime.datetime.strptime(latest_str, "%Y%m%d").date()
        base = latest_dt - datetime.timedelta(days=30)

    if status.get("bad_price_count", 0) > 0 and status.get("bad_price_first_date"):
        try:
            bd = parse_trade_date(status["bad_price_first_date"])
            base = min(base, bd - datetime.timedelta(days=30))
        except Exception:
            pass

    actual = max(start_floor, base)
    return actual.strftime("%Y%m%d")


def init_db():
    """初始化数据库表结构（如果不存在）"""
    try:
        # 确保数据目录存在
        os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # 原有 ETF 行情表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS etf_daily_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_time DATETIME NOT NULL,
                    trade_date DATE NOT NULL,
                    etf_code VARCHAR(10) NOT NULL,
                    etf_name VARCHAR(20) NOT NULL,
                    price REAL NOT NULL,
                    ma20 REAL NOT NULL,
                    bias_rate REAL NOT NULL,
                    daily_change REAL,
                    turnover REAL,
                    volume REAL,
                    open REAL,
                    high REAL,
                    low REAL,
                    prev_close REAL,
                    UNIQUE(trade_date, etf_code)
                )
            """)
            # 兼容历史库：若旧表缺少新列，则补齐（不影响既有逻辑）
            cursor.execute("PRAGMA table_info(etf_daily_metrics)")
            existing_cols = {r[1] for r in cursor.fetchall()}
            if "open" not in existing_cols:
                cursor.execute("ALTER TABLE etf_daily_metrics ADD COLUMN open REAL DEFAULT NULL")
            
            # 新增：市场指标表（保留旧表，向后兼容）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_indicator_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    indicator_name TEXT NOT NULL,
                    indicator_value REAL NOT NULL,
                    source TEXT,
                    updated_at TEXT,
                    extra_json TEXT,
                    UNIQUE(trade_date, indicator_name)
                )
            """)
            
            # 新增：ETF 市场代理表（推荐方案）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_proxy_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    proxy_name TEXT NOT NULL,
                    proxy_value REAL NOT NULL,
                    source TEXT,
                    updated_at TEXT,
                    extra_json TEXT,
                    UNIQUE(trade_date, proxy_name)
                )
            """)

            # 新增：ETF 估值表（不污染 etf_daily_metrics）
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS etf_valuation_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    etf_code TEXT NOT NULL,
                    etf_name TEXT,
                    pe_ttm REAL,
                    pe_percentile_3y REAL,
                    pe_bucket_3y TEXT,
                    dividend_yield REAL,
                    valuation_source TEXT,
                    source_index_code TEXT,
                    source_index_name TEXT,
                    raw_payload TEXT,
                    record_time TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(trade_date, etf_code)
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_etf_valuation_daily_code_date
                ON etf_valuation_daily(etf_code, trade_date)
                """
            )

            cursor.execute("PRAGMA table_info(etf_valuation_daily)")
            existing_cols = {r[1] for r in cursor.fetchall()}
            if "source_index_code" not in existing_cols:
                cursor.execute("ALTER TABLE etf_valuation_daily ADD COLUMN source_index_code TEXT DEFAULT NULL")
            if "source_index_name" not in existing_cols:
                cursor.execute("ALTER TABLE etf_valuation_daily ADD COLUMN source_index_name TEXT DEFAULT NULL")
            if "raw_payload" not in existing_cols:
                cursor.execute("ALTER TABLE etf_valuation_daily ADD COLUMN raw_payload TEXT DEFAULT NULL")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS index_valuation_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date DATE NOT NULL,
                    index_code TEXT NOT NULL,
                    index_name TEXT,
                    pe1 REAL,
                    pe2 REAL,
                    dividend_yield_1 REAL,
                    dividend_yield_2 REAL,
                    source TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(trade_date, index_code)
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS macro_rate_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date DATE NOT NULL,
                    indicator_name TEXT NOT NULL,
                    indicator_value REAL NOT NULL,
                    source TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(trade_date, indicator_name)
                )
                """
            )
            
            conn.commit()
            print(f"✅ 数据库连接/初始化成功: {DB_PATH}")
    except Exception as e:
        print(f"❌ 数据库初始化失败: {e}")


def update_etf_valuation(etf_list: List[Dict[str, Any]], *, valuation_mode: str = "standard") -> None:
    """
    可选步骤：通过 akshare 的中证指数估值接口获取估值数据，并写入 etf_valuation_daily。
    注意：当前是“ETF -> 对应指数代码”映射后把指数估值落入 ETF 估值表；主表 etf_daily_metrics 不改动。
    """
    mode = str(valuation_mode or "standard").strip().lower()
    if mode not in ("standard", "deep"):
        mode = "standard"
    deep_note_printed = False
    source_used_counts: dict[str, int] = {}

    total = 0
    ok = 0
    skipped_no_mapping = 0
    skipped_empty = 0
    failed_fetch = 0
    failed_parse = 0
    inserted_rows = 0
    updated_rows = 0
    per_etf_row_count: dict[str, int] = {}
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for etf in etf_list:
            code = str(etf.get("code") or "").strip()
            name = str(etf.get("name") or code).strip()
            if not code:
                continue
            total += 1
            idx_code = ETF_TO_CSINDEX_INDEX_CODE.get(code)
            if not idx_code:
                skipped_no_mapping += 1
                print(f"⚠️  估值跳过：未配置 ETF→指数映射: {code} ({name})", file=sys.stderr)
                continue
            try:
                df_val, source_used, note = fetch_index_valuation_multi_source(index_code=str(idx_code), mode=mode)
                if mode == "deep" and note and not deep_note_printed:
                    print(f"ℹ️  估值 deep mode: {note}", file=sys.stderr)
                    deep_note_printed = True
                source_used_counts[source_used] = int(source_used_counts.get(source_used, 0) + 1)
            except KeyError as e:
                failed_parse += 1
                print(f"❌  估值解析失败: {code} ({name}) index={idx_code} err={e}", file=sys.stderr)
                continue
            except Exception as e:
                failed_fetch += 1
                print(f"❌  估值拉取失败: {code} ({name}) index={idx_code} err={e}", file=sys.stderr)
                continue
            if df_val.empty:
                skipped_empty += 1
                print(f"⚠️  估值为空：{code} ({name}) index={idx_code}", file=sys.stderr)
                continue

            rows = []
            for _, r in df_val.iterrows():
                trade_date = str(r.get("trade_date") or "").strip()
                if not trade_date:
                    continue
                rows.append(
                    (
                        trade_date,
                        code,
                        name,
                        r.get("pe_ttm", None),
                        None,
                        None,
                        r.get("dividend_yield", None),
                        source_used,
                        r.get("source_index_code", None),
                        r.get("source_index_name", None),
                        r.get("raw_payload", None),
                    )
                )
            if not rows:
                failed_parse += 1
                print(f"⚠️  估值无有效行：{code} ({name}) index={idx_code}", file=sys.stderr)
                continue

            try:
                dates = sorted({x[0] for x in rows})
                existing: set[str] = set()
                if dates:
                    placeholders = ",".join(["?"] * len(dates))
                    cursor.execute(
                        f"SELECT trade_date FROM etf_valuation_daily WHERE etf_code = ? AND trade_date IN ({placeholders})",
                        [code, *dates],
                    )
                    existing = {str(x[0]) for x in cursor.fetchall() if x and x[0]}
                ins = int(len(dates) - len(existing))
                upd = int(len(existing))

                cursor.executemany(
                    """
                    INSERT INTO etf_valuation_daily
                        (trade_date, etf_code, etf_name, pe_ttm, pe_percentile_3y, pe_bucket_3y, dividend_yield, valuation_source, source_index_code, source_index_name, raw_payload)
                    VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(trade_date, etf_code) DO UPDATE SET
                        etf_name=excluded.etf_name,
                        pe_ttm=excluded.pe_ttm,
                        pe_percentile_3y=excluded.pe_percentile_3y,
                        pe_bucket_3y=excluded.pe_bucket_3y,
                        dividend_yield=excluded.dividend_yield,
                        valuation_source=excluded.valuation_source,
                        source_index_code=excluded.source_index_code,
                        source_index_name=excluded.source_index_name,
                        raw_payload=excluded.raw_payload
                    """,
                    rows,
                )
                conn.commit()
                ok += 1
                per_etf_row_count[code] = int(len(rows))
                inserted_rows += ins
                updated_rows += upd
                print(
                    f"✅  估值写入成功: {code} ({name}) index={idx_code} rows={len(rows)} inserted={ins} updated={upd} source={source_used}"
                )
            except Exception as e:
                failed_parse += 1
                print(f"❌  估值写入失败: {code} ({name}) index={idx_code} err={e}", file=sys.stderr)

    print(
        "🏷️  估值更新完成: "
        f"total={total} ok={ok} skipped_no_mapping={skipped_no_mapping} "
        f"skipped_empty={skipped_empty} "
        f"failed_fetch={failed_fetch} failed_parse={failed_parse} "
        f"inserted_rows={inserted_rows} updated_rows={updated_rows} source_used={source_used_counts} "
        f"per_etf_rows={per_etf_row_count}",
        file=sys.stderr,
    )


def update_dividend_valuation_sources(
    *,
    index_codes: List[str],
    cn10y_secid: str = "171.CN10Y",
    indicator_name: str = "CN10Y",
) -> None:
    idx_codes = [str(x).strip() for x in (index_codes or []) if str(x).strip()]
    if not idx_codes:
        idx_codes = ["000922"]

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        total_idx = 0
        ok_idx = 0
        idx_rows_written = 0
        for code in idx_codes:
            total_idx += 1
            df, source_used, note = fetch_dividend_index_indicators_multi_source(index_code=code)
            if df is None or getattr(df, "empty", True):
                print(f"⚠️  红利指数估值为空: index={code} source={source_used} note={note}", file=sys.stderr)
                continue

            rows = []
            for _, r in df.iterrows():
                td_raw = r.get("trade_date")
                if td_raw is None:
                    continue
                td = parse_trade_date(td_raw).isoformat()
                rows.append(
                    (
                        td,
                        str(r.get("index_code") or code),
                        r.get("index_name", None),
                        r.get("pe1", None),
                        r.get("pe2", None),
                        r.get("dividend_yield_1", None),
                        r.get("dividend_yield_2", None),
                        str(r.get("source") or source_used),
                    )
                )
            if not rows:
                print(f"⚠️  红利指数估值无有效行: index={code} source={source_used}", file=sys.stderr)
                continue

            cursor.executemany(
                """
                INSERT INTO index_valuation_daily(
                    trade_date, index_code, index_name, pe1, pe2, dividend_yield_1, dividend_yield_2, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_date, index_code) DO UPDATE SET
                    index_name=excluded.index_name,
                    pe1=excluded.pe1,
                    pe2=excluded.pe2,
                    dividend_yield_1=excluded.dividend_yield_1,
                    dividend_yield_2=excluded.dividend_yield_2,
                    source=excluded.source,
                    updated_at=CURRENT_TIMESTAMP
                """,
                rows,
            )
            conn.commit()
            ok_idx += 1
            idx_rows_written += int(len(rows))
            print(f"✅  红利指数估值写入成功: index={code} rows={len(rows)} source={source_used}")

        df_rate = fetch_cn10y_eastmoney_kline(secid=cn10y_secid)
        rate_rows_written = 0
        if df_rate is None or getattr(df_rate, "empty", True):
            print(f"⚠️  CN10Y 拉取为空: secid={cn10y_secid}", file=sys.stderr)
        else:
            rows = []
            for _, r in df_rate.iterrows():
                td_raw = r.get("trade_date")
                v = r.get("indicator_value")
                if td_raw is None or v is None:
                    continue
                td = parse_trade_date(td_raw).isoformat()
                rows.append((td, str(indicator_name), float(v), str(r.get("source") or "eastmoney_push2his")))
            if rows:
                cursor.executemany(
                    """
                    INSERT INTO macro_rate_daily(trade_date, indicator_name, indicator_value, source)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(trade_date, indicator_name) DO UPDATE SET
                        indicator_value=excluded.indicator_value,
                        source=excluded.source,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    rows,
                )
                conn.commit()
                rate_rows_written = int(len(rows))
                print(f"✅  CN10Y 写入成功: rows={rate_rows_written} source=eastmoney_push2his")

    print(
        f"🏷️  红利估值源更新完成: index_total={total_idx} index_ok={ok_idx} index_rows={idx_rows_written} "
        f"cn10y_rows={rate_rows_written}",
        file=sys.stderr,
    )

def get_latest_date_in_db(code: str) -> str:
    """获取数据库中某只 ETF 的最新数据日期"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(trade_date) FROM etf_daily_metrics WHERE etf_code = ?", (code,))
            result = cursor.fetchone()
            if result and result[0]:
                return result[0].replace('-', '')
    except Exception:
        pass
    return get_effective_start_date(code).strftime("%Y%m%d")


def compute_backfill_range(status: Dict[str, Any]) -> Optional[tuple[str, str]]:
    """若库中最早日期晚于 START_DATE，则返回需要补历史的 [start, end]（end 为最早日期前一日）。"""
    if not status.get("min_date"):
        return None
    try:
        mn = parse_trade_date(status["min_date"])
        start_floor = get_effective_start_date(str(status.get("code") or ""))
        if mn <= start_floor:
            return None
        end_dt = mn - datetime.timedelta(days=1)
        return (start_floor.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))
    except Exception:
        return None

def fetch_and_process_data(etf_info: Dict, start_date: str) -> pd.DataFrame:
    """获取并处理单只 ETF 数据 (分批次获取)"""
    return fetch_and_process_data_range(etf_info, start_date=start_date, end_date=END_DATE)


def fetch_and_process_data_range(etf_info: Dict, start_date: str, end_date: str) -> pd.DataFrame:
    """获取并处理单只 ETF 数据 (分批次获取)，支持指定结束日期（用于补历史缺口）。"""
    code = etf_info['code']
    name = etf_info['name']
    
    print(f"⏳ 正在获取 {code} ({name}) 数据 ({start_date} -> {end_date})...")
    
    all_dfs = []
    start_dt = datetime.datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.datetime.strptime(end_date, "%Y%m%d")
    span_days = int((end_dt - start_dt).days)
    batch_days = 365 if span_days >= 400 else 90
    
    # 如果已经是最新，不需要更新
    if start_dt >= end_dt:
        print(f"  -> 已是最新数据，无需更新")
        return pd.DataFrame()
        
    current_start = start_dt
    
    while current_start <= end_dt:
        current_end = min(current_start + datetime.timedelta(days=batch_days), end_dt) 
        batch_start_str = current_start.strftime("%Y%m%d")
        batch_end_str = current_end.strftime("%Y%m%d")
        
        print(f"  -> 获取批次: {batch_start_str} - {batch_end_str} ...", end="", flush=True)
        
        try:
            df_batch = DataFetcher.get_etf_hist(
                symbol=code,
                start_date=batch_start_str,
                end_date=batch_end_str
            )
            
            if df_batch is not None and not df_batch.empty:
                all_dfs.append(df_batch)
                print(" ✅")
            else:
                print(" ⚠️ (空)")
                
        except Exception as e:
            print(f" ❌ ({e})")
            
        time.sleep(1.5)
        current_start = current_end + datetime.timedelta(days=1)

    if not all_dfs:
        return pd.DataFrame()
        
    try:
        df = pd.concat(all_dfs, ignore_index=True)
        df.drop_duplicates(subset=['trade_date'], keep='last', inplace=True)
        df.sort_values('trade_date', inplace=True)
        
        # 为了计算正确的 MA20，我们不仅需要新数据，还需要获取数据库中最近的 20 天数据拼接起来计算
        # 这里为了简化和确保准确性，如果发现有新数据，最好从数据库读出历史一起算，或者直接覆盖重算
        # 作为通用脚本，直接基于抓取到的数据算 MA20（如果是增量更新且天数少于20天，前面的 MA20 会为空，
        # 所以对于增量，我们需要把 start_date 往前推 30 天来抓取，以保证 MA20 连续）
        
        df['ma20'] = df['close'].rolling(window=20).mean()
        # 修正：计算乖离率时乘以 100，使其成为真正的百分比数值
        df['bias_rate'] = (df['close'] - df['ma20']) / df['ma20'] * 100
        df['daily_change'] = df['change_pct'] / 100.0
        
        # 移除 MA20 计算产生的空值
        df.dropna(subset=['ma20', 'bias_rate'], inplace=True)
        
        # 计算昨收
        df['prev_close'] = df['close'] / (1 + df['daily_change'])
        
        return df
    except Exception as e:
        print(f"❌ {code} 数据处理失败: {e}")
        return pd.DataFrame()

def save_to_db(df: pd.DataFrame, etf_info: Dict):
    """批量写入数据库"""
    if df is None or df.empty:
        return
        
    code = etf_info['code']
    name = etf_info['name']
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            data_to_insert = []
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for _, row in df.iterrows():
                # 兼容 akshare 数据格式，处理日期字符串
                trade_date_str = row['trade_date']
                if isinstance(trade_date_str, datetime.date):
                    trade_date_str = trade_date_str.strftime('%Y-%m-%d')
                elif isinstance(trade_date_str, str) and len(trade_date_str) == 10:
                    pass # already YYYY-MM-DD
                    
                data_to_insert.append((
                    current_time,
                    trade_date_str,
                    code,
                    name,
                    row['close'],
                    row['ma20'],
                    row['bias_rate'],
                    row['daily_change'],
                    row['turnover'],
                    row['volume'],
                    row.get('open', None),
                    row['high'],
                    row['low'],
                    row['prev_close']
                ))
            
            cursor.executemany("""
                INSERT OR REPLACE INTO etf_daily_metrics 
                (record_time, trade_date, etf_code, etf_name, price, ma20, bias_rate, daily_change, turnover, volume, open, high, low, prev_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data_to_insert)
            
            conn.commit()
            print(f"✅ {code} 更新成功: 写入/更新 {len(data_to_insert)} 条数据")
            
    except Exception as e:
        print(f"❌ {code} 数据库写入失败: {e}")

def fetch_market_valuation_data() -> List[Dict]:
    """
    获取 A 股整体估值数据 (PB/PE)
    返回: 标准化的指标记录列表
    """
    import akshare as ak
    records = []
    source = "akshare"
    updated_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        print("⏳ 正在获取 A 股整体 PB 数据...")
        df_pb = ak.stock_a_all_pb()
        if df_pb is not None and not df_pb.empty:
            for _, row in df_pb.iterrows():
                try:
                    trade_date = str(row.get('日期', '')).replace('-', '')
                    if len(trade_date) == 8:
                        trade_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
                    value = float(row.get('pb', row.get('PB', 0)))
                    records.append({
                        'trade_date': trade_date,
                        'indicator_name': 'a_all_pb',
                        'indicator_value': value,
                        'source': source,
                        'updated_at': updated_at
                    })
                except Exception as e:
                    continue
        print(f"  -> PB 数据: 获取 {len([r for r in records if r['indicator_name'] == 'a_all_pb'])} 条")
    except Exception as e:
        print(f"  ⚠️ PB 数据获取失败: {e}")
    
    time.sleep(1)
    
    try:
        print("⏳ 正在获取 A 股整体 PE 数据...")
        df_pe = ak.stock_a_all_pe()
        if df_pe is not None and not df_pe.empty:
            for _, row in df_pe.iterrows():
                try:
                    trade_date = str(row.get('日期', '')).replace('-', '')
                    if len(trade_date) == 8:
                        trade_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
                    value = float(row.get('pe', row.get('PE', 0)))
                    records.append({
                        'trade_date': trade_date,
                        'indicator_name': 'a_all_pe',
                        'indicator_value': value,
                        'source': source,
                        'updated_at': updated_at
                    })
                except Exception as e:
                    continue
        print(f"  -> PE 数据: 获取 {len([r for r in records if r['indicator_name'] == 'a_all_pe'])} 条")
    except Exception as e:
        print(f"  ⚠️ PE 数据获取失败: {e}")
    
    return records

def fetch_market_activity_data() -> List[Dict]:
    """
    获取市场活跃度数据（沪深300 ETF 510300 成交额作为 proxy，使用 DataFetcher fallback 机制）
    返回: 标准化的指标记录列表
    """
    records = []
    source = "DataFetcher"
    updated_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        print("⏳ 正在获取市场活跃度数据（沪深300 ETF 510300）...")
        df_activity = DataFetcher.get_etf_hist(
            symbol="510300",
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        if df_activity is not None and not df_activity.empty:
            for _, row in df_activity.iterrows():
                try:
                    trade_date = row['trade_date']
                    if isinstance(trade_date, datetime.date):
                        trade_date = trade_date.strftime('%Y-%m-%d')
                    turnover = float(row.get('turnover', 0)) if pd.notna(row.get('turnover')) else 0.0
                    volume = float(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0.0
                    
                    if turnover > 0:
                        records.append({
                            'trade_date': trade_date,
                            'indicator_name': 'market_turnover',
                            'indicator_value': turnover,
                            'source': source,
                            'updated_at': updated_at
                        })
                    if volume > 0:
                        records.append({
                            'trade_date': trade_date,
                            'indicator_name': 'market_volume',
                            'indicator_value': volume,
                            'source': source,
                            'updated_at': updated_at
                        })
                except Exception as e:
                    continue
        print(f"  -> 活跃度数据: 获取 {len(records)} 条")
    except Exception as e:
        print(f"  ⚠️ 活跃度数据获取失败: {e}")
    
    return records

def upsert_market_indicators(conn: sqlite3.Connection, records: List[Dict]) -> int:
    """
    批量 upsert 市场指标数据
    返回: 实际写入/更新的行数
    """
    if not records:
        return 0
    
    cursor = conn.cursor()
    data_to_insert = []
    
    for r in records:
        data_to_insert.append((
            r['trade_date'],
            r['indicator_name'],
            r['indicator_value'],
            r.get('source'),
            r.get('updated_at'),
            r.get('extra_json')
        ))
    
    cursor.executemany("""
        INSERT OR REPLACE INTO market_indicator_daily
        (trade_date, indicator_name, indicator_value, source, updated_at, extra_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, data_to_insert)
    
    conn.commit()
    return len(data_to_insert)

def update_market_indicators():
    """市场指标更新主逻辑：抓取 + 入库"""
    print("\n" + "=" * 60)
    print("📊 开始更新市场指标数据")
    print("=" * 60)
    
    all_records = []
    
    # 1. 获取估值数据
    try:
        val_records = fetch_market_valuation_data()
        all_records.extend(val_records)
    except Exception as e:
        print(f"❌ 估值数据更新异常: {e}")
    
    # 2. 获取活跃度数据
    try:
        act_records = fetch_market_activity_data()
        all_records.extend(act_records)
    except Exception as e:
        print(f"❌ 活跃度数据更新异常: {e}")
    
    # 3. 入库
    if all_records:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                count = upsert_market_indicators(conn, all_records)
                print(f"\n✅ 市场指标更新完成: 写入/更新 {count} 条记录")
                
                # 统计各指标
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT indicator_name, MIN(trade_date), MAX(trade_date), COUNT(*)
                    FROM market_indicator_daily
                    GROUP BY indicator_name
                """)
                print("\n📈 指标汇总:")
                for row in cursor.fetchall():
                    print(f"  {row[0]}: {row[1]} ~ {row[2]}, 共 {row[3]} 条")
        except Exception as e:
            print(f"❌ 市场指标入库失败: {e}")
    else:
        print("⚠️  未获取到任何市场指标数据")

def compute_market_regime(conn: sqlite3.Connection, lookback_days: int = 252, trade_date: Optional[str] = None) -> Optional[Dict]:
    """
    计算市场状态 (最小实现版)
    逻辑: 基于 PB 历史百分位判断
      - pb_percentile < 0.30 -> AGGRESSIVE
      - 0.30 <= pb_percentile <= 0.70 -> BALANCED
      - pb_percentile > 0.70 -> DEFENSIVE
    """
    if trade_date is None:
        trade_date = datetime.date.today().strftime('%Y-%m-%d')
    
    cursor = conn.cursor()
    
    # 获取 PB 数据
    cursor.execute("""
        SELECT trade_date, indicator_value
        FROM market_indicator_daily
        WHERE indicator_name = 'a_all_pb'
        ORDER BY trade_date
    """)
    pb_rows = cursor.fetchall()
    
    if len(pb_rows) < 30:
        return None
    
    df_pb = pd.DataFrame(pb_rows, columns=['trade_date', 'value'])
    df_pb['value'] = pd.to_numeric(df_pb['value'], errors='coerce')
    df_pb.dropna(inplace=True)
    
    if df_pb.empty:
        return None
    
    # 找到最接近 target_date 的值
    df_pb['date_obj'] = pd.to_datetime(df_pb['trade_date'])
    target_dt = pd.to_datetime(trade_date)
    df_pb['diff'] = abs(df_pb['date_obj'] - target_dt)
    closest_idx = df_pb['diff'].idxmin()
    current_pb = df_pb.loc[closest_idx, 'value']
    
    # 计算过去 lookback_days 的百分位
    lookback_start = target_dt - datetime.timedelta(days=lookback_days)
    lookback_df = df_pb[df_pb['date_obj'] >= lookback_start].copy()
    
    if len(lookback_df) < 60:
        lookback_df = df_pb.tail(min(252, len(df_pb))).copy()
    
    lookback_df['rank'] = lookback_df['value'].rank(pct=True)
    pb_percentile = lookback_df[lookback_df['date_obj'] == df_pb.loc[closest_idx, 'date_obj']]['rank'].values[0]
    
    # 判断状态
    if pb_percentile < 0.30:
        regime = "AGGRESSIVE"
    elif pb_percentile > 0.70:
        regime = "DEFENSIVE"
    else:
        regime = "BALANCED"
    
    return {
        'trade_date': df_pb.loc[closest_idx, 'trade_date'],
        'pb_value': float(current_pb),
        'pb_percentile': float(pb_percentile),
        'market_regime': regime,
        'computed_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def compute_market_proxy_factors(conn: sqlite3.Connection) -> List[Dict]:
    """
    从现有 ETF 行情数据计算市场代理因子
    返回: 标准化的 market_proxy_daily 记录列表
    """
    records = []
    source = "etf_proxy"
    updated_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    proxy_codes = list(MARKET_PROXY_ETFS.keys())
    
    cursor = conn.cursor()
    placeholders = ','.join(['?' for _ in proxy_codes])
    
    cursor.execute(f"""
        SELECT trade_date, etf_code, price, ma20, bias_rate, daily_change, volume
        FROM etf_daily_metrics
        WHERE etf_code IN ({placeholders})
        ORDER BY trade_date, etf_code
    """, proxy_codes)
    
    rows = cursor.fetchall()
    if not rows:
        return records
    
    df_raw = pd.DataFrame(rows, columns=['trade_date', 'etf_code', 'price', 'ma20', 'bias_rate', 'daily_change', 'volume'])
    
    dates = sorted(df_raw['trade_date'].unique())
    
    for trade_date in dates:
        df_day = df_raw[df_raw['trade_date'] == trade_date].copy()
        
        if len(df_day) < 2:
            continue
        
        # 计算各因子
        composite_ret = 0.0
        composite_bias = 0.0
        momentum_score = 0.0
        breadth_score = 0.0
        total_weight = 0.0
        
        up_count = 0
        total_count = 0
        
        for _, row in df_day.iterrows():
            code = row['etf_code']
            if code not in MARKET_PROXY_ETFS:
                continue
            
            weight = MARKET_PROXY_ETFS[code]['weight']
            
            ret = float(row['daily_change'] or 0.0)
            bias_pct = float(row['bias_rate'] or 0.0)
            ma20 = float(row['ma20'] or 0.0)
            
            # 验证 MA20 可用
            if pd.isna(ma20) or ma20 <= 0:
                continue
            
            # 转换 bias：百分比 → 小数 (e.g., -5.0 → -0.05)
            bias = bias_pct / 100.0
            
            # 安全限制：bias 范围 [-0.5, 0.5]
            if bias < -0.5 or bias > 0.5:
                continue
            
            total_weight += weight
            
            composite_ret += ret * weight
            composite_bias += bias * weight
            
            # 短期动量: 基于 bias (小数)
            if bias > 0:
                momentum_score += weight
            elif bias < -0.02:
                momentum_score -= weight * 0.5
            
            # 涨跌统计
            total_count += 1
            if ret > 0:
                up_count += 1
        
        if total_weight > 0:
            composite_ret /= total_weight
            composite_bias /= total_weight
        
        # 市场广度 (涨跌比)
        if total_count > 0:
            breadth_score = up_count / total_count
        
        records.append({
            'trade_date': trade_date,
            'proxy_name': 'market_composite_ret',
            'proxy_value': composite_ret,
            'source': source,
            'updated_at': updated_at
        })
        
        records.append({
            'trade_date': trade_date,
            'proxy_name': 'market_composite_bias',
            'proxy_value': composite_bias,
            'source': source,
            'updated_at': updated_at
        })
        
        records.append({
            'trade_date': trade_date,
            'proxy_name': 'market_momentum_score',
            'proxy_value': momentum_score,
            'source': source,
            'updated_at': updated_at
        })
        
        records.append({
            'trade_date': trade_date,
            'proxy_name': 'market_breadth_score',
            'proxy_value': breadth_score,
            'source': source,
            'updated_at': updated_at
        })
    
    return records

def upsert_market_proxy(conn: sqlite3.Connection, records: List[Dict]) -> int:
    """批量 upsert 市场代理数据"""
    if not records:
        return 0
    
    cursor = conn.cursor()
    data_to_insert = []
    
    for r in records:
        data_to_insert.append((
            r['trade_date'],
            r['proxy_name'],
            r['proxy_value'],
            r.get('source'),
            r.get('updated_at'),
            r.get('extra_json')
        ))
    
    cursor.executemany("""
        INSERT OR REPLACE INTO market_proxy_daily
        (trade_date, proxy_name, proxy_value, source, updated_at, extra_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, data_to_insert)
    
    conn.commit()
    return len(data_to_insert)

def validate_market_proxy(conn: sqlite3.Connection) -> Dict[str, Any]:
    """自动验证 market_proxy 数据质量"""
    result = {
        "pass": True,
        "checks": {},
        "stats": {}
    }
    
    cursor = conn.cursor()
    
    # 验证 1: market_composite_bias 数值范围
    cursor.execute("""
        SELECT MIN(proxy_value), MAX(proxy_value), AVG(proxy_value)
        FROM market_proxy_daily
        WHERE proxy_name = 'market_composite_bias'
    """)
    row = cursor.fetchone()
    if row and row[0] is not None:
        min_bias, max_bias, avg_bias = row
        result["stats"]["market_composite_bias"] = {
            "min": min_bias,
            "max": max_bias,
            "avg": avg_bias
        }
        
        range_ok = (min_bias > -0.5) and (max_bias < 0.5)
        avg_ok = (-0.15 <= avg_bias <= 0.15)
        result["checks"]["range_check"] = range_ok and avg_ok
        if not (range_ok and avg_ok):
            result["pass"] = False
    
    # 验证 2: 无极端值 (|bias| > 1)
    cursor.execute("""
        SELECT COUNT(*)
        FROM market_proxy_daily
        WHERE proxy_name = 'market_composite_bias'
          AND ABS(proxy_value) > 1.0
    """)
    extreme_count = cursor.fetchone()[0]
    result["checks"]["no_extreme_values"] = (extreme_count == 0)
    if extreme_count > 0:
        result["pass"] = False
    
    # 验证 3: 时间连续性 (最近 30 天)
    cursor.execute("""
        SELECT trade_date
        FROM market_proxy_daily
        WHERE proxy_name = 'market_composite_bias'
        ORDER BY trade_date DESC
        LIMIT 30
    """)
    recent_dates = [r[0] for r in cursor.fetchall()]
    result["checks"]["time_continuity"] = (len(recent_dates) >= 20)
    if len(recent_dates) < 20:
        result["pass"] = False
    
    return result

def update_market_proxy():
    """市场代理数据更新主逻辑：从现有 ETF 数据计算 + 入库 + 验证"""
    print("\n" + "=" * 60)
    print("📊 开始更新 ETF 市场代理数据")
    print("=" * 60)
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # 先清空旧数据，避免混合
            cursor = conn.cursor()
            cursor.execute("DELETE FROM market_proxy_daily")
            conn.commit()
            print("ℹ️  已清空旧 market_proxy_daily 数据")
            
            records = compute_market_proxy_factors(conn)
            
            if records:
                count = upsert_market_proxy(conn, records)
                print(f"✅ 市场代理数据更新完成: 写入/更新 {count} 条记录")
                
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT proxy_name, MIN(trade_date), MAX(trade_date), COUNT(*)
                    FROM market_proxy_daily
                    GROUP BY proxy_name
                """)
                print("\n📈 代理因子汇总:")
                for row in cursor.fetchall():
                    print(f"  {row[0]}: {row[1]} ~ {row[2]}, 共 {row[3]} 条")
                
                # 执行验证
                print("\n" + "=" * 60)
                print("🔍 开始验证数据质量")
                print("=" * 60)
                validation = validate_market_proxy(conn)
                
                # 输出验证结果
                print("\n[VALIDATION RESULT]")
                if "market_composite_bias" in validation["stats"]:
                    s = validation["stats"]["market_composite_bias"]
                    print(f"\nmarket_composite_bias:")
                    print(f"  - min: {s['min']:.6f}")
                    print(f"  - max: {s['max']:.6f}")
                    print(f"  - avg: {s['avg']:.6f}")
                
                print("\nCHECK:")
                rc = "PASS" if validation["checks"].get("range_check") else "FAIL"
                print(f"  ✔ range check: {rc}")
                nev = "PASS" if validation["checks"].get("no_extreme_values") else "FAIL"
                print(f"  ✔ no extreme values: {nev}")
                tc = "PASS" if validation["checks"].get("time_continuity") else "FAIL"
                print(f"  ✔ time continuity: {tc}")
                
                print("\nFINAL STATUS:")
                print("  " + ("✅ PASS" if validation["pass"] else "❌ FAIL"))
                
            else:
                print("⚠️  未计算到任何市场代理数据（可能 ETF 数据不足）")
    except Exception as e:
        print(f"❌ 市场代理数据更新失败: {e}")

def main():
    global DB_PATH, START_DATE
    import argparse
    parser = argparse.ArgumentParser(description="ETF Database Updater")
    parser.add_argument("--full", action="store_true", help="强制全量回填 (忽略现有数据)")
    parser.add_argument("--db-path", default=DB_PATH, help="SQLite 数据库路径")
    parser.add_argument("--start-date-floor", default=START_DATE, help="全局最早抓取起点 (YYYYMMDD)")
    parser.add_argument(
        "--universe",
        default=None,
        help="标的清单 JSON（含 etf_list）；缺省则尝试数据库同目录或 scripts 下的 etf_universe.json",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅体检数据库（缺失、异常 price、滞后），不写行情；若有标的需更新则退出码 1",
    )
    parser.add_argument(
        "--smart",
        action="store_true",
        help="先体检再更新：只对「需更新」的标的拉数（仍为增量区间，非全表重下）",
    )
    parser.add_argument(
        "--backfill-missing",
        action="store_true",
        help="补齐历史缺失：若最早日期晚于「有效起点=max(START_DATE,基金成立日)」，则只抓有效起点 -> (最早日期-1) 的缺口区间",
    )
    parser.add_argument(
        "--skip-market",
        action="store_true",
        help="跳过市场指标更新（仅更新 ETF 行情）",
    )
    parser.add_argument(
        "--skip-proxy",
        action="store_true",
        help="跳过 ETF 市场代理数据更新",
    )
    parser.add_argument(
        "--update-valuation",
        action="store_true",
        help="可选：更新 ETF 估值表 etf_valuation_daily（akshare 中证指数估值；需 ETF→指数映射）",
    )
    parser.add_argument(
        "--valuation-mode",
        choices=["standard", "deep"],
        default="standard",
        help="估值更新模式：standard=仅主源；deep=预留多源/更深历史结构（当前回退到主源）",
    )
    parser.add_argument(
        "--update-dividend-valuation",
        action="store_true",
        help="可选：更新红利估值锚数据（index_valuation_daily + macro_rate_daily：中证指数 indicator xls + EastMoney CN10Y）",
    )
    parser.add_argument(
        "--dividend-index-codes",
        type=str,
        default="000922",
        help="红利估值锚：指数代码列表，逗号分隔（默认：000922）",
    )
    parser.add_argument(
        "--cn10y-secid",
        type=str,
        default="171.CN10Y",
        help="EastMoney secid for CN10Y kline (default: 171.CN10Y)",
    )
    parser.add_argument(
        "--only-dividend-valuation",
        action="store_true",
        help="只更新红利估值锚数据（index_valuation_daily + macro_rate_daily），不更新 ETF 行情/市场指标/市场代理",
    )
    args = parser.parse_args()

    DB_PATH = args.db_path
    START_DATE = str(getattr(args, "start_date_floor", START_DATE) or START_DATE).replace("-", "").strip()

    if args.universe and not os.path.isfile(args.universe):
        print(f"❌ 找不到标的文件: {args.universe}", file=sys.stderr)
        sys.exit(1)
    universe_path = args.universe or resolve_universe_path(DB_PATH)
    etf_list = load_etf_list(universe_path)

    print(f"🚀 ETF 数据库任务")
    print(f"📁 数据库路径: {DB_PATH}")
    if universe_path:
        print(f"📄 标的清单: {universe_path}")
    else:
        print("📄 标的清单: 内置默认列表")
    print("-" * 50)

    init_db()

    if getattr(args, "only_dividend_valuation", False):
        idx_codes = [x.strip() for x in str(getattr(args, "dividend_index_codes", "")).split(",") if x.strip()]
        update_dividend_valuation_sources(
            index_codes=idx_codes,
            cn10y_secid=str(getattr(args, "cn10y_secid", "171.CN10Y")),
            indicator_name="CN10Y",
        )
        print("-" * 50)
        print("🏁 红利估值锚更新完成!")
        sys.exit(0)

    with sqlite3.connect(DB_PATH) as conn:
        statuses = {etf["code"]: check_etf_data_status(conn, etf["code"], etf["name"]) for etf in etf_list}

    need_count = sum(1 for s in statuses.values() if s["needs_update"])
    if args.check_only:
        print_check_report(etf_list, statuses, DB_PATH)
        print(f"\n共 {len(etf_list)} 只标的，需更新: {need_count} 只")
        sys.exit(1 if need_count else 0)

    if args.smart:
        print_check_report(etf_list, statuses, DB_PATH)
    else:
        exp = last_expected_eod_date()
        if need_count == 0:
            print(f"📋 体检：{len(etf_list)} 只标的均正常（期望最近交易日≈ {exp}）")
        else:
            print_check_report(etf_list, statuses, DB_PATH)

    print("-" * 50)
    if args.smart:
        print("🧠 智能模式：仅更新体检未通过的标的（增量拉数）")

    for etf in etf_list:
        st = statuses[etf["code"]]
        backfill_range = compute_backfill_range(st)
        needs_backfill = backfill_range is not None
        if args.smart and not args.full and (not st["needs_update"]) and (not needs_backfill):
            print(f"⏭️  跳过 {etf['code']} ({etf['name']})：体检通过")
            continue

        if args.backfill_missing and (not args.full) and needs_backfill:
            bf_start, bf_end = backfill_range
            df = fetch_and_process_data_range(etf, start_date=bf_start, end_date=bf_end)
            save_to_db(df, etf)
            time.sleep(2)

        actual_start = compute_actual_start(etf["code"], args.full, st)
        df = fetch_and_process_data(etf, actual_start)
        save_to_db(df, etf)
        time.sleep(2)

    print("-" * 50)
    
    # 更新市场指标数据（保留旧逻辑，向后兼容）
    if not args.skip_market:
        update_market_indicators()
    
    # 更新 ETF 市场代理数据（推荐新方案）
    if not args.skip_proxy:
        update_market_proxy()

    if getattr(args, "update_valuation", False):
        print("-" * 50)
        print("🏷️  开始更新 ETF 估值数据（etf_valuation_daily）")
        update_etf_valuation(etf_list, valuation_mode=str(getattr(args, "valuation_mode", "standard")))

    if getattr(args, "update_dividend_valuation", False):
        print("-" * 50)
        print("🏷️  开始更新红利估值锚数据（index_valuation_daily + macro_rate_daily）")
        idx_codes = [x.strip() for x in str(getattr(args, "dividend_index_codes", "")).split(",") if x.strip()]
        update_dividend_valuation_sources(
            index_codes=idx_codes,
            cn10y_secid=str(getattr(args, "cn10y_secid", "171.CN10Y")),
            indicator_name="CN10Y",
        )
    
    print("-" * 50)
    print("🏁 数据库更新任务完成!")

if __name__ == "__main__":
    main()
