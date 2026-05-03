from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class EtfMetadata:
    code: str
    name: str
    inception_date: datetime.date


_RAW: Dict[str, tuple[str, str]] = {
    "159209": ("红利质量 ETF（招商中证全指红利质量 ETF）", "2025-03-12"),
    "159361": ("A500ETF（易方达中证 A500ETF）", "2024-11-12"),
    "510300": ("沪深 300ETF（华泰柏瑞沪深 300ETF）", "2012-05-04"),
    "510500": ("中证 500ETF（南方中证 500ETF）", "2013-02-06"),
    "159531": ("南方中证 2000ETF", "2023-09-07"),
    "510150": ("消费 ETF（招商上证消费 80ETF）", "2010-12-08"),
    "159992": ("医疗 ETF（银华中证创新药产业 ETF）", "2020-03-20"),
    "510410": ("资源 ETF（博时上证自然资源 ETF）", "2012-04-10"),
    "588000": ("科创 50ETF（华夏上证科创板 50 成份 ETF）", "2020-09-28"),
    "513050": ("中概互联 ETF（易方达中证海外中国互联网 50ETF）", "2017-01-04"),
    "512880": ("证券 ETF（国泰中证全指证券公司 ETF）", "2016-07-26"),
    "515080": ("中证红利 ETF（招商中证红利 ETF）", "2019-11-28"),
    "511130": ("30 年国债 ETF（博时上证 30 年期国债 ETF）", "2024-03-20"),
    "518880": ("黄金 ETF（华安黄金易 ETF）", "2013-07-18"),
    "562060": ("华宝标普中国 A 股红利机会 ETF", "2023-12-08"),
    "515880": ("通信 ETF（国泰中证全指通信设备 ETF）", "2019-08-16"),
    "513500": ("标普 500ETF（博时标普 500ETF）", "2013-12-05"),
    "159501": ("嘉实纳斯达克 100ETF（QDII）", "2023-05-31"),
    "159941": ("纳指ETF", "2015-07-13"),
    "159740": ("恒生科技 ETF（大成恒生科技 ETF）", "2021-05-18"),
}


ETF_METADATA: Dict[str, EtfMetadata] = {
    code: EtfMetadata(
        code=code,
        name=name,
        inception_date=datetime.date.fromisoformat(dt),
    )
    for code, (name, dt) in _RAW.items()
}


def get_etf_metadata(code: str) -> Optional[EtfMetadata]:
    return ETF_METADATA.get(str(code))


def get_inception_date(code: str) -> Optional[datetime.date]:
    m = get_etf_metadata(code)
    return m.inception_date if m is not None else None


def get_display_name(code: str) -> Optional[str]:
    m = get_etf_metadata(code)
    return m.name if m is not None else None
