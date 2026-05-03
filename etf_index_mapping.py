"""
ETF -> 中证指数估值接口 symbol 映射表。

用于把“指数估值”落到 ETF 粒度的 etf_valuation_daily 表中（etf_code 仍为 ETF 代码）。

说明：
- 值为 None 表示尚未配置/不确定（估值更新会 warning 并跳过，不会中断全流程）
- 这里的指数代码用于 akshare: stock_zh_index_value_csindex(symbol=...)
"""

ETF_TO_CSINDEX_INDEX_CODE: dict[str, str | None] = {
    "510300": "000300",
    "510500": "000905",
    "515080": "000922",
    "159209": "932315",
    "159361": "000510",
}

VALUATION_SOURCE_CSINDEX = "akshare_csindex"


def valuation_source_tag(*, mode: str) -> str:
    m = str(mode).strip().lower()
    if m not in ("standard", "deep"):
        m = "standard"
    return f"{VALUATION_SOURCE_CSINDEX}_{m}"
