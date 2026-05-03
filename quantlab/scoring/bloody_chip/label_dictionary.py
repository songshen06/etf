from typing import Dict

# 标签字典：将机器标签映射为人话短语和情绪极性
# 格式: { "TAG": {"text": "中文人话", "sentiment": "positive/negative/neutral"} }
LABEL_DICTIONARY: Dict[str, Dict[str, str]] = {
    # Drawdown Damage
    "DD_MAX_DAMAGE": {
        "text": "经历了极度惨烈的深度回撤，价格已处于绝对底部区域",
        "sentiment": "positive" # 对于买入机会而言是正面的
    },
    "DD_PARTIAL_DAMAGE": {
        "text": "经历了较深幅度的回撤，具备一定的跌出来的空间",
        "sentiment": "positive"
    },
    "DD_TOO_SHALLOW": {
        "text": "近期回撤幅度较浅，尚未形成明显的超跌血筹特征",
        "sentiment": "negative"
    },

    # Chip Structure
    "CHIP_OVERSOLD": {
        "text": "筹码结构呈现极度超卖状态",
        "sentiment": "positive"
    },
    "CHIP_NORMAL_BIAS": {
        "text": "筹码乖离率处于正常区间",
        "sentiment": "neutral"
    },
    "CHIP_MOMENTUM_RECOVERING": {
        "text": "并且动能已经开始出现企稳修复的迹象",
        "sentiment": "positive"
    },
    "CHIP_MOMENTUM_WEAK": {
        "text": "但短期动能依然疲弱，尚未见底企稳",
        "sentiment": "negative"
    },

    # Reversal Potential
    "REV_HIGH_VOLUME": {
        "text": "近期出现显著的底部放量特征，有资金进场异动",
        "sentiment": "positive"
    },
    "REV_NORMAL_VOLUME": {
        "text": "近期成交量平淡，未见明显的底部资金抢筹迹象",
        "sentiment": "neutral"
    },
    "REV_STRONG_MOMENTUM": {
        "text": "且短期反转势头强劲",
        "sentiment": "positive"
    },
    "REV_WEAK_MOMENTUM": {
        "text": "且反转动能匮乏，缺乏向上催化",
        "sentiment": "negative"
    }
}
