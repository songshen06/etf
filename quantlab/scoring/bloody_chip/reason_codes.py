from enum import Enum

class ReasonCode(str, Enum):
    # Drawdown Damage
    DD_TOO_SHALLOW = "DD_TOO_SHALLOW"
    DD_MAX_DAMAGE = "DD_MAX_DAMAGE"
    DD_PARTIAL_DAMAGE = "DD_PARTIAL_DAMAGE"

    # Chip Structure
    CHIP_OVERSOLD = "CHIP_OVERSOLD"
    CHIP_NORMAL_BIAS = "CHIP_NORMAL_BIAS"
    CHIP_MOMENTUM_RECOVERING = "CHIP_MOMENTUM_RECOVERING"
    CHIP_MOMENTUM_WEAK = "CHIP_MOMENTUM_WEAK"

    # Reversal Potential
    REV_HIGH_VOLUME = "REV_HIGH_VOLUME"
    REV_NORMAL_VOLUME = "REV_NORMAL_VOLUME"
    REV_STRONG_MOMENTUM = "REV_STRONG_MOMENTUM"
    REV_WEAK_MOMENTUM = "REV_WEAK_MOMENTUM"

    # MVPs
    VAL_NO_DATA = "VAL_NO_DATA"
    SENT_NO_DATA = "SENT_NO_DATA"
    
    @classmethod
    def describe(cls, code: str) -> str:
        descriptions = {
            cls.DD_TOO_SHALLOW: "Drawdown is too shallow to be considered 'bloody'.",
            cls.DD_MAX_DAMAGE: "Drawdown is extremely deep, signaling max damage.",
            cls.DD_PARTIAL_DAMAGE: "Drawdown is significant but not maximal.",
            cls.CHIP_OVERSOLD: "Bias rate shows oversold conditions.",
            cls.CHIP_NORMAL_BIAS: "Bias rate is normal, no extreme chip compression.",
            cls.CHIP_MOMENTUM_RECOVERING: "Momentum is showing signs of recovery.",
            cls.CHIP_MOMENTUM_WEAK: "Momentum remains weak.",
            cls.REV_HIGH_VOLUME: "Volume ratio indicates a surge, possible reversal.",
            cls.REV_NORMAL_VOLUME: "Volume is normal, no strong reversal signal.",
            cls.REV_STRONG_MOMENTUM: "Recent 20d momentum is strong.",
            cls.REV_WEAK_MOMENTUM: "Recent 20d momentum is weak.",
            cls.VAL_NO_DATA: "Valuation compression is currently disabled/MVP.",
            cls.SENT_NO_DATA: "Sentiment extreme is currently disabled/MVP."
        }
        return descriptions.get(code, "Unknown reason.")
