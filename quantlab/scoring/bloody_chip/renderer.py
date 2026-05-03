from typing import List, Dict, Any, Callable
from quantlab.scoring.bloody_chip.models import BloodyChipScoreResult, DimensionScore

class ExplanationRenderer:
    def __init__(self, result: BloodyChipScoreResult):
        self.result = result

    def render_summary_line(self) -> str:
        dd_ds = self.result.dimension_scores.get("drawdown_damage", DimensionScore(score=0, evidence={}, reason_codes=[]))
        rev_ds = self.result.dimension_scores.get("reversal_potential", DimensionScore(score=0, evidence={}, reason_codes=[]))
        dd_score = float(dd_ds.score or 0.0)
        rev_score = float(rev_ds.score or 0.0)
        cat = self.result.category
        if cat == "STANDARD_BLOODY_CHIP":
            return "标准血筹：回撤充分且技术反转条件具备，属于左侧机会（技术口径）。"
        if cat == "WEAK_BLOODY_CHIP":
            return "弱血筹：回撤达到门槛但确认度一般，偏观察/试探（技术口径）。"
        if cat == "EARLY_REVERSAL":
            if dd_score < 2.0 and rev_score >= 5.0:
                return "存在早期反转迹象，但回撤不足，未形成血筹（技术口径）。"
            return "存在早期反转迹象，仍需进一步确认（技术口径）。"
        return "非候选：回撤不足或信号缺失，不具备典型血筹机会（技术口径）。"

    def render_drawdown_section(self) -> str:
        ds = self.result.dimension_scores.get("drawdown_damage", DimensionScore(score=0, evidence={}, reason_codes=[]))
        dd_score = float(ds.score or 0.0)
        if dd_score < 2.0:
            return "回撤不足，不具备典型带血特征。"
        if dd_score < 3.0:
            return "回撤偏浅，未达到标准血筹要求。"
        if dd_score < 7.0:
            return "已出现显著杀跌，具备左侧空间。"
        return "深度杀跌已发生，左侧空间充足。"

    def render_reversal_section(self) -> str:
        ds = self.result.dimension_scores.get("reversal_potential", DimensionScore(score=0, evidence={}, reason_codes=[]))
        dd_ds = self.result.dimension_scores.get("drawdown_damage", DimensionScore(score=0, evidence={}, reason_codes=[]))
        tags = ds.reason_codes or []
        dd_score = float(dd_ds.score or 0.0)
        vol_high = "REV_HIGH_VOLUME" in tags
        mom_strong = "REV_STRONG_MOMENTUM" in tags
        mom_weak = "REV_WEAK_MOMENTUM" in tags
        prefix = "存在早期反转迹象，但未形成血筹；" if dd_score < 2.0 and (vol_high or mom_strong or float(ds.score or 0.0) >= 5.0) else ""
        if vol_high and mom_weak:
            return f"{prefix}放量但动能偏弱，趋势未确认。"
        if vol_high and mom_strong:
            return f"{prefix}放量与动能同步转强，反转确认度较高。"
        if (not vol_high) and mom_strong:
            return f"{prefix}动能改善但未放量，反转仍需确认。"
        if prefix:
            return f"{prefix}反转信号不足，仍需等待确认。"
        return "反转信号不足，仍需等待确认。"

    def get_full_narrative(self) -> Dict[str, Any]:
        return {
            "summary": self.render_summary_line(),
            "drawdown": self.render_drawdown_section(),
            "reversal": self.render_reversal_section(),
        }

def dedup_by_group(results: List[BloodyChipScoreResult], group_key: Callable[[BloodyChipScoreResult], str]) -> List[BloodyChipScoreResult]:
    best: Dict[str, BloodyChipScoreResult] = {}
    for r in results:
        g = group_key(r)
        if g not in best or r.total_score > best[g].total_score:
            best[g] = r
    return list(best.values())
