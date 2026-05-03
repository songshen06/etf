from typing import Dict
from .models import DimensionScore

def build_summary(total_score: float, category: str, dimension_scores: Dict[str, DimensionScore]) -> str:
    """
    Build a human-readable summary of the Bloody Chip Score.
    """
    parts = [f"Category: {category}", f"Total: {total_score:.2f}/10 (technical only)"]
    
    # summarize by dimension
    if "drawdown_damage" in dimension_scores:
        dd = dimension_scores["drawdown_damage"]
        parts.append(f"Drawdown Damage: {dd.score:.2f} ({', '.join(dd.reason_codes)})")
        
    if "chip_structure" in dimension_scores:
        cs = dimension_scores["chip_structure"]
        parts.append(f"CHIP: {cs.score:.2f} ({', '.join(cs.reason_codes)})")
        
    if "reversal_potential" in dimension_scores:
        rp = dimension_scores["reversal_potential"]
        parts.append(f"REV: {rp.score:.2f} ({', '.join(rp.reason_codes)})")
    
    return " | ".join(parts)
