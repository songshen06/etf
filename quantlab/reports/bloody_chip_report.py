import json
import csv
from pathlib import Path
from typing import Dict, Any
from quantlab.scoring.bloody_chip.models import BloodyChipScoreResult
from quantlab.scoring.bloody_chip.renderer import ExplanationRenderer

def generate_explain_artifacts(result: BloodyChipScoreResult, output_dir: Path):
    """
    Generate the artifacts for the explain command:
    - report.md
    - score_result.json
    - dimension_scores.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. score_result.json
    json_path = output_dir / "score_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))
        
    # 2. dimension_scores.csv
    csv_path = output_dir / "dimension_scores.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dimension", "score", "reason_codes", "evidence"])
        for dim in ("drawdown_damage", "chip_structure", "reversal_potential"):
            if dim not in result.dimension_scores:
                continue
            d_score = result.dimension_scores[dim]
            writer.writerow([
                dim,
                round(d_score.score, 4),
                "|".join(d_score.reason_codes),
                json.dumps(d_score.evidence, ensure_ascii=False)
            ])
            
    # 3. report.md
    md_path = output_dir / "report.md"
    narrative = ExplanationRenderer(result).get_full_narrative()
    md_content = [
        f"# Bloody Chip Explanation Report: {result.etf_code}",
        f"**Date:** {result.snapshot_date}",
        f"**Category:** {result.category}",
        f"**Total Score:** {result.total_score:.2f} / 10.0",
        "",
        "## Summary",
        narrative.get("summary", ""),
        "",
        "## Drawdown",
        narrative.get("drawdown", ""),
        "",
        "## Reversal",
        narrative.get("reversal", ""),
        "",
        "## Dimension Scores"
    ]
    
    for dim in ("drawdown_damage", "chip_structure", "reversal_potential"):
        if dim not in result.dimension_scores:
            continue
        d_score = result.dimension_scores[dim]
        md_content.append(f"- **{dim}**: {d_score.score:.2f} ({', '.join(d_score.reason_codes) if d_score.reason_codes else 'None'})")
        
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    return {
        "json": json_path,
        "csv": csv_path,
        "md": md_path
    }
