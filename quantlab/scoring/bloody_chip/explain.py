from typing import Dict, Any
from pathlib import Path
from .engine import score_etf
from .config_loader import load_config
from .renderer import ExplanationRenderer
import pandas as pd

def run_explain(etf_code: str, df: pd.DataFrame, config_path: Path | None = None) -> Dict[str, Any]:
    """
    Run scoring and gather explanation data.
    """
    config = load_config(config_path)
    result = score_etf(etf_code, df, config)
    renderer = ExplanationRenderer(result)
    
    return {
        "result": result,
        "config": config,
        "narrative": renderer.get_full_narrative(),
        "category": result.category,
        "scale": {"min": result.scale_min, "max": result.scale_max}
    }
