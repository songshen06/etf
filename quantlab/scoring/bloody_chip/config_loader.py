import yaml
from pathlib import Path
from typing import Optional
from .models import BloodyChipConfig

def load_config(path: Optional[str | Path] = None) -> BloodyChipConfig:
    if path is None:
        # Default to repo root / configs / bloody_chip_etf.yaml
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        path = repo_root / "configs" / "bloody_chip_etf.yaml"
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    return BloodyChipConfig.model_validate(data)
