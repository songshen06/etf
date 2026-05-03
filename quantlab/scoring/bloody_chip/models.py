from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class DimensionConfig(BaseModel):
    weight: float
    max_score: float = 10.0
    rules: Dict[str, Any] = Field(default_factory=dict)

class BloodyChipConfig(BaseModel):
    dimensions: Dict[str, DimensionConfig]

class DimensionScore(BaseModel):
    score: float
    evidence: Dict[str, Any]
    reason_codes: List[str]

class BloodyChipScoreResult(BaseModel):
    total_score: float
    scale_min: float = 0.0
    scale_max: float = 10.0
    snapshot_date: str
    etf_code: str
    dimension_scores: Dict[str, DimensionScore]
    summary: str
    category: str = "NOT_CANDIDATE"
    
    @property
    def is_bloody_chip(self) -> bool:
        return self.category in ("STANDARD_BLOODY_CHIP", "WEAK_BLOODY_CHIP")
