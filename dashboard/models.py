from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class MarketConfig(BaseModel):
    condition_id: str
    question: str
    token1: str
    token2: str
    param_type: str
    min_size: float
    max_size: float
    trade_size: float
    neg_risk: str
    max_spread: float
    # Add other fields as needed

class Config(BaseModel):
    selected_markets: List[Dict[str, Any]]
    hyperparameters: Dict[str, Dict[str, Any]]
