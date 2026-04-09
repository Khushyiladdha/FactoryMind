"""
factory_mind/models.py
Typed Pydantic models for FactoryMind OpenEnv environment.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class FactoryObs(BaseModel):
    inventory: Dict[str, float] = Field(
        description="Stock levels: cells (panels), glass (sqm), eva (sqm), backsheet (sqm)"
    )
    demand_hist: List[float] = Field(
        description="Last 7 days of order demand in MW"
    )
    capacity: float = Field(ge=0, description="Daily production capacity in MW")
    costs: Dict[str, float] = Field(
        description="Per-unit costs $/unit: cells, glass, eva, backsheet, storage"
    )
    events: List[str] = Field(default_factory=list)
    step_count: int = Field(ge=0)
    current_profit: float
    task_id: str
    done: bool = Field(default=False)


class FactoryAction(BaseModel):
    reorder: Dict[str, float] = Field(default_factory=dict)
    schedule_mw: float = Field(default=0.0, ge=0.0, le=70.0)
    forecast_next_3days: List[float] = Field(default_factory=list)


class FactoryReward(BaseModel):
    """Structured reward breakdown. grader_score is ALWAYS a float, never null."""
    total: float
    profit_delta: float
    stockout_penalty: float
    overstock_penalty: float
    forecast_accuracy: float
    # CRITICAL: default 0.5 (not None) so JSON never has null score field
    grader_score: float = Field(
        default=0.5,
        description="Task grader score strictly in (0,1). 0.5 until episode ends."
    )


class EpisodeState(BaseModel):
    obs: FactoryObs
    episode_rewards: List[float] = Field(default_factory=list)
    task_id: str
    optimal_profit: Optional[float] = Field(default=None)