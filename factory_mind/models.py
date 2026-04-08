"""
factory_mind/models.py
Typed Pydantic models for FactoryMind OpenEnv environment.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class FactoryObs(BaseModel):
    """Observation returned by reset() and step()."""
    inventory: Dict[str, float] = Field(
        description="Stock levels: cells (panels), glass (sqm), eva (sqm), backsheet (sqm)"
    )
    demand_hist: List[float] = Field(
        description="Last 7 days of order demand in MW"
    )
    capacity: float = Field(
        ge=0, description="Daily production capacity in MW"
    )
    costs: Dict[str, float] = Field(
        description="Per-unit costs $/unit: cells, glass, eva, backsheet, storage"
    )
    events: List[str] = Field(
        default_factory=list,
        description="Active disruption events e.g. ['supplier_delay_glass', 'monsoon_dip']"
    )
    step_count: int = Field(ge=0, description="Current step in episode")
    current_profit: float = Field(description="Cumulative profit this episode")
    task_id: str = Field(description="Active task identifier")
    done: bool = Field(default=False, description="Whether episode has ended")


class FactoryAction(BaseModel):
    """Action submitted by the agent each step."""
    reorder: Dict[str, float] = Field(
        default_factory=dict,
        description="Material quantities to order this step e.g. {'cells': 10000}"
    )
    schedule_mw: float = Field(
        default=0.0, ge=0.0, le=70.0,
        description="MW of panels to produce this step"
    )
    forecast_next_3days: List[float] = Field(
        default_factory=list,
        description="Agent demand forecast for next 3 days in MW e.g. [160, 180, 150]"
    )


class FactoryReward(BaseModel):
    """Structured reward breakdown returned in step info."""
    total: float = Field(description="Total reward this step (0.0–1.0 range for grader)")
    profit_delta: float = Field(description="Profit change component")
    stockout_penalty: float = Field(description="Penalty for stockouts")
    overstock_penalty: float = Field(description="Penalty for excess inventory costs")
    forecast_accuracy: float = Field(description="Reward for accurate demand forecast")
    grader_score: Optional[float] = Field(
        default=None,
        description="Final task grader score 0.0–1.0, set on episode end"
    )


class EpisodeState(BaseModel):
    """Full internal state — returned by state() endpoint."""
    obs: FactoryObs
    episode_rewards: List[float] = Field(default_factory=list)
    task_id: str
    optimal_profit: Optional[float] = Field(
        default=None, description="OR-Tools optimal for hard/expert tasks"
    )
