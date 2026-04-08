"""
server/app.py
FastAPI server exposing the FactoryMind OpenEnv API.
Endpoints: POST /reset, POST /step, GET /state, GET /health
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional

from factory_mind.env import FactoryMindEnv
from factory_mind.models import FactoryAction

app = FastAPI(
    title="FactoryMind",
    description="70MW Solar Factory RL Environment — OpenEnv compliant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global env instance (one per worker)
# ---------------------------------------------------------------------------
_env = FactoryMindEnv()


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy_reorder"


class StepRequest(BaseModel):
    reorder: Dict[str, float] = {}
    schedule_mw: float = 0.0
    forecast_next_3days: list = []


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe — must return 200 for validator."""
    return {"status": "ok", "env": "factory-mind", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest):
    """Start a new episode. Returns initial observation."""
    valid = ["easy_reorder", "medium_spike", "hard_risk", "full_chain"]
    if req.task_id not in valid:
        raise HTTPException(status_code=400, detail=f"task_id must be one of {valid}")
    obs = _env.reset(task_id=req.task_id)
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest) -> StepResponse:
    """Execute one action in the environment."""
    try:
        action = FactoryAction(
            reorder=req.reorder,
            schedule_mw=req.schedule_mw,
            forecast_next_3days=req.forecast_next_3days,
        )
        obs, reward, done, info = _env.step(action)
        return StepResponse(
            observation=obs.model_dump(),
            reward=reward,
            done=done,
            info=info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    """Return current full episode state."""
    try:
        s = _env.state()
        return s.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def list_tasks():
    """Enumerate available tasks and their metadata."""
    return {
        "tasks": [
            {"id": "easy_reorder",  "name": "Basic Reorder",       "difficulty": "easy",   "max_steps": 5,  "target_score": 0.92},
            {"id": "medium_spike",  "name": "Demand Spike",         "difficulty": "medium", "max_steps": 10, "target_score": 0.78},
            {"id": "hard_risk",     "name": "Risk Optimization",    "difficulty": "hard",   "max_steps": 20, "target_score": 0.55},
            {"id": "full_chain",    "name": "Full Supply Chain",    "difficulty": "expert", "max_steps": 25, "target_score": 0.38},
        ]
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()