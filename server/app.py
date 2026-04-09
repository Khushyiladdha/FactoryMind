"""
server/app.py
FastAPI server exposing the FactoryMind OpenEnv API.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional
import numpy as np

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

_env = FactoryMindEnv()


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


def _safe_score(v: float) -> float:
    """Ensure score is strictly between 0 and 1."""
    v = float(v)
    if v <= 0.0:
        return 0.05
    if v >= 1.0:
        return 0.99
    return round(v, 4)


def _sanitize_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure grader_score in info is strictly between 0 and 1."""
    if "grader_score" in info:
        info["grader_score"] = _safe_score(info["grader_score"])
    return info


@app.get("/health")
def health():
    return {"status": "ok", "env": "factory-mind", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = None):
    if req is None:
        req = ResetRequest()
    valid = ["easy_reorder", "medium_spike", "hard_risk", "full_chain"]
    if req.task_id not in valid:
        raise HTTPException(status_code=400, detail=f"task_id must be one of {valid}")
    obs = _env.reset(task_id=req.task_id)
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest) -> StepResponse:
    try:
        action = FactoryAction(
            reorder=req.reorder,
            schedule_mw=req.schedule_mw,
            forecast_next_3days=req.forecast_next_3days,
        )
        obs, reward, done, info = _env.step(action)

        # Sanitize grader_score strictly between 0 and 1
        info = _sanitize_info(info)

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
    try:
        s = _env.state()
        return s.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def list_tasks():
    """Enumerate tasks — scores strictly between 0 and 1."""
    return {
        "tasks": [
            {"id": "easy_reorder", "name": "Basic Reorder",    "difficulty": "easy",   "max_steps": 5,  "target_score": 0.76},
            {"id": "medium_spike", "name": "Demand Spike",     "difficulty": "medium", "max_steps": 10, "target_score": 0.50},
            {"id": "hard_risk",    "name": "Risk Optimization","difficulty": "hard",   "max_steps": 20, "target_score": 0.55},
            {"id": "full_chain",   "name": "Full Supply Chain","difficulty": "expert", "max_steps": 25, "target_score": 0.49},
        ]
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

application = app

if __name__ == "__main__":
    main()