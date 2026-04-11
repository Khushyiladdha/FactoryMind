"""
FactoryMind Baseline Inference Script
=====================================
Mandatory stdout format:
  [START] task=<task_name> env=factory_mind model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   task=<task_name> success=<true|false> steps=<n> score=<0.XX> rewards=<r1,r2,...>

Env variables required:
  API_BASE_URL   LLM endpoint
  MODEL_NAME     Model ID
  HF_TOKEN       HF / API key (NO default)
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — EXACTLY as hackathon requires
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://cherrykl-factorymind.hf.space")

# NO default for HF_TOKEN (required by spec)
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK   = "factory_mind"
TASKS       = ["easy_reorder", "medium_spike", "hard_risk", "full_chain"]
MAX_STEPS   = 25
TEMPERATURE = 0.3
MAX_TOKENS  = 256

# ---------------------------------------------------------------------------
# OpenAI client (required by spec — must use OpenAI Client)
# ---------------------------------------------------------------------------
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# Env HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json=action,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# CRITICAL: Score sanitizer — STRICTLY between 0 and 1, NEVER 0.0 or 1.0
# ---------------------------------------------------------------------------

def safe_score(score: Optional[float]) -> float:
    """Guarantee score is strictly in (0, 1) — never 0.0, never 1.0.
    
    Uses epsilon = 0.0001 as the minimum distance from boundaries.
    After rounding, re-checks boundaries to catch floating point edge cases.
    """
    EPS = 0.0001
    if score is None:
        return 0.05
    s = float(score)
    # Handle NaN/Inf
    if s != s or s == float('inf') or s == float('-inf'):
        return 0.05
    # Clamp to strict open interval with epsilon margin
    if s <= EPS:
        return 0.05
    if s >= 1.0 - EPS:
        return 0.95
    # Round, then re-check (rounding can push to boundary)
    s = round(s, 4)
    if s <= 0.0:
        return 0.05
    if s >= 1.0:
        return 0.95
    return s


# ---------------------------------------------------------------------------
# Stdout log helpers — EXACT format from sample inference script
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_str  = "true" if done else "false"
    error_str = str(error) if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """CRITICAL: includes task= and score= fields that the validator parses."""
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    # score MUST be strictly between 0 and 1
    score = safe_score(score)
    print(
        f"[END] task={task} success={success_str} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous AI agent managing a 70MW solar panel factory.
Maximize profit by managing raw material reorders, production scheduling, and demand forecasting.

Materials: cells, glass, eva, backsheet.
Usage per MW: cells=286, glass=2.4sqm, eva=2.1sqm, backsheet=2.1sqm.
Revenue per MW: $280,000. Avoid stockouts and overstock.

Respond ONLY with valid JSON — no explanation, no markdown:
{
  "reorder": {"cells": 0, "glass": 0, "eva": 0, "backsheet": 0},
  "schedule_mw": 55.0,
  "forecast_next_3days": [160.0, 160.0, 160.0]
}"""


def build_prompt(obs: dict) -> str:
    return (
        f"Inventory: {json.dumps(obs.get('inventory', {}))}\n"
        f"Demand history (7 days): {obs.get('demand_hist', [])}\n"
        f"Capacity: {obs.get('capacity', 70)} MW\n"
        f"Costs: {json.dumps(obs.get('costs', {}))}\n"
        f"Events: {obs.get('events', [])}\n"
        f"Step: {obs.get('step_count', 0)} | Profit: ${obs.get('current_profit', 0):,.0f}\n"
        f"Task: {obs.get('task_id', '')}\n\n"
        f"Respond with JSON action only."
    )


def parse_action(text: str) -> dict:
    """Parse LLM JSON response into action dict. Robust to markdown fences."""
    text = text.strip()
    for fence in ["```json", "```"]:
        if fence in text:
            text = text.split(fence)[-1].split("```")[0].strip()
    try:
        a = json.loads(text)
        return {
            "reorder": {k: float(v) for k, v in a.get("reorder", {}).items()
                        if k in ("cells", "glass", "eva", "backsheet")},
            "schedule_mw": float(max(0, min(70, a.get("schedule_mw", 40.0)))),
            "forecast_next_3days": [float(x) for x in a.get("forecast_next_3days", [])][:3],
        }
    except Exception:
        return {"reorder": {}, "schedule_mw": 40.0, "forecast_next_3days": [160.0, 160.0, 160.0]}


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> dict:
    """Run one full episode. Returns summary dict with grader_score."""
    obs       = env_reset(task_id)
    rewards: List[float] = []
    steps     = 0
    done      = False
    grader_score: Optional[float] = None
    history: List[Dict[str, str]] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    while not done and steps < MAX_STEPS:
        steps += 1
        last_error: Optional[str] = None
        action_str = "null"

        # --- LLM call ---
        try:
            history.append({"role": "user", "content": build_prompt(obs)})
            context = history[-6:]  # keep context small

            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + context,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = (resp.choices[0].message.content or "").strip()
            history.append({"role": "assistant", "content": raw})
            action = parse_action(raw)
            action_str = json.dumps(action, separators=(",", ":"))

        except Exception as e:
            last_error = str(e)[:120]
            action = {"reorder": {}, "schedule_mw": 35.0, "forecast_next_3days": [160.0, 160.0, 160.0]}
            action_str = json.dumps(action, separators=(",", ":"))

        # --- Env step ---
        try:
            result = env_step(action)
            obs    = result["observation"]
            reward = float(result["reward"])
            done   = bool(result["done"])
            info   = result.get("info", {})
            if done and "grader_score" in info:
                grader_score = float(info["grader_score"])
        except Exception as e:
            last_error = str(e)[:120]
            reward = 0.01  # NOT 0.0 — avoid exact boundary
            done   = True
            grader_score = 0.05  # NOT 0.0 — strict (0,1)

        rewards.append(reward)
        log_step(step=steps, action=action_str, reward=reward, done=done, error=last_error)

    # --- Compute final score ---
    # Use grader_score if available, otherwise derive from rewards
    if grader_score is not None:
        final_score = safe_score(grader_score)
    else:
        # Fallback: normalize rewards to (0, 1)
        total = sum(rewards)
        if total > 0:
            final_score = safe_score(min(total / 1_000_000, 0.95))
        else:
            final_score = 0.05

    success = final_score >= 0.3

    # CRITICAL: log_end with task= and score= fields
    log_end(task=task_id, success=success, steps=steps, score=final_score, rewards=rewards)
    print("", flush=True)

    return {
        "task_id": task_id,
        "grader_score": final_score,
        "steps": steps,
        "total_reward": sum(rewards),
        "success": success,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results: List[dict] = []
    for task_id in TASKS:
        try:
            summary = run_episode(task_id)
            results.append(summary)
        except Exception as e:
            # Even on total failure, emit valid [START] and [END] with safe score
            print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)
            print(
                f"[END] task={task_id} success=false steps=0 "
                f"score=0.0500 rewards=0.00",
                flush=True,
            )
            results.append({
                "task_id": task_id,
                "grader_score": 0.05,
                "success": False,
            })
        time.sleep(1)

    # Summary table
    print("\n=== FactoryMind Baseline Results ===", flush=True)
    print(f"{'Task':<20} {'Score':>8} {'Success':>9}", flush=True)
    print("-" * 42, flush=True)
    for r in results:
        score = r.get("grader_score", 0.05)
        print(f"{r['task_id']:<20} {score:>8.4f} {str(r.get('success', False)):>9}", flush=True)


if __name__ == "__main__":
    main()