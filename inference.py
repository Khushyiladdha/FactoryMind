"""
inference.py — FactoryMind Baseline Inference Script
=====================================================
Mandatory stdout format:
  [START] task=<task_name> env=factory_mind model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Env variables required:
  API_BASE_URL   LLM endpoint  (default: HuggingFace router)
  MODEL_NAME     Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       HF / API key
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

MAX_STEPS_OVERRIDE = None   # None = use task default
TEMPERATURE = 0.3
MAX_TOKENS = 256
STEP_TIMEOUT = 60           # seconds per LLM call

TASKS = ["easy_reorder", "medium_spike", "hard_risk", "full_chain"]
BENCHMARK = "factory_mind"

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# Env helpers (HTTP calls to FastAPI server)
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> Dict[str, Any]:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous AI agent controlling a 70MW solar panel factory.
Your goal: maximize profit by managing raw material reorders, production scheduling, and demand forecasting.

Materials: cells (panels), glass (sqm), eva (sqm), backsheet (sqm).
Usage per MW: cells=286, glass=2.4, eva=2.1, backsheet=2.1.
Price per MW sold: $280,000. Avoid stockouts and overstock.

You MUST respond ONLY with a valid JSON object matching this schema:
{
  "reorder": {"cells": <float>, "glass": <float>, "eva": <float>, "backsheet": <float>},
  "schedule_mw": <float 0-70>,
  "forecast_next_3days": [<float>, <float>, <float>]
}
No explanation. No markdown. JSON only."""


def build_user_prompt(obs: Dict[str, Any]) -> str:
    return f"""Current observation:
Inventory: {json.dumps(obs.get('inventory', {}), indent=2)}
Demand history (last 7 days): {obs.get('demand_hist', [])}
Capacity: {obs.get('capacity', 70)} MW/day
Costs: {json.dumps(obs.get('costs', {}), indent=2)}
Active events: {obs.get('events', [])}
Step: {obs.get('step_count', 0)} | Current profit: ${obs.get('current_profit', 0):,.0f}
Task: {obs.get('task_id', '')}

Decide your action now. Respond with JSON only."""


# ---------------------------------------------------------------------------
# LLM action parser
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Dict[str, Any]:
    """Parse LLM response into action dict. Robust to markdown fences."""
    text = text.strip()
    # Strip markdown fences
    for fence in ["```json", "```"]:
        if fence in text:
            text = text.split(fence)[-1].split("```")[0].strip()
    try:
        action = json.loads(text)
        # Validate/coerce types
        reorder = {k: float(v) for k, v in action.get("reorder", {}).items()
                   if k in ("cells", "glass", "eva", "backsheet")}
        schedule_mw = float(action.get("schedule_mw", 0.0))
        forecast = [float(x) for x in action.get("forecast_next_3days", [])]
        return {
            "reorder": reorder,
            "schedule_mw": max(0.0, min(70.0, schedule_mw)),
            "forecast_next_3days": forecast[:3],
        }
    except Exception:
        # Fallback: safe default action
        return {"reorder": {}, "schedule_mw": 40.0, "forecast_next_3days": [160.0, 160.0, 160.0]}


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> Dict[str, Any]:
    """Run one full episode for a task. Returns summary dict."""
    obs = env_reset(task_id)
    rewards: List[float] = []
    step_n = 0
    done = False
    grader_score = None
    last_error = None
    action_str = "null"

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    conversation: List[Dict[str, str]] = []

    while not done:
        step_n += 1
        last_error = None

        try:
            # Build prompt
            user_msg = build_user_prompt(obs)
            conversation.append({"role": "user", "content": user_msg})

            # Truncate context to last 6 messages to stay within limits
            context = conversation[-6:]

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + context,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=STEP_TIMEOUT,
            )
            raw_text = response.choices[0].message.content or ""
            conversation.append({"role": "assistant", "content": raw_text})

            action = parse_action(raw_text)
            action_str = json.dumps(action, separators=(",", ":"))

        except Exception as e:
            last_error = str(e)[:120]
            action = {"reorder": {}, "schedule_mw": 35.0, "forecast_next_3days": [160.0, 160.0, 160.0]}
            action_str = json.dumps(action, separators=(",", ":"))

        # Step environment
        try:
            result = env_step(action)
            obs = result["observation"]
            reward = float(result["reward"])
            done = bool(result["done"])
            info = result.get("info", {})
            if done and "grader_score" in info:
                grader_score = float(info["grader_score"])
        except Exception as e:
            last_error = str(e)[:120]
            reward = 0.0
            done = True
            grader_score = 0.0

        rewards.append(reward)
        error_str = last_error if last_error else "null"
        done_str = "true" if done else "false"
        print(
            f"[STEP] step={step_n} action={action_str} reward={reward:.2f} "
            f"done={done_str} error={error_str}",
            flush=True,
        )

    success = (grader_score is not None and grader_score > 0.3)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step_n} rewards={rewards_str}", flush=True)
    print("", flush=True)

    return {
        "task_id": task_id,
        "grader_score": grader_score,
        "steps": step_n,
        "total_reward": sum(rewards),
        "success": success,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = []
    for task_id in TASKS:
        try:
            summary = run_episode(task_id)
            results.append(summary)
        except Exception as e:
            print(f"[END] success=false steps=0 rewards=", flush=True)
            results.append({"task_id": task_id, "error": str(e), "grader_score": 0.0})
        time.sleep(1)  # Brief pause between tasks

    # Summary table
    print("\n=== FactoryMind Baseline Results ===", flush=True)
    print(f"{'Task':<20} {'Grader Score':>14} {'Steps':>8} {'Success':>9}", flush=True)
    print("-" * 55, flush=True)
    for r in results:
        score = r.get("grader_score") or 0.0
        steps = r.get("steps", 0)
        success = r.get("success", False)
        print(f"{r['task_id']:<20} {score:>14.4f} {steps:>8} {str(success):>9}", flush=True)


if __name__ == "__main__":
    main()
