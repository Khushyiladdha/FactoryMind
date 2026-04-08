"""
inference.py — FactoryMind Baseline Inference Script

MANDATORY env vars:
  API_BASE_URL      LLM endpoint
  MODEL_NAME        Model identifier
  HF_TOKEN          HuggingFace / API key (NO default)

Stdout format (STRICT):
  [START] task=<n> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import json
import os
import time

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://cherrykl-factorymind.hf.space")

# NO default for HF_TOKEN (required by spec)
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK   = "factory-mind"          # FIX 2: match project name
TASKS       = ["easy_reorder", "medium_spike", "hard_risk", "full_chain"]
MAX_STEPS   = 25                       # FIX 3: match longest task
TEMPERATURE = 0.3
MAX_TOKENS  = 256

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# Env HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json=action,                   # FastAPI expects flat action fields directly
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Stdout log helpers — EXACT format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    done_str  = "true" if done else "false"
    error_str = str(error) if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} rewards={rewards_str}",
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
# Single episode
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> dict:
    obs     = env_reset(task_id)
    rewards = []
    steps   = 0
    done    = False
    grader  = None
    history = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    while not done and steps < MAX_STEPS:
        steps += 1
        last_error = None
        action_str = "null"

        try:
            history.append({"role": "user", "content": build_prompt(obs)})
            context = history[-4:]  # keep context small for performance

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
            last_error = str(e)[:100]
            action = {"reorder": {}, "schedule_mw": 35.0, "forecast_next_3days": [160.0, 160.0, 160.0]}
            action_str = json.dumps(action, separators=(",", ":"))

        try:
            result = env_step(action)
            obs    = result["observation"]
            reward = float(result["reward"])
            done   = bool(result["done"])
            info   = result.get("info", {})
            if done and "grader_score" in info:
                grader = float(info["grader_score"])
        except Exception as e:
            last_error = str(e)[:100]
            reward = 0.0
            done   = True
            grader = 0.0

        rewards.append(reward)
        log_step(step=steps, action=action_str, reward=reward, done=done, error=last_error)

    success = bool(done)              # FIX 1: success = task completed, not custom threshold
    log_end(success=success, steps=steps, rewards=rewards)
    print("", flush=True)

    return {"task_id": task_id, "grader_score": grader, "steps": steps, "success": success}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results = []
    for task_id in TASKS:
        try:
            results.append(run_episode(task_id))
        except Exception as e:
            log_end(success=False, steps=0, rewards=[])
            results.append({"task_id": task_id, "grader_score": 0.0, "success": False})
        time.sleep(1)

    print("\n=== FactoryMind Baseline Results ===", flush=True)
    print(f"{'Task':<20} {'Score':>8} {'Success':>9}", flush=True)
    print("-" * 42, flush=True)
    for r in results:
        score = r.get("grader_score") or 0.0
        print(f"{r['task_id']:<20} {score:>8.4f} {str(r.get('success', False)):>9}", flush=True)


if __name__ == "__main__":
    main()