"""
test_local.py — smoke test all 4 tasks without an LLM.
Run: python test_local.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from factory_mind.env import FactoryMindEnv
from factory_mind.models import FactoryAction

TASKS = ["easy_reorder", "medium_spike", "hard_risk", "full_chain"]


def heuristic_action(obs) -> FactoryAction:
    """Simple rule-based agent for smoke testing."""
    inv = obs.inventory
    # Reorder anything below threshold
    reorder = {}
    thresholds = {"cells": 20_000, "glass": 10_000, "eva": 1_000, "backsheet": 1_000}
    for mat, thresh in thresholds.items():
        if inv.get(mat, 0) < thresh:
            reorder[mat] = thresh * 0.5

    # Schedule ~80% of capacity
    schedule = 55.0

    # Naive forecast: repeat last demand
    last_d = obs.demand_hist[-1] if obs.demand_hist else 160.0
    forecast = [last_d, last_d, last_d]

    return FactoryAction(
        reorder=reorder,
        schedule_mw=schedule,
        forecast_next_3days=forecast,
    )


def run_task(task_id: str):
    env = FactoryMindEnv()
    obs = env.reset(task_id=task_id)
    print(f"\n{'='*50}")
    print(f"Task: {task_id}")
    print(f"  Initial EVA: {obs.inventory.get('eva', 0):.0f}  Events: {obs.events}")

    rewards = []
    grader_score = None

    for step in range(30):
        if obs.done:
            break
        action = heuristic_action(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            grader_score = info.get("grader_score")
            break

    total = sum(rewards)
    print(f"  Steps: {len(rewards)} | Total reward: {total:.4f} | Grader: {grader_score:.4f}")
    assert grader_score is not None, "Grader score must be set on done!"
    assert 0.0 <= grader_score <= 1.0, f"Grader out of range: {grader_score}"
    print(f"  ✅ PASS — grader={grader_score:.4f}")
    return grader_score


if __name__ == "__main__":
    scores = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as e:
            print(f"  ❌ FAIL — {task_id}: {e}")
            import traceback; traceback.print_exc()
            scores[task_id] = 0.0

    print(f"\n{'='*50}")
    print("Summary:")
    for tid, sc in scores.items():
        status = "✅" if sc > 0 else "❌"
        print(f"  {status} {tid:<20} {sc:.4f}")

    all_pass = all(s > 0 for s in scores.values())
    if all_pass:
        print("\n🎉 All tasks passed smoke test!")
    else:
        print("\n⚠️  Some tasks failed — check above.")
        sys.exit(1)
