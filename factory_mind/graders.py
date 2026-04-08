"""
factory_mind/graders.py
Deterministic, reproducible graders for all 4 FactoryMind tasks.
Each returns a float in [0.0, 1.0].
"""
from typing import Dict, Any, List
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, clamp result to [0, 1]."""
    if denominator == 0:
        return default
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _mse(pred: List[float], true: List[float]) -> float:
    if not pred or not true:
        return 1e6
    length = min(len(pred), len(true))
    return float(np.mean([(p - t) ** 2 for p, t in zip(pred[:length], true[:length])]))


def _variance(values: List[float]) -> float:
    if len(values) < 2:
        return 1.0
    return float(np.var(values))


# ---------------------------------------------------------------------------
# OR-Tools / PuLP optimal solver for hard + expert tasks
# ---------------------------------------------------------------------------

def _solve_optimal_profit(state: Dict[str, Any]) -> float:
    """
    Simple LP via PuLP: maximize revenue - costs subject to capacity + inventory.
    Returns optimal profit. Falls back to heuristic if PuLP unavailable.
    """
    try:
        import pulp  # type: ignore

        inv = state.get("inventory", {})
        costs = state.get("costs", {})
        capacity = state.get("capacity", 70.0)
        demand = state.get("demand_hist", [160.0] * 7)
        avg_demand = float(np.mean(demand))

        # Usage rates per MW: cells=286, glass=2.4sqm, eva=2.1sqm, backsheet=2.1sqm
        usage = {"cells": 286.0, "glass": 2.4, "eva": 2.1, "backsheet": 2.1}
        price_per_mw = 280_000.0  # $280k/MW revenue

        prob = pulp.LpProblem("factory_opt", pulp.LpMaximize)
        mw = pulp.LpVariable("mw", lowBound=0, upBound=capacity)

        max_by_inv = min(
            inv.get(k, 0) / usage.get(k, 1) for k in usage
        )
        prob += mw <= max_by_inv
        prob += mw <= avg_demand

        revenue = price_per_mw * mw
        raw_cost = pulp.lpSum(
            costs.get(k, 0.1) * usage[k] * mw for k in usage
        )
        prob += revenue - raw_cost

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        return max(float(pulp.value(prob.objective)), 1.0)

    except Exception:
        # Heuristic fallback
        inv = state.get("inventory", {})
        capacity = state.get("capacity", 70.0)
        demand = float(np.mean(state.get("demand_hist", [160.0])))
        feasible_mw = min(capacity, demand, inv.get("cells", 0) / 286)
        return max(feasible_mw * 280_000.0 * 0.6, 1.0)


# ---------------------------------------------------------------------------
# Task Graders
# ---------------------------------------------------------------------------

def grade_easy_reorder(final_state: Dict[str, Any], episode_actions: List[Dict]) -> float:
    """
    Task 1 — Basic Reorder.
    Score = how close the agent's EVA reorder was to the optimal 1500 units.
    Bonus for acting within first 2 steps.
    """
    optimal_reorder = 1500.0
    best_reorder = 0.0
    best_step = len(episode_actions)

    for i, action in enumerate(episode_actions):
        reorder = action.get("reorder", {})
        eva_order = float(reorder.get("eva", 0.0))
        if eva_order > best_reorder:
            best_reorder = eva_order
            best_step = i

    base_score = 1.0 - abs(best_reorder - optimal_reorder) / optimal_reorder
    base_score = float(np.clip(base_score, 0.0, 1.0))

    # Bonus: acted early
    early_bonus = 0.1 if best_step <= 1 else 0.0

    # Partial credit: any EVA order at all
    if best_reorder == 0:
        return 0.05

    return float(np.clip(base_score + early_bonus, 0.0, 1.0))


def grade_medium_spike(final_state: Dict[str, Any], episode_actions: List[Dict]) -> float:
    """
    Task 2 — Demand Spike.
    Score = forecast accuracy (50%) + schedule fill rate (50%).
    """
    true_demand = [200.0, 240.0, 220.0]

    # Best forecast across all steps
    best_forecast_score = 0.0
    total_scheduled = 0.0
    total_demand = sum(true_demand)

    for action in episode_actions:
        forecast = action.get("forecast_next_3days", [])
        if len(forecast) >= 3:
            mse = _mse(forecast, true_demand)
            var = _variance(true_demand)
            acc = float(np.clip(1.0 - mse / max(var, 1.0), 0.0, 1.0))
            best_forecast_score = max(best_forecast_score, acc)

        total_scheduled += float(action.get("schedule_mw", 0.0))

    fill_rate = _safe_ratio(total_scheduled, total_demand * 0.8)  # target 80%
    fill_score = float(np.clip(fill_rate, 0.0, 1.0))

    return float(np.clip(0.5 * best_forecast_score + 0.5 * fill_score, 0.0, 1.0))


def grade_hard_risk(final_state: Dict[str, Any], episode_actions: List[Dict]) -> float:
    """
    Task 3 — Risk Optimization.
    Score = agent_profit / or_tools_optimal_profit.
    Penalizes ignoring active events.
    """
    agent_profit = float(final_state.get("current_profit", 0.0))
    optimal = _solve_optimal_profit(final_state)

    profit_ratio = _safe_ratio(max(agent_profit, 0.0), optimal)

    # Penalize if agent never accounted for events (no reorder during delay)
    events = final_state.get("events", [])
    handled_events = 0
    for action in episode_actions:
        reorder = action.get("reorder", {})
        if "supplier_delay_glass" in events and reorder.get("cells", 0) > 0:
            handled_events += 1
            break

    event_penalty = 0.0 if handled_events > 0 or not events else 0.1

    return float(np.clip(profit_ratio - event_penalty, 0.0, 1.0))


def grade_full_chain(final_state: Dict[str, Any], episode_actions: List[Dict]) -> float:
    """
    Task 4 — Full Chain.
    Composite: 0.25 * each prior metric + 0.25 * sharpe(step_rewards).
    """
    # Partial scores from sub-tasks
    easy = grade_easy_reorder(final_state, episode_actions) * 0.25
    med = grade_medium_spike(final_state, episode_actions) * 0.25
    hard = grade_hard_risk(final_state, episode_actions) * 0.25

    # Sharpe-like score on reward trajectory
    rewards = final_state.get("_episode_rewards", [])
    if len(rewards) > 1:
        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards)) + 1e-8
        sharpe_raw = mean_r / std_r
        sharpe_score = float(np.clip(sharpe_raw / 3.0, 0.0, 1.0))  # normalise ~3 is great
    else:
        sharpe_score = 0.0

    rush_penalty = 0.05 if final_state.get("_rush_missed", False) else 0.0

    return float(np.clip(easy + med + hard + 0.25 * sharpe_score - rush_penalty, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

GRADERS = {
    "easy_reorder": grade_easy_reorder,
    "medium_spike": grade_medium_spike,
    "hard_risk": grade_hard_risk,
    "full_chain": grade_full_chain,
}


def run_grader(task_id: str, final_state: Dict[str, Any], episode_actions: List[Dict]) -> float:
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task_id!r}. Valid: {list(GRADERS)}")
    return grader(final_state, episode_actions)
