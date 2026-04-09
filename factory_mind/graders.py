"""
factory_mind/graders.py
Deterministic, reproducible graders for all 4 FactoryMind tasks.
Each returns a float in [0.0, 1.0].
"""
from typing import Dict, Any, List
import numpy as np


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
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


def _solve_optimal_profit(state: Dict[str, Any]) -> float:
    try:
        import pulp  # type: ignore

        inv = state.get("inventory", {})
        costs = state.get("costs", {})
        capacity = state.get("capacity", 70.0)
        demand = state.get("demand_hist", [160.0] * 7)
        avg_demand = float(np.mean(demand))

        usage = {"cells": 286.0, "glass": 2.4, "eva": 2.1, "backsheet": 2.1}
        price_per_mw = 280_000.0

        prob = pulp.LpProblem("factory_opt", pulp.LpMaximize)
        mw = pulp.LpVariable("mw", lowBound=0, upBound=capacity)

        max_by_inv = min(inv.get(k, 0) / usage.get(k, 1) for k in usage)
        prob += mw <= max_by_inv
        prob += mw <= avg_demand

        revenue = price_per_mw * mw
        raw_cost = pulp.lpSum(costs.get(k, 0.1) * usage[k] * mw for k in usage)
        prob += revenue - raw_cost

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        return max(float(pulp.value(prob.objective)), 1.0)

    except Exception:
        inv = state.get("inventory", {})
        capacity = state.get("capacity", 70.0)
        demand = float(np.mean(state.get("demand_hist", [160.0])))
        feasible_mw = min(capacity, demand, inv.get("cells", 0) / 286)
        return max(feasible_mw * 280_000.0 * 0.6, 1.0)


def grade_easy_reorder(final_state: Dict[str, Any], episode_actions: List[Dict]) -> float:
    optimal_reorder = 1500.0
    best_reorder = 0.0
    best_step = len(episode_actions)

    for i, action in enumerate(episode_actions):
        reorder = action.get("reorder", {})
        eva_order = float(reorder.get("eva", 0.0))
        if eva_order > best_reorder:
            best_reorder = eva_order
            best_step = i

    if best_reorder == 0:
        return 0.05

    # Generous: any EVA reorder scores at least 0.5
    base_score = 1.0 - abs(best_reorder - optimal_reorder) / (optimal_reorder * 2)
    base_score = float(np.clip(base_score, 0.5, 1.0))

    early_bonus = 0.1 if best_step <= 1 else 0.0

    return float(np.clip(base_score + early_bonus, 0.0, 1.0))


def grade_medium_spike(final_state: Dict[str, Any], episode_actions: List[Dict]) -> float:
    true_demand = [200.0, 240.0, 220.0]

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
        elif len(forecast) >= 1:
            # Partial credit for any forecast attempt
            best_forecast_score = max(best_forecast_score, 0.3)

        total_scheduled += float(action.get("schedule_mw", 0.0))

    # More generous fill rate: target 60% instead of 80%
    fill_rate = _safe_ratio(total_scheduled, total_demand * 0.6)
    fill_score = float(np.clip(fill_rate, 0.0, 1.0))

    # Bonus: agent ordered glass during shortage
    glass_order_bonus = 0.0
    for action in episode_actions:
        if float(action.get("reorder", {}).get("glass", 0)) > 0:
            glass_order_bonus = 0.1
            break

    return float(np.clip(0.45 * best_forecast_score + 0.45 * fill_score + glass_order_bonus, 0.0, 1.0))


def grade_hard_risk(final_state: Dict[str, Any], episode_actions: List[Dict]) -> float:
    agent_profit = float(final_state.get("current_profit", 0.0))
    optimal = _solve_optimal_profit(final_state)

    profit_ratio = _safe_ratio(max(agent_profit, 0.0), optimal)

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
    easy = grade_easy_reorder(final_state, episode_actions) * 0.25
    med = grade_medium_spike(final_state, episode_actions) * 0.25
    hard = grade_hard_risk(final_state, episode_actions) * 0.25

    rewards = final_state.get("_episode_rewards", [])
    if len(rewards) > 1:
        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards)) + 1e-8
        sharpe_raw = mean_r / std_r
        sharpe_score = float(np.clip(sharpe_raw / 3.0, 0.0, 1.0))
    else:
        sharpe_score = 0.0

    # Bonus: agent placed any reorders (shows proactive planning)
    reorder_bonus = 0.0
    for action in episode_actions:
        if any(float(v) > 0 for v in action.get("reorder", {}).values()):
            reorder_bonus = 0.05
            break

    rush_penalty = 0.05 if final_state.get("_rush_missed", False) else 0.0

    return float(np.clip(easy + med + hard + 0.25 * sharpe_score + reorder_bonus - rush_penalty, 0.0, 1.0))


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
    try:
        score = float(grader(final_state, episode_actions))
    except Exception:
        score = 0.05
    if score <= 0.0:
        score = 0.05
    if score >= 1.0:
        score = 0.99
    return round(float(np.clip(score, 0.05, 0.99)), 4)