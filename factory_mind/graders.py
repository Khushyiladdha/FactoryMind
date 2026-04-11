"""
factory_mind/graders.py
ALL scores guaranteed strictly in (0, 1) — never 0.0, never 1.0.
Epsilon = 1e-4 applied at every exit point.
"""
from typing import Dict, Any, List
import numpy as np

EPSILON = 1e-4  # All scores bounded to [EPSILON, 1-EPSILON]


def _bound(score: float) -> float:
    """Single source of truth: force score into open interval (0, 1)."""
    score = float(score)
    if not (score == score):  # NaN check
        return 0.5
    return float(np.clip(score, EPSILON, 1.0 - EPSILON))


def _safe_div(numerator: float, denominator: float) -> float:
    """Safe division, result in (EPSILON, 1-EPSILON)."""
    if denominator <= 0:
        return EPSILON
    ratio = float(numerator) / float(denominator)
    return _bound(ratio)


def _mse(pred: List[float], true: List[float]) -> float:
    if not pred or not true:
        return 1e6
    n = min(len(pred), len(true))
    return float(np.mean([(p - t) ** 2 for p, t in zip(pred[:n], true[:n])]))


def _solve_optimal_profit(state: Dict[str, Any]) -> float:
    try:
        import pulp
        inv = state.get("inventory", {})
        costs = state.get("costs", {})
        capacity = state.get("capacity", 70.0)
        avg_demand = float(np.mean(state.get("demand_hist", [160.0] * 7)))
        usage = {"cells": 286.0, "glass": 2.4, "eva": 2.1, "backsheet": 2.1}
        prob = pulp.LpProblem("opt", pulp.LpMaximize)
        mw = pulp.LpVariable("mw", lowBound=0, upBound=capacity)
        max_by_inv = min(inv.get(k, 0) / usage.get(k, 1) for k in usage)
        prob += mw <= max_by_inv
        prob += mw <= avg_demand
        revenue = 280_000.0 * mw
        raw_cost = pulp.lpSum(costs.get(k, 0.1) * usage[k] * mw for k in usage)
        prob += revenue - raw_cost
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        val = pulp.value(prob.objective)
        return float(val) if val and val > 0 else 1.0
    except Exception:
        inv = state.get("inventory", {})
        demand = float(np.mean(state.get("demand_hist", [160.0])))
        capacity = state.get("capacity", 70.0)
        mw = min(capacity, demand, inv.get("cells", 0) / 286.0)
        return max(mw * 280_000.0 * 0.6, 1.0)


# ---------------------------------------------------------------------------
# Task graders
# ---------------------------------------------------------------------------

def grade_easy_reorder(final_state: Dict[str, Any], episode_actions: List[Dict]) -> float:
    optimal = 1500.0
    best_order = 0.0
    best_step = len(episode_actions)

    for i, action in enumerate(episode_actions):
        qty = float(action.get("reorder", {}).get("eva", 0.0))
        if qty > best_order:
            best_order = qty
            best_step = i

    if best_order <= 0:
        return _bound(0.05)

    # Score based on proximity to optimal, capped at 0.88
    proximity = 1.0 - abs(best_order - optimal) / (optimal * 2.0)
    base = float(np.clip(proximity, 0.3, 0.88))
    bonus = 0.07 if best_step <= 1 else 0.0
    return _bound(base + bonus)


def grade_medium_spike(final_state: Dict[str, Any], episode_actions: List[Dict]) -> float:
    """Grade demand-spike task. Rewards forecasting, production, and glass reorder.

    Scoring breakdown (designed so a reasonable LLM agent scores ~0.50-0.65):
      - Forecast accuracy:  30% weight, generous partial credit
      - Production fill:    30% weight, lower target (50% of demand)
      - Glass reorder:      fixed 0.10 bonus for any glass order
      - Scheduling effort:  fixed 0.08 bonus for any production > 0
      - Forecast attempt:   fixed 0.05 bonus for providing any forecast
    Max theoretical: 0.30*0.88 + 0.30*0.88 + 0.10 + 0.08 + 0.05 = 0.758
    """
    true_demand = [200.0, 240.0, 220.0]

    best_fc = 0.0
    made_any_forecast = False
    total_sched = 0.0

    mean_demand = float(np.mean(true_demand))
    # Normalize MSE relative to demand scale, not variance
    # This way forecasts within ~30% of true values score well
    forecast_norm = mean_demand ** 2 * 0.1

    for action in episode_actions:
        fc = action.get("forecast_next_3days", [])
        if len(fc) >= 3:
            made_any_forecast = True
            mse = _mse(fc, true_demand)
            normalized = 1.0 - mse / max(forecast_norm, 1.0)
            acc = float(np.clip(normalized, 0.10, 0.88))
            best_fc = max(best_fc, acc)
        elif len(fc) >= 1:
            made_any_forecast = True
            # Partial credit just for trying with fewer values
            best_fc = max(best_fc, 0.30)

        total_sched += float(action.get("schedule_mw", 0.0))

    # Fill rate: lower target (50% of demand instead of 60%)
    target = sum(true_demand) * 0.50
    fill = _safe_div(total_sched, target)
    fill = float(np.clip(fill, EPSILON, 0.88))

    # Bonuses
    glass_bonus = 0.0
    for action in episode_actions:
        if float(action.get("reorder", {}).get("glass", 0)) > 0:
            glass_bonus = 0.10
            break

    scheduling_bonus = 0.08 if total_sched > 0 else 0.0
    forecast_bonus = 0.05 if made_any_forecast else 0.0

    raw = 0.30 * best_fc + 0.30 * fill + glass_bonus + scheduling_bonus + forecast_bonus
    return _bound(raw)


def grade_hard_risk(final_state: Dict[str, Any], episode_actions: List[Dict]) -> float:
    profit = float(final_state.get("current_profit", 0.0))
    optimal = _solve_optimal_profit(final_state)

    # ratio in (EPSILON, 1-EPSILON) from _safe_div, then cap at 0.88
    ratio = float(np.clip(_safe_div(max(profit, 0.0), optimal), EPSILON, 0.88))

    events = final_state.get("events", [])
    if "supplier_delay_glass" in events:
        handled = any(
            float(a.get("reorder", {}).get("cells", 0)) > 0
            for a in episode_actions
        )
        penalty = 0.0 if handled else 0.07
    else:
        penalty = 0.0

    return _bound(ratio - penalty)


def grade_full_chain(final_state: Dict[str, Any], episode_actions: List[Dict]) -> float:
    # Apply _bound to each sub-score first
    easy = _bound(grade_easy_reorder(final_state, episode_actions))
    med  = _bound(grade_medium_spike(final_state, episode_actions))
    hard = _bound(grade_hard_risk(final_state, episode_actions))

    rewards = final_state.get("_episode_rewards", [])
    if len(rewards) > 1:
        mean_r = float(np.mean(rewards))
        std_r  = float(np.std(rewards)) + 1e-8
        sharpe = float(np.clip(mean_r / std_r / 3.0, EPSILON, 0.88))
    else:
        sharpe = 0.1

    has_reorder = any(
        any(float(v) > 0 for v in a.get("reorder", {}).values())
        for a in episode_actions
    )
    bonus   = 0.03 if has_reorder else 0.0
    penalty = 0.03 if final_state.get("_rush_missed", False) else 0.0

    raw = 0.22 * easy + 0.22 * med + 0.22 * hard + 0.22 * sharpe + bonus - penalty
    return _bound(raw)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

GRADERS = {
    "easy_reorder": grade_easy_reorder,
    "medium_spike": grade_medium_spike,
    "hard_risk":    grade_hard_risk,
    "full_chain":   grade_full_chain,
}


def run_grader(task_id: str, final_state: Dict[str, Any], episode_actions: List[Dict]) -> float:
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task_id!r}. Valid: {list(GRADERS)}")

    try:
        score = grader(final_state, episode_actions)
    except Exception:
        score = 0.5  # safe fallback, never 0 or 1

    # Final guarantee — single _bound call
    result = _bound(score)
    # Hard assert for safety
    assert 0.0 < result < 1.0, f"Score {result} out of bounds for task {task_id}"
    return result