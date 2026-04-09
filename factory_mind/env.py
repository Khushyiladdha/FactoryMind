"""
factory_mind/env.py
Core FactoryMind environment: step() / reset() / state()
Simulates a 70MW solar panel factory with realistic supply-chain dynamics.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from factory_mind.models import FactoryObs, FactoryAction, FactoryReward, EpisodeState
from factory_mind.graders import run_grader

MATERIALS = ["cells", "glass", "eva", "backsheet"]

USAGE_RATE: Dict[str, float] = {
    "cells": 286.0,
    "glass": 2.4,
    "eva": 2.1,
    "backsheet": 2.1,
}

BASE_COSTS: Dict[str, float] = {
    "cells": 0.25,
    "glass": 0.08,
    "eva": 0.15,
    "backsheet": 0.10,
    "storage": 0.002,
}

PRICE_PER_MW = 280_000.0
MAX_CAPACITY = 70.0

LOW_STOCK_THRESHOLDS: Dict[str, float] = {
    "cells": 20_000,
    "glass": 8_000,
    "eva": 1_000,
    "backsheet": 1_000,
}

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy_reorder": {
        "inventory": {"cells": 50_000, "glass": 30_000, "eva": 500, "backsheet": 2_500},
        "demand_hist": [160.0, 162.0, 158.0, 161.0, 160.0, 159.0, 161.0],
        "events": [],
        "max_steps": 5,
        "cost_multiplier": 1.0,
        "demand_multiplier": 1.0,
        "rush_shipment": False,
    },
    "medium_spike": {
        "inventory": {"cells": 50_000, "glass": 20_000, "eva": 2_000, "backsheet": 2_500},
        "demand_hist": [160.0, 170.0, 180.0, 195.0, 210.0, 225.0, 240.0],
        "events": ["glass_shortage"],
        "max_steps": 10,
        "cost_multiplier": 1.0,
        "demand_multiplier": 1.2,
        "rush_shipment": False,
    },
    "hard_risk": {
        "inventory": {"cells": 45_000, "glass": 15_000, "eva": 1_800, "backsheet": 2_000},
        "demand_hist": [160.0, 155.0, 150.0, 145.0, 130.0, 128.0, 125.0],
        "events": ["supplier_delay_glass", "monsoon_dip"],
        "max_steps": 20,
        "cost_multiplier": 1.15,
        "demand_multiplier": 0.8,
        "rush_shipment": False,
    },
    "full_chain": {
        "inventory": {"cells": 40_000, "glass": 12_000, "eva": 1_500, "backsheet": 1_800},
        "demand_hist": [160.0, 155.0, 148.0, 200.0, 240.0, 220.0, 195.0],
        "events": ["supplier_delay_glass", "monsoon_dip", "rush_shipment"],
        "max_steps": 25,
        "cost_multiplier": 1.15,
        "demand_multiplier": 1.0,
        "rush_shipment": True,
    },
}


class FactoryMindEnv:
    """FactoryMind OpenEnv-compliant environment."""

    def __init__(self) -> None:
        self._state: Optional[FactoryObs] = None
        self._task_id: str = "easy_reorder"
        self._episode_actions: List[Dict[str, Any]] = []
        self._episode_rewards: List[float] = []
        self._rng: np.random.Generator = np.random.default_rng(42)
        self._rush_missed: bool = False
        self._optimal_profit: Optional[float] = None
        self._task_config: Dict[str, Any] = {}
        self._prev_forecast_error: Optional[float] = None

    def reset(self, task_id: str = "easy_reorder") -> FactoryObs:
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id {task_id!r}. Valid: {list(TASK_CONFIGS)}")

        cfg = copy.deepcopy(TASK_CONFIGS[task_id])
        self._task_id = task_id
        self._task_config = cfg
        self._episode_actions = []
        self._episode_rewards = []
        self._rush_missed = False
        self._prev_forecast_error = None

        seed = 42 + list(TASK_CONFIGS.keys()).index(task_id)
        self._rng = np.random.default_rng(seed)

        costs = {k: v * cfg["cost_multiplier"] for k, v in BASE_COSTS.items()}

        self._state = FactoryObs(
            inventory=cfg["inventory"],
            demand_hist=cfg["demand_hist"],
            capacity=MAX_CAPACITY,
            costs=costs,
            events=cfg["events"],
            step_count=0,
            current_profit=0.0,
            task_id=task_id,
            done=False,
        )
        return self._state

    def step(self, action: FactoryAction) -> Tuple[FactoryObs, float, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        cfg = self._task_config
        state = self._state
        action_dict = action.model_dump()
        self._episode_actions.append(action_dict)

        # --- 1. Reorders
        delayed_materials = set()
        for ev in state.events:
            if "delay" in ev:
                parts = ev.split("_")
                if len(parts) >= 3:
                    delayed_materials.add(parts[-1])

        new_inventory = dict(state.inventory)
        reorder_cost = 0.0
        for mat, qty in action.reorder.items():
            qty = max(0.0, float(qty))
            if mat in delayed_materials:
                qty *= 0.5
            if mat in new_inventory and mat in state.costs:
                new_inventory[mat] = new_inventory[mat] + qty
                reorder_cost += qty * state.costs.get(mat, 0.1)

        # --- 2. Production
        schedule_mw = float(action.schedule_mw)
        schedule_mw = min(schedule_mw, MAX_CAPACITY)
        for mat, rate in USAGE_RATE.items():
            max_by_mat = new_inventory.get(mat, 0.0) / rate
            schedule_mw = min(schedule_mw, max_by_mat)
        schedule_mw = max(0.0, schedule_mw)

        for mat, rate in USAGE_RATE.items():
            new_inventory[mat] = max(0.0, new_inventory.get(mat, 0.0) - schedule_mw * rate)

        # --- 3. Demand
        base_demand = cfg["demand_hist"][-1] if cfg["demand_hist"] else 160.0
        demand_mult = cfg.get("demand_multiplier", 1.0)
        if "monsoon_dip" in state.events:
            demand_mult *= 0.7
        noise = float(self._rng.normal(0, 3.0))
        realised_demand = max(0.0, base_demand * demand_mult + noise)
        new_demand_hist = list(state.demand_hist[1:]) + [realised_demand]

        # --- 4. Revenue & costs
        sold_mw = min(schedule_mw, realised_demand)
        revenue = sold_mw * PRICE_PER_MW

        storage_cost = sum(
            v * state.costs.get("storage", 0.002)
            for v in new_inventory.values()
        )

        stockout_mw = max(0.0, realised_demand - schedule_mw)
        stockout_penalty_val = stockout_mw * PRICE_PER_MW * 0.3

        rush_penalty_val = 0.0
        if "rush_shipment" in state.events and state.step_count >= cfg["max_steps"] - 3:
            if sold_mw < realised_demand * 0.9:
                rush_penalty_val = 50_000.0
                self._rush_missed = True

        profit_delta = revenue - reorder_cost - storage_cost - stockout_penalty_val - rush_penalty_val
        new_profit = state.current_profit + profit_delta

        # --- 5. Dense reward
        max_possible_profit = MAX_CAPACITY * PRICE_PER_MW
        profit_reward = float(np.clip(profit_delta / max(max_possible_profit, 1.0), -1.0, 1.0))

        stockout_ratio = stockout_mw / max(realised_demand, 1.0)
        stockout_reward = -0.2 * stockout_ratio

        overstock_units = sum(max(0.0, v - 60_000.0) for v in new_inventory.values())
        overstock_reward = -0.1 * float(np.clip(overstock_units / 100_000.0, 0.0, 1.0))

        forecast = action.forecast_next_3days
        forecast_reward = 0.0
        if len(forecast) >= 1:
            simple_true = [realised_demand] * min(3, len(forecast))
            mse = float(np.mean([(p - t) ** 2 for p, t in zip(forecast, simple_true)]))
            forecast_reward = 0.1 * float(np.clip(1.0 - mse / max(realised_demand ** 2, 1.0), 0.0, 1.0))

        # Urgency penalty: low stock on any critical material
        urgency_penalty = 0.0
        for mat, threshold in LOW_STOCK_THRESHOLDS.items():
            if new_inventory.get(mat, 0) < threshold:
                urgency_penalty -= 0.05

        # Forecast improvement bonus
        forecast_improvement = 0.0
        if len(forecast) >= 1:
            curr_forecast_error = abs(forecast[0] - realised_demand)
            if self._prev_forecast_error is not None and curr_forecast_error < self._prev_forecast_error:
                forecast_improvement = 0.1
            self._prev_forecast_error = curr_forecast_error

        # Proactive reorder bonus
        proactive_bonus = 0.0
        for mat, threshold in LOW_STOCK_THRESHOLDS.items():
            ordered = float(action.reorder.get(mat, 0))
            stock_before = state.inventory.get(mat, 0)
            if ordered > 0 and stock_before < threshold * 2:
                proactive_bonus += 0.05

        raw_reward = (
            0.4 * profit_reward
            + stockout_reward
            + overstock_reward
            + forecast_reward
            + urgency_penalty
            + forecast_improvement
            + proactive_bonus
        )
        step_reward = float(np.clip(raw_reward, -0.99, 0.99))
        if step_reward == 0.0:
            step_reward = 0.01
        step_reward = round(step_reward, 4)
        self._episode_rewards.append(step_reward)

        # --- 6. Termination
        new_step = state.step_count + 1
        done = new_step >= cfg["max_steps"]

        # --- 7. Update state
        self._state = FactoryObs(
            inventory=new_inventory,
            demand_hist=new_demand_hist,
            capacity=MAX_CAPACITY,
            costs=state.costs,
            events=state.events,
            step_count=new_step,
            current_profit=new_profit,
            task_id=self._task_id,
            done=done,
        )

        # --- 8. Info + grader
        info: Dict[str, Any] = {
            "reward_breakdown": FactoryReward(
                total=step_reward,
                profit_delta=profit_delta,
                stockout_penalty=-stockout_reward,
                overstock_penalty=-overstock_reward,
                forecast_accuracy=forecast_reward,
            ).model_dump(),
            "realised_demand": realised_demand,
            "sold_mw": sold_mw,
            "scheduled_mw": schedule_mw,
        }

        if done:
            final_state_dict = self._state.model_dump()
            final_state_dict["_episode_rewards"] = self._episode_rewards
            final_state_dict["_rush_missed"] = self._rush_missed
            grader_score = run_grader(self._task_id, final_state_dict, self._episode_actions)
            # Strictly between 0 and 1 — never exactly 0.0 or 1.0
            grader_score = float(np.clip(grader_score, 0.01, 0.99))
            info["grader_score"] = grader_score
            self._state = FactoryObs(**{**self._state.model_dump()})

        return self._state, step_reward, done, info

    def state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return EpisodeState(
            obs=self._state,
            episode_rewards=self._episode_rewards,
            task_id=self._task_id,
            optimal_profit=self._optimal_profit,
        )