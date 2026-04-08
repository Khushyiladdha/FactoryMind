"""
tests/test_env.py
Full pytest suite for FactoryMind.
Run: pytest tests/ -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from factory_mind.env import FactoryMindEnv, TASK_CONFIGS, MATERIALS, USAGE_RATE
from factory_mind.models import FactoryObs, FactoryAction, FactoryReward, EpisodeState
from factory_mind.graders import run_grader, GRADERS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return FactoryMindEnv()


def heuristic_action(obs: FactoryObs) -> FactoryAction:
    """Simple deterministic agent for testing."""
    inv = obs.inventory
    reorder = {}
    thresholds = {"cells": 20_000, "glass": 10_000, "eva": 1_000, "backsheet": 1_000}
    for mat, thresh in thresholds.items():
        if inv.get(mat, 0) < thresh:
            reorder[mat] = thresh * 0.5
    last_d = obs.demand_hist[-1] if obs.demand_hist else 160.0
    return FactoryAction(
        reorder=reorder,
        schedule_mw=55.0,
        forecast_next_3days=[last_d, last_d, last_d],
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_obs_creation(self):
        obs = FactoryObs(
            inventory={"cells": 50000, "glass": 30000, "eva": 500, "backsheet": 2500},
            demand_hist=[160.0] * 7,
            capacity=70.0,
            costs={"cells": 0.25, "glass": 0.08, "eva": 0.15, "backsheet": 0.10, "storage": 0.002},
            events=[],
            step_count=0,
            current_profit=0.0,
            task_id="easy_reorder",
        )
        assert obs.step_count == 0
        assert obs.done is False

    def test_action_schedule_clamp(self):
        # schedule_mw must be 0–70
        with pytest.raises(Exception):
            FactoryAction(schedule_mw=-5.0)
        with pytest.raises(Exception):
            FactoryAction(schedule_mw=999.0)

    def test_action_defaults(self):
        action = FactoryAction(schedule_mw=40.0)
        assert action.reorder == {}
        assert action.forecast_next_3days == []

    def test_reward_model(self):
        r = FactoryReward(
            total=0.5,
            profit_delta=100.0,
            stockout_penalty=0.1,
            overstock_penalty=0.05,
            forecast_accuracy=0.2,
        )
        assert r.grader_score is None


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_all_tasks(self, env):
        for task_id in TASK_CONFIGS:
            obs = env.reset(task_id=task_id)
            assert isinstance(obs, FactoryObs)
            assert obs.task_id == task_id
            assert obs.step_count == 0
            assert obs.done is False
            assert obs.current_profit == 0.0
            assert len(obs.demand_hist) == 7
            for mat in MATERIALS:
                assert mat in obs.inventory

    def test_reset_invalid_task(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="nonexistent_task")

    def test_reset_clears_history(self, env):
        obs = env.reset("easy_reorder")
        action = heuristic_action(obs)
        env.step(action)
        # Reset again — should be clean
        obs2 = env.reset("easy_reorder")
        assert obs2.step_count == 0
        assert obs2.current_profit == 0.0

    def test_reset_reproducible(self, env):
        """Two resets of same task must yield identical initial state."""
        obs1 = env.reset("medium_spike")
        obs2 = env.reset("medium_spike")
        assert obs1.inventory == obs2.inventory
        assert obs1.demand_hist == obs2.demand_hist
        assert obs1.events == obs2.events


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_before_reset_raises(self, env):
        action = FactoryAction(schedule_mw=40.0)
        with pytest.raises(RuntimeError, match="reset"):
            env.step(action)

    def test_step_returns_correct_types(self, env):
        obs = env.reset("easy_reorder")
        action = heuristic_action(obs)
        new_obs, reward, done, info = env.step(action)
        assert isinstance(new_obs, FactoryObs)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_increments_step_count(self, env):
        obs = env.reset("easy_reorder")
        for i in range(3):
            if obs.done:
                break
            action = heuristic_action(obs)
            obs, _, done, _ = env.step(action)

    def test_step_after_done_raises(self, env):
        obs = env.reset("easy_reorder")
        # Run to completion
        for _ in range(10):
            if obs.done:
                break
            action = heuristic_action(obs)
            obs, _, done, _ = env.step(action)
        # Now try to step again — should raise
        with pytest.raises(RuntimeError, match="done"):
            env.step(FactoryAction(schedule_mw=40.0))

    def test_reward_in_valid_range(self, env):
        """Step rewards should be in a reasonable range."""
        obs = env.reset("medium_spike")
        for _ in range(5):
            if obs.done:
                break
            action = heuristic_action(obs)
            obs, reward, done, info = env.step(action)
            assert -2.0 <= reward <= 2.0, f"Reward out of range: {reward}"

    def test_inventory_never_negative(self, env):
        """Production must never drive inventory below zero."""
        obs = env.reset("hard_risk")
        for _ in range(25):
            if obs.done:
                break
            # Aggressive scheduling to stress inventory
            action = FactoryAction(schedule_mw=70.0)
            obs, _, done, _ = env.step(action)
            for mat in MATERIALS:
                assert obs.inventory.get(mat, 0) >= 0.0, f"{mat} went negative"

    def test_info_has_reward_breakdown(self, env):
        obs = env.reset("easy_reorder")
        action = heuristic_action(obs)
        _, _, _, info = env.step(action)
        assert "reward_breakdown" in info
        rb = info["reward_breakdown"]
        assert "total" in rb
        assert "profit_delta" in rb

    def test_grader_score_on_done(self, env):
        """grader_score must appear in info when episode ends."""
        obs = env.reset("easy_reorder")
        last_info = {}
        for _ in range(10):
            if obs.done:
                break
            action = heuristic_action(obs)
            obs, _, done, last_info = env.step(action)
        assert "grader_score" in last_info
        score = last_info["grader_score"]
        assert 0.0 <= score <= 1.0, f"Grader score out of range: {score}"

    def test_demand_hist_length_constant(self, env):
        """demand_hist must always stay length 7."""
        obs = env.reset("medium_spike")
        for _ in range(8):
            if obs.done:
                break
            action = heuristic_action(obs)
            obs, _, done, _ = env.step(action)
            assert len(obs.demand_hist) == 7

    def test_zero_schedule_no_crash(self, env):
        """Zero production action must be handled gracefully."""
        obs = env.reset("easy_reorder")
        action = FactoryAction(schedule_mw=0.0, reorder={}, forecast_next_3days=[])
        new_obs, reward, done, info = env.step(action)
        assert isinstance(reward, float)

    def test_events_propagate(self, env):
        """Events must persist across steps."""
        obs = env.reset("hard_risk")
        initial_events = set(obs.events)
        for _ in range(5):
            if obs.done:
                break
            action = heuristic_action(obs)
            obs, _, _, _ = env.step(action)
        assert set(obs.events) == initial_events


# ---------------------------------------------------------------------------
# Full episode tests
# ---------------------------------------------------------------------------

class TestFullEpisode:
    @pytest.mark.parametrize("task_id", list(TASK_CONFIGS.keys()))
    def test_full_episode_completes(self, env, task_id):
        """Every task must complete within max_steps with a valid grader score."""
        obs = env.reset(task_id)
        cfg = TASK_CONFIGS[task_id]
        grader_score = None
        steps = 0

        for _ in range(cfg["max_steps"] + 5):  # +5 safety margin
            if obs.done:
                break
            action = heuristic_action(obs)
            obs, reward, done, info = env.step(action)
            steps += 1
            if done:
                grader_score = info.get("grader_score")
                break

        assert obs.done, f"Episode did not finish for {task_id}"
        assert grader_score is not None, f"No grader_score for {task_id}"
        assert 0.0 <= grader_score <= 1.0

    @pytest.mark.parametrize("task_id", list(TASK_CONFIGS.keys()))
    def test_episode_reproducible(self, task_id):
        """Two identical runs must yield identical grader scores."""
        scores = []
        for _ in range(2):
            e = FactoryMindEnv()
            obs = e.reset(task_id)
            gs = None
            cfg = TASK_CONFIGS[task_id]
            for _ in range(cfg["max_steps"] + 5):
                if obs.done:
                    break
                action = heuristic_action(obs)
                obs, _, done, info = e.step(action)
                if done:
                    gs = info.get("grader_score")
                    break
            scores.append(gs)
        assert scores[0] == scores[1], f"Non-deterministic: {scores}"


# ---------------------------------------------------------------------------
# State API tests
# ---------------------------------------------------------------------------

class TestStateAPI:
    def test_state_before_reset_raises(self, env):
        with pytest.raises(RuntimeError):
            env.state()

    def test_state_returns_episode_state(self, env):
        env.reset("easy_reorder")
        s = env.state()
        assert isinstance(s, EpisodeState)
        assert s.task_id == "easy_reorder"
        assert isinstance(s.obs, FactoryObs)

    def test_state_tracks_rewards(self, env):
        obs = env.reset("medium_spike")
        for _ in range(3):
            if obs.done:
                break
            obs, _, _, _ = env.step(heuristic_action(obs))
        s = env.state()
        assert len(s.episode_rewards) == 3


# ---------------------------------------------------------------------------
# Grader unit tests
# ---------------------------------------------------------------------------

class TestGraders:
    def test_all_graders_exist(self):
        for task_id in TASK_CONFIGS:
            assert task_id in GRADERS, f"Missing grader for {task_id}"

    def test_grader_score_range(self):
        """All graders must return values in [0, 1]."""
        dummy_state = {
            "inventory": {"cells": 20000, "glass": 8000, "eva": 800, "backsheet": 1000},
            "demand_hist": [160.0] * 7,
            "capacity": 70.0,
            "costs": {"cells": 0.25, "storage": 0.002},
            "events": [],
            "current_profit": 500_000.0,
            "_episode_rewards": [0.1, 0.2, 0.15],
            "_rush_missed": False,
        }
        dummy_actions = [
            {"reorder": {"eva": 1500}, "schedule_mw": 55.0, "forecast_next_3days": [160, 165, 162]},
            {"reorder": {}, "schedule_mw": 60.0, "forecast_next_3days": [200, 240, 220]},
        ]
        for task_id in GRADERS:
            score = run_grader(task_id, dummy_state, dummy_actions)
            assert 0.0 <= score <= 1.0, f"{task_id} grader returned {score}"

    def test_easy_reorder_perfect_action(self):
        """Perfect EVA reorder should score near 1.0."""
        state = {
            "inventory": {"cells": 50000, "glass": 30000, "eva": 500, "backsheet": 2500},
            "demand_hist": [160.0] * 7,
            "capacity": 70.0,
            "costs": {},
            "events": [],
            "current_profit": 0.0,
            "_episode_rewards": [],
            "_rush_missed": False,
        }
        actions = [{"reorder": {"eva": 1500.0}, "schedule_mw": 55.0, "forecast_next_3days": []}]
        score = run_grader("easy_reorder", state, actions)
        assert score >= 0.85, f"Perfect action should score >= 0.85, got {score}"

    def test_easy_reorder_no_action(self):
        """No reorder should score near 0."""
        state = {
            "inventory": {"cells": 50000, "glass": 30000, "eva": 500, "backsheet": 2500},
            "demand_hist": [160.0] * 7,
            "capacity": 70.0,
            "costs": {},
            "events": [],
            "current_profit": 0.0,
            "_episode_rewards": [],
            "_rush_missed": False,
        }
        actions = [{"reorder": {}, "schedule_mw": 55.0, "forecast_next_3days": []}]
        score = run_grader("easy_reorder", state, actions)
        assert score <= 0.15, f"No reorder should score <= 0.15, got {score}"

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            run_grader("nonexistent", {}, [])

    def test_grader_deterministic(self):
        """Same inputs must always return the same score."""
        state = {
            "inventory": {"cells": 30000, "glass": 12000, "eva": 1000, "backsheet": 1500},
            "demand_hist": [155.0] * 7,
            "capacity": 70.0,
            "costs": {"cells": 0.25, "storage": 0.002},
            "events": ["supplier_delay_glass"],
            "current_profit": 200_000.0,
            "_episode_rewards": [0.1, 0.05, 0.08],
            "_rush_missed": False,
        }
        actions = [{"reorder": {"cells": 5000}, "schedule_mw": 50.0, "forecast_next_3days": [150, 145, 130]}]
        scores = [run_grader("hard_risk", state, actions) for _ in range(5)]
        assert len(set(scores)) == 1, f"Non-deterministic grader: {scores}"


# ---------------------------------------------------------------------------
# Usage-rate sanity tests
# ---------------------------------------------------------------------------

class TestUsageRates:
    def test_materials_complete(self):
        for mat in MATERIALS:
            assert mat in USAGE_RATE, f"Missing usage rate for {mat}"

    def test_usage_rates_positive(self):
        for mat, rate in USAGE_RATE.items():
            assert rate > 0, f"Usage rate for {mat} must be positive"
