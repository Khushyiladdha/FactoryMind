"""
tests/test_api.py
Integration tests for the FastAPI server endpoints.
Run: pytest tests/test_api.py -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from fastapi.testclient import TestClient
    from server.app import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi/httpx not installed")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_body(self, client):
        r = client.get("/health")
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestResetEndpoint:
    def test_reset_easy(self, client):
        r = client.post("/reset", json={"task_id": "easy_reorder"})
        assert r.status_code == 200
        data = r.json()
        assert data["task_id"] == "easy_reorder"
        assert data["step_count"] == 0
        assert data["done"] is False
        assert "inventory" in data
        assert "demand_hist" in data

    @pytest.mark.parametrize("task_id", ["easy_reorder", "medium_spike", "hard_risk", "full_chain"])
    def test_reset_all_tasks(self, client, task_id):
        r = client.post("/reset", json={"task_id": task_id})
        assert r.status_code == 200
        assert r.json()["task_id"] == task_id

    def test_reset_invalid_task(self, client):
        r = client.post("/reset", json={"task_id": "fake_task"})
        assert r.status_code == 400

    def test_reset_default_task(self, client):
        r = client.post("/reset", json={})
        assert r.status_code == 200
        assert r.json()["task_id"] == "easy_reorder"


class TestStepEndpoint:
    def test_step_after_reset(self, client):
        client.post("/reset", json={"task_id": "easy_reorder"})
        r = client.post("/step", json={
            "reorder": {"eva": 1500.0},
            "schedule_mw": 55.0,
            "forecast_next_3days": [160.0, 162.0, 161.0],
        })
        assert r.status_code == 200
        data = r.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data

    def test_step_reward_is_float(self, client):
        client.post("/reset", json={"task_id": "easy_reorder"})
        r = client.post("/step", json={"schedule_mw": 40.0})
        assert isinstance(r.json()["reward"], float)

    def test_step_before_reset_returns_400(self, client):
        # Force a fresh server state: reset then run to done, then step again
        client.post("/reset", json={"task_id": "easy_reorder"})
        for _ in range(10):
            r = client.post("/step", json={"schedule_mw": 55.0})
            if r.json().get("done"):
                break
        # Episode is done — next step should 400
        r = client.post("/step", json={"schedule_mw": 55.0})
        assert r.status_code == 400

    def test_grader_score_on_completion(self, client):
        client.post("/reset", json={"task_id": "easy_reorder"})
        last = None
        for _ in range(10):
            r = client.post("/step", json={
                "reorder": {"eva": 1500.0},
                "schedule_mw": 55.0,
                "forecast_next_3days": [160.0, 160.0, 160.0],
            })
            last = r.json()
            if last["done"]:
                break
        assert last is not None and last["done"]
        assert "grader_score" in last["info"]
        score = last["info"]["grader_score"]
        assert 0.0 <= score <= 1.0


class TestStateEndpoint:
    def test_state_after_reset(self, client):
        client.post("/reset", json={"task_id": "medium_spike"})
        r = client.get("/state")
        assert r.status_code == 200
        data = r.json()
        assert "obs" in data
        assert "episode_rewards" in data
        assert data["task_id"] == "medium_spike"


class TestTasksEndpoint:
    def test_tasks_lists_four(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200
        tasks = r.json()["tasks"]
        assert len(tasks) == 4

    def test_tasks_have_required_fields(self, client):
        r = client.get("/tasks")
        for t in r.json()["tasks"]:
            assert "id" in t
            assert "difficulty" in t
            assert "max_steps" in t
            assert "target_score" in t
