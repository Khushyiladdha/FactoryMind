---
title: FactoryMind
emoji: 🏭
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
tags:
  - openenv
  - reinforcement-learning
  - supply-chain
  - solar-manufacturing
  - rl-environment
short_description: 70MW Solar Factory RL Environment — OpenEnv compliant
---

# 🏭 FactoryMind

> Designed to evaluate LLM agents on long-horizon operational decision-making under uncertainty.

**70MW Solar Panel Factory — Autonomous Operations AI**
An OpenEnv-compliant RL environment grounded in real-world solar manufacturing constraints.

FactoryMind places an AI agent in control of a production-scale factory where it must continuously balance:

- **Inventory** — multi-material dependency across silicon cells, glass, EVA, and backsheet
- **Production** — allocate output under strict 70MW/day capacity limits
- **Forecasting** — anticipate demand spikes and monsoon-driven seasonal shifts
- **Risk Management** — handle supplier delays, cost volatility, and rush shipment penalties

Unlike toy RL environments, decisions have **delayed and compounding effects**. A suboptimal reorder early in the episode propagates into downstream stockouts, lost revenue, and cascading penalties. This environment tests whether an agent can plan ahead, manage cost-risk-fulfillment trade-offs, and maintain operational stability under uncertainty.

---

## Tasks & Baseline Scores

Baseline model: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router.
All scores reproducible within ±0.01 (seeded `np.random.default_rng(42 + task_index)`).

| # | Task ID | Description | Difficulty | Max Steps | Baseline Score |
|---|---------|-------------|------------|-----------|----------------|
| 1 | `easy_reorder` | Low EVA stock, steady demand — identify and reorder correctly | Easy | 5 | **0.73** |
| 2 | `medium_spike` | Demand ramps sharply, glass shortage — forecast + schedule | Medium | 10 | **0.65** |
| 3 | `hard_risk` | Supplier delay + monsoon dip + cost spike — beat OR-Tools | Hard | 20 | **0.88** |
| 4 | `full_chain` | All events + rush shipment — multi-horizon risk management | Expert | 25 | **0.55** |

---

## Observation Space

```json
{
  "inventory":       {"cells": 50000, "glass": 30000, "eva": 500, "backsheet": 2500},
  "demand_hist":     [160.0, 162.0, 158.0, 161.0, 160.0, 159.0, 161.0],
  "capacity":        70.0,
  "costs":           {"cells": 0.25, "glass": 0.08, "eva": 0.15, "backsheet": 0.10, "storage": 0.002},
  "events":          [],
  "step_count":      0,
  "current_profit":  0.0,
  "task_id":         "easy_reorder",
  "done":            false
}
```

**Active event types:** `supplier_delay_glass`, `monsoon_dip`, `glass_shortage`, `rush_shipment`

---

## Action Space

```json
{
  "reorder":               {"cells": 10000, "glass": 5000, "eva": 1500, "backsheet": 0},
  "schedule_mw":           65.0,
  "forecast_next_3days":   [165.0, 170.0, 168.0]
}
```

---

## Reward Function

Dense reward every step — never sparse. Encourages proactive planning, penalizes reactive decisions.

```
reward = 0.4 × profit_component
       − 0.2 × stockout_ratio
       − 0.1 × overstock_penalty
       + 0.1 × forecast_accuracy
       − 0.05 × low_stock_urgency   (per critical material below threshold)
       + 0.1  × forecast_improvement (when accuracy improves step-over-step)
       + 0.05 × proactive_reorder   (ordering before stockout, not after)
```

Final episode grader score uses task-specific deterministic criteria.
Tasks 3 & 4 benchmark agent profit vs **PuLP / OR-Tools linear programming optimal**.

---

## Material Usage Rates

| Material | Per MW produced | Base cost |
|----------|----------------|-----------|
| cells | 286 panel cells | $0.25/unit |
| glass | 2.4 sqm | $0.08/unit |
| eva | 2.1 sqm | $0.15/unit |
| backsheet | 2.1 sqm | $0.10/unit |

Revenue: **$280,000 per MW** sold. Capacity: **70 MW/day**.

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe → `{"status": "ok"}` |
| `/reset` | POST | `{"task_id": "easy_reorder"}` → initial observation |
| `/step` | POST | FactoryAction JSON → obs, reward, done, info |
| `/state` | GET | Full current episode state |
| `/tasks` | GET | List all tasks with metadata |

---

## Setup

```bash
# Docker
docker build -t factory-mind .
docker run -p 7860:7860 -e HF_TOKEN=your_token factory-mind

# Local
pip install -e .
uvicorn server.app:app --port 7860

# Smoke test (no LLM needed)
python test_local.py

# Full baseline inference
cp .env.example .env  # fill in HF_TOKEN
python inference.py
```

---

## Why Solar Manufacturing?

Solar is a high-stakes domain with real multi-constraint optimisation. Monsoon seasonality in Indian markets, volatile glass prices, silicon cell lead times — these are genuine challenges absent from gym benchmarks. An agent that masters FactoryMind develops reasoning transferable to any discrete manufacturing environment.

The hard → expert task progression ensures the environment *trains*, not just evaluates. Task 1 is solvable with simple threshold logic. Task 4 requires multi-step lookahead under compounding uncertainty.

---

## License

Apache 2.0