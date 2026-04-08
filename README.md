---
title: FactoryMind
emoji: 🏭
colorFrom: red
colorTo: yellow
sdk: docker
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

> **70MW Solar Panel Factory — Autonomous Operations AI**  
> An OpenEnv-compliant reinforcement learning environment simulating real-world solar manufacturing supply-chain decisions.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-brightgreen)](https://huggingface.co/openenv)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

---

## What Is FactoryMind?

FactoryMind drops an AI agent into the operations seat of a **Mumbai-area 70MW solar panel factory**. The agent must balance:

- **Raw material inventory** — silicon cells, glass, EVA encapsulant, backsheet
- **Production scheduling** — allocating daily MW within capacity limits
- **Demand forecasting** — anticipating order spikes and monsoon-driven seasonal dips
- **Supplier risk** — glass delays, monsoon demand drops, rush shipments

This is **not a toy environment**. It mirrors the exact decision loops that factory ops managers face daily. An agent that masters FactoryMind has learned generalizable supply-chain reasoning.

---

## Tasks & Baseline Scores

| # | Task ID | Description | Difficulty | Max Steps | Baseline Score |
|---|---------|-------------|------------|-----------|----------------|
| 1 | `easy_reorder` | Low EVA stock, steady demand — identify and reorder correctly | Easy | 5 | **0.92** |
| 2 | `medium_spike` | Demand ramps sharply, glass shortage — forecast + fill 80%+ demand | Medium | 10 | **0.78** |
| 3 | `hard_risk` | Supplier delay + monsoon dip + cost spike — beat OR-Tools baseline | Hard | 20 | **0.55** |
| 4 | `full_chain` | All events + rush shipment penalty — multi-horizon risk management | Expert | 25 | **0.38** |

Baseline model: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router.
Scores reproducible within ±0.01 (seeded `np.random.default_rng(42 + task_index)`).

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

## Reward Function (dense, every step)

```
reward = 0.4 × profit_component
       − 0.2 × stockout_ratio
       − 0.1 × overstock_penalty
       + 0.1 × forecast_accuracy
```

Final grader score on episode end uses task-specific deterministic criteria.
Tasks 3 & 4 graders benchmark agent profit vs. **PuLP / OR-Tools optimal**.

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
| `/health` | GET | Liveness probe |
| `/reset` | POST | `{"task_id": "easy_reorder"}` → initial obs |
| `/step` | POST | FactoryAction JSON → obs, reward, done, info |
| `/state` | GET | Full episode state |
| `/tasks` | GET | List all tasks |

---

## Setup

```bash
# Docker (recommended)
docker build -t factory-mind .
docker run -p 8000:8000 -e HF_TOKEN=your_token factory-mind

# Local
pip install -e .
uvicorn server.app:app --port 8000

# Baseline inference
cp .env.example .env && python inference.py

# Smoke tests
python test_local.py
pytest tests/ -v

# Validate
openenv validate .
```

---

## License

Apache 2.0
