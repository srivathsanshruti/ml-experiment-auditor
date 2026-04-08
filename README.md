---
title: ML Experiment Auditor
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ML Experiment Auditor

An OpenEnv environment where an AI agent reads synthetic ML experiment logs and detects real problems: data leakage, bad configs, and reproducibility issues.

## Tasks
- **Task 1 (Easy):** Detect config errors — NaN loss, learning rate > 10, no validation split
- **Task 2 (Medium):** Detect data leakage — test metrics reported before train/test split
- **Task 3 (Hard):** Full audit — leakage + metric inconsistency + reproducibility gaps

## API Endpoints
- `GET /health` — health check
- `POST /reset` — start a new episode
- `POST /step` — take an action
- `GET /state` — current episode state

## Setup
```bash
pip install openenv-core
```
