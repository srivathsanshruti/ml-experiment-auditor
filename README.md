# ML Experiment Auditor - OpenEnv Environment

An OpenEnv environment for the **Meta × HuggingFace OpenEnv Hackathon**.

---
## Environment Description

### Concept
The agent is given a realistic (synthetic) ML experiment log and must identify specific failure modes. All graders are fully **deterministic** — no LLM judge, just string/value matching against ground truth.

### Action Space

| Action | Fields | Description |
|--------|--------|-------------|
| `inspect` | `section`: `full` \| `config` \| `training` \| `evaluation` | Read the log or a named section |
| `submit` | `findings`: `{...}` | Submit structured findings for grading |

### Observation Space

Each step returns an `Observation` object:

```json
{
  "content": "<log text or grading feedback>",
  "task_id": "task1",
  "task_description": "...",
  "step": 2,
  "done": false,
  "metadata": {}
}
```

---

## Tasks

### Task 1 — Config Error Detection *(easy)*

**Goal:** Detect obvious configuration errors in the experiment log.

**Issues to find:**
- `nan_loss` — training loss values are NaN
- `high_learning_rate` — learning rate > 10
- `no_validation_split` — no validation split defined

**Submission format:**
```json
{"issues": ["nan_loss", "high_learning_rate", "no_validation_split"]}
```

**Scoring:** `len(correct_issues_detected) / 3.0`

---

### Task 2 — Data Leakage Detection *(medium)*

**Goal:** Detect data leakage where test-set metrics appear *before* the train/test split step in the log timeline.

**Submission format:**
```json
{
  "data_leakage_detected": true,
  "leakage_type": "test_evaluation_before_split"
}
```

**Scoring:**
- +0.5 for `data_leakage_detected: true`
- +0.5 for correct `leakage_type`

---

### Task 3 — Full ML Audit *(hard)*

**Goal:** Perform a comprehensive audit and identify all of:
- `data_leakage` — test stats computed before split, or scaler fitted on full dataset
- `metric_inconsistency` — reported accuracy (0.9234) does not match confusion-matrix accuracy (0.8697)
- `no_random_seed` — no `random_state` in split or model training
- `unpinned_requirements` — dependencies listed without version pins

**Submission format:**
```json
{
  "issues": [
    "data_leakage",
    "metric_inconsistency",
    "no_random_seed",
    "unpinned_requirements"
  ]
}
```

**Scoring:** `len(correct_issues_detected) / 4.0`

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install "openenv-core[core]" fastapi uvicorn[standard] pydantic requests openai
```

### 2. Validate the environment

```bash
openenv validate
```

### 3. Start the server

```bash
uvicorn my_env.server.app:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker build -f my_env/server/Dockerfile -t ml-audit-env .
docker run -p 8000:8000 ml-audit-env
```

### 4. Run the inference script

```bash
export API_BASE_URL="http://localhost:8000"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_..."
export INFERENCE_API="https://api-inference.huggingface.co/v1"

python inference.py
```

---

## Baseline Scores

Scores achievable by a capable LLM reading the full log once:

| Task | Difficulty | Expected Baseline |
|------|-----------|-------------------|
| task1 | easy | 1.00 |
| task2 | medium | 1.00 |
| task3 | hard | 0.75 – 1.00 |
| **Average** | — | **~0.92** |

A random agent that submits empty findings scores **0.0** on all tasks.

---

## File Structure

```
my_env/
├── __init__.py              # Package exports: MLAuditClient
├── models.py                # Pydantic models (Action, Observation, ...)
├── client.py                # HTTP client for the environment server
├── openenv.yaml             # OpenEnv spec (spec_version: 1)
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI application
    ├── ml_audit_environment.py  # Logs, graders, session state
    └── Dockerfile           # Container for the server
inference.py                 # Agent script (OpenAI-compatible client)
README.md                    # This file
```

---

## REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all tasks |
| `POST` | `/reset` | Start episode: `{"task_id": "task1"}` |
| `POST` | `/step` | Take step: `{"session_id": "...", "action_type": "inspect"\|"submit", ...}` |
