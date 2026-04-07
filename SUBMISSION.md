# ML Experiment Auditor — Hackathon Submission

## Project Overview

An OpenEnv reinforcement learning environment where an AI agent audits synthetic ML experiment logs to detect real engineering problems: data leakage, misconfigured training runs, and reproducibility failures.

## Environment Description

The environment presents the agent with realistic ML experiment logs and grades its ability to identify specific issues. Three tasks of increasing difficulty are included.

### Tasks

| Task | Name | Difficulty | Max Score |
|------|------|------------|-----------|
| task1 | Config Error Detection | Easy | 1.0 |
| task2 | Data Leakage Detection | Medium | 1.0 |
| task3 | Full ML Audit | Hard | 1.0 |

**Task 1 — Config Error Detection**
The agent reads a ResNet-18 training log and identifies: NaN loss values, learning rate above 10, and missing validation split. Graded by set intersection (1/3 per correct issue).

**Task 2 — Data Leakage Detection**
The agent detects that test-set metrics were computed before the train/test split (timestamp evidence in log). Graded 0.5 for detecting leakage + 0.5 for correct leakage type.

**Task 3 — Full ML Audit**
The agent audits an XGBoost credit risk experiment for: data leakage (scaler fitted on full dataset), metric inconsistency (confusion matrix vs reported accuracy mismatch), missing random seeds, and unpinned dependency versions. Graded by set intersection (0.25 per correct issue).

## Grading

All graders are deterministic — no LLM judge. Scores are computed by comparing submitted findings against hardcoded ground truth.

## Baseline Results

Running `meta-llama/Llama-3.3-70B-Instruct` via HuggingFace Inference API:

| Task | Score |
|------|-------|
| task1 | 1.00 |
| task2 | 1.00 |
| task3 | 1.00 |
| **Average** | **1.00** |

## Setup

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API base URL | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `meta-llama/Llama-3.3-70B-Instruct` |
| `HF_TOKEN` | HuggingFace API token | Required |
| `ENV_SERVER_URL` | Environment server URL | `http://localhost:8000` |

### Running Locally

```bash
# Start environment server
cd my_env
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run inference (in a separate terminal)
export HF_TOKEN=your_token_here
python3 inference.py
```

### Running the Dashboard

```bash
python3 -m streamlit run dashboard.py
```

## File Structure

```
ml-experiment-auditor/
├── inference.py              # Agent inference script (harness entry point)
├── dashboard.py              # Streamlit dashboard
├── Dockerfile                # HuggingFace Space deployment
├── my_env/
│   ├── models.py             # Action and Observation types
│   ├── client.py             # HTTP client wrapper
│   ├── openenv.yaml          # OpenEnv environment spec
│   ├── pyproject.toml        # Package configuration
│   ├── uv.lock               # Locked dependencies
│   └── server/
│       ├── app.py            # FastAPI server (create_app)
│       ├── ml_audit_environment.py  # Environment logic + graders
│       └── Dockerfile        # Server-only Docker deployment
```

## Technical Details

- Built with `openenv-core` framework
- FastAPI server auto-generated via `create_app()`
- Episode state persisted in module-level store (supports concurrent sessions)
- OpenAI-compatible client for all LLM calls
- Fully deterministic graders — reproducible scores across runs
