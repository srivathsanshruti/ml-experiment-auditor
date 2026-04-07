"""
ML Experiment Auditor — inference script.

Environment variables:
  API_BASE_URL    - LLM API base URL (default: https://router.huggingface.co/v1)
  MODEL_NAME      - Model identifier for inference (default: meta-llama/Llama-3.3-70B-Instruct)
  HF_TOKEN        - HuggingFace API token (required)
  ENV_SERVER_URL  - ML Audit environment server URL (default: http://localhost:8000)

Stdout format:
  [START] task=<task_name> env=ml_audit model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get(
    "API_BASE_URL", "https://router.huggingface.co/v1"
).rstrip("/")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")
ENV_SERVER_URL: str = os.environ.get("ENV_SERVER_URL", "http://localhost:8000").rstrip("/")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------

_llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Per-task system prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: Dict[str, str] = {
    "task1": (
        "You are an expert ML experiment auditor. "
        "Carefully read the experiment log and identify configuration errors.\n\n"
        "Return ONLY a JSON object in this exact format — nothing else:\n"
        '{"issues": ["<issue1>", "<issue2>", ...]}\n\n'
        "Valid issue identifiers (include ALL that are present):\n"
        "  nan_loss            — any training loss value is NaN or inf\n"
        "  high_learning_rate  — learning rate is greater than 10\n"
        "  no_validation_split — validation split is missing, 'none', or 0\n\n"
        "Include ONLY identifiers that are genuinely present. Return nothing but JSON."
    ),
    "task2": (
        "You are an expert ML experiment auditor specialising in data leakage.\n\n"
        "Data leakage occurs when test-set information is used before the "
        "train/test split. Specifically: test metrics computed at an earlier "
        "timestamp than the split step indicate leakage.\n\n"
        "Return ONLY a JSON object in this exact format — nothing else:\n"
        '{"data_leakage_detected": <true|false>, "leakage_type": "<type>"}\n\n'
        "Valid leakage_type values:\n"
        "  test_evaluation_before_split — test metrics computed before split\n"
        "  none                         — no leakage detected\n\n"
        "Return nothing but the JSON object."
    ),
    "task3": (
        "You are an expert ML experiment auditor. "
        "Perform a comprehensive audit of the experiment log.\n\n"
        "Return ONLY a JSON object in this exact format — nothing else:\n"
        '{"issues": ["<issue1>", "<issue2>", ...]}\n\n'
        "Valid issue identifiers (include ALL that are present):\n"
        "  data_leakage          — test-set statistics/metrics computed before the "
        "train/test split, OR preprocessing (e.g. scaler) fitted on the full dataset "
        "before splitting\n"
        "  metric_inconsistency  — a reported metric value contradicts other reported "
        "values (e.g. accuracy from confusion matrix differs from reported accuracy)\n"
        "  no_random_seed        — random_state / seed absent in the split, model "
        "training, or both\n"
        "  unpinned_requirements — packages listed without pinned versions "
        "(e.g. 'numpy' instead of 'numpy==1.24.0')\n\n"
        "Return nothing but the JSON object."
    ),
}

# ---------------------------------------------------------------------------
# Logging helpers — exact harness format
# ---------------------------------------------------------------------------


def _log_start(task_id: str) -> None:
    print(
        f"[START] task={task_id} env=ml_audit model={MODEL_NAME}",
        flush=True,
    )


def _log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    error_str = error if error is not None else "null"
    done_str = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def _log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(
        f"[END] success={success_str} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment API helpers
# ---------------------------------------------------------------------------


def _env_reset(task_id: str) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_SERVER_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _env_step(
    episode_id: str,
    action_type: str,
    section: str = "full",
    findings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    action: Dict[str, Any] = {"action_type": action_type, "section": section}
    if findings is not None:
        action["findings"] = findings
    resp = requests.post(
        f"{ENV_SERVER_URL}/step",
        json={"action": action, "episode_id": episode_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _call_llm(system_prompt: str, log_content: str) -> str:
    completion = _llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the ML experiment log to audit:\n\n{log_content}"},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return completion.choices[0].message.content.strip()


def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------


def run_task(task_id: str) -> float:
    system_prompt = SYSTEM_PROMPTS[task_id]
    rewards: List[float] = []
    step = 0
    last_error: Optional[str] = None

    _log_start(task_id)

    # Step 1: reset environment, retrieve experiment log
    reset_data = _env_reset(task_id)
    obs = reset_data.get("observation", {})
    episode_id: str = obs.get("episode_id", "")
    log_content: str = obs.get("content", "")
    step += 1
    rewards.append(0.0)
    _log_step(step, "inspect:full", 0.0, done=False)

    # Step 2: call LLM to analyse the log
    raw = _call_llm(system_prompt, log_content)
    findings = _parse_json(raw)
    step += 1
    rewards.append(0.0)
    _log_step(step, f"analyze_log:{task_id}", 0.0, done=False)

    # Step 3: submit findings, receive graded reward
    step_data = _env_step(episode_id=episode_id, action_type="submit", findings=findings)
    final_reward: float = step_data.get("reward") or 0.0
    step += 1
    rewards.append(final_reward)
    _log_step(step, f"submit:{json.dumps(findings, separators=(',', ':'))}", final_reward, done=True)

    success = final_reward > 0.0
    _log_end(success=success, steps=step, rewards=rewards)

    return final_reward


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    task_ids = ["task1", "task2", "task3"]
    total_reward = 0.0

    for task_id in task_ids:
        try:
            reward = run_task(task_id)
            total_reward += reward
        except Exception as exc:
            _log_end(success=False, steps=0, rewards=[0.0])

    avg = total_reward / len(task_ids)
    print(f"Average reward: {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
