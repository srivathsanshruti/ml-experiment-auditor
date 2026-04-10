from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional
import requests
from openai import OpenAI

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
API_KEY: str = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "dummy"))
MODEL_NAME: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_SERVER_URL: str = os.environ.get("ENV_SERVER_URL", "https://sshruti115-ml-experiment-auditor.hf.space").rstrip("/")

_llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPTS: Dict[str, str] = {
    "task1": (
        "You are an expert ML experiment auditor. "
        "Read the experiment log and identify configuration errors.\n\n"
        "Return ONLY this JSON format:\n"
        '{"issues": ["<issue1>", "<issue2>", ...]}\n\n'
        "Valid issue identifiers:\n"
        "  nan_loss            - any training loss value is NaN or inf\n"
        "  high_learning_rate  - learning rate is greater than 10\n"
        "  no_validation_split - validation split is missing or none\n\n"
        "Return nothing but JSON."
    ),
    "task2": (
        "You are an expert ML experiment auditor specialising in data leakage.\n\n"
        "Return ONLY this JSON format:\n"
        '{"data_leakage_detected": true, "leakage_type": "test_evaluation_before_split"}\n\n'
        "Valid leakage_type values:\n"
        "  test_evaluation_before_split\n"
        "  none\n\n"
        "Return nothing but JSON."
    ),
    "task3": (
        "You are an expert ML experiment auditor. "
        "Perform a comprehensive audit.\n\n"
        "Return ONLY this JSON format:\n"
        '{"issues": ["<issue1>", "<issue2>", ...]}\n\n'
        "Valid issue identifiers:\n"
        "  data_leakage\n"
        "  metric_inconsistency\n"
        "  no_random_seed\n"
        "  unpinned_requirements\n\n"
        "Return nothing but JSON."
    ),
}


def _log_start(task_id: str) -> None:
    print(json.dumps({"type": "START", "task_id": task_id, "env": "ml_audit", "model": MODEL_NAME}), flush=True)


def _log_step(step: int, action: str, reward: float, done: bool) -> None:
    print(json.dumps({"type": "STEP", "step": step, "action": action, "reward": reward, "done": done}), flush=True)


def _log_end(task_id: str, final_reward: float, steps: int) -> None:
    print(json.dumps({"type": "END", "task_id": task_id, "final_reward": final_reward, "steps": steps}), flush=True)


def _env_reset(task_id: str) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_SERVER_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _env_step(episode_id: str, action_type: str, findings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    action: Dict[str, Any] = {"action_type": action_type, "section": "full"}
    if findings is not None:
        action["findings"] = findings
    resp = requests.post(
        f"{ENV_SERVER_URL}/step",
        json={"action": action, "episode_id": episode_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _call_llm(system_prompt: str, log_content: str) -> str:
    completion = _llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Audit this ML experiment log:\n\n{log_content}"},
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


def run_task(task_id: str) -> float:
    _log_start(task_id)
    step = 0

    reset_data = _env_reset(task_id)
    obs = reset_data.get("observation", {})
    episode_id: str = obs.get("episode_id", "")
    log_content: str = obs.get("content", "")
    step += 1
    _log_step(step, "inspect:full", 0.0, done=False)

    raw = _call_llm(SYSTEM_PROMPTS[task_id], log_content)
    findings = _parse_json(raw)
    step += 1
    _log_step(step, f"analyze:{task_id}", 0.0, done=False)

    step_data = _env_step(episode_id=episode_id, action_type="submit", findings=findings)
    final_reward: float = step_data.get("reward") or 0.0
    step += 1
    _log_step(step, f"submit:{json.dumps(findings, separators=(',', ':'))}", final_reward, done=True)

    _log_end(task_id, final_reward, step)
    return final_reward


def main() -> None:
    task_ids = ["task1", "task2", "task3"]
    total_reward = 0.0
    for task_id in task_ids:
        try:
            reward = run_task(task_id)
            total_reward += reward
        except Exception as exc:
            print(json.dumps({"type": "END", "task_id": task_id, "final_reward": 0.0, "steps": 0, "error": str(exc)}), flush=True)
    avg = total_reward / len(task_ids)
    print(json.dumps({"type": "SUMMARY", "average_reward": round(avg, 4)}), flush=True)


if __name__ == "__main__":
    main()
