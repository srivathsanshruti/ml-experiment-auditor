"""
ML Experiment Auditor — Streamlit Dashboard
Run with: python3 -m streamlit run dashboard.py
"""
import json
import os
import subprocess
import sys

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ML Experiment Auditor", layout="wide")

st.title("ML Experiment Auditor")
st.caption("Detects data leakage, bad configs, and reproducibility issues in ML experiment logs.")

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    env_server = st.text_input(
        "Environment Server URL",
        value=os.environ.get("ENV_SERVER_URL", "http://localhost:8000"),
    )
    api_base = st.text_input(
        "LLM API Base URL",
        value=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
    )
    hf_token = st.text_input(
        "HuggingFace Token",
        value=os.environ.get("HF_TOKEN", ""),
        type="password",
        help="Get one at huggingface.co/settings/tokens (Read permission)",
    )
    model_name = st.text_input(
        "Model",
        value=os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct"),
    )

    st.divider()

    if st.button("Check Server Health"):
        import requests
        try:
            r = requests.get(f"{env_server}/health", timeout=5)
            if r.status_code == 200:
                st.success("Server is healthy")
            else:
                st.error(f"Server returned {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach server: {e}")

    st.divider()
    run_btn = st.button("Run Inference", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

TASK_META = {
    "task1": {"label": "Task 1 — Config Error Detection", "difficulty": "Easy"},
    "task2": {"label": "Task 2 — Data Leakage Detection", "difficulty": "Medium"},
    "task3": {"label": "Task 3 — Full ML Audit",          "difficulty": "Hard"},
}

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

if not run_btn:
    st.subheader("Tasks Overview")
    cols = st.columns(3)
    descs = [
        ("Config Error Detection", "Easy",   "Detect NaN loss, learning rate > 10, missing validation split."),
        ("Data Leakage Detection", "Medium", "Detect test metrics computed before train/test split."),
        ("Full ML Audit",          "Hard",   "Leakage + metric inconsistency + missing seeds + unpinned deps."),
    ]
    for col, (name, diff, desc) in zip(cols, descs):
        with col:
            st.markdown(f"**{name}**")
            st.caption(f"Difficulty: {diff}")
            st.caption(desc)

else:
    if not hf_token:
        st.error("Please enter your HuggingFace token in the sidebar.")
        st.stop()

    env = os.environ.copy()
    env["ENV_SERVER_URL"] = env_server
    env["API_BASE_URL"] = api_base
    env["HF_TOKEN"] = hf_token
    env["MODEL_NAME"] = model_name

    task_rewards: dict = {}
    current_task = None

    st.subheader("Results")
    metric_placeholders = {}
    step_placeholders = {}

    cols = st.columns(3)
    for col, task_id in zip(cols, TASK_META):
        with col:
            meta = TASK_META[task_id]
            st.markdown(f"**{meta['label']}**")
            st.caption(f"Difficulty: {meta['difficulty']}")
            metric_placeholders[task_id] = st.empty()
            step_placeholders[task_id] = st.container()

    st.divider()
    chart_placeholder = st.empty()

    process = subprocess.Popen(
        [sys.executable, "inference.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("[START]"):
            parts = dict(p.split("=", 1) for p in line[7:].strip().split() if "=" in p)
            current_task = parts.get("task")
            if current_task and current_task in metric_placeholders:
                metric_placeholders[current_task].info("Running...")

        elif line.startswith("[STEP]"):
            parts = dict(p.split("=", 1) for p in line[6:].strip().split() if "=" in p)
            if current_task and current_task in step_placeholders:
                with step_placeholders[current_task]:
                    reward = float(parts.get("reward", 0))
                    reward_str = f" | reward: {reward:.2f}" if reward else ""
                    st.write(f"Step {parts.get('step')}: `{parts.get('action', '')[:60]}`{reward_str}")

        elif line.startswith("[END]"):
            parts = dict(p.split("=", 1) for p in line[5:].strip().split() if "=" in p)
            if current_task:
                rewards_list = [float(r) for r in parts.get("rewards", "0").split(",")]
                final_reward = rewards_list[-1] if rewards_list else 0.0
                task_rewards[current_task] = final_reward
                success = parts.get("success") == "true"
                if success:
                    metric_placeholders[current_task].metric(
                        "Final Reward",
                        f"{final_reward:.4f}",
                        delta=f"{final_reward - 0.5:+.2f} vs 0.5 baseline",
                    )
                else:
                    metric_placeholders[current_task].error("Failed")

    process.wait()

    stderr = process.stderr.read()
    if stderr and not task_rewards:
        with st.expander("Debug output"):
            st.code(stderr[:3000])

    if task_rewards:
        df = pd.DataFrame(
            [{"Task": k, "Reward": v} for k, v in task_rewards.items()]
        )
        chart_placeholder.subheader("Reward Summary")
        chart_placeholder.bar_chart(df.set_index("Task"), y="Reward", use_container_width=True)

        avg = sum(task_rewards.values()) / len(task_rewards)
        cols = st.columns(4)
        for col, (tid, reward) in zip(cols, task_rewards.items()):
            col.metric(TASK_META[tid]["label"].split("—")[1].strip(), f"{reward:.4f}")
        cols[3].metric("Average", f"{avg:.4f}")
