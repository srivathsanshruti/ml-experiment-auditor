"""
ML Experiment Auditor — core environment logic.

Provides synthetic experiment logs, deterministic graders, and session state.
Since the openenv HTTP server creates a fresh Environment instance per request,
episode state is stored in a module-level dict keyed by episode_id.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

try:
    from ..models import MLAuditAction, MLAuditObservation
except ImportError:
    from models import MLAuditAction, MLAuditObservation

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

TASK_INFO: Dict[str, Dict[str, Any]] = {
    "task1": {
        "name": "Config Error Detection",
        "description": (
            "Detect obvious configuration errors: NaN loss values, "
            "learning rate above 10, and missing validation split."
        ),
        "difficulty": "easy",
        "max_steps": 10,
    },
    "task2": {
        "name": "Data Leakage Detection",
        "description": (
            "Detect data leakage where test-set metrics are reported "
            "before the train/test split is performed."
        ),
        "difficulty": "medium",
        "max_steps": 15,
    },
    "task3": {
        "name": "Full ML Audit",
        "description": (
            "Comprehensive audit: detect data leakage, metric inconsistency "
            "(reported vs confusion-matrix accuracy), missing random seeds, "
            "and unpinned dependency versions."
        ),
        "difficulty": "hard",
        "max_steps": 20,
    },
}

# ---------------------------------------------------------------------------
# Synthetic experiment logs
# ---------------------------------------------------------------------------

_LOG_TASK1 = """\
=== ML EXPERIMENT LOG ===
Experiment ID : exp_20240115_001
Model         : ResNet-18
Dataset       : CIFAR-10
Started       : 2024-01-15 10:00:00

[CONFIG]
  learning_rate    : 15.0
  epochs           : 100
  batch_size       : 32
  optimizer        : Adam
  weight_decay     : 0.0001
  validation_split : none
  random_seed      : 42
  device           : cuda

[TRAINING]
2024-01-15 10:00:10  Training started
2024-01-15 10:01:00  Epoch  1/100 | loss: nan | train_acc: 0.0512
2024-01-15 10:02:00  Epoch  2/100 | loss: nan | train_acc: 0.0489
2024-01-15 10:03:00  Epoch  3/100 | loss: nan | train_acc: 0.0501
2024-01-15 10:04:00  Epoch  4/100 | loss: nan | train_acc: 0.0498
2024-01-15 10:05:00  Epoch  5/100 | loss: nan | train_acc: 0.0512
2024-01-15 10:06:00  Early stopping triggered: loss did not improve

[EVALUATION]
  No validation metrics available (validation_split = none)
  Test evaluation skipped.

=== END OF LOG ===
"""

_LOG_TASK2 = """\
=== ML EXPERIMENT LOG ===
Experiment ID : exp_20240115_002
Model         : BERT-base-uncased (fine-tuned)
Dataset       : sentiment_analysis_50k
Started       : 2024-01-15 09:00:00

[DATA LOADING]
2024-01-15 09:00:00  Loading full dataset (50,000 samples)
2024-01-15 09:00:03  Dataset loaded successfully

[EVALUATION]
2024-01-15 09:00:05  Evaluating on test set...
2024-01-15 09:00:07  Test Accuracy  : 0.9214
2024-01-15 09:00:07  Test F1 Score  : 0.9187
2024-01-15 09:00:07  Test AUC-ROC   : 0.9731

[DATA PREPROCESSING]
2024-01-15 09:00:10  Performing train/test split (80% train, 20% test)
2024-01-15 09:00:11  Train samples: 40,000  |  Test samples: 10,000
2024-01-15 09:00:12  Applying tokenization...
2024-01-15 09:00:45  Tokenization complete

[TRAINING]
2024-01-15 09:01:00  Training started (random_seed: 123)
2024-01-15 09:05:00  Epoch 1/5 | loss: 0.4231 | val_acc: 0.8901
2024-01-15 09:09:00  Epoch 2/5 | loss: 0.3102 | val_acc: 0.9123
2024-01-15 09:13:00  Epoch 3/5 | loss: 0.2541 | val_acc: 0.9201
2024-01-15 09:17:00  Epoch 4/5 | loss: 0.2103 | val_acc: 0.9214
2024-01-15 09:21:00  Epoch 5/5 | loss: 0.1876 | val_acc: 0.9214
2024-01-15 09:21:00  Training complete

[FINAL EVALUATION]
2024-01-15 09:22:00  Final test accuracy: 0.9214

=== END OF LOG ===
"""

_LOG_TASK3 = """\
=== ML EXPERIMENT LOG ===
Experiment ID : exp_20240115_003
Model         : XGBoost Classifier
Dataset       : credit_risk_100k
Started       : 2024-01-15 14:00:00

[ENVIRONMENT]
  Python       : 3.9
  Dependencies : numpy, pandas, scikit-learn, xgboost, matplotlib

[DATA LOADING]
2024-01-15 14:00:00  Loading dataset (100,000 samples, 47 features)
2024-01-15 14:00:02  Dataset loaded

[INITIAL STATISTICS]
2024-01-15 14:00:03  Computing test set statistics...
2024-01-15 14:00:04  Test set mean_target: 0.2341
2024-01-15 14:00:04  Test set feature correlations computed

[DATA PREPROCESSING]
2024-01-15 14:00:05  Performing train/test split (75% train, 25% test, no random_state)
2024-01-15 14:00:06  Train samples: 75,000  |  Test samples: 25,000
2024-01-15 14:00:07  Fitting StandardScaler on full dataset (all 100,000 samples)
2024-01-15 14:00:08  Scaler fitted and applied to all splits

[FEATURE ENGINEERING]
2024-01-15 14:00:10  Creating derived features...
2024-01-15 14:00:15  Feature engineering complete (47 -> 89 features)

[TRAINING]
2024-01-15 14:00:16  Training XGBoost (no random_state set)
2024-01-15 14:02:00  Training complete

[EVALUATION]
2024-01-15 14:02:05  Confusion Matrix (test set, 25,000 samples):
    True  Positives : 3842
    True  Negatives : 17901
    False Positives : 1203
    False Negatives : 2054
2024-01-15 14:02:05  Accuracy from confusion matrix : 0.8697
2024-01-15 14:02:06  Reported Test Accuracy         : 0.9234
2024-01-15 14:02:06  Reported Test AUC-ROC          : 0.9102
2024-01-15 14:02:06  Reported Test F1 Score         : 0.7234

=== END OF LOG ===
"""

EXPERIMENT_LOGS: Dict[str, str] = {
    "task1": _LOG_TASK1,
    "task2": _LOG_TASK2,
    "task3": _LOG_TASK3,
}

LOG_SECTIONS: Dict[str, Dict[str, str]] = {
    "task1": {
        "config": (
            "[CONFIG]\n"
            "  learning_rate    : 15.0\n"
            "  epochs           : 100\n"
            "  batch_size       : 32\n"
            "  optimizer        : Adam\n"
            "  validation_split : none\n"
            "  random_seed      : 42\n"
        ),
        "training": (
            "[TRAINING]\n"
            "2024-01-15 10:01:00  Epoch  1/100 | loss: nan | train_acc: 0.0512\n"
            "2024-01-15 10:02:00  Epoch  2/100 | loss: nan | train_acc: 0.0489\n"
            "2024-01-15 10:06:00  Early stopping triggered: loss did not improve\n"
        ),
        "evaluation": (
            "[EVALUATION]\n"
            "  No validation metrics available (validation_split = none)\n"
            "  Test evaluation skipped.\n"
        ),
    },
    "task2": {
        "config": "(No explicit [CONFIG] block — see DATA PREPROCESSING for split settings)\n",
        "training": (
            "[TRAINING]\n"
            "2024-01-15 09:01:00  Training started (random_seed: 123)\n"
            "2024-01-15 09:05:00  Epoch 1/5 | loss: 0.4231 | val_acc: 0.8901\n"
            "2024-01-15 09:21:00  Epoch 5/5 | loss: 0.1876 | val_acc: 0.9214\n"
        ),
        "evaluation": (
            "[EVALUATION — timestamps matter!]\n"
            "2024-01-15 09:00:05  Evaluating on test set...\n"
            "2024-01-15 09:00:07  Test Accuracy  : 0.9214\n"
            "\n"
            "[DATA PREPROCESSING]\n"
            "2024-01-15 09:00:10  Performing train/test split (80% train, 20% test)\n"
        ),
    },
    "task3": {
        "config": (
            "[ENVIRONMENT]\n"
            "  Python       : 3.9\n"
            "  Dependencies : numpy, pandas, scikit-learn, xgboost, matplotlib\n"
            "\n"
            "NOTE: No version pins on any dependency.\n"
        ),
        "training": (
            "[TRAINING]\n"
            "2024-01-15 14:00:16  Training XGBoost (no random_state set)\n"
            "2024-01-15 14:02:00  Training complete\n"
        ),
        "evaluation": (
            "[EVALUATION]\n"
            "  Accuracy from confusion matrix : 0.8697\n"
            "  Reported Test Accuracy         : 0.9234\n"
            "  (TP=3842, TN=17901, FP=1203, FN=2054 → (3842+17901)/25000 = 0.8697)\n"
        ),
    },
}

# ---------------------------------------------------------------------------
# Ground truth & deterministic graders
# ---------------------------------------------------------------------------

GROUND_TRUTH: Dict[str, Any] = {
    "task1": {"issues": {"nan_loss", "high_learning_rate", "no_validation_split"}},
    "task2": {
        "data_leakage_detected": True,
        "leakage_type": "test_evaluation_before_split",
    },
    "task3": {
        "issues": {
            "data_leakage",
            "metric_inconsistency",
            "no_random_seed",
            "unpinned_requirements",
        }
    },
}


def _grade_task1(findings: Dict[str, Any]) -> Tuple[float, str]:
    submitted = set(findings.get("issues", []))
    ground_truth = GROUND_TRUTH["task1"]["issues"]
    correct = submitted & ground_truth
    false_pos = submitted - ground_truth
    score = len(correct) / len(ground_truth)
    detail = (
        f"Detected {len(correct)}/{len(ground_truth)}. "
        f"Correct: {sorted(correct)}. "
        f"False positives: {sorted(false_pos)}."
    )
    return round(score, 4), detail


def _grade_task2(findings: Dict[str, Any]) -> Tuple[float, str]:
    score = 0.0
    parts = []
    if findings.get("data_leakage_detected") is True:
        score += 0.5
        parts.append("leakage_detected=True (+0.5)")
    else:
        parts.append("leakage_detected missing/False (+0.0)")
    if findings.get("leakage_type") == GROUND_TRUTH["task2"]["leakage_type"]:
        score += 0.5
        parts.append("leakage_type correct (+0.5)")
    else:
        parts.append(f"leakage_type '{findings.get('leakage_type', '')}' wrong (+0.0)")
    return round(score, 4), " | ".join(parts)


def _grade_task3(findings: Dict[str, Any]) -> Tuple[float, str]:
    submitted = set(findings.get("issues", []))
    ground_truth = GROUND_TRUTH["task3"]["issues"]
    correct = submitted & ground_truth
    false_pos = submitted - ground_truth
    score = len(correct) / len(ground_truth)
    detail = (
        f"Detected {len(correct)}/{len(ground_truth)}. "
        f"Correct: {sorted(correct)}. "
        f"False positives: {sorted(false_pos)}."
    )
    return round(score, 4), detail


GRADERS = {
    "task1": _grade_task1,
    "task2": _grade_task2,
    "task3": _grade_task3,
}

# ---------------------------------------------------------------------------
# Per-episode state (module-level, persists across request-scoped env instances)
# ---------------------------------------------------------------------------

class _EpisodeState:
    __slots__ = ("task_id", "step_count", "done", "max_steps")

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.step_count = 0
        self.done = False
        self.max_steps = TASK_INFO[task_id]["max_steps"]


_EPISODE_STORE: Dict[str, _EpisodeState] = {}

# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------


class MLAuditEnvironment(Environment):
    """ML Experiment Auditor environment.

    The HTTP server creates a fresh instance per request; all episode state
    is stored in the module-level ``_EPISODE_STORE`` dict.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State()

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task1", **kwargs: Any) -> MLAuditObservation:  # type: ignore[override]
        if task_id not in TASK_INFO:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid values: {list(TASK_INFO)}"
            )
        episode_id = str(uuid4())
        _EPISODE_STORE[episode_id] = _EpisodeState(task_id)
        self._state = State(episode_id=episode_id, step_count=0)

        info = TASK_INFO[task_id]
        return MLAuditObservation(
            content=EXPERIMENT_LOGS[task_id],
            task_id=task_id,
            task_description=info["description"],
            step=0,
            done=False,
            reward=0.0,
            episode_id=episode_id,
            metadata={
                "difficulty": info["difficulty"],
                "max_steps": info["max_steps"],
            },
        )

    def step(  # type: ignore[override]
        self,
        action: MLAuditAction,
        episode_id: str = "",
        **kwargs: Any,
    ) -> MLAuditObservation:
        ep = _EPISODE_STORE.get(episode_id)
        if ep is None:
            raise ValueError(
                f"Unknown episode_id '{episode_id}'. Call /reset first to get one."
            )
        if ep.done:
            raise ValueError("Episode already finished. Call /reset to start a new one.")

        ep.step_count += 1
        if ep.step_count >= ep.max_steps:
            ep.done = True
        self._state = State(episode_id=episode_id, step_count=ep.step_count)

        task_id = ep.task_id
        info = TASK_INFO[task_id]

        if action.action_type == "inspect":
            sec = (action.section or "full").lower()
            if sec == "full":
                content = EXPERIMENT_LOGS[task_id]
            else:
                content = LOG_SECTIONS[task_id].get(
                    sec,
                    f"Unknown section '{sec}'. Valid: full, config, training, evaluation",
                )
            return MLAuditObservation(
                content=content,
                task_id=task_id,
                task_description=info["description"],
                step=ep.step_count,
                done=ep.done,
                reward=0.0,
                metadata={"episode_id": episode_id, "section": sec},
            )

        if action.action_type == "submit":
            findings = action.findings or {}
            reward, detail = GRADERS[task_id](findings)
            ep.done = True
            return MLAuditObservation(
                content=f"Submission graded. {detail}",
                task_id=task_id,
                task_description=info["description"],
                step=ep.step_count,
                done=True,
                reward=reward,
                metadata={"episode_id": episode_id, "grading_detail": detail},
            )

        raise ValueError(
            f"Unknown action_type '{action.action_type}'. "
            "Valid values: 'inspect', 'submit'."
        )

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="ml-experiment-auditor",
            description=(
                "An AI agent reads synthetic ML experiment logs and detects "
                "real problems: data leakage, bad configs, reproducibility gaps."
            ),
            version="1.0.0",
            author="ML Audit Team",
        )
