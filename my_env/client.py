"""
HTTP client for the ML Experiment Auditor OpenEnv environment.

Wraps the openenv HTTP API and tracks episode_id across reset/step calls.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from .models import MLAuditAction, MLAuditObservation


class MLAuditClient:
    """Client for the ML Audit environment server.

    Usage::

        client = MLAuditClient("http://localhost:8000")
        obs_data = client.reset("task1")
        episode_id = obs_data["metadata"]["episode_id"]
        result = client.step(episode_id, "inspect", section="full")
        result = client.step(episode_id, "submit", findings={"issues": ["nan_loss"]})
        print(result["reward"])
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._http = requests.Session()
        self._episode_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def health(self) -> bool:
        try:
            resp = self._http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def metadata(self) -> Dict[str, Any]:
        resp = self._http.get(f"{self.base_url}/metadata")
        resp.raise_for_status()
        return resp.json()

    def schema(self) -> Dict[str, Any]:
        resp = self._http.get(f"{self.base_url}/schema")
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> List[str]:
        """Return valid task IDs by inspecting the action schema."""
        return ["task1", "task2", "task3"]

    # ------------------------------------------------------------------
    # Core environment interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task1") -> Dict[str, Any]:
        """Reset the environment for the given task.

        Returns the raw observation dict (includes metadata.episode_id).
        """
        resp = self._http.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
        )
        resp.raise_for_status()
        data = resp.json()
        obs = data.get("observation", {})
        self._episode_id = obs.get("episode_id")
        return data

    def step(
        self,
        episode_id: str,
        action_type: str,
        section: str = "full",
        findings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute one step.

        Args:
            episode_id: Returned in observation.metadata.episode_id after reset.
            action_type: 'inspect' or 'submit'.
            section: Section name for inspect ('full', 'config', 'training', 'evaluation').
            findings: Structured findings dict for submit.
        """
        action: Dict[str, Any] = {"action_type": action_type, "section": section}
        if findings is not None:
            action["findings"] = findings

        payload: Dict[str, Any] = {
            "action": action,
            "episode_id": episode_id,
        }
        resp = self._http.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def inspect(self, episode_id: str, section: str = "full") -> Dict[str, Any]:
        return self.step(episode_id, "inspect", section=section)

    def submit_findings(
        self, episode_id: str, findings: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.step(episode_id, "submit", findings=findings)
