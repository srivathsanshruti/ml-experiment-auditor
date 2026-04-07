"""
Data models for the ML Experiment Auditor environment.

Extends openenv base types for type-safe serialisation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MLAuditAction(Action):
    """Action for the ML Audit environment.

    The agent can inspect log sections or submit structured findings.
    """

    action_type: str = Field(
        ...,
        description="'inspect' to read a log section, 'submit' to grade findings",
    )
    section: Optional[str] = Field(
        default="full",
        description="Log section for inspect: 'full', 'config', 'training', 'evaluation'",
    )
    findings: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured findings dict for submit action",
    )


class MLAuditObservation(Observation):
    """Observation from the ML Audit environment."""

    content: str = Field(default="", description="Log content or grading feedback")
    task_id: str = Field(default="", description="Current task identifier")
    task_description: str = Field(default="", description="Human-readable task goal")
    step: int = Field(default=0, description="Current step within the episode")
    episode_id: str = Field(default="", description="Episode identifier for /step calls")
