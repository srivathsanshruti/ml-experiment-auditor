"""
FastAPI application for the ML Experiment Auditor OpenEnv environment.

Uses openenv.core.env_server.http_server.create_app to generate all
required endpoints: /health, /metadata, /schema, /reset, /step, /state,
/mcp, /openapi.json, etc.

Usage:
    # Development:
    uvicorn my_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn my_env.server.app:app --host 0.0.0.0 --port 8000

    # Via entry point (uv run server):
    python -m my_env.server.app
"""
from __future__ import annotations

from openenv.core.env_server.http_server import create_app

try:
    from ..models import MLAuditAction, MLAuditObservation
    from .ml_audit_environment import MLAuditEnvironment
except ImportError:
    from models import MLAuditAction, MLAuditObservation
    from server.ml_audit_environment import MLAuditEnvironment


# Create the fully-featured app (simulation mode: /reset + /step + /state)
app = create_app(
    MLAuditEnvironment,
    MLAuditAction,
    MLAuditObservation,
    env_name="ml_audit",
    max_concurrent_envs=10,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for uv run server or direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
