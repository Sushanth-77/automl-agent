"""
Shared utility to get the current pipeline run directory.
All agents should import this instead of hardcoding RUNS_DIR / "current".
"""
from __future__ import annotations

import os
from pathlib import Path


def get_run_dir() -> Path:
    """
    Return the current pipeline run directory.

    Precedence:
    1. AUTOML_RUN_DIR env var (set by graph.run_pipeline)
    2. RUNS_DIR / "current" fallback (for standalone testing)
    """
    env_dir = os.getenv("AUTOML_RUN_DIR")
    if env_dir:
        path = Path(env_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Fallback
    from config import RUNS_DIR
    fallback = RUNS_DIR / "current"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback
