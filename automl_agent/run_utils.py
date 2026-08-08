"""
Shared utility to get the current pipeline run directory.
All agents should import this instead of hardcoding RUNS_DIR / "current".
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


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


def list_runs(runs_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Scan the runs/ directory and return metadata for each completed run.

    Each entry contains:
      - run_id: str (folder name like "20260808_202220")
      - run_dir: str (absolute path)
      - dataset: str (inferred from state.json dataset_path)
      - task_type: str
      - best_model: str
      - best_metric_value: float | None
      - stop_reason: str
      - n_models: int
      - n_iterations: int
      - has_report: bool
      - timestamp: str (human-readable from run_id)

    Sorted newest-first. Excludes "current" and "uploads" pseudo-dirs.
    """
    from config import RUNS_DIR as _RUNS_DIR, PRIMARY_METRICS

    _dir = Path(runs_dir) if runs_dir else _RUNS_DIR

    runs: list[dict[str, Any]] = []
    for sub in sorted(_dir.iterdir(), reverse=True):
        # Skip non-run directories
        if not sub.is_dir() or sub.name in ("current", "uploads"):
            continue
        state_path = sub / "state.json"
        if not state_path.exists():
            continue

        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        task_type = state.get("task_type", "?")
        primary_metric = PRIMARY_METRICS.get(task_type, "f1_weighted")

        # Find best metric value
        best_val: float | None = None
        best_model = state.get("_current_best_model_id") or state.get("stop_reason", "?")
        for r in state.get("eval_results", []):
            if r.get("is_best"):
                best_val = r.get("metrics", {}).get(primary_metric)
                best_model = r.get("model_id", best_model)
                break

        # Parse timestamp from run_id (YYYYMMDD_HHMMSS)
        run_id = sub.name
        try:
            from datetime import datetime
            ts = datetime.strptime(run_id, "%Y%m%d_%H%M%S")
            timestamp = ts.strftime("%d %b %Y, %H:%M:%S")
        except ValueError:
            timestamp = run_id

        dataset_path = state.get("dataset_path", "")
        dataset_name = Path(dataset_path).stem if dataset_path else "?"

        runs.append({
            "run_id": run_id,
            "run_dir": str(sub),
            "dataset": dataset_name,
            "task_type": task_type,
            "best_model": best_model,
            "primary_metric": primary_metric,
            "best_metric_value": best_val,
            "stop_reason": state.get("stop_reason", "?"),
            "n_models": len(state.get("trained_models", [])),
            "n_iterations": state.get("iteration", 0),
            "has_report": (sub / "report.md").exists(),
            "timestamp": timestamp,
            "state": state,  # full state for loading into dashboard
        })

    return runs
