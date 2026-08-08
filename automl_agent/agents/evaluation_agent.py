"""
Evaluation Agent.

Evaluates each trained model, picks the best by primary metric,
and appends results to eval_results.

Output appended to PipelineState:
  - eval_results
  - _current_best_model_id
  - _previous_best_metric (for plateau detection)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from automl_agent.state import EvalResult, PipelineState
from automl_agent.tools.model_tools import evaluate_model, load_model
from config import PRIMARY_METRICS

logger = logging.getLogger(__name__)


def run_evaluation_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Evaluation Agent.

    Reads: state["trained_models"], state["task_type"], state["iteration"]
    Appends: state["eval_results"]
    Updates: state["_current_best_model_id"], state["_previous_best_metric"]
    """
    logger.info("▶ Evaluation Agent starting...")

    task_type = state["task_type"]
    iteration = state.get("iteration", 0)
    primary_metric = PRIMARY_METRICS[task_type]

    # Load test split saved by Training Agent
    from automl_agent.run_utils import get_run_dir
    run_dir = get_run_dir()
    X_test = pd.read_parquet(str(run_dir / "X_test.parquet"))
    y_test_df = pd.read_parquet(str(run_dir / "y_test.parquet"))
    y_test = y_test_df.iloc[:, 0]

    trained_models = state.get("trained_models", [])
    # Only evaluate models from this iteration
    current_iter_models = [m for m in trained_models if m.get("iteration") == iteration]

    eval_results = list(state.get("eval_results", []))
    best_metric = float("inf") if primary_metric == "rmse" else -float("inf")
    best_model_id = state.get("_current_best_model_id", "")

    # Higher is better for all metrics except RMSE/MAE
    higher_is_better = primary_metric not in ("rmse", "mae")

    for model_entry in current_iter_models:
        model_id = model_entry["model_id"]
        artifact_path = model_entry.get("artifact_path")

        if not artifact_path or model_entry.get("error"):
            logger.warning(f"  Skipping '{model_id}' — no artifact (training failed).")
            continue

        try:
            estimator = load_model(artifact_path)
            metrics = evaluate_model(estimator, X_test, y_test, task_type)
            primary_val = metrics.get(primary_metric, 0.0)

            is_best = (
                (higher_is_better and primary_val > best_metric) or
                (not higher_is_better and primary_val < best_metric)
            )
            if is_best:
                best_metric = primary_val
                best_model_id = model_id

            result: EvalResult = {
                "model_id": model_id,
                "iteration": iteration,
                "metrics": metrics,
                "is_best": False,  # will update after loop
            }
            eval_results.append(result)
            logger.info(f"  ✓ {model_id}: {primary_metric}={primary_val}")

        except Exception as e:
            logger.error(f"  ✗ Evaluation failed for '{model_id}': {e}")

    # Mark best model across ALL eval results
    for r in eval_results:
        r["is_best"] = r["model_id"] == best_model_id

    prev_best = state.get("_previous_best_metric", None)
    logger.info(f"  ✓ Best model this run: '{best_model_id}' ({primary_metric}={best_metric:.4f})")

    return {
        **state,
        "eval_results": eval_results,
        "_current_best_model_id": best_model_id,
        "_previous_best_metric": best_metric,
    }
