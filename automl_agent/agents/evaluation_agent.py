"""
Evaluation Agent.

Evaluates each trained model, picks the best by primary metric,
and appends results to eval_results.

Output appended to PipelineState:
  - eval_results
  - _current_best_model_id
  - _previous_best_metric (for plateau detection)

Bug fixes:
  B1: _previous_best_metric is only updated when a model actually succeeds,
      preventing the plateau checker from seeing inf vs inf.
  B4: When all models in an iteration fail, the global best model/metric from
      previous iterations is preserved rather than reset to inf.
"""
from __future__ import annotations

import logging

import pandas as pd

from automl_agent.run_utils import get_run_dir
from automl_agent.state import EvalResult, PipelineState
from automl_agent.tools.model_tools import cross_validate_model, evaluate_model, load_model
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
    higher_is_better = primary_metric not in ("rmse", "mae")

    # Load test split saved by Training Agent
    run_dir = get_run_dir()
    X_test = pd.read_parquet(str(run_dir / "X_test.parquet"))
    y_test_df = pd.read_parquet(str(run_dir / "y_test.parquet"))
    y_test = y_test_df.iloc[:, 0]

    # Load training set for cross-validation
    X_train_path = run_dir / "X_train.parquet"
    y_train_path = run_dir / "y_test.parquet"  # y_test shares index but we need y_train
    X_train = pd.read_parquet(str(run_dir / "X_train.parquet")) if (run_dir / "X_train.parquet").exists() else None
    # We'll re-derive y_train from the full parquet split saved by training_agent
    y_train = None
    if X_train is not None:
        # Load full df to derive y_train (same split seed=42)
        try:
            _df_path = state.get("_feature_df_path") or state.get("_cleaned_df_path", "")
            if _df_path:
                import pandas as _pd_local
                _full_df = _pd_local.read_parquet(_df_path)
                from automl_agent.tools.model_tools import split_data as _split
                _, _, y_train, _ = _split(_full_df, state["target_column"], test_size=0.2, random_state=42)
        except Exception:
            pass  # CV just won't run if we can't reconstruct y_train

    trained_models = state.get("trained_models", [])
    # Only evaluate models from THIS iteration
    current_iter_models = [m for m in trained_models if m.get("iteration") == iteration]

    eval_results = list(state.get("eval_results", []))

    # Per-iteration best (reset each call)
    iter_best_metric = float("inf") if not higher_is_better else -float("inf")
    iter_best_model_id = ""

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

            is_iter_best = (
                (higher_is_better and primary_val > iter_best_metric) or
                (not higher_is_better and primary_val < iter_best_metric)
            )
            if is_iter_best:
                iter_best_metric = primary_val
                iter_best_model_id = model_id

            result: EvalResult = {
                "model_id": model_id,
                "iteration": iteration,
                "metrics": metrics,
                "is_best": False,  # updated below
                "cv_mean": None,
                "cv_std": None,
                "cv_folds": None,
            }

            # Cross-validation on training set (5-fold)
            if X_train is not None and y_train is not None:
                try:
                    cv_mean, cv_std = cross_validate_model(
                        estimator, X_train, y_train,
                        task_type=task_type, cv_folds=5,
                    )
                    result["cv_mean"] = cv_mean
                    result["cv_std"] = cv_std
                    result["cv_folds"] = 5
                    logger.info(f"    CV: {primary_metric}={cv_mean:.4f} ± {cv_std:.4f} (5-fold)")
                except Exception as cv_err:
                    logger.warning(f"    CV failed for '{model_id}': {cv_err}")

            eval_results.append(result)
            logger.info(f"  ✓ {model_id}: {primary_metric}={primary_val:.4f}")

        except Exception as e:
            logger.error(f"  ✗ Evaluation failed for '{model_id}': {e}")

    # ── Global best tracking (B1 + B4 fix) ───────────────────────────────────
    prev_best_model_id = state.get("_current_best_model_id", "")
    prev_best_metric = state.get("_previous_best_metric", None)

    if iter_best_model_id:
        # At least one model succeeded this iteration
        if prev_best_metric is None:
            # First ever evaluation — accept this iteration's result
            global_best_model_id = iter_best_model_id
            global_best_metric = iter_best_metric
        else:
            improved = (
                (higher_is_better and iter_best_metric > prev_best_metric) or
                (not higher_is_better and iter_best_metric < prev_best_metric)
            )
            global_best_model_id = iter_best_model_id if improved else prev_best_model_id
            global_best_metric = iter_best_metric if improved else prev_best_metric
    else:
        # B4: All models failed — preserve historical best to avoid clobbering plateau state
        logger.warning("  All models this iteration failed — preserving previous best.")
        global_best_model_id = prev_best_model_id
        global_best_metric = prev_best_metric if prev_best_metric is not None else (
            float("inf") if not higher_is_better else -float("inf")
        )

    # Mark the global best across all eval results
    for r in eval_results:
        r["is_best"] = r["model_id"] == global_best_model_id

    best_display = f"{global_best_metric:.4f}" if global_best_metric not in (float("inf"), -float("inf")) else "N/A"
    logger.info(f"  ✓ Best model this run: '{global_best_model_id}' ({primary_metric}={best_display})")

    return {
        **state,
        "eval_results": eval_results,
        "_current_best_model_id": global_best_model_id,
        "_previous_best_metric": global_best_metric,
    }
