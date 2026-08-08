"""
Training Agent.

Trains each candidate model, optionally using Optuna HPO to find
best hyperparameters before fitting on the full training set.

Optuna is bypassed in mock mode (n_trials=1) to keep tests instant.
In live mode: OPTUNA_TRIALS trials (default=20, set via env or config.py).

Output appended to PipelineState:
  - trained_models (list of TrainedModel)
"""
from __future__ import annotations

import logging

import pandas as pd

from automl_agent.llm_client import get_mock_mode
from automl_agent.run_utils import get_run_dir
from automl_agent.state import PipelineState, TrainedModel
from automl_agent.tools.model_tools import optuna_tune, split_data, train_model
from config import OPTUNA_TRIALS

logger = logging.getLogger(__name__)


def run_training_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Training Agent.

    Reads: state["_feature_df_path"], state["target_column"],
           state["candidate_models"], state["task_type"], state["iteration"]
    Appends: state["trained_models"]
    Also stores split data paths for reuse by Evaluation/Error Analysis.
    """
    logger.info("▶ Training Agent starting...")

    mock_mode = get_mock_mode()
    df_path = state.get("_feature_df_path") or state.get("_cleaned_df_path")
    df = pd.read_parquet(df_path)
    target = state["target_column"]
    task_type = state["task_type"]
    iteration = state.get("iteration", 0)
    candidates = state.get("candidate_models", [])

    # Optuna trials: 1 in mock (no real search), OPTUNA_TRIALS in live
    n_trials = 1 if mock_mode else OPTUNA_TRIALS

    logger.info(f"  Dataset shape: {df.shape}, target='{target}', task='{task_type}'")
    logger.info(f"  Training {len(candidates)} candidate(s) — iteration {iteration}"
                f" | HPO trials: {n_trials}")

    # Split data (consistent seed per run)
    X_train, X_test, y_train, y_test = split_data(df, target, test_size=0.2, random_state=42)

    # Save splits for downstream agents
    run_dir = get_run_dir()
    X_test.to_parquet(str(run_dir / "X_test.parquet"), index=False)
    y_test.to_frame().to_parquet(str(run_dir / "y_test.parquet"), index=False)
    X_train.to_parquet(str(run_dir / "X_train.parquet"), index=False)

    artifact_dir = run_dir / "models"
    trained_models = list(state.get("trained_models", []))

    for candidate in candidates:
        # Rebase model ID to the *current* iteration so each loop creates new entries
        original_id = candidate["model_id"]
        bare_id = original_id.split("_", 1)[1] if original_id.startswith("iter") else original_id
        model_id = f"iter{iteration}_{bare_id}"
        model_family = candidate["model_family"]
        base_params = candidate.get("params", {})

        logger.info(f"    Tuning '{model_id}' ({model_family}) with {n_trials} Optuna trial(s)...")
        try:
            # ── HPO: find best params ────────────────────────────────────────
            tuned_params, cv_score = optuna_tune(
                X_train, y_train,
                model_family=model_family,
                task_type=task_type,
                n_trials=n_trials,
                cv_folds=3,
                random_state=42,
            )
            # Merge base_params (class_weight, etc.) into tuned params
            merged_params = {**base_params, **tuned_params}
            logger.info(f"    ✓ HPO done — cv_score={cv_score:.4f} | best_params={tuned_params}")

            # ── Final fit on full training set ───────────────────────────────
            iter_candidate = {**candidate, "model_id": model_id, "params": merged_params}
            estimator, artifact_path = train_model(
                X_train, y_train,
                model_config=iter_candidate,
                task_type=task_type,
                artifact_dir=artifact_dir,
            )
            trained_entry: TrainedModel = {
                "model_id": model_id,
                "model_family": model_family,
                "params": merged_params,
                "artifact_path": artifact_path,
                "iteration": iteration,
            }
            trained_models.append(trained_entry)
            logger.info(f"    ✓ '{model_id}' trained → {artifact_path}")

        except Exception as e:
            logger.error(f"    ✗ Failed to train/tune '{model_id}': {e}")
            trained_models.append({
                "model_id": model_id,
                "model_family": model_family,
                "params": base_params,
                "artifact_path": None,
                "iteration": iteration,
                "error": str(e),
            })

    logger.info(f"  ✓ Training complete. {len(trained_models)} total models in state.")
    return {**state, "trained_models": trained_models}
