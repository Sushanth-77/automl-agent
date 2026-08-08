"""
Training Agent.

Trains each candidate model using the feature-engineered dataset.
Mostly mechanical — minimal LLM reasoning (just logs what it trained).

Output appended to PipelineState:
  - trained_models (list of TrainedModel)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from automl_agent.state import PipelineState, TrainedModel
from automl_agent.tools.model_tools import split_data, train_model

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

    df_path = state.get("_feature_df_path") or state.get("_cleaned_df_path")
    df = pd.read_parquet(df_path)
    target = state["target_column"]
    task_type = state["task_type"]
    iteration = state.get("iteration", 0)
    candidates = state.get("candidate_models", [])

    logger.info(f"  Dataset shape: {df.shape}, target='{target}', task='{task_type}'")
    logger.info(f"  Training {len(candidates)} candidate(s) — iteration {iteration}")

    # Split data (consistent seed per run)
    X_train, X_test, y_train, y_test = split_data(df, target, test_size=0.2, random_state=42)

    # Save splits for downstream agents
    from automl_agent.run_utils import get_run_dir
    run_dir = get_run_dir()
    X_test.to_parquet(str(run_dir / "X_test.parquet"), index=False)
    y_test.to_frame().to_parquet(str(run_dir / "y_test.parquet"), index=False)
    X_train.to_parquet(str(run_dir / "X_train.parquet"), index=False)

    artifact_dir = run_dir / "models"
    trained_models = list(state.get("trained_models", []))

    for candidate in candidates:
        model_id = candidate["model_id"]
        model_family = candidate["model_family"]
        params = candidate.get("params", {})

        logger.info(f"    Training '{model_id}' ({model_family}) ...")
        try:
            estimator, artifact_path = train_model(
                X_train, y_train,
                model_config=candidate,
                task_type=task_type,
                artifact_dir=artifact_dir,
            )
            trained_entry: TrainedModel = {
                "model_id": model_id,
                "model_family": model_family,
                "params": params,
                "artifact_path": artifact_path,
                "iteration": iteration,
            }
            trained_models.append(trained_entry)
            logger.info(f"    ✓ '{model_id}' trained → {artifact_path}")
        except Exception as e:
            logger.error(f"    ✗ Failed to train '{model_id}': {e}")
            # Don't crash the pipeline — log and continue
            trained_models.append({
                "model_id": model_id,
                "model_family": model_family,
                "params": params,
                "artifact_path": None,
                "iteration": iteration,
                "error": str(e),
            })

    logger.info(f"  ✓ Training complete. {len(trained_models)} total models in state.")
    return {**state, "trained_models": trained_models}
