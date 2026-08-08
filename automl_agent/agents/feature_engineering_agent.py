"""
Feature Engineering Agent.

Reads ONLY critic-approved diagnoses and proposes + applies concrete feature changes.
Never acts on rejected diagnoses.

Output appended to PipelineState:
  - feature_changes
  - _feature_df_path (updated path to parquet with engineered features)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from automl_agent.llm_client import get_llm, get_mock_mode, invoke_llm
from automl_agent.state import FeatureChange, PipelineState
from automl_agent.tools.feature_tools import (
    add_aggregate_feature,
    add_interaction_feature,
    apply_class_weighting,
    bin_feature,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a feature engineering agent in an ML pipeline.
You receive VERIFIED diagnoses of model failures (verified by a critic agent).
Your job: propose 1-3 concrete, targeted feature engineering changes that directly address each diagnosis.

Rules:
- Only act on the approved diagnoses provided — do not invent new problems.
- Each change must be directly tied to a specific diagnosis_id.
- Prefer simple, interpretable changes. Don't engineer noise.
- Available change types:
    * "add_interaction": multiply or divide two existing columns → specify col_a, col_b, operation
    * "bin_feature": bin a continuous feature → specify col, strategy (quantile/uniform), n_bins
    * "add_aggregate": group-level aggregate → specify groupby_col, agg_col, agg_fn (mean/std/count)
    * "apply_class_weight": change model to use balanced class weighting (for imbalance issues)
    * "no_change": if the diagnosis cannot be addressed by feature engineering

- For class_imbalance issues: prefer "apply_class_weight" over feature changes.
- For feature_missing issues: prefer imputation was done in cleaning; use "add_aggregate" to capture group info.
- For distribution_shift: use "bin_feature" or "add_interaction" to help the model capture the pattern.

Available columns in the dataset: {available_columns}

Respond ONLY with valid JSON:
{
  "proposed_changes": [
    {
      "diagnosis_id": "<id>",
      "change_type": "add_interaction" | "bin_feature" | "add_aggregate" | "apply_class_weight" | "no_change",
      "description": "<plain-English description of what this does>",
      "justification": "<why this addresses the cited diagnosis>",
      "params": {
        // For add_interaction: { "col_a": "...", "col_b": "...", "operation": "multiply" }
        // For bin_feature:      { "col": "...", "strategy": "quantile", "n_bins": 4 }
        // For add_aggregate:    { "groupby_col": "...", "agg_col": "...", "agg_fn": "mean" }
        // For apply_class_weight: {}
        // For no_change: {}
      }
    }
  ]
}
"""


def run_feature_engineering_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Feature Engineering Agent.

    Reads: state["critic_review"], state["error_analysis"],
           state["_feature_df_path"], state["candidate_models"],
           state["iteration"]
    Appends: state["feature_changes"]
    Updates: state["_feature_df_path"], state["candidate_models"] (for class weighting)
    """
    logger.info("▶ Feature Engineering Agent starting...")
    mock_mode = get_mock_mode()
    llm = get_llm(mock_mode=mock_mode)

    iteration = state.get("iteration", 0)
    critic_reviews = state.get("critic_review", [])
    error_analysis = state.get("error_analysis", [])

    # Only approved diagnoses from this iteration
    approved_ids = {
        r["diagnosis_id"]
        for r in critic_reviews
        if r.get("verdict") == "supported" and r.get("iteration") == iteration
    }
    approved_diagnoses = [
        d for d in error_analysis
        if d["diagnosis_id"] in approved_ids
    ]

    if not approved_diagnoses:
        logger.info("  No approved diagnoses — skipping feature engineering.")
        return state

    # Load current feature df
    df_path = state.get("_feature_df_path") or state.get("_cleaned_df_path")
    df = pd.read_parquet(df_path)
    target = state["target_column"]
    available_columns = [c for c in df.columns if c != target]

    system = SYSTEM_PROMPT.replace("{available_columns}", str(available_columns))

    user_prompt = f"""Approved diagnoses for iteration {iteration}:
{json.dumps(approved_diagnoses, indent=2)}

Available feature columns: {available_columns}

Propose targeted feature engineering changes."""

    raw_response = invoke_llm(
        llm, system, user_prompt,
        agent_name="feature_engineering",
        mock_mode=mock_mode,
    )

    try:
        parsed = json.loads(raw_response)
        proposals = parsed.get("proposed_changes", [])
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
        proposals = parsed.get("proposed_changes", [])

    feature_changes = list(state.get("feature_changes", []))
    candidate_models = list(state.get("candidate_models", []))

    for proposal in proposals:
        change_type = proposal.get("change_type", "no_change")
        params = proposal.get("params", {})
        diagnosis_id = proposal.get("diagnosis_id", "")
        description = proposal.get("description", "")
        justification = proposal.get("justification", "")

        log_entry = None
        try:
            if change_type == "add_interaction":
                df, log_entry = add_interaction_feature(
                    df,
                    col_a=params.get("col_a", ""),
                    col_b=params.get("col_b", ""),
                    operation=params.get("operation", "multiply"),
                )
                logger.info(f"  ✓ Added interaction: {log_entry.get('new_column')}")

            elif change_type == "bin_feature":
                df, log_entry = bin_feature(
                    df,
                    col=params.get("col", ""),
                    strategy=params.get("strategy", "quantile"),
                    n_bins=int(params.get("n_bins", 4)),
                )
                logger.info(f"  ✓ Binned feature: {log_entry.get('new_column')}")

            elif change_type == "add_aggregate":
                df, log_entry = add_aggregate_feature(
                    df,
                    groupby_col=params.get("groupby_col", ""),
                    agg_col=params.get("agg_col", ""),
                    agg_fn=params.get("agg_fn", "mean"),
                )
                logger.info(f"  ✓ Added aggregate: {log_entry.get('new_column')}")

            elif change_type == "apply_class_weight":
                # Apply to each candidate model config
                updated_candidates = []
                for model_cfg in candidate_models:
                    updated_cfg, log_entry = apply_class_weighting(model_cfg)
                    updated_candidates.append(updated_cfg)
                candidate_models = updated_candidates
                log_entry = log_entry or {"change_type": "class_weight", "applied": True}
                logger.info("  ✓ Applied class weighting to candidate models.")

            elif change_type == "no_change":
                logger.info(f"  — No change proposed for diagnosis '{diagnosis_id}'.")
                continue

        except Exception as e:
            logger.error(f"  ✗ Feature change '{change_type}' failed: {e}")
            continue

        if log_entry:
            feature_change: FeatureChange = {
                "iteration": iteration,
                "diagnosis_id": diagnosis_id,
                "change_type": change_type,
                "description": description,
                "justification": justification,
                "columns_affected": log_entry.get("columns_affected",
                                                  [log_entry.get("new_column", "")]),
            }
            feature_changes.append(feature_change)

    # Save updated dataframe
    from automl_agent.run_utils import get_run_dir
    run_dir = get_run_dir()
    feature_path = str(run_dir / f"features_iter{iteration}.parquet")
    df.to_parquet(feature_path, index=False)
    logger.info(f"  ✓ Feature-engineered data saved → {feature_path}")

    return {
        **state,
        "feature_changes": feature_changes,
        "candidate_models": candidate_models,
        "_feature_df_path": feature_path,
    }
