"""
Error Analysis Agent — the core differentiator.

Uses diagnostic tools (confusion matrix, misclassified samples, slice performance)
and asks the LLM to produce a written, structured diagnosis of WHY the model is failing.

Output appended to PipelineState:
  - error_analysis (list of ErrorAnalysisEntry)
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import pandas as pd

from automl_agent.llm_client import get_llm, get_mock_mode, invoke_llm
from automl_agent.state import ErrorAnalysisEntry, PipelineState
from automl_agent.tools.model_tools import (
    get_confusion_matrix,
    get_misclassified_samples,
    get_residuals,
    get_worst_predictions,
    load_model,
    slice_performance,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_CLASSIFICATION = """You are an expert ML error analysis agent.
You will be given:
  1. Confusion matrix with per-class precision/recall
  2. The top misclassified samples (worst model errors)
  3. Slice performance — how the model performs across different feature subgroups

Your task: write a detailed, evidence-based diagnosis of WHY the model is failing.
Be specific. Reference the actual numbers and patterns you see.
Do NOT make generic statements like "the model needs more data" without citing evidence.

Common failure patterns to look for:
  - Class imbalance: one class dominates predictions
  - Systematic feature-based errors: errors cluster where a specific feature is missing/extreme
  - Distribution shift: a subgroup the model consistently underperforms on
  - Threshold miscalibration: false positive / false negative rate asymmetry

Respond ONLY with valid JSON:
{
  "diagnoses": [
    {
      "issue": "<short tag: class_imbalance | feature_missing | distribution_shift | threshold | other>",
      "affected_class": "<class label or null>",
      "evidence_cited": "<specific numbers from the data: e.g. 'minority recall=0.61 vs majority=0.89'>",
      "reasoning": "<2-4 sentences: what the data shows and why this is the likely cause>",
      "structured_tags": { "<key>": "<value>" }
    }
  ]
}
"""

SYSTEM_PROMPT_REGRESSION = """You are an expert ML error analysis agent.
You will be given:
  1. Residual statistics (mean, std, max absolute residual)
  2. The worst predictions (largest absolute errors)
  3. Slice performance — how the model performs across different feature subgroups

Your task: write a detailed, evidence-based diagnosis of WHY the model is failing.
Be specific. Reference the actual numbers and patterns you see.

Common regression failure patterns:
  - Systematic under/over-prediction for a specific range
  - High-leverage outliers dominating error
  - Feature interaction not captured by current model
  - Heteroscedasticity: error variance varies with predicted value

Respond ONLY with valid JSON:
{
  "diagnoses": [
    {
      "issue": "<short tag: underprediction | overprediction | outlier_driven | heteroscedasticity | feature_interaction | other>",
      "affected_class": null,
      "evidence_cited": "<specific numbers from the data>",
      "reasoning": "<2-4 sentences>",
      "structured_tags": { "<key>": "<value>" }
    }
  ]
}
"""


def run_error_analysis_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Error Analysis Agent.

    Reads: state["_current_best_model_id"], state["trained_models"],
           state["task_type"], state["iteration"]
    Appends: state["error_analysis"]
    """
    logger.info("▶ Error Analysis Agent starting...")
    mock_mode = get_mock_mode()
    llm = get_llm(mock_mode=mock_mode)

    task_type = state["task_type"]
    iteration = state.get("iteration", 0)
    best_model_id = state.get("_current_best_model_id", "")
    trained_models = state.get("trained_models", [])

    # Find best model entry
    best_entry = next(
        (m for m in trained_models if m["model_id"] == best_model_id), None
    )
    if not best_entry or not best_entry.get("artifact_path"):
        logger.warning("  No best model found — skipping error analysis.")
        return state

    estimator = load_model(best_entry["artifact_path"])

    # Load test split
    from automl_agent.run_utils import get_run_dir
    run_dir = get_run_dir()
    X_test = pd.read_parquet(str(run_dir / "X_test.parquet"))
    y_test_df = pd.read_parquet(str(run_dir / "y_test.parquet"))
    y_test = y_test_df.iloc[:, 0]

    # Gather diagnostics
    diagnostics: dict = {"model_id": best_model_id, "task_type": task_type}

    if task_type == "classification":
        cm_data = get_confusion_matrix(estimator, X_test, y_test)
        misclassified = get_misclassified_samples(estimator, X_test, y_test, n=20)
        diagnostics["confusion_matrix"] = cm_data
        diagnostics["misclassified_summary"] = {
            "n_misclassified": len(misclassified),
            "sample_rows": misclassified.head(5).to_dict(orient="records"),
        }
        system_prompt = SYSTEM_PROMPT_CLASSIFICATION
    else:
        residual_data = get_residuals(estimator, X_test, y_test)
        worst_preds = get_worst_predictions(estimator, X_test, y_test, n=20)
        diagnostics["residual_stats"] = residual_data
        diagnostics["worst_predictions_summary"] = {
            "n_worst": len(worst_preds),
            "sample_rows": worst_preds.head(5).to_dict(orient="records"),
        }
        system_prompt = SYSTEM_PROMPT_REGRESSION

    # Slice performance on first few columns
    slice_cols = [c for c in X_test.columns[:5] if X_test[c].nunique() <= 10]
    slice_results = {}
    for col in slice_cols[:3]:
        slice_results[col] = slice_performance(estimator, X_test, y_test, col, task_type)
    diagnostics["slice_performance"] = slice_results

    user_prompt = f"""Current iteration: {iteration}
Model: {best_model_id}

Diagnostic data:
{json.dumps(diagnostics, indent=2, default=str)}

Diagnose the model's failures."""

    raw_response = invoke_llm(
        llm, system_prompt, user_prompt,
        agent_name="error_analysis",
        mock_mode=mock_mode,
    )

    try:
        parsed = json.loads(raw_response)
        raw_diagnoses = parsed.get("diagnoses", [])
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
        raw_diagnoses = parsed.get("diagnoses", [])

    error_analysis = list(state.get("error_analysis", []))
    for diag in raw_diagnoses:
        diagnosis_id = f"diag_{iteration}_{uuid.uuid4().hex[:6]}"
        entry: ErrorAnalysisEntry = {
            "diagnosis_id": diagnosis_id,
            "iteration": iteration,
            "model_id": best_model_id,
            "issue": diag.get("issue", "unknown"),
            "affected_class": diag.get("affected_class"),
            "evidence_cited": diag.get("evidence_cited", ""),
            "reasoning": diag.get("reasoning", ""),
            "structured_tags": diag.get("structured_tags", {}),
        }
        error_analysis.append(entry)
        logger.info(f"  ✓ Diagnosis: [{entry['issue']}] {entry['evidence_cited'][:80]}...")

    logger.info(f"  ✓ {len(raw_diagnoses)} diagnoses produced.")
    return {**state, "error_analysis": error_analysis}
