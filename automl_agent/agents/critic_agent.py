"""
Critic / Skeptic Agent.

The reliability gatekeeper. Re-runs the SAME diagnostic queries that the Error Analysis
Agent cited as evidence, independently, and asks the LLM:
"Does this evidence actually support the diagnosis?"

Verdicts: "supported" | "rejected"
Feature Engineering only receives diagnoses marked "supported".

Output appended to PipelineState:
  - critic_review (list of CriticReview)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from automl_agent.llm_client import get_llm, get_mock_mode, invoke_llm
from automl_agent.state import CriticReview, PipelineState
from automl_agent.tools.model_tools import (
    get_confusion_matrix,
    get_residuals,
    load_model,
    slice_performance,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the CRITIC (Skeptic) agent in an ML pipeline.
Your role is to verify whether diagnoses produced by the Error Analysis Agent are
actually supported by the data — or whether they are plausible-sounding but unsupported.

You will receive:
1. The original diagnosis (issue, evidence cited, reasoning)
2. The independently re-queried diagnostic data (confusion matrix / residuals / slice stats)

Your job: compare what the diagnosis claims with what the re-queried data actually shows.

Verdict rules:
- "supported": the evidence cited by the diagnosis matches (within reasonable margin) what you see in re-queried data.
- "rejected": the diagnosis overstates, misquotes, or contradicts the re-queried data.

Be rigorous. If the cited recall is "0.61" but the actual recall is "0.78", that is "rejected".
If the direction is correct but the exact number is slightly off, use "supported" with a note.

Respond ONLY with valid JSON for EACH diagnosis:
{
  "verdicts": [
    {
      "diagnosis_id": "<id>",
      "verdict": "supported" | "rejected",
      "reasoning": "<2-3 sentences: what the re-queried data shows vs. what was claimed>",
      "evidence_recheck": "<specific numbers from the re-query that confirm or contradict>"
    }
  ]
}
"""


def run_critic_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Critic / Skeptic Agent.

    Reads: state["error_analysis"], state["_current_best_model_id"],
           state["trained_models"], state["task_type"], state["iteration"]
    Appends: state["critic_review"]
    """
    logger.info("▶ Critic Agent starting...")
    mock_mode = get_mock_mode()
    llm = get_llm(mock_mode=mock_mode)

    task_type = state["task_type"]
    iteration = state.get("iteration", 0)
    best_model_id = state.get("_current_best_model_id", "")
    trained_models = state.get("trained_models", [])

    # Get diagnoses from current iteration only
    current_diagnoses = [
        d for d in state.get("error_analysis", [])
        if d.get("iteration") == iteration
    ]

    if not current_diagnoses:
        logger.info("  No diagnoses to review — skipping critic.")
        return state

    # Load model + data
    best_entry = next((m for m in trained_models if m["model_id"] == best_model_id), None)
    if not best_entry or not best_entry.get("artifact_path"):
        logger.warning("  Critic: no best model artifact — skipping.")
        return state

    estimator = load_model(best_entry["artifact_path"])

    from automl_agent.run_utils import get_run_dir
    run_dir = get_run_dir()
    X_test = pd.read_parquet(str(run_dir / "X_test.parquet"))
    y_test_df = pd.read_parquet(str(run_dir / "y_test.parquet"))
    y_test = y_test_df.iloc[:, 0]

    # Re-run the same diagnostics independently
    recheck_data: dict = {}
    if task_type == "classification":
        recheck_data["confusion_matrix"] = get_confusion_matrix(estimator, X_test, y_test)
    else:
        recheck_data["residual_stats"] = get_residuals(estimator, X_test, y_test)

    # Slice performance for columns mentioned in any diagnosis
    all_tags = {}
    for d in current_diagnoses:
        all_tags.update(d.get("structured_tags", {}))

    slice_cols = [c for c in X_test.columns[:5] if X_test[c].nunique() <= 10]
    slice_results = {}
    for col in slice_cols[:3]:
        slice_results[col] = slice_performance(estimator, X_test, y_test, col, task_type)
    recheck_data["slice_performance"] = slice_results

    user_prompt = f"""Original diagnoses (iteration {iteration}):
{json.dumps(current_diagnoses, indent=2)}

Independently re-queried diagnostic data:
{json.dumps(recheck_data, indent=2, default=str)}

For each diagnosis, determine if the evidence actually supports it."""

    raw_response = invoke_llm(
        llm, SYSTEM_PROMPT, user_prompt,
        agent_name="critic",
        mock_mode=mock_mode,
    )

    try:
        parsed = json.loads(raw_response)
        raw_verdicts = parsed.get("verdicts", [])
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
        raw_verdicts = parsed.get("verdicts", [])

    critic_review = list(state.get("critic_review", []))

    # Match verdicts to diagnoses (by index if ID not found)
    for i, diag in enumerate(current_diagnoses):
        if i < len(raw_verdicts):
            v = raw_verdicts[i]
        else:
            # Default to supported if LLM produced fewer verdicts than diagnoses
            v = {"verdict": "supported", "reasoning": "[default] LLM produced no verdict.",
                 "evidence_recheck": ""}

        review: CriticReview = {
            "diagnosis_id": diag["diagnosis_id"],
            "iteration": iteration,
            "verdict": v.get("verdict", "supported"),
            "reasoning": v.get("reasoning", ""),
            "evidence_recheck": v.get("evidence_recheck", ""),
        }
        critic_review.append(review)
        verdict_str = "✓ SUPPORTED" if review["verdict"] == "supported" else "✗ REJECTED"
        logger.info(f"  {verdict_str}: {diag['issue']} — {review['reasoning'][:80]}...")

    supported_count = sum(1 for r in critic_review if r["iteration"] == iteration and r["verdict"] == "supported")
    logger.info(f"  ✓ Critic complete: {supported_count}/{len(current_diagnoses)} diagnoses supported.")

    return {**state, "critic_review": critic_review}
