"""
Arbiter Agent.

Reads both the Aggressive and Conservative proposals plus the dataset summary.
Picks one proposal, blends them, or requests a hybrid — with a written justification
tied explicitly to the actual dataset characteristics.

Produces candidate_models list for Training Agent.

Output written to PipelineState:
  - arbiter_decision
  - candidate_models
"""
from __future__ import annotations

import json
import logging
import uuid

from automl_agent.llm_client import get_llm, get_mock_mode, invoke_llm
from automl_agent.state import ArbiterDecision, CandidateModel, PipelineState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the ARBITER agent in a model selection debate.
You have received two competing strategy proposals: one AGGRESSIVE (raw performance),
one CONSERVATIVE (robustness/interpretability). You must decide which to follow.

Your decision must be grounded in the actual dataset characteristics:
- Small datasets (< 1000 rows): favour Conservative to avoid overfitting.
- Large datasets (> 10000 rows): Aggressive has more room to shine.
- High class imbalance (ratio > 5): note this explicitly.
- High null/noise: favour Conservative.

You may:
  A) Choose the Aggressive proposal as-is.
  B) Choose the Conservative proposal as-is.
  C) Create a BLEND: pick the Conservative model family but allow wider params than Conservative proposed,
     OR pick the Aggressive model family but add regularisation constraints.

You MUST provide dataset-specific justification (reference actual numbers from the summary).
Do NOT use generic statements like "it depends" without specifics.

After picking a strategy, also list up to 3 candidate model configurations to train
(e.g., the chosen model at different param settings, or one baseline + one main choice).

Respond ONLY with valid JSON:
{
  "chosen_strategy": "aggressive" | "conservative" | "blend",
  "model_family": "<chosen family>",
  "hyperparam_ranges": { "<param>": [<min>, <max>] },
  "justification": "<3-5 sentences citing actual dataset numbers>",
  "candidate_models": [
    {
      "model_id": "<unique_id>",
      "model_family": "<family>",
      "params": { "<param>": <value> }
    },
    ...
  ]
}
"""


def run_arbiter_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Arbiter Agent.

    Reads: state["strategy_proposals"], state["raw_df_summary"], state["task_type"]
    Writes: state["arbiter_decision"], state["candidate_models"]
    """
    logger.info("▶ Arbiter Agent starting...")
    mock_mode = get_mock_mode()
    llm = get_llm(mock_mode=mock_mode)

    proposals = state.get("strategy_proposals", [])
    summary = state.get("raw_df_summary", {})
    task_type = state.get("task_type", "classification")
    iteration = state.get("iteration", 0)

    agg = next((p for p in proposals if p["agent"] == "aggressive"), {})
    con = next((p for p in proposals if p["agent"] == "conservative"), {})

    user_prompt = f"""Dataset characteristics:
- Rows: {summary.get('n_rows', '?')}
- Columns: {summary.get('n_cols', '?')}
- Task type: {task_type}
- Imbalance ratio: {summary.get('imbalance_ratio', 'N/A')}
- Class balance: {json.dumps(summary.get('class_balance', {}), indent=2)}

AGGRESSIVE proposal:
{json.dumps(agg, indent=2)}

CONSERVATIVE proposal:
{json.dumps(con, indent=2)}

Current iteration: {iteration}
Make your arbiter decision and produce candidate model configurations."""

    raw_response = invoke_llm(
        llm, SYSTEM_PROMPT, user_prompt,
        agent_name="arbiter",
        mock_mode=mock_mode,
    )

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}

    arbiter_decision: ArbiterDecision = {
        "chosen_strategy": parsed.get("chosen_strategy", "conservative"),
        "model_family": parsed.get("model_family", "random_forest"),
        "hyperparam_ranges": parsed.get("hyperparam_ranges", {}),
        "justification": parsed.get("justification", ""),
    }

    # Build candidate_models from LLM output OR auto-generate from chosen family
    raw_candidates = parsed.get("candidate_models", [])
    candidate_models: list[CandidateModel] = []

    if raw_candidates:
        for c in raw_candidates:
            model_id = c.get("model_id") or f"model_{uuid.uuid4().hex[:6]}"
            candidate_models.append({
                "model_id": f"iter{iteration}_{model_id}",
                "model_family": c.get("model_family", arbiter_decision["model_family"]),
                "params": c.get("params", {}),
            })
    else:
        # Fallback: generate 2 candidates from arbiter decision
        family = arbiter_decision["model_family"]
        ranges = arbiter_decision["hyperparam_ranges"]
        # Baseline (conservative end of ranges) + main (midpoint)
        def _pick_val(v, idx=0):
            if isinstance(v, list) and len(v) >= 2:
                return v[idx]
            return v

        baseline_params = {k: _pick_val(v, 0) for k, v in ranges.items()}
        main_params = {k: _pick_val(v, 1) for k, v in ranges.items()}

        candidate_models = [
            {"model_id": f"iter{iteration}_baseline", "model_family": family, "params": baseline_params},
            {"model_id": f"iter{iteration}_main", "model_family": family, "params": main_params},
        ]

    logger.info(f"  ✓ Arbiter chose: {arbiter_decision['chosen_strategy']} → {arbiter_decision['model_family']}")
    logger.info(f"  ✓ {len(candidate_models)} candidate models queued.")

    return {
        **state,
        "arbiter_decision": arbiter_decision,
        "candidate_models": candidate_models,
    }
