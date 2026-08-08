"""
Aggressive Strategy Agent.

Proposes the model family and hyperparameter search space that maximises raw performance.
Allowed to choose high-variance models (XGBoost, LightGBM) and wide search spaces.
Must explicitly acknowledge the overfitting risk it is accepting.

Output appended to PipelineState:
  - strategy_proposals (one entry with agent="aggressive")
"""
from __future__ import annotations

import json
import logging

from automl_agent.llm_client import get_llm, get_mock_mode, invoke_llm
from automl_agent.state import PipelineState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the AGGRESSIVE strategy agent in a model selection debate.
Your goal: propose the model family and hyperparameter search space that maximises raw predictive performance.
You may choose high-variance, complex models (XGBoost, LightGBM, deep Random Forests).
You may propose wide hyperparameter ranges that allow the search to explore aggressively.

You MUST:
1. Justify your choice based on the actual dataset characteristics provided.
2. Explicitly acknowledge the overfitting risk you are accepting (be specific — mention dataset size, noise, etc.).
3. Choose from these model families ONLY: logistic_regression, random_forest, gradient_boosting, xgboost, lightgbm

Respond ONLY with valid JSON:
{
  "model_family": "<one of the supported families>",
  "hyperparam_ranges": {
    "<param_name>": [<min>, <max>]  // or a list of discrete values
  },
  "justification": "<2-3 sentences: why this model for THIS dataset>",
  "acknowledged_tradeoff": "<1-2 sentences: what overfitting risk you accept and why it's worth it>"
}
"""


def run_aggressive_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Aggressive Strategy Agent.

    Reads: state["raw_df_summary"], state["task_type"]
    Appends: state["strategy_proposals"] (agent="aggressive")
    """
    logger.info("▶ Aggressive Strategy Agent starting...")
    mock_mode = get_mock_mode()
    llm = get_llm(mock_mode=mock_mode)

    summary = state.get("raw_df_summary", {})
    task_type = state.get("task_type", "classification")

    user_prompt = f"""Dataset summary:
- Rows: {summary.get('n_rows', '?')}
- Columns: {summary.get('n_cols', '?')}
- Task type: {task_type}
- Imbalance ratio: {summary.get('imbalance_ratio', 'N/A')}
- Class balance: {json.dumps(summary.get('class_balance', {}), indent=2)}

Full column profile (abbreviated):
{json.dumps({k: {p: v for p, v in info.items() if p in ('dtype', 'null_pct', 'n_unique', 'is_numeric')}
             for k, info in summary.get('columns', {}).items()}, indent=2)}

Propose an AGGRESSIVE strategy optimising for raw performance."""

    raw_response = invoke_llm(
        llm, SYSTEM_PROMPT, user_prompt,
        agent_name="aggressive",
        mock_mode=mock_mode,
    )

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        # Extract JSON block if wrapped in markdown
        import re
        match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}

    proposal = {
        "agent": "aggressive",
        "model_family": parsed.get("model_family", "xgboost"),
        "hyperparam_ranges": parsed.get("hyperparam_ranges", {}),
        "justification": parsed.get("justification", ""),
        "acknowledged_tradeoff": parsed.get("acknowledged_tradeoff", ""),
    }
    logger.info(f"  ✓ Aggressive proposal: {proposal['model_family']}")

    proposals = list(state.get("strategy_proposals", []))
    proposals.append(proposal)

    return {**state, "strategy_proposals": proposals}
