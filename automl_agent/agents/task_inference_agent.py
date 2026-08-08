"""
Task-Type Inference Agent.

Inspects the target column's statistics and asks the LLM to decide:
  "classification" | "regression"

Output appended to PipelineState:
  - task_type
  - task_type_reasoning
"""
from __future__ import annotations

import json
import logging

from automl_agent.llm_client import get_llm, get_mock_mode, invoke_llm
from automl_agent.state import PipelineState
from automl_agent.tools.data_tools import inspect_target_column

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a machine learning task-type inference expert.
You will be given statistical properties of a target column from a tabular dataset.
Your job is to decide whether this is a CLASSIFICATION or REGRESSION task.

Rules:
- CLASSIFICATION: target has a small number of discrete categories (typically ≤ 20 unique values),
  or has an object/string dtype, or is clearly binary (0/1, True/False, Yes/No).
- REGRESSION: target is a continuous numeric variable with many unique values (typically > 20),
  a wide range, and no natural class structure.
- When uncertain (e.g., integer target with ~15 unique values), lean toward CLASSIFICATION
  if values are clearly labels, REGRESSION if they represent a measured quantity.

Respond ONLY with a valid JSON object in this exact format:
{
  "task_type": "classification" or "regression",
  "reasoning": "one or two sentences explaining your decision based on the statistics"
}
"""


def run_task_inference_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Task-Type Inference.

    Reads: state["dataset_path"], state["target_column"]
    Writes: state["task_type"], state["task_type_reasoning"]
    """
    import pandas as pd

    logger.info("▶ Task Inference Agent starting...")
    mock_mode = get_mock_mode()
    llm = get_llm(mock_mode=mock_mode)

    # Load dataset
    df = pd.read_parquet(state["_cleaned_df_path"]) if state.get("_cleaned_df_path") \
        else pd.read_csv(state["dataset_path"])

    target = state["target_column"]
    profile = inspect_target_column(df, target)
    logger.info(f"  Target column profile: {json.dumps(profile, indent=2)}")

    user_prompt = f"""Target column statistics:
{json.dumps(profile, indent=2)}

Determine the task type."""

    raw_response = invoke_llm(
        llm, SYSTEM_PROMPT, user_prompt,
        agent_name="task_inference",
        mock_mode=mock_mode,
    )

    try:
        parsed = json.loads(raw_response)
        task_type = parsed["task_type"].lower().strip()
        reasoning = parsed["reasoning"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"  LLM response parse error: {e}. Defaulting to classification.")
        task_type = "classification"
        reasoning = f"Parse error ({e}); defaulted to classification."

    if task_type not in ("classification", "regression"):
        logger.warning(f"  Unknown task_type '{task_type}'. Defaulting to classification.")
        task_type = "classification"

    logger.info(f"  ✓ Task type: {task_type}")
    logger.info(f"  ✓ Reasoning: {reasoning}")

    return {
        **state,
        "task_type": task_type,
        "task_type_reasoning": reasoning,
    }
