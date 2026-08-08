"""
Data Cleaning Agent.

Profiles the dataset and asks the LLM to produce a column-by-column cleaning plan.
Executes the plan deterministically using data_tools, appending to cleaning_log.

Output appended to PipelineState:
  - raw_df_summary
  - cleaning_log
  - _cleaned_df_path (internal — path to parquet after cleaning)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from automl_agent.llm_client import get_llm, get_mock_mode, invoke_llm
from automl_agent.state import PipelineState
from automl_agent.tools.data_tools import (
    drop_column,
    encode_categoricals,
    impute_missing,
    profile_dataset,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert data-cleaning agent for tabular ML datasets.
You will receive a JSON profile of a dataset (dtypes, null counts, cardinality, etc.).
Your job: produce a column-by-column cleaning plan.

For each column that needs cleaning, decide one of:
  - "impute_missing":  fill nulls. Specify strategy: "median", "mean", "mode", "constant:<value>"
  - "encode_categoricals": encode a categorical column. Strategy: "label", "onehot", "ordinal:<c1>,<c2>,..."
  - "drop_column": remove a column entirely. Give a reason.
  - "no_action": column is clean, no action needed.

Guidelines:
- Drop columns that are >50% null AND low-information (IDs, free-text, hash-like).
- Drop columns that appear to be ID columns (unique per row, non-numeric).
- Impute numeric nulls with "median" (robust to skew) unless the column is clearly symmetric.
- Impute low-cardinality categorical nulls with "mode".
- One-hot encode categoricals with ≤ 8 unique values; label-encode binary or ordinal ones.
- NEVER modify the target column.
- NEVER include the target column in your cleaning plan.

Respond ONLY with a valid JSON object:
{
  "cleaning_plan": [
    {
      "column": "<col_name>",
      "action": "impute_missing" | "encode_categoricals" | "drop_column" | "no_action",
      "strategy": "<strategy_string>",   // only for impute_missing and encode_categoricals
      "reason": "<one sentence justification>"
    },
    ...
  ]
}
"""


def run_data_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Data Cleaning.

    Reads: state["dataset_path"], state["target_column"]
    Writes: state["raw_df_summary"], state["cleaning_log"], state["_cleaned_df_path"]
    """
    logger.info("▶ Data Cleaning Agent starting...")
    mock_mode = get_mock_mode()
    llm = get_llm(mock_mode=mock_mode)

    # Load raw dataset
    df = pd.read_csv(state["dataset_path"])
    target = state["target_column"]
    logger.info(f"  Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")

    # Profile
    raw_summary = profile_dataset(df, target)
    logger.info(f"  Dataset profiled: {len(raw_summary['columns'])} columns.")

    # Build a compact summary for the LLM (avoid sending raw data)
    compact_profile = {
        "n_rows": raw_summary["n_rows"],
        "n_cols": raw_summary["n_cols"],
        "target_column": target,
        "columns": {},
    }
    for col, info in raw_summary["columns"].items():
        if info.get("is_target"):
            continue
        compact_profile["columns"][col] = {
            "dtype": info["dtype"],
            "null_pct": info["null_pct"],
            "n_unique": info["n_unique"],
            "is_numeric": info["is_numeric"],
            "likely_id": info.get("likely_id_column", False),
        }
        if not info["is_numeric"] and "top_values" in info:
            compact_profile["columns"][col]["top_values"] = info["top_values"]

    user_prompt = f"Dataset profile:\n{json.dumps(compact_profile, indent=2)}\n\nProduce a cleaning plan."

    raw_response = invoke_llm(
        llm, SYSTEM_PROMPT, user_prompt,
        agent_name="data_cleaning",
        mock_mode=mock_mode,
    )

    try:
        parsed = json.loads(raw_response)
        cleaning_plan = parsed["cleaning_plan"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"  LLM response parse error: {e}. Using empty cleaning plan.")
        cleaning_plan = []

    logger.info(f"  Cleaning plan has {len(cleaning_plan)} steps.")

    # Execute the plan
    cleaning_log = list(state.get("cleaning_log", []))
    for step in cleaning_plan:
        col = step.get("column", "")
        action = step.get("action", "no_action")
        reason = step.get("reason", "")
        strategy = step.get("strategy", "")

        if col == target or col not in df.columns:
            continue  # never touch target; skip missing cols

        if action == "drop_column":
            df, log_entry = drop_column(df, col, reason)
            cleaning_log.append(log_entry)
            logger.info(f"    Dropped '{col}': {reason}")

        elif action == "impute_missing":
            df, log_entry = impute_missing(df, col, strategy or "median")
            log_entry["reason"] = reason
            cleaning_log.append(log_entry)
            logger.info(f"    Imputed '{col}' ({strategy}): {reason}")

        elif action == "encode_categoricals":
            df, log_entry = encode_categoricals(df, col, strategy or "label")
            log_entry["reason"] = reason
            cleaning_log.append(log_entry)
            logger.info(f"    Encoded '{col}' ({strategy}): {reason}")

        elif action == "no_action":
            logger.debug(f"    No action on '{col}'.")

    # ── Safety fallback ──────────────────────────────────────────────────────
    # Any non-numeric columns not handled by the cleaning plan will crash sklearn.
    # Drop them with a logged warning rather than failing at training time.
    # Note: pandas 3.x uses StringDtype for some string columns; check with is_numeric_dtype.
    import pandas.api.types as pat
    remaining_str_cols = [
        c for c in df.columns
        if not pat.is_numeric_dtype(df[c]) and c != target
    ]
    for col in remaining_str_cols:
        df = df.drop(columns=[col])
        cleaning_log.append({
            "column": col,
            "action": "drop_column",
            "reason": "[auto-fallback] Non-numeric column not handled by cleaning plan; "
                      "dropped to prevent training failure.",
        })
        logger.warning(f"    [fallback] Dropped remaining non-numeric column '{col}'")

    # Ensure target is numeric (label-encode if not already)
    if not pat.is_numeric_dtype(df[target]):
        df[target] = pd.Categorical(df[target]).codes
        cleaning_log.append({
            "column": target,
            "action": "encode_target",
            "reason": "[auto-fallback] Target column was non-numeric; label-encoded.",
        })
        logger.warning(f"    [fallback] Label-encoded non-numeric target '{target}'")

    # Save cleaned dataframe
    from automl_agent.run_utils import get_run_dir
    run_dir = get_run_dir()
    cleaned_path = str(run_dir / "cleaned.parquet")
    df.to_parquet(cleaned_path, index=False)
    logger.info(f"  Cleaned data saved to {cleaned_path}")
    logger.info(f"  {len(cleaning_log)} cleaning steps logged.")

    return {
        **state,
        "raw_df_summary": raw_summary,
        "cleaning_log": cleaning_log,
        "_cleaned_df_path": cleaned_path,
        "_feature_df_path": cleaned_path,  # initialise feature path same as cleaned
    }
