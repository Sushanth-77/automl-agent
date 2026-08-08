"""
Feature engineering tools — deterministic transformations applied by Feature Engineering Agent.

All functions return (modified_df, log_entry_dict).
No LLM calls — these are the "hands" the agent uses after its reasoning step.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def add_interaction_feature(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    operation: str = "multiply",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Create a new feature by combining two existing columns.

    Operations: "multiply", "divide", "add", "subtract", "ratio"
    Returns (modified_df, log_entry).
    """
    df = df.copy()

    if col_a not in df.columns:
        raise ValueError(f"Column '{col_a}' not found.")
    if col_b not in df.columns:
        raise ValueError(f"Column '{col_b}' not found.")

    new_col = f"{col_a}_x_{col_b}" if operation == "multiply" else f"{col_a}_{operation}_{col_b}"

    if operation == "multiply":
        df[new_col] = df[col_a] * df[col_b]
    elif operation == "divide" or operation == "ratio":
        df[new_col] = df[col_a] / (df[col_b].replace(0, np.nan))
    elif operation == "add":
        df[new_col] = df[col_a] + df[col_b]
    elif operation == "subtract":
        df[new_col] = df[col_a] - df[col_b]
    else:
        raise ValueError(f"Unknown operation: '{operation}'")

    return df, {
        "change_type": "add_interaction_feature",
        "new_column": new_col,
        "source_columns": [col_a, col_b],
        "operation": operation,
        "null_count_new": int(df[new_col].isna().sum()),
    }


def bin_feature(
    df: pd.DataFrame,
    col: str,
    strategy: str = "quantile",
    n_bins: int = 4,
    labels: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Bin a continuous feature into discrete categories.

    Strategies: "quantile" (equal-frequency), "uniform" (equal-width), "kmeans"
    Returns (modified_df, log_entry).
    """
    df = df.copy()
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")

    new_col = f"{col}_binned"

    if strategy == "quantile":
        df[new_col] = pd.qcut(
            df[col], q=n_bins, labels=labels, duplicates="drop"
        ).astype(str)
    elif strategy == "uniform":
        df[new_col] = pd.cut(
            df[col], bins=n_bins, labels=labels
        ).astype(str)
    elif strategy == "kmeans":
        try:
            from sklearn.preprocessing import KBinsDiscretizer
            kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans")
            valid_mask = df[col].notna()
            binned = kbd.fit_transform(df.loc[valid_mask, [col]])
            df.loc[valid_mask, new_col] = binned[:, 0].astype(int).astype(str)
        except Exception as e:
            raise RuntimeError(f"KMeans binning failed: {e}") from e
    else:
        raise ValueError(f"Unknown binning strategy: '{strategy}'")

    return df, {
        "change_type": "bin_feature",
        "original_column": col,
        "new_column": new_col,
        "strategy": strategy,
        "n_bins": n_bins,
        "value_counts": df[new_col].value_counts().to_dict(),
    }


def add_aggregate_feature(
    df: pd.DataFrame,
    groupby_col: str,
    agg_col: str,
    agg_fn: str = "mean",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Add a group-level aggregate as a new feature.

    agg_fn: "mean", "median", "std", "min", "max", "count"
    Returns (modified_df, log_entry).
    """
    df = df.copy()
    if groupby_col not in df.columns:
        raise ValueError(f"Groupby column '{groupby_col}' not found.")
    if agg_col not in df.columns:
        raise ValueError(f"Aggregate column '{agg_col}' not found.")

    new_col = f"{agg_col}_{agg_fn}_by_{groupby_col}"
    group_agg = df.groupby(groupby_col)[agg_col].transform(agg_fn)
    df[new_col] = group_agg

    return df, {
        "change_type": "add_aggregate_feature",
        "new_column": new_col,
        "groupby_column": groupby_col,
        "aggregate_column": agg_col,
        "aggregation_fn": agg_fn,
        "null_count_new": int(df[new_col].isna().sum()),
    }


def apply_class_weighting(
    model_config: dict[str, Any],
    weight: str = "balanced",
) -> dict[str, Any]:
    """
    Update a model config to use class weighting.
    Works for: logistic_regression, random_forest, lightgbm.
    Does NOT modify XGBoost configs (needs scale_pos_weight instead).

    Returns updated model_config dict.
    """
    config = dict(model_config)
    params = dict(config.get("params", {}))

    family = config.get("model_family", "").lower()
    if family in ("logistic_regression", "random_forest", "lightgbm"):
        params["class_weight"] = weight
        config["params"] = params
        log_entry = {
            "change_type": "class_weight",
            "model_family": family,
            "weight_applied": weight,
            "note": f"Set class_weight='{weight}' in model params.",
        }
    elif family in ("xgboost", "gradient_boosting"):
        # For XGBoost, recommend scale_pos_weight for binary; skip silently for others
        log_entry = {
            "change_type": "class_weight",
            "model_family": family,
            "note": "XGBoost/GBM class_weight not directly supported; "
                    "consider scale_pos_weight for binary tasks.",
        }
    else:
        log_entry = {
            "change_type": "class_weight",
            "model_family": family,
            "note": f"class_weight not applied — unknown family '{family}'.",
        }

    return config, log_entry


def drop_feature(
    df: pd.DataFrame,
    col: str,
    reason: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Remove an engineered or redundant feature.
    Returns (modified_df, log_entry).
    """
    df = df.copy()
    if col not in df.columns:
        return df, {"change_type": "drop_feature", "column": col, "note": "Column not found — skipped."}

    df = df.drop(columns=[col])
    return df, {
        "change_type": "drop_feature",
        "column": col,
        "reason": reason,
    }
