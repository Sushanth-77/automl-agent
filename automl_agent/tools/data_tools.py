"""
Data tools — pure Python/pandas utilities used by Data Agent and Task Inference Agent.

All functions return (modified_df, log_entry_dict) so callers can append to cleaning_log.
None of these functions call the LLM — they are deterministic tools.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


# ── Target column inspection ───────────────────────────────────────────────────

def inspect_target_column(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """
    Return a structured profile of the target column used by Task Inference Agent.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame. "
                         f"Available columns: {list(df.columns)}")

    col = df[target].dropna()
    n_unique = int(col.nunique())
    dtype_str = str(col.dtype)
    value_counts = col.value_counts(normalize=True).head(10).to_dict()
    # Convert keys to strings for JSON serialisability
    value_counts = {str(k): round(float(v), 4) for k, v in value_counts.items()}

    # Distribution shape hints
    is_numeric = pd.api.types.is_numeric_dtype(col)
    if is_numeric:
        q_range = float(col.quantile(0.75) - col.quantile(0.25))
        std_dev = float(col.std())
        mean_val = float(col.mean())
    else:
        q_range = std_dev = mean_val = None

    return {
        "column_name": target,
        "dtype": dtype_str,
        "n_unique": n_unique,
        "n_samples": int(len(col)),
        "null_count": int(df[target].isna().sum()),
        "value_counts_normalized": value_counts,
        "is_numeric": is_numeric,
        "iqr": q_range,
        "std_dev": std_dev,
        "mean": mean_val,
    }


# ── Dataset profiling ──────────────────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    """
    Return a rich profile of the full dataset used by Data Agent.
    """
    profile: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": {},
    }

    for col in df.columns:
        col_data = df[col]
        null_count = int(col_data.isna().sum())
        null_pct = round(null_count / len(df) * 100, 2)
        n_unique = int(col_data.nunique())
        dtype_str = str(col_data.dtype)
        is_numeric = pd.api.types.is_numeric_dtype(col_data)

        col_profile: dict[str, Any] = {
            "dtype": dtype_str,
            "null_count": null_count,
            "null_pct": null_pct,
            "n_unique": n_unique,
            "is_numeric": is_numeric,
            "is_target": col == target,
        }

        if is_numeric and null_count < len(df):
            col_profile["mean"] = round(float(col_data.mean()), 4)
            col_profile["std"] = round(float(col_data.std()), 4)
            col_profile["min"] = round(float(col_data.min()), 4)
            col_profile["max"] = round(float(col_data.max()), 4)
        else:
            top_values = col_data.value_counts().head(5).to_dict()
            col_profile["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        # Flag potential ID columns
        if n_unique == len(df) and not is_numeric:
            col_profile["likely_id_column"] = True

        profile["columns"][col] = col_profile

    # Class balance for target (if classification candidate)
    if target and target in df.columns:
        target_col = df[target].dropna()
        if target_col.nunique() <= 20:
            balance = target_col.value_counts(normalize=True).to_dict()
            profile["class_balance"] = {str(k): round(float(v), 4) for k, v in balance.items()}
            # Imbalance ratio (majority / minority)
            vals = list(balance.values())
            if len(vals) >= 2:
                profile["imbalance_ratio"] = round(max(vals) / min(vals), 2)

    return profile


# ── Cleaning operations ────────────────────────────────────────────────────────

def impute_missing(
    df: pd.DataFrame, col: str, strategy: str = "median"
) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing values in col.

    Strategies: "median", "mean", "mode", "constant:<value>", "ffill", "bfill"
    Returns (modified_df, log_entry).
    """
    df = df.copy()
    before_null = int(df[col].isna().sum())

    if before_null == 0:
        return df, {
            "column": col, "action": "impute_missing", "strategy": strategy,
            "note": "No missing values found — skipped.", "nulls_filled": 0,
        }

    if strategy == "median":
        fill_value = df[col].median()
        df[col] = df[col].fillna(fill_value)
    elif strategy == "mean":
        fill_value = df[col].mean()
        df[col] = df[col].fillna(fill_value)
    elif strategy == "mode":
        fill_value = df[col].mode()[0]
        df[col] = df[col].fillna(fill_value)
    elif strategy.startswith("constant:"):
        fill_value = strategy.split(":", 1)[1]
        # Try to cast to existing dtype
        try:
            fill_value = type(df[col].dropna().iloc[0])(fill_value)
        except Exception:
            pass
        df[col] = df[col].fillna(fill_value)
    elif strategy == "ffill":
        fill_value = "forward-fill"
        df[col] = df[col].ffill()
    elif strategy == "bfill":
        fill_value = "backward-fill"
        df[col] = df[col].bfill()
    else:
        raise ValueError(f"Unknown imputation strategy: '{strategy}'")

    after_null = int(df[col].isna().sum())
    return df, {
        "column": col,
        "action": "impute_missing",
        "strategy": strategy,
        "fill_value": str(fill_value) if strategy not in ("ffill", "bfill") else strategy,
        "nulls_before": before_null,
        "nulls_after": after_null,
        "nulls_filled": before_null - after_null,
    }


def encode_categoricals(
    df: pd.DataFrame, col: str, strategy: str = "label"
) -> tuple[pd.DataFrame, dict]:
    """
    Encode a categorical column.

    Strategies: "label", "onehot", "ordinal:<cat1>,<cat2>,..."
    Returns (modified_df, log_entry).
    """
    df = df.copy()
    original_n_cols = len(df.columns)

    if strategy == "label":
        df[col] = pd.Categorical(df[col]).codes
        new_cols = [col]
    elif strategy == "onehot":
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)
        new_cols = list(dummies.columns)
    elif strategy.startswith("ordinal:"):
        categories = [c.strip() for c in strategy.split(":", 1)[1].split(",")]
        df[col] = pd.Categorical(df[col], categories=categories, ordered=True).codes
        new_cols = [col]
    else:
        raise ValueError(f"Unknown encoding strategy: '{strategy}'")

    return df, {
        "column": col,
        "action": "encode_categoricals",
        "strategy": strategy,
        "new_columns": new_cols,
        "n_cols_before": original_n_cols,
        "n_cols_after": len(df.columns),
    }


def drop_column(df: pd.DataFrame, col: str, reason: str) -> tuple[pd.DataFrame, dict]:
    """
    Drop a column from the DataFrame.
    Returns (modified_df, log_entry).
    """
    if col not in df.columns:
        return df, {
            "column": col, "action": "drop_column", "reason": reason,
            "note": "Column not found — skipped.",
        }
    df = df.drop(columns=[col])
    return df, {
        "column": col,
        "action": "drop_column",
        "reason": reason,
    }
