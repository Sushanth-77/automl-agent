"""
LLM client — wraps langchain-google-genai with mock mode support.

Usage:
    from automl_agent.llm_client import get_llm, invoke_llm

    llm = get_llm()
    response = invoke_llm(llm, system_prompt, user_prompt)
    # returns a plain string

Mock mode (MOCK_MODE=true or --mock CLI flag):
    Returns deterministic canned responses based on the agent_name tag.
    No API calls are made. Quota is not consumed.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ── Canned mock responses keyed by agent name ──────────────────────────────────
_MOCK_RESPONSES: dict[str, Any] = {
    "task_inference": {
        "task_type": "classification",
        "reasoning": (
            "[MOCK] Target column has 2 unique values (0, 1) with object-like dtype → "
            "binary classification."
        ),
    },
    "data_cleaning": {
        "cleaning_plan": [
            {
                "column": "Age",
                "action": "impute_missing",
                "strategy": "median",
                "reason": "[MOCK] Age is 20% null, numeric, best imputed with median to avoid outlier skew.",
            },
            {
                "column": "Cabin",
                "action": "drop_column",
                "reason": "[MOCK] Cabin is 77% null and high-cardinality — too sparse to impute reliably.",
            },
            {
                "column": "Sex",
                "action": "encode_categoricals",
                "strategy": "label",
                "reason": "[MOCK] Sex is binary categorical → label encode.",
            },
            {
                "column": "Embarked",
                "action": "encode_categoricals",
                "strategy": "onehot",
                "reason": "[MOCK] Embarked has 3 categories with no ordinal relationship → one-hot encode.",
            },
        ]
    },
    "aggressive": {
        "model_family": "xgboost",
        "hyperparam_ranges": {
            "n_estimators": [100, 500],
            "max_depth": [3, 10],
            "learning_rate": [0.01, 0.3],
            "subsample": [0.6, 1.0],
        },
        "justification": (
            "[MOCK] XGBoost with wide search space maximises raw F1. "
            "Dataset is moderate-sized (~1k rows) so some overfitting risk is acceptable."
        ),
        "acknowledged_tradeoff": (
            "[MOCK] Wide hyperparameter ranges increase overfitting risk, "
            "especially on small folds; cross-validation is critical."
        ),
    },
    "conservative": {
        "model_family": "logistic_regression",
        "hyperparam_ranges": {
            "C": [0.001, 1.0],
            "max_iter": [200, 500],
            "solver": ["lbfgs"],
        },
        "justification": (
            "[MOCK] Regularised logistic regression is interpretable, low-variance, "
            "and robust on small datasets with mixed feature types."
        ),
        "acknowledged_tradeoff": (
            "[MOCK] Logistic regression may underfit complex non-linear patterns "
            "present in this dataset."
        ),
    },
    "arbiter": {
        "chosen_strategy": "conservative",
        "model_family": "random_forest",
        "hyperparam_ranges": {
            "n_estimators": [100, 300],
            "max_depth": [4, 8],
            "min_samples_leaf": [2, 10],
        },
        "justification": (
            "[MOCK] Dataset is small (~900 rows after cleaning). "
            "Conservative's overfitting concern wins. "
            "Choosing Random Forest as a moderate blend — more expressive than logistic "
            "regression but with built-in regularisation via min_samples_leaf."
        ),
    },
    "error_analysis": {
        "diagnoses": [
            {
                "issue": "class_imbalance",
                "affected_class": "minority",
                "evidence_cited": (
                    "[MOCK] Minority class recall is 0.61 vs. majority recall 0.89. "
                    "Confusion matrix shows 38% of minority samples misclassified."
                ),
                "reasoning": (
                    "[MOCK] The model has learned to favour the majority class. "
                    "Applying class weighting or oversampling should improve minority recall."
                ),
                "structured_tags": {"issue": "class_imbalance", "severity": "high"},
            }
        ]
    },
    "critic": {
        "verdicts": [
            {
                "verdict": "supported",
                "reasoning": (
                    "[MOCK] Re-queried slice metrics confirm minority recall = 0.61, "
                    "majority recall = 0.89. Diagnosis is substantiated."
                ),
                "evidence_recheck": "[MOCK] Slice stats re-run; values match cited evidence.",
            }
        ]
    },
    "feature_engineering": {
        "proposed_changes": [
            {
                "change_type": "class_weight",
                "description": "[MOCK] Apply class_weight='balanced' to address class imbalance.",
                "justification": (
                    "[MOCK] Critic-approved diagnosis: class imbalance. "
                    "Balanced weighting directly penalises majority-class errors."
                ),
                "columns_affected": ["target"],
            }
        ]
    },
    "report": {
        "executive_summary": (
            "[MOCK] AutoML Agent ran 1 iteration on the dataset. "
            "Task inferred as classification. "
            "Strategy debate chose a conservative Random Forest approach. "
            "Error analysis identified class imbalance; critic confirmed. "
            "Feature engineering applied balanced class weighting. "
            "Final model achieved F1=0.82."
        )
    },
}


def _is_mock_mode() -> bool:
    """Check if mock mode is active via env var or CLI flag."""
    if os.getenv("MOCK_MODE", "false").lower() == "true":
        return True
    if "--mock" in sys.argv:
        return True
    return False


def get_llm(mock_mode: bool | None = None):
    """
    Return a LangChain-compatible LLM.

    In mock mode: returns None (callers must use invoke_llm which handles None).
    In real mode: returns a ChatGoogleGenerativeAI instance.
    Raises RuntimeError if GOOGLE_API_KEY is missing in real mode.
    """
    if mock_mode is None:
        mock_mode = _is_mock_mode()

    if mock_mode:
        return None  # mock mode — no LLM needed

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. "
            "Get a free key from https://aistudio.google.com and add it to your .env file.\n"
            "  echo 'GOOGLE_API_KEY=your_key_here' >> .env\n"
            "Or run in mock mode:  --mock  /  MOCK_MODE=true"
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        raise ImportError(
            "langchain-google-genai is not installed. Run: pip install langchain-google-genai"
        ) from e

    model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.3,
    )


def invoke_llm(
    llm,
    system_prompt: str,
    user_prompt: str,
    agent_name: str = "unknown",
    mock_mode: bool | None = None,
) -> str:
    """
    Invoke the LLM and return the response as a plain string.

    In mock mode, returns the canned JSON response for the given agent_name.
    In real mode, sends the system + user prompts to Gemini and returns the text.
    """
    if mock_mode is None:
        mock_mode = _is_mock_mode()

    if mock_mode or llm is None:
        return _get_mock_response(agent_name, user_prompt)

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content


def _get_mock_response(agent_name: str, user_prompt: str = "") -> str:
    """
    Return a deterministic mock response for the given agent.

    For task_inference: inspects n_unique from user_prompt to return correct task type.
    For data_cleaning: generates a real cleaning plan from the actual column profile
                       instead of returning hardcoded Titanic column names.
    All other agents return canned JSON responses.
    """
    if agent_name == "task_inference":
        # Try to extract n_unique from the prompt JSON
        try:
            import re
            m = re.search(r'"n_unique":\s*(\d+)', user_prompt)
            n_unique = int(m.group(1)) if m else 2
        except Exception:
            n_unique = 2

        if n_unique > 20:
            return json.dumps({
                "task_type": "regression",
                "reasoning": (
                    f"[MOCK] Target column has {n_unique} unique values with a wide numeric range "
                    "→ continuous regression task. SalePrice-style targets are predicted as real values."
                ),
            })
        else:
            return json.dumps({
                "task_type": "classification",
                "reasoning": (
                    f"[MOCK] Target column has {n_unique} unique values (0, 1) "
                    "→ binary classification."
                ),
            })

    if agent_name == "data_cleaning":
        # Build a real cleaning plan from the actual column profile in the user_prompt.
        # This makes mock mode work correctly on ANY dataset, not just Titanic.
        try:
            # Extract the JSON profile from the prompt
            profile_str = user_prompt.replace("Dataset profile:\n", "").replace("\n\nProduce a cleaning plan.", "")
            profile = json.loads(profile_str)
            columns = profile.get("columns", {})
        except Exception:
            columns = {}

        cleaning_plan = []
        for col, info in columns.items():
            dtype = info.get("dtype", "object")
            null_pct = info.get("null_pct", 0.0)
            n_unique = info.get("n_unique", 1)
            is_numeric = info.get("is_numeric", False)
            likely_id = info.get("likely_id", False)

            # Drop likely ID columns or >80% null
            if likely_id or null_pct > 80:
                cleaning_plan.append({
                    "column": col,
                    "action": "drop_column",
                    "strategy": "",
                    "reason": (
                        f"[MOCK] '{col}' has {null_pct:.1f}% nulls or looks like an ID column "
                        "— too sparse/unique to be informative."
                    ),
                })
            # Impute numeric nulls
            elif is_numeric and null_pct > 0.0:
                cleaning_plan.append({
                    "column": col,
                    "action": "impute_missing",
                    "strategy": "median",
                    "reason": f"[MOCK] '{col}' is numeric with {null_pct:.1f}% nulls → impute median.",
                })
            # Encode low-cardinality categoricals
            elif not is_numeric and n_unique <= 2:
                cleaning_plan.append({
                    "column": col,
                    "action": "encode_categoricals",
                    "strategy": "label",
                    "reason": f"[MOCK] '{col}' is binary categorical → label encode.",
                })
            elif not is_numeric and n_unique <= 8:
                cleaning_plan.append({
                    "column": col,
                    "action": "encode_categoricals",
                    "strategy": "onehot",
                    "reason": f"[MOCK] '{col}' has {n_unique} categories → one-hot encode.",
                })
            elif not is_numeric and n_unique > 8:
                # High-cardinality string → drop (safety; live LLM might handle differently)
                cleaning_plan.append({
                    "column": col,
                    "action": "drop_column",
                    "strategy": "",
                    "reason": (
                        f"[MOCK] '{col}' has {n_unique} unique string values — "
                        "too high-cardinality for simple encoding; dropping."
                    ),
                })
            else:
                cleaning_plan.append({
                    "column": col,
                    "action": "no_action",
                    "strategy": "",
                    "reason": f"[MOCK] '{col}' is clean numeric — no action needed.",
                })

        return json.dumps({"cleaning_plan": cleaning_plan}, indent=2)

    canned = _MOCK_RESPONSES.get(agent_name, {"mock": f"[MOCK] No canned response for '{agent_name}'"})
    return json.dumps(canned, indent=2)


def get_mock_mode() -> bool:
    """Exported helper so other modules can check mock mode without importing internals."""
    return _is_mock_mode()
