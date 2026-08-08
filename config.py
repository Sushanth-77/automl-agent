"""
Central configuration for AutoML Agent pipeline.
All agents import from here — never hardcode values in agent files.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RUNS_DIR = ROOT_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")

# ── Pipeline defaults ─────────────────────────────────────────────────────────
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))
METRIC_PLATEAU_THRESHOLD: float = 0.005   # Δ < 0.5% for 2 consecutive iters → stop
MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() == "true"

# ── Model families available to Strategy agents ──────────────────────────────
SUPPORTED_MODEL_FAMILIES = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",   # sklearn GBM
    "xgboost",
    "lightgbm",
]

# ── Metric names by task type ─────────────────────────────────────────────────
PRIMARY_METRICS = {
    "classification": "f1_weighted",
    "regression": "rmse",
}

CLASSIFICATION_METRICS = ["accuracy", "f1_weighted", "precision_weighted",
                           "recall_weighted", "roc_auc"]
REGRESSION_METRICS = ["rmse", "mae", "r2"]

# ── Optuna HPO ───────────────────────────────────────────────────────────────
# Number of trials per model in live mode. In mock mode the training_agent
# falls back to a single trial (no real search needed).
OPTUNA_TRIALS: int = int(os.getenv("OPTUNA_TRIALS", "20"))
