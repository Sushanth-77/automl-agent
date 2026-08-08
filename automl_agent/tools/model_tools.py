"""
Model tools — training, evaluation, and diagnostic utilities.

These are deterministic Python functions (no LLM). They are called by
Training Agent, Evaluation Agent, Error Analysis Agent, and Critic Agent.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


# ── Model factory ─────────────────────────────────────────────────────────────

def _build_estimator(model_family: str, params: dict[str, Any], task_type: str):
    """Build a sklearn-compatible estimator from family name + params dict."""
    family = model_family.lower()

    if family == "logistic_regression":
        base = LogisticRegression(**{k: v for k, v in params.items()
                                     if k in ("C", "max_iter", "solver", "class_weight",
                                               "penalty", "random_state")})
        return Pipeline([("scaler", StandardScaler()), ("clf", base)])

    elif family == "ridge":
        base = Ridge(**{k: v for k, v in params.items()
                        if k in ("alpha", "max_iter", "random_state")})
        return Pipeline([("scaler", StandardScaler()), ("reg", base)])

    elif family == "random_forest":
        kwargs = {k: v for k, v in params.items()
                  if k in ("n_estimators", "max_depth", "min_samples_leaf",
                            "min_samples_split", "class_weight", "random_state", "n_jobs")}
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("n_jobs", -1)
        if task_type == "classification":
            return RandomForestClassifier(**kwargs)
        return RandomForestRegressor(**kwargs)

    elif family == "gradient_boosting":
        kwargs = {k: v for k, v in params.items()
                  if k in ("n_estimators", "max_depth", "learning_rate",
                            "subsample", "random_state")}
        kwargs.setdefault("random_state", 42)
        if task_type == "classification":
            return GradientBoostingClassifier(**kwargs)
        return GradientBoostingRegressor(**kwargs)

    elif family == "xgboost":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        kwargs = {k: v for k, v in params.items()
                  if k in ("n_estimators", "max_depth", "learning_rate",
                            "subsample", "colsample_bytree", "random_state",
                            "scale_pos_weight", "use_label_encoder")}
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("verbosity", 0)
        if task_type == "classification":
            return XGBClassifier(**kwargs)
        return XGBRegressor(**kwargs)

    elif family == "lightgbm":
        if not _HAS_LGB:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
        kwargs = {k: v for k, v in params.items()
                  if k in ("n_estimators", "max_depth", "learning_rate",
                            "subsample", "colsample_bytree", "random_state",
                            "class_weight", "num_leaves")}
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("verbose", -1)
        if task_type == "classification":
            return LGBMClassifier(**kwargs)
        return LGBMRegressor(**kwargs)

    else:
        raise ValueError(f"Unknown model family: '{model_family}'")


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_config: dict[str, Any],
    task_type: str,
    artifact_dir: str | Path = "runs/models",
) -> tuple[Any, str]:
    """
    Train a model from a config dict and save it as a joblib artifact.
    Returns (fitted_estimator, artifact_path).
    """
    import joblib

    model_family = model_config["model_family"]
    params = model_config.get("params", {})
    model_id = model_config["model_id"]

    estimator = _build_estimator(model_family, params, task_type)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(X_train, y_train)

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = str(artifact_dir / f"{model_id}.joblib")
    joblib.dump(estimator, artifact_path)

    return estimator, artifact_path


def load_model(artifact_path: str):
    """Load a saved estimator from disk."""
    import joblib
    return joblib.load(artifact_path)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
) -> dict[str, float]:
    """
    Compute evaluation metrics. Returns a flat dict of metric_name → value.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = estimator.predict(X_test)

    metrics: dict[str, float] = {}

    if task_type == "classification":
        metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
        metrics["f1_weighted"] = round(
            float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4
        )
        metrics["precision_weighted"] = round(
            float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4
        )
        metrics["recall_weighted"] = round(
            float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4
        )
        # ROC-AUC (binary only; skip if multiclass without predict_proba)
        try:
            if hasattr(estimator, "predict_proba"):
                y_prob = estimator.predict_proba(X_test)
                n_classes = y_prob.shape[1]
                if n_classes == 2:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob[:, 1])), 4)
                else:
                    metrics["roc_auc"] = round(
                        float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")), 4
                    )
        except Exception:
            pass
    else:
        mse = mean_squared_error(y_test, y_pred)
        metrics["rmse"] = round(float(np.sqrt(mse)), 4)
        metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
        metrics["r2"] = round(float(r2_score(y_test, y_pred)), 4)

    return metrics


# ── Diagnostic tools ──────────────────────────────────────────────────────────

def get_confusion_matrix(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Return confusion matrix as a structured dict (not a figure — avoids display deps)."""
    y_pred = estimator.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    classes = [str(c) for c in sorted(y_test.unique())]
    return {
        "matrix": cm.tolist(),
        "classes": classes,
        "n_classes": len(classes),
        "per_class_recall": {
            cls: round(float(cm[i, i] / cm[i].sum()), 4) if cm[i].sum() > 0 else 0.0
            for i, cls in enumerate(classes)
        },
        "per_class_precision": {
            cls: round(float(cm[i, i] / cm[:, i].sum()), 4) if cm[:, i].sum() > 0 else 0.0
            for i, cls in enumerate(classes)
        },
    }


def get_residuals(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Compute residuals for regression models."""
    y_pred = estimator.predict(X_test)
    residuals = y_test.values - y_pred
    return {
        "residuals": residuals.tolist(),
        "mean_residual": round(float(residuals.mean()), 4),
        "std_residual": round(float(residuals.std()), 4),
        "max_abs_residual": round(float(np.abs(residuals).max()), 4),
        "pct_within_10pct": round(
            float((np.abs(residuals) / (np.abs(y_test.values) + 1e-8) < 0.1).mean()), 4
        ),
    }


def get_misclassified_samples(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n: int = 20,
) -> pd.DataFrame:
    """Return the n most confidently misclassified samples."""
    y_pred = estimator.predict(X_test)
    mask = y_pred != y_test.values
    wrong = X_test[mask].copy()
    wrong["true_label"] = y_test[mask].values
    wrong["predicted_label"] = y_pred[mask]

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X_test[mask])
        wrong["confidence"] = proba.max(axis=1)
        wrong = wrong.sort_values("confidence", ascending=False)

    return wrong.head(n)


def get_worst_predictions(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n: int = 20,
) -> pd.DataFrame:
    """Return the n samples with the largest absolute residual (regression)."""
    y_pred = estimator.predict(X_test)
    result = X_test.copy()
    result["true_value"] = y_test.values
    result["predicted_value"] = y_pred
    result["abs_residual"] = np.abs(y_test.values - y_pred)
    return result.sort_values("abs_residual", ascending=False).head(n)


def slice_performance(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    by_column: str,
    task_type: str,
) -> dict[str, Any]:
    """
    Compute performance metrics split by the unique values of by_column.
    Returns {value: {metric: score}} for each unique value.
    """
    if by_column not in X_test.columns:
        return {"error": f"Column '{by_column}' not found in X_test."}

    y_pred = estimator.predict(X_test)
    slices: dict[str, Any] = {}

    for val in X_test[by_column].unique():
        mask = X_test[by_column] == val
        n = int(mask.sum())
        if n < 5:
            continue  # skip slices too small to be meaningful

        y_true_slice = y_test[mask]
        y_pred_slice = y_pred[mask]

        if task_type == "classification":
            slices[str(val)] = {
                "n": n,
                "accuracy": round(float(accuracy_score(y_true_slice, y_pred_slice)), 4),
                "f1_weighted": round(
                    float(f1_score(y_true_slice, y_pred_slice, average="weighted",
                                   zero_division=0)), 4
                ),
            }
        else:
            mse = mean_squared_error(y_true_slice, y_pred_slice)
            slices[str(val)] = {
                "n": n,
                "rmse": round(float(np.sqrt(mse)), 4),
                "mae": round(float(mean_absolute_error(y_true_slice, y_pred_slice)), 4),
            }

    return slices


# ── Data splitting ─────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split for classification; random for regression."""
    X = df.drop(columns=[target])
    y = df[target]

    # Use stratify only for low-cardinality targets
    stratify = y if y.nunique() <= 20 else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

    return X_train, X_test, y_train, y_test


# ── Optuna HPO ────────────────────────────────────────────────────────────────

def optuna_tune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_family: str,
    task_type: str,
    n_trials: int = 20,
    cv_folds: int = 3,
    random_state: int = 42,
) -> tuple[dict[str, Any], float]:
    """
    Run Bayesian hyperparameter optimisation with Optuna.

    Returns (best_params, best_cv_score) for the given model_family and task.
    In mock mode callers should pass n_trials=1 to skip real search.

    The search space is deliberately narrow to keep runtime reasonable:
      - RandomForest / GBM: n_estimators, max_depth, min_samples_leaf
      - XGBoost / LightGBM: n_estimators, max_depth, learning_rate, subsample
      - LogisticRegression / Ridge: C / alpha
    """
    import optuna
    from sklearn.model_selection import cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    primary_metric = "f1_weighted" if task_type == "classification" else "neg_root_mean_squared_error"
    direction = "maximize"  # cross_val_score maximises both (neg_rmse is maximised toward 0)

    def objective(trial: optuna.Trial) -> float:
        family = model_family.lower()
        params: dict[str, Any] = {}

        if family == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
        elif family in ("gradient_boosting",):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
        elif family == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        elif family == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
        elif family == "logistic_regression":
            params = {
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "max_iter": trial.suggest_int("max_iter", 200, 1000),
            }
        elif family == "ridge":
            params = {
                "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True),
            }
        else:
            return 0.0

        params["random_state"] = random_state
        estimator = _build_estimator(model_family, params, task_type)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                estimator, X_train, y_train,
                cv=cv_folds,
                scoring=primary_metric,
                n_jobs=-1,
            )
        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {k: v for k, v in study.best_params.items()}
    best_params["random_state"] = random_state
    best_score = study.best_value

    return best_params, best_score
