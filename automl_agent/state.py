"""
PipelineState — the shared state TypedDict passed between all LangGraph nodes.

Rules:
  - Every agent APPENDS to list fields; it never overwrites history.
  - Non-list fields (task_type, iteration, etc.) may be updated in place.
  - The full state at any point IS the audit log / demo artifact.
"""
from __future__ import annotations

from typing import Any, TypedDict


class StrategyProposal(TypedDict):
    agent: str          # "aggressive" | "conservative"
    model_family: str
    hyperparam_ranges: dict[str, Any]
    justification: str
    acknowledged_tradeoff: str


class ArbiterDecision(TypedDict):
    chosen_strategy: str    # "aggressive" | "conservative" | "blend"
    model_family: str
    hyperparam_ranges: dict[str, Any]
    justification: str


class CandidateModel(TypedDict):
    model_id: str
    model_family: str
    params: dict[str, Any]


class TrainedModel(TypedDict):
    model_id: str
    model_family: str
    params: dict[str, Any]
    artifact_path: str
    iteration: int


class EvalResult(TypedDict):
    model_id: str
    iteration: int
    metrics: dict[str, float]
    is_best: bool
    # Cross-validation scores (added Feature 2)
    cv_mean: float | None
    cv_std: float | None
    cv_folds: int | None


class ErrorAnalysisEntry(TypedDict):
    diagnosis_id: str
    iteration: int
    model_id: str
    issue: str              # e.g. "class_imbalance", "feature_missing", "distribution_shift"
    affected_class: str | None
    evidence_cited: str     # natural-language description of the evidence
    reasoning: str          # full LLM reasoning
    structured_tags: dict[str, Any]


class CriticReview(TypedDict):
    diagnosis_id: str
    iteration: int
    verdict: str            # "supported" | "rejected"
    reasoning: str
    evidence_recheck: str   # what the critic actually re-queried


class FeatureChange(TypedDict):
    iteration: int
    diagnosis_id: str       # which approved diagnosis triggered this
    change_type: str        # "add_interaction" | "bin_feature" | "aggregate" | "class_weight"
    description: str
    justification: str
    columns_affected: list[str]


class PipelineState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    dataset_path: str
    target_column: str

    # ── Task inference ────────────────────────────────────────────────────────
    task_type: str                  # "classification" | "regression"
    task_type_reasoning: str

    # ── Data profiling & cleaning ─────────────────────────────────────────────
    raw_df_summary: dict[str, Any]  # dtypes, nulls, cardinality, class balance
    cleaning_log: list[dict]        # list of {column, action, reason, before_stats, after_stats}

    # ── Strategy debate ───────────────────────────────────────────────────────
    strategy_proposals: list[StrategyProposal]
    arbiter_decision: ArbiterDecision

    # ── Models ────────────────────────────────────────────────────────────────
    candidate_models: list[CandidateModel]
    trained_models: list[TrainedModel]

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_results: list[EvalResult]

    # ── Diagnosis + critic loop ───────────────────────────────────────────────
    error_analysis: list[ErrorAnalysisEntry]
    critic_review: list[CriticReview]

    # ── Feature engineering ───────────────────────────────────────────────────
    feature_changes: list[FeatureChange]

    # ── Orchestration ─────────────────────────────────────────────────────────
    iteration: int
    max_iterations: int
    stop_reason: str | None

    # ── Report ────────────────────────────────────────────────────────────────
    report_sections: dict[str, str]  # section_name → markdown text

    # ── Feature importance (updated per evaluation) ────────────────────────────
    feature_importance: dict[str, float]  # feature_name → importance score

    # ── Calibration / prediction intervals (F4) ────────────────────────────────
    calibration_data: dict[str, Any]      # reliability diagram data (classification)
    prediction_intervals: dict[str, Any]  # bootstrap intervals (regression)

    # ── Internal (not shown in report) ───────────────────────────────────────
    _cleaned_df_path: str           # path to parquet after cleaning
    _feature_df_path: str           # path to parquet after feature engineering
    _current_best_model_id: str
    _previous_best_metric: float
