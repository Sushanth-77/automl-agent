"""
Report Agent.

Reads the full PipelineState and generates a structured markdown report
covering every decision, disagreement, and correction logged throughout the run.

Output written to PipelineState:
  - report_sections (dict of section_name → markdown text)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from automl_agent.llm_client import get_llm, get_mock_mode, invoke_llm
from automl_agent.state import PipelineState
from config import PRIMARY_METRICS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a technical report writer for an AutoML pipeline.
You will be given a complete log of what the pipeline did — including task inference,
data cleaning decisions, strategy debate, model evaluation across iterations,
error diagnoses, critic verdicts, and feature engineering changes.

Write an executive summary section (3-5 paragraphs) that:
1. States the task type inferred and why.
2. Summarises the strategy debate outcome and what the arbiter decided.
3. Describes the key failure pattern identified and whether it survived critic review.
4. States what feature engineering was applied and whether it helped (cite metric changes).
5. Gives the final model recommendation.

Be specific. Reference actual numbers. Do NOT use placeholder text.
Write in a professional but readable tone.

Respond ONLY with valid JSON:
{
  "executive_summary": "<markdown text, may use ### headings and bullet points>"
}
"""


def _build_iteration_table(eval_results: list[dict], primary_metric: str) -> str:
    """Build a markdown iteration-by-iteration metric table."""
    if not eval_results:
        return "_No evaluation results._"

    rows = []
    for r in eval_results:
        is_best = "⭐" if r.get("is_best") else ""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in r.get("metrics", {}).items())
        rows.append(f"| {r['iteration']} | {r['model_id']} | {metrics_str} | {is_best} |")

    header = "| Iter | Model ID | Metrics | Best |"
    separator = "|------|----------|---------|------|"
    return "\n".join([header, separator] + rows)


def _build_cleaning_section(cleaning_log: list[dict]) -> str:
    if not cleaning_log:
        return "_No cleaning steps logged._"
    lines = []
    for entry in cleaning_log:
        col = entry.get("column", "?")
        action = entry.get("action", "?")
        reason = entry.get("reason", "")
        lines.append(f"- **{col}** → `{action}`: {reason}")
    return "\n".join(lines)


def _build_debate_section(proposals: list[dict], arbiter: dict) -> str:
    lines = []
    for p in proposals:
        lines.append(f"\n**{p['agent'].upper()} Agent** proposed `{p['model_family']}`:")
        lines.append(f"> {p.get('justification', '')}")
        lines.append(f"> *Acknowledged tradeoff:* {p.get('acknowledged_tradeoff', '')}")

    if arbiter:
        lines.append(f"\n**Arbiter Decision:** `{arbiter.get('chosen_strategy', '')}` → `{arbiter.get('model_family', '')}`")
        lines.append(f"> {arbiter.get('justification', '')}")

    return "\n".join(lines)


def _build_diagnosis_section(error_analysis: list[dict], critic_review: list[dict]) -> str:
    lines = []
    verdict_map = {r["diagnosis_id"]: r for r in critic_review}

    for diag in error_analysis:
        review = verdict_map.get(diag["diagnosis_id"], {})
        verdict = review.get("verdict", "pending")
        icon = "✅" if verdict == "supported" else ("❌" if verdict == "rejected" else "⏳")
        lines.append(f"\n{icon} **[Iter {diag['iteration']}] {diag['issue']}**")
        lines.append(f"  - Evidence: {diag.get('evidence_cited', '')}")
        lines.append(f"  - Reasoning: {diag.get('reasoning', '')}")
        if review:
            lines.append(f"  - Critic: {review.get('reasoning', '')}")

    return "\n".join(lines) if lines else "_No diagnoses logged._"


def _build_feature_section(feature_changes: list[dict]) -> str:
    if not feature_changes:
        return "_No feature changes applied._"
    lines = []
    for fc in feature_changes:
        lines.append(f"- **[Iter {fc['iteration']}] {fc['change_type']}**: {fc['description']}")
        lines.append(f"  - *Justification:* {fc['justification']}")
    return "\n".join(lines)


def run_report_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Report Agent.

    Reads: full PipelineState
    Writes: state["report_sections"]
    Also saves report as markdown to runs/current/report.md
    """
    logger.info("▶ Report Agent starting...")
    mock_mode = get_mock_mode()
    llm = get_llm(mock_mode=mock_mode)

    task_type = state.get("task_type", "?")
    primary_metric = PRIMARY_METRICS.get(task_type, "f1_weighted")
    dataset_path = state.get("dataset_path", "?")
    target = state.get("target_column", "?")

    # Build LLM summary prompt
    summary_context = {
        "task_type": task_type,
        "task_type_reasoning": state.get("task_type_reasoning", ""),
        "dataset": dataset_path,
        "target_column": target,
        "iterations_run": state.get("iteration", 0),
        "stop_reason": state.get("stop_reason", "max_iterations"),
        "strategy_debate_outcome": state.get("arbiter_decision", {}),
        "n_cleaning_steps": len(state.get("cleaning_log", [])),
        "eval_results_summary": [
            {
                "iteration": r["iteration"],
                "model_id": r["model_id"],
                "primary_metric": r.get("metrics", {}).get(primary_metric),
                "is_best": r["is_best"],
            }
            for r in state.get("eval_results", [])
        ],
        "diagnoses_summary": [
            {
                "iteration": d["iteration"],
                "issue": d["issue"],
                "evidence": d["evidence_cited"][:200],
            }
            for d in state.get("error_analysis", [])
        ],
        "critic_verdicts": [
            {
                "diagnosis_id": r["diagnosis_id"],
                "verdict": r["verdict"],
            }
            for r in state.get("critic_review", [])
        ],
        "feature_changes": [fc["description"] for fc in state.get("feature_changes", [])],
    }

    raw_response = invoke_llm(
        llm, SYSTEM_PROMPT,
        f"Pipeline run summary:\n{json.dumps(summary_context, indent=2)}",
        agent_name="report",
        mock_mode=mock_mode,
    )

    try:
        parsed = json.loads(raw_response)
        executive_summary = parsed.get("executive_summary", raw_response)
    except json.JSONDecodeError:
        executive_summary = raw_response

    # Assemble full report sections
    sections = {
        "executive_summary": executive_summary,
        "task_inference": (
            f"**Task type:** `{task_type}`\n\n"
            f"**Reasoning:** {state.get('task_type_reasoning', '')}"
        ),
        "data_cleaning": _build_cleaning_section(state.get("cleaning_log", [])),
        "strategy_debate": _build_debate_section(
            state.get("strategy_proposals", []),
            state.get("arbiter_decision", {}),
        ),
        "iteration_metrics": _build_iteration_table(
            state.get("eval_results", []), primary_metric
        ),
        "error_analysis_and_critic": _build_diagnosis_section(
            state.get("error_analysis", []),
            state.get("critic_review", []),
        ),
        "feature_engineering": _build_feature_section(state.get("feature_changes", [])),
        "conclusion": (
            f"**Stop reason:** {state.get('stop_reason', 'max_iterations')}\n\n"
            f"**Best model:** `{state.get('_current_best_model_id', 'N/A')}`\n\n"
            f"**Total iterations:** {state.get('iteration', 0)}"
        ),
    }

    # Render full markdown report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_report = f"""# AutoML Agent — Pipeline Report
**Generated:** {timestamp}
**Dataset:** `{dataset_path}`
**Target:** `{target}` | **Task type:** `{task_type}`

---

## Executive Summary

{sections['executive_summary']}

---

## 1. Task Type Inference

{sections['task_inference']}

---

## 2. Data Cleaning

{sections['data_cleaning']}

---

## 3. Strategy Debate

{sections['strategy_debate']}

---

## 4. Iteration Metrics

{sections['iteration_metrics']}

---

## 5. Error Analysis & Critic Reviews

{sections['error_analysis_and_critic']}

---

## 6. Feature Engineering

{sections['feature_engineering']}

---

## 7. Conclusion

{sections['conclusion']}
"""

    # Save to disk
    from automl_agent.run_utils import get_run_dir
    run_dir = get_run_dir()
    report_path = str(run_dir / "report.md")
    Path(report_path).write_text(full_report, encoding="utf-8")
    logger.info(f"  ✓ Report saved → {report_path}")

    return {**state, "report_sections": sections}
