"""
LangGraph StateGraph — wires all agents into the AutoML pipeline.

Node order:
  task_inference → data_cleaning → strategy_debate →
  training → evaluation → [stop OR error_analysis → critic → feature_engineering → training loop]

Stopping logic (explicit — see _should_stop()):
  1. iteration == max_iterations                             → stop
  2. Primary metric plateau (Δ < threshold) for 2 iters    → stop
  3. Critic rejected ALL diagnoses (nothing to act on)      → stop
  Otherwise: increment iteration, loop back to training.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Literal

from langgraph.graph import END, StateGraph

from automl_agent.agents.critic_agent import run_critic_agent
from automl_agent.agents.data_agent import run_data_agent
from automl_agent.agents.error_analysis_agent import run_error_analysis_agent
from automl_agent.agents.evaluation_agent import run_evaluation_agent
from automl_agent.agents.feature_engineering_agent import run_feature_engineering_agent
from automl_agent.agents.report_agent import run_report_agent
from automl_agent.agents.strategy_debate.aggressive_agent import run_aggressive_agent
from automl_agent.agents.strategy_debate.arbiter_agent import run_arbiter_agent
from automl_agent.agents.strategy_debate.conservative_agent import run_conservative_agent
from automl_agent.agents.task_inference_agent import run_task_inference_agent
from automl_agent.agents.training_agent import run_training_agent
from automl_agent.state import PipelineState
from config import METRIC_PLATEAU_THRESHOLD, PRIMARY_METRICS, RUNS_DIR

logger = logging.getLogger(__name__)


# ── Stopping logic ─────────────────────────────────────────────────────────────

def _get_metric_history(state: PipelineState) -> list[float]:
    """Extract primary metric values across iterations in order."""
    task_type = state.get("task_type", "classification")
    primary_metric = PRIMARY_METRICS[task_type]
    higher_is_better = primary_metric not in ("rmse", "mae")

    # Get best metric per iteration
    by_iter: dict[int, float] = {}
    for r in state.get("eval_results", []):
        val = r.get("metrics", {}).get(primary_metric)
        if val is None:
            continue
        it = r["iteration"]
        if it not in by_iter:
            by_iter[it] = val
        else:
            if higher_is_better:
                by_iter[it] = max(by_iter[it], val)
            else:
                by_iter[it] = min(by_iter[it], val)

    return [by_iter[i] for i in sorted(by_iter.keys())]


def _is_plateau(history: list[float], task_type: str, threshold: float) -> bool:
    """Return True if the last 2 iterations show < threshold improvement."""
    if len(history) < 2:
        return False
    delta = abs(history[-1] - history[-2])
    primary_metric = PRIMARY_METRICS[task_type]
    # For RMSE: lower is better, so improvement = reduction
    return delta < threshold


def _all_diagnoses_rejected(state: PipelineState) -> bool:
    """Return True if every diagnosis in the current iteration was rejected by the critic."""
    iteration = state.get("iteration", 0)
    current_reviews = [
        r for r in state.get("critic_review", [])
        if r.get("iteration") == iteration
    ]
    if not current_reviews:
        return False
    return all(r["verdict"] == "rejected" for r in current_reviews)


def _should_stop(state: PipelineState) -> Literal["stop", "continue"]:
    """
    Central stopping criterion.

    Returns "stop" if any of:
      1. iteration >= max_iterations
      2. Metric plateaued for 2 consecutive iterations
      3. Critic rejected all diagnoses (nothing left to improve)
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    task_type = state.get("task_type", "classification")

    # Rule 1: Hit iteration cap
    if iteration >= max_iter:
        logger.info(f"  🛑 Stop: reached max_iterations ({max_iter})")
        return "stop"

    # Rule 2: Plateau
    history = _get_metric_history(state)
    if _is_plateau(history, task_type, METRIC_PLATEAU_THRESHOLD):
        logger.info(f"  🛑 Stop: metric plateau detected (history={[f'{v:.4f}' for v in history]})")
        return "stop"

    # Rule 3: All diagnoses rejected
    if _all_diagnoses_rejected(state):
        logger.info("  🛑 Stop: all diagnoses rejected by critic — nothing actionable.")
        return "stop"

    logger.info(f"  ↩ Continue: iteration {iteration + 1}/{max_iter}")
    return "continue"


# ── Node wrappers that inject state updates ────────────────────────────────────

def node_task_inference(state: PipelineState) -> PipelineState:
    return run_task_inference_agent(state)


def node_data_cleaning(state: PipelineState) -> PipelineState:
    return run_data_agent(state)


def node_aggressive(state: PipelineState) -> PipelineState:
    return run_aggressive_agent(state)


def node_conservative(state: PipelineState) -> PipelineState:
    return run_conservative_agent(state)


def node_arbiter(state: PipelineState) -> PipelineState:
    return run_arbiter_agent(state)


def node_training(state: PipelineState) -> PipelineState:
    return run_training_agent(state)


def node_evaluation(state: PipelineState) -> PipelineState:
    return run_evaluation_agent(state)


def node_error_analysis(state: PipelineState) -> PipelineState:
    return run_error_analysis_agent(state)


def node_critic(state: PipelineState) -> PipelineState:
    return run_critic_agent(state)


def node_feature_engineering(state: PipelineState) -> PipelineState:
    return run_feature_engineering_agent(state)


def node_increment_iteration(state: PipelineState) -> PipelineState:
    """Increment the iteration counter before looping back to training."""
    return {**state, "iteration": state.get("iteration", 0) + 1}


def node_set_stop_reason(state: PipelineState) -> PipelineState:
    """Record why we stopped."""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    task_type = state.get("task_type", "classification")

    reason: str
    if iteration >= max_iter:
        reason = f"max_iterations ({max_iter}) reached"
    elif _is_plateau(_get_metric_history(state), task_type, METRIC_PLATEAU_THRESHOLD):
        reason = "metric plateau"
    elif _all_diagnoses_rejected(state):
        reason = "all diagnoses rejected by critic"
    else:
        reason = "pipeline complete"

    return {**state, "stop_reason": reason}


def node_report(state: PipelineState) -> PipelineState:
    return run_report_agent(state)


# ── Build graph ────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the AutoML LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("task_inference", node_task_inference)
    graph.add_node("data_cleaning", node_data_cleaning)
    graph.add_node("aggressive_strategy", node_aggressive)
    graph.add_node("conservative_strategy", node_conservative)
    graph.add_node("arbiter", node_arbiter)
    graph.add_node("training", node_training)
    graph.add_node("evaluation", node_evaluation)
    graph.add_node("error_analysis", node_error_analysis)
    graph.add_node("critic", node_critic)
    graph.add_node("feature_engineering", node_feature_engineering)
    graph.add_node("increment_iteration", node_increment_iteration)
    graph.add_node("set_stop_reason", node_set_stop_reason)
    graph.add_node("report", node_report)

    # Entry point
    graph.set_entry_point("task_inference")

    # Linear flow: task inference → cleaning → debate → training → eval
    graph.add_edge("task_inference", "data_cleaning")
    graph.add_edge("data_cleaning", "aggressive_strategy")
    graph.add_edge("aggressive_strategy", "conservative_strategy")
    graph.add_edge("conservative_strategy", "arbiter")
    graph.add_edge("arbiter", "training")
    graph.add_edge("training", "evaluation")

    # After evaluation: check stopping criteria
    graph.add_conditional_edges(
        "evaluation",
        _should_stop,
        {
            "stop": "set_stop_reason",
            "continue": "error_analysis",
        },
    )

    # Analysis + critic + feature eng → increment iteration → back to training
    graph.add_edge("error_analysis", "critic")
    graph.add_edge("critic", "feature_engineering")
    graph.add_edge("feature_engineering", "increment_iteration")
    graph.add_edge("increment_iteration", "training")

    # Stop path → report → END
    graph.add_edge("set_stop_reason", "report")
    graph.add_edge("report", END)

    return graph.compile()


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_pipeline(
    dataset_path: str,
    target_column: str,
    max_iterations: int = 3,
    run_id: str | None = None,
) -> PipelineState:
    """
    Entry point: build the graph and run the full pipeline.

    Saves the complete PipelineState JSON to runs/<run_id>/state.json.
    Returns the final state.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Point "current" to this run using a pointer file (works on all OS without admin rights)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write a pointer so agents know where "current" is
    pointer_file = RUNS_DIR / "current_run.txt"
    pointer_file.write_text(str(run_dir), encoding="utf-8")

    # Also create/update the "current" directory directly (copy approach, not symlink)
    current_dir = RUNS_DIR / "current"
    current_dir.mkdir(parents=True, exist_ok=True)

    # Monkey-patch: write a redirect so that agents loading from RUNS_DIR/"current" 
    # actually use run_dir. We do this by making current_dir == run_dir via an env var.
    import os
    os.environ["AUTOML_RUN_DIR"] = str(run_dir)

    initial_state: PipelineState = {
        "dataset_path": dataset_path,
        "target_column": target_column,
        "task_type": "",
        "task_type_reasoning": "",
        "raw_df_summary": {},
        "cleaning_log": [],
        "strategy_proposals": [],
        "arbiter_decision": {},
        "candidate_models": [],
        "trained_models": [],
        "eval_results": [],
        "error_analysis": [],
        "critic_review": [],
        "feature_changes": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "stop_reason": None,
        "report_sections": {},
        "_cleaned_df_path": "",
        "_feature_df_path": "",
        "_current_best_model_id": "",
        "_previous_best_metric": None,
    }

    graph = build_graph()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info(f"🚀 AutoML Agent pipeline starting | run_id={run_id}")
    logger.info(f"   Dataset: {dataset_path} | Target: {target_column} | Max iters: {max_iterations}")

    final_state = graph.invoke(initial_state)

    # Save full state as JSON
    state_path = run_dir / "state.json"
    # Filter out internal _ keys for the saved artifact
    public_state = {k: v for k, v in final_state.items() if not k.startswith("_")}
    state_path.write_text(
        json.dumps(public_state, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"✅ Pipeline complete! State saved → {state_path}")
    logger.info(f"   Stop reason: {final_state.get('stop_reason', 'N/A')}")
    logger.info(f"   Best model: {final_state.get('_current_best_model_id', 'N/A')}")

    return final_state
