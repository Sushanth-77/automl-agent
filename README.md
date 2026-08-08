# AutoML Agent 🤖

> A multi-agent autonomous ML system that infers its own task type, debates modeling strategies, diagnoses model failures, verifies its diagnoses via a critic agent, and iteratively engineers features — all logged in natural language.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-purple.svg)](https://github.com/langchain-ai/langgraph)
[![Gemini](https://img.shields.io/badge/Gemini-2.5--flash-orange.svg)](https://aistudio.google.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red.svg)](https://streamlit.io)

---

## Architecture

```
              ┌─────────────────────┐
              │  Task-Type Inference │  ← classification vs regression, from target column
              └──────────┬───────────┘
                          ▼
                 ┌────────────────────┐
                 │   Orchestrator      │  ← LangGraph StateGraph, stopping logic
                 └─────────┬──────────┘
                            │
   ┌────────┬────────┬─────┴─────┬──────────────┬─────────────┬──────────────┬────────┐
   ▼        ▼        ▼           ▼              ▼             ▼              ▼        ▼
Data      Strategy  Training  Evaluation   Error        Critic/       Feature   Report
Agent     Debate    Agent     Agent        Analysis     Skeptic       Eng.      Agent
(clean)   (Agg vs               (metrics)  Agent        Agent         Agent
          Cons. +                          (diagnose)   (verify the   (approved
          Arbiter)                                      diagnosis)    diagnoses
                                                                       only)
```

**Stopping criteria** (explicit — interviewers will ask):
1. `iteration == max_iterations` → stop
2. Primary metric plateau (Δ < 0.5% for 2 consecutive iterations) → stop
3. Critic rejected **all** diagnoses (nothing actionable) → stop

---

## Agents

| Agent | Role | Key outputs |
|-------|------|-------------|
| **Task Inference** | Classifies target column as classification/regression | `task_type`, `task_type_reasoning` |
| **Data Cleaning** | LLM-planned imputation, encoding, column drops | `cleaning_log`, cleaned parquet |
| **Aggressive Strategy** | Proposes high-variance model + wide search space | `strategy_proposals[0]` |
| **Conservative Strategy** | Proposes regularised, robust model | `strategy_proposals[1]` |
| **Arbiter** | Picks/blends strategies with dataset-specific justification | `arbiter_decision`, `candidate_models` |
| **Training** | Trains candidate models, saves artifacts | `trained_models` |
| **Evaluation** | Computes metrics, selects best model | `eval_results` |
| **Error Analysis** | Diagnoses failures from confusion matrix + slices | `error_analysis` |
| **Critic / Skeptic** | Verifies each diagnosis against re-queried data | `critic_review` |
| **Feature Engineering** | Applies only critic-approved feature changes | `feature_changes` |
| **Report** | Assembles full markdown pipeline report | `report_sections` |

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/Sushanth-77/automl-agent.git
cd automl-agent
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. API key (required for live mode)

Get a free Gemini API key from [aistudio.google.com](https://aistudio.google.com):

```bash
copy .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_key_here
```

### 3. Run the Streamlit dashboard

```bash
streamlit run app.py
```

### 4. CLI usage

```bash
# Mock mode (no API calls — fast for testing)
python run.py --dataset titanic --target Survived --iterations 3 --mock

# Live mode (requires GOOGLE_API_KEY)
python run.py --dataset adult --target income --iterations 2

# Regression dataset
python run.py --dataset house_prices --target SalePrice --iterations 3
```

---

## Datasets

| Dataset | Task | Messiness | Source |
|---------|------|-----------|--------|
| **Titanic** | Binary classification (`Survived`) | 20% null Age, 77% null Deck, mixed types | seaborn built-in |
| **Adult Income** | Binary classification (`income >50K`) | ~7% nulls, `?` values, categorical imbalance | sklearn / UCI |
| **House Prices** | Regression (`SalePrice`) | Many high-null features, skewed target | OpenML / GitHub |

---

## The State Transcript IS the Demo

Every agent decision, disagreement, and correction is logged in `PipelineState`. When you run the pipeline, you'll see:

```json
{
  "task_type": "classification",
  "task_type_reasoning": "Target has 2 unique values (0/1) → binary classification.",
  "cleaning_log": [
    {"column": "Age", "action": "impute_missing", "strategy": "median", "reason": "..."},
    {"column": "Cabin", "action": "drop_column", "reason": "77% null, low-information..."}
  ],
  "strategy_proposals": [
    {"agent": "aggressive", "model_family": "xgboost", "justification": "..."},
    {"agent": "conservative", "model_family": "logistic_regression", "justification": "..."}
  ],
  "arbiter_decision": {"chosen_strategy": "conservative", "justification": "Dataset is small..."},
  "error_analysis": [{"issue": "class_imbalance", "evidence_cited": "Minority recall=0.61..."}],
  "critic_review": [{"verdict": "supported", "reasoning": "Re-queried slice confirms..."}],
  "feature_changes": [{"change_type": "apply_class_weight", "justification": "..."}]
}
```

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Agent orchestration | LangGraph |
| LLM reasoning | Google Gemini API (`gemini-2.5-flash`) via `langchain-google-genai` |
| ML backend | scikit-learn, XGBoost, LightGBM |
| Hyperparam search | RandomizedSearchCV / Optuna |
| Interpretability | SHAP |
| UI | Streamlit |
| State logging | JSON per run in `runs/` |

---

## Resume Framing

> Built AutoML Agent, a multi-agent autonomous ML system (LangGraph + Gemini API) that infers its own task type, debates competing modeling strategies between adversarial agents, diagnoses model failures, verifies its diagnoses via a critic agent before acting on them, and iteratively engineers features in response — closing the loop without human intervention. Demonstrated measurable metric improvement across iterations on 3 tabular datasets spanning classification and regression; full agent reasoning, disagreement, and correction logged and surfaced in a Streamlit dashboard.

---

## Project Structure

```
automl-agent/
├── automl_agent/
│   ├── state.py                      # PipelineState TypedDict (the contract)
│   ├── llm_client.py                 # Gemini wrapper + mock mode
│   ├── graph.py                      # LangGraph orchestration + stopping logic
│   ├── config.py                     # Central config
│   ├── agents/
│   │   ├── task_inference_agent.py
│   │   ├── data_agent.py
│   │   ├── strategy_debate/
│   │   │   ├── aggressive_agent.py
│   │   │   ├── conservative_agent.py
│   │   │   └── arbiter_agent.py
│   │   ├── training_agent.py
│   │   ├── evaluation_agent.py
│   │   ├── error_analysis_agent.py
│   │   ├── critic_agent.py
│   │   ├── feature_engineering_agent.py
│   │   └── report_agent.py
│   ├── tools/
│   │   ├── data_tools.py
│   │   ├── model_tools.py
│   │   └── feature_tools.py
│   └── data/
│       └── loader.py
├── app.py                            # Streamlit dashboard
├── run.py                            # CLI entry point
├── config.py                         # Top-level config
├── requirements.txt
├── .env.example
└── README.md
```
