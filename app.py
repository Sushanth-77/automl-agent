"""
AutoML Agent — Streamlit Dashboard

Run with:
  streamlit run app.py

Features:
  - Dataset selector + target column override
  - Max iterations slider + mock mode toggle
  - "Run Pipeline" button with live agent-by-agent log
  - Iteration metric table (before/after comparison)
  - Diagnosis + critic verdict side-by-side
  - Feature engineering changes log
  - Full final report view
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* Header */
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -1px;
    margin-bottom: 0;
}

.sub-header {
    color: #8892b0;
    font-size: 1rem;
    font-weight: 400;
    margin-top: 0;
}

/* Agent log cards */
.agent-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    backdrop-filter: blur(10px);
}

.agent-card.running {
    border-color: rgba(102, 126, 234, 0.6);
    box-shadow: 0 0 20px rgba(102, 126, 234, 0.1);
}

.agent-card.complete {
    border-color: rgba(72, 199, 142, 0.4);
}

.agent-card.error {
    border-color: rgba(255, 100, 100, 0.4);
}

/* Metric cards */
.metric-card {
    background: rgba(102, 126, 234, 0.08);
    border: 1px solid rgba(102, 126, 234, 0.15);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

/* Verdict badges */
.verdict-supported {
    background: rgba(72, 199, 142, 0.15);
    color: #48c78e;
    border: 1px solid rgba(72, 199, 142, 0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    font-weight: 600;
}

.verdict-rejected {
    background: rgba(255, 100, 100, 0.15);
    color: #ff6464;
    border: 1px solid rgba(255, 100, 100, 0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 15, 26, 0.95) !important;
    border-right: 1px solid rgba(102, 126, 234, 0.15);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
}

/* Code font for model IDs etc */
code {
    font-family: 'JetBrains Mono', monospace;
    background: rgba(102, 126, 234, 0.1);
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 0.85em;
}

/* Dividers */
hr {
    border-color: rgba(102, 126, 234, 0.15);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 8px;
    padding: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 11])
with col_title:
    st.markdown('<p class="main-header">🤖 AutoML Agent</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Multi-agent autonomous ML pipeline · Infers · Debates · Diagnoses · Improves</p>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")

    dataset_choice = st.selectbox(
        "Dataset",
        options=["titanic", "adult_income", "house_prices"],
        format_func=lambda x: {
            "titanic": "🚢 Titanic (Classification)",
            "adult_income": "💰 Adult Income (Classification)",
            "house_prices": "🏠 House Prices (Regression)",
        }[x],
        index=0,
        key="dataset_choice",
    )

    default_targets = {
        "titanic": "Survived",
        "adult_income": "income",
        "house_prices": "SalePrice",
    }
    target_col = st.text_input(
        "Target Column",
        value=default_targets[dataset_choice],
        key="target_col",
        help="Column to predict. Leave as default or override.",
    )

    max_iterations = st.slider(
        "Max Improvement Iterations",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        key="max_iterations",
        help="Pipeline loops (error analysis → feature eng → retrain) up to this many times.",
    )

    mock_mode = st.toggle(
        "🔧 Mock Mode (no API calls)",
        value=True,
        key="mock_mode",
        help="Use canned LLM responses for fast testing without burning API quota.",
    )

    st.markdown("---")
    st.markdown("### 🔑 API Key Status")
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if api_key:
        st.success("GOOGLE_API_KEY ✓ set")
    else:
        st.warning("GOOGLE_API_KEY not set\n\nMock mode required.", icon="⚠️")

    st.markdown("---")
    st.markdown(
        "**How it works:**\n"
        "1. **Task Inference** — infers classification vs regression\n"
        "2. **Data Cleaning** — LLM-planned imputation & encoding\n"
        "3. **Strategy Debate** — Aggressive vs Conservative, Arbiter decides\n"
        "4. **Training** — trains candidate models\n"
        "5. **Evaluation** — computes metrics\n"
        "6. **Error Analysis** — diagnoses failures\n"
        "7. **Critic** — verifies diagnoses before acting\n"
        "8. **Feature Eng** — applies approved fixes\n"
        "9. **Report** — full written summary",
        unsafe_allow_html=False,
    )

# ── Main area ──────────────────────────────────────────────────────────────────

# Session state initialisation
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "pipeline_log" not in st.session_state:
    st.session_state.pipeline_log = []
if "is_running" not in st.session_state:
    st.session_state.is_running = False

tab_run, tab_results, tab_report = st.tabs(["🚀 Run Pipeline", "📊 Results", "📝 Report"])

# ── Tab 1: Run Pipeline ────────────────────────────────────────────────────────
with tab_run:
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown(f"""
        **Selected:** `{dataset_choice}` → target: `{target_col}` | 
        **Iterations:** {max_iterations} | 
        **Mode:** {'🔧 Mock' if mock_mode else '🤖 Live (Gemini API)'}
        """)
    with col_btn:
        run_clicked = st.button("▶ Run Pipeline", key="run_btn", type="primary")

    # Log container
    log_container = st.container()

    if run_clicked:
        st.session_state.pipeline_log = []
        st.session_state.pipeline_state = None

        # Set env vars
        if mock_mode:
            os.environ["MOCK_MODE"] = "true"
        else:
            os.environ.pop("MOCK_MODE", None)

        from automl_agent.data.loader import load_dataset
        from automl_agent.graph import run_pipeline

        with st.status("🚀 Running AutoML Agent pipeline...", expanded=True) as status:

            # Load dataset
            st.write(f"📂 Loading `{dataset_choice}` dataset...")
            try:
                dataset_info = load_dataset(dataset_choice)
                target = target_col or dataset_info.target_column
                st.write(f"✅ Dataset loaded: {dataset_info.df.shape[0]} rows × {dataset_info.df.shape[1]} cols")
            except Exception as e:
                st.error(f"❌ Dataset load failed: {e}")
                status.update(label="❌ Failed", state="error")
                st.stop()

            # Agent progress display
            agent_names = [
                ("🔍", "Task Inference", "Inferring classification vs regression from target column..."),
                ("🧹", "Data Cleaning", "LLM planning column-by-column cleaning strategy..."),
                ("⚔️", "Strategy Debate (Aggressive)", "Proposing high-performance model strategy..."),
                ("🛡️", "Strategy Debate (Conservative)", "Proposing robust/interpretable strategy..."),
                ("⚖️", "Arbiter", "Deciding between strategies based on dataset characteristics..."),
                ("🏋️", "Training", "Training candidate models..."),
                ("📊", "Evaluation", "Computing metrics on held-out test set..."),
                ("🔬", "Error Analysis", "Diagnosing model failures from confusion matrix & slices..."),
                ("🕵️", "Critic / Skeptic", "Verifying diagnoses against re-queried data..."),
                ("🔧", "Feature Engineering", "Applying approved feature changes..."),
                ("📝", "Report", "Generating full pipeline report..."),
            ]

            for icon, name, desc in agent_names:
                st.write(f"{icon} **{name}**: {desc}")

            # Run pipeline
            try:
                final_state = run_pipeline(
                    dataset_path=dataset_info.dataset_path,
                    target_column=target,
                    max_iterations=max_iterations,
                )
                st.session_state.pipeline_state = final_state
                status.update(
                    label=f"✅ Pipeline complete! Stop: {final_state.get('stop_reason', 'done')}",
                    state="complete",
                )
                st.success(
                    f"✅ Done! Best model: `{final_state.get('_current_best_model_id', 'N/A')}` | "
                    f"Iterations: {final_state.get('iteration', 0)} | "
                    f"Stop: {final_state.get('stop_reason', 'N/A')}"
                )
            except Exception as e:
                status.update(label=f"❌ Pipeline error: {e}", state="error")
                st.error(f"Pipeline failed: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")


# ── Tab 2: Results ─────────────────────────────────────────────────────────────
with tab_results:
    state = st.session_state.get("pipeline_state")

    if state is None:
        st.info("Run the pipeline first to see results.", icon="ℹ️")
    else:
        # Quick metrics
        task_type = state.get("task_type", "?")
        best_model = state.get("_current_best_model_id", "N/A")
        stop_reason = state.get("stop_reason", "N/A")
        iterations = state.get("iteration", 0)

        from config import PRIMARY_METRICS
        primary_metric = PRIMARY_METRICS.get(task_type, "f1_weighted")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Task Type", task_type.capitalize())
        with m2:
            st.metric("Iterations Run", iterations)
        with m3:
            st.metric("Stop Reason", stop_reason)
        with m4:
            # Find best metric value
            best_val = None
            for r in state.get("eval_results", []):
                if r.get("is_best"):
                    best_val = r.get("metrics", {}).get(primary_metric)
            st.metric(f"Best {primary_metric}", f"{best_val:.4f}" if best_val else "N/A")

        st.markdown("---")

        # ── Metric evolution table
        st.markdown("### 📈 Metric Evolution Across Iterations")
        eval_results = state.get("eval_results", [])
        if eval_results:
            import pandas as pd
            rows = []
            for r in eval_results:
                row = {"Iteration": r["iteration"], "Model ID": r["model_id"]}
                row.update(r.get("metrics", {}))
                row["Best"] = "⭐" if r.get("is_best") else ""
                rows.append(row)
            df_eval = pd.DataFrame(rows)
            st.dataframe(df_eval, use_container_width=True, hide_index=True)
        else:
            st.info("No evaluation results yet.")

        st.markdown("---")

        # ── Error Analysis + Critic verdicts
        st.markdown("### 🔬 Error Diagnoses & Critic Verdicts")
        diagnoses = state.get("error_analysis", [])
        critic_reviews = state.get("critic_review", [])
        verdict_map = {r["diagnosis_id"]: r for r in critic_reviews}

        if diagnoses:
            for diag in diagnoses:
                review = verdict_map.get(diag["diagnosis_id"], {})
                verdict = review.get("verdict", "pending")
                verdict_color = "#48c78e" if verdict == "supported" else (
                    "#ff6464" if verdict == "rejected" else "#ffd700"
                )
                verdict_icon = "✅" if verdict == "supported" else ("❌" if verdict == "rejected" else "⏳")

                with st.expander(
                    f"[Iter {diag['iteration']}] {diag['issue'].replace('_', ' ').title()} "
                    f"{verdict_icon} {verdict.upper() if verdict != 'pending' else 'PENDING'}",
                    expanded=False,
                ):
                    col_diag, col_critic = st.columns(2)
                    with col_diag:
                        st.markdown("**📋 Error Analysis Diagnosis**")
                        st.markdown(f"**Issue:** `{diag['issue']}`")
                        if diag.get("affected_class"):
                            st.markdown(f"**Affected class:** `{diag['affected_class']}`")
                        st.markdown(f"**Evidence cited:** {diag.get('evidence_cited', '')}")
                        st.markdown(f"**Reasoning:** {diag.get('reasoning', '')}")

                    with col_critic:
                        st.markdown(f"**🕵️ Critic Verdict: :{('green' if verdict == 'supported' else 'red')}[{verdict.upper()}]**")
                        if review:
                            st.markdown(f"**Reasoning:** {review.get('reasoning', '')}")
                            st.markdown(f"**Evidence recheck:** {review.get('evidence_recheck', '')}")
        else:
            st.info("No error analyses produced yet.")

        st.markdown("---")

        # ── Feature changes
        st.markdown("### 🔧 Feature Engineering Changes")
        feature_changes = state.get("feature_changes", [])
        if feature_changes:
            for fc in feature_changes:
                st.markdown(
                    f"**[Iter {fc['iteration']}]** `{fc['change_type']}` — {fc['description']}\n\n"
                    f"*Justification (from approved diagnosis `{fc['diagnosis_id']}`):* {fc['justification']}"
                )
                st.markdown("---")
        else:
            st.info("No feature changes applied.")

        st.markdown("---")

        # ── Cleaning log
        with st.expander("🧹 Data Cleaning Log", expanded=False):
            cleaning_log = state.get("cleaning_log", [])
            if cleaning_log:
                import pandas as pd
                df_clean = pd.DataFrame(cleaning_log)
                st.dataframe(df_clean, use_container_width=True, hide_index=True)
            else:
                st.info("No cleaning steps logged.")

        # ── Strategy debate
        with st.expander("⚔️ Strategy Debate", expanded=False):
            proposals = state.get("strategy_proposals", [])
            arbiter = state.get("arbiter_decision", {})

            for p in proposals:
                st.markdown(f"**{p['agent'].upper()} Agent** → `{p['model_family']}`")
                st.markdown(f"> {p.get('justification', '')}")
                st.markdown(f"> *Tradeoff acknowledged:* {p.get('acknowledged_tradeoff', '')}")
                st.markdown("")

            if arbiter:
                st.markdown(
                    f"**⚖️ Arbiter Decision:** `{arbiter.get('chosen_strategy', '')}` → "
                    f"`{arbiter.get('model_family', '')}`"
                )
                st.markdown(f"> {arbiter.get('justification', '')}")

        # ── Full state JSON
        with st.expander("🗂️ Full State JSON (audit log)", expanded=False):
            public_state = {k: v for k, v in state.items() if not k.startswith("_")}
            st.json(public_state)


# ── Tab 3: Report ─────────────────────────────────────────────────────────────
with tab_report:
    state = st.session_state.get("pipeline_state")

    if state is None:
        st.info("Run the pipeline first to generate the report.", icon="ℹ️")
    else:
        from automl_agent.run_utils import get_run_dir
        report_path = get_run_dir() / "report.md"

        if report_path.exists():
            report_md = report_path.read_text(encoding="utf-8")
            st.markdown(report_md)

            st.download_button(
                label="⬇️ Download Report (Markdown)",
                data=report_md,
                file_name="automl_agent_report.md",
                mime="text/markdown",
                key="download_report",
            )
        else:
            sections = state.get("report_sections", {})
            if sections:
                for section_name, content in sections.items():
                    st.markdown(f"### {section_name.replace('_', ' ').title()}")
                    st.markdown(content)
                    st.markdown("---")
            else:
                st.warning("No report generated yet. The pipeline may not have completed.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#4a5568; font-size:0.8rem;'>"
    "AutoML Agent · LangGraph + Gemini API · "
    "State transcript is the demo · Built for interviews"
    "</center>",
    unsafe_allow_html=True,
)
