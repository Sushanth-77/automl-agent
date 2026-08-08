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
        value=default_targets.get(dataset_choice, "target"),
        key="target_col",
        help="Column to predict. Leave as default or override.",
    )

    # ── Custom CSV Upload (F5) ─────────────────────────────────────────────
    with st.expander("📁 Upload Your Own CSV", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            key="csv_uploader",
            help="Upload any tabular CSV. After uploading, select the target column below.",
        )
        if uploaded_file is not None:
            import pandas as _pd_up
            import tempfile
            from pathlib import Path as _Path

            # Read to detect columns
            try:
                _df_preview = _pd_up.read_csv(uploaded_file)
                uploaded_file.seek(0)  # reset for saving

                st.write(f"**{uploaded_file.name}** — {_df_preview.shape[0]} rows × {_df_preview.shape[1]} cols")
                st.dataframe(_df_preview.head(3), use_container_width=True, hide_index=True)

                custom_target = st.selectbox(
                    "Select target column",
                    options=list(_df_preview.columns),
                    key="custom_target_col",
                )

                if st.button("✅ Use this dataset", key="use_custom_csv"):
                    # Save to a stable temp path
                    upload_dir = Path("runs") / "uploads"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    save_path = upload_dir / uploaded_file.name
                    save_path.write_bytes(uploaded_file.read())
                    st.session_state.custom_dataset_path = str(save_path)
                    st.session_state.custom_target_col = custom_target
                    st.success(f"Custom dataset ready! Target: `{custom_target}`")
            except Exception as _e:
                st.error(f"Failed to read CSV: {_e}")

    # Show custom dataset status
    if st.session_state.get("custom_dataset_path"):
        st.info(
            f"📁 Custom: `{Path(st.session_state.custom_dataset_path).name}` → "
            f"`{st.session_state.get('custom_target_col', '?')}`"
        )
        if st.button("❌ Clear custom dataset", key="clear_custom"):
            st.session_state.custom_dataset_path = None
            st.session_state.custom_target_col = None
            st.rerun()

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

    # ── Run History Browser (F6) ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🕐 Past Runs")
    try:
        from automl_agent.run_utils import list_runs
        past_runs = list_runs()
        if past_runs:
            for run in past_runs[:10]:  # show last 10
                metric_str = (
                    f"{run['primary_metric']}={run['best_metric_value']:.4f}"
                    if run["best_metric_value"] is not None else "N/A"
                )
                label = (
                    f"📅 {run['timestamp']} | {run['dataset']} | {metric_str}"
                )
                with st.expander(label, expanded=False):
                    st.markdown(f"**Run ID:** `{run['run_id']}`")
                    st.markdown(f"**Task:** {run['task_type']} | **Models:** {run['n_models']} | **Iters:** {run['n_iterations']}")
                    st.markdown(f"**Best model:** `{run['best_model']}`")
                    st.markdown(f"**Stop reason:** {run['stop_reason']}")
                    col_load, col_dir = st.columns(2)
                    with col_load:
                        if st.button("📥 Load into dashboard", key=f"load_{run['run_id']}"):
                            st.session_state.pipeline_state = run["state"]
                            st.success("Loaded! Switch to Results tab.")
                            st.rerun()
                    with col_dir:
                        st.markdown(f"`{run['run_dir']}`")
        else:
            st.info("No completed runs yet. Run the pipeline first.")
    except Exception as _hist_err:
        st.warning(f"Could not load run history: {_hist_err}")

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

        from automl_agent.data.loader import load_dataset, load_from_path
        from automl_agent.graph import run_pipeline

        with st.status("🚀 Running AutoML Agent pipeline...", expanded=True) as status:

            # Load dataset — custom upload takes priority
            custom_path = st.session_state.get("custom_dataset_path")
            custom_target = st.session_state.get("custom_target_col")
            try:
                if custom_path and custom_target:
                    st.write(f"📂 Loading custom dataset: `{Path(custom_path).name}`...")
                    dataset_info = load_from_path(custom_path, custom_target)
                    target = custom_target
                else:
                    st.write(f"📂 Loading `{dataset_choice}` dataset...")
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
                # Cross-validation columns
                cv_mean = r.get("cv_mean")
                cv_std = r.get("cv_std")
                cv_folds = r.get("cv_folds")
                if cv_mean is not None and cv_std is not None:
                    row["CV mean"] = cv_mean
                    row["CV std"] = cv_std
                    row["CV folds"] = cv_folds
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

        # ── Feature Importance (F3) ────────────────────────────────────────────
        st.markdown("### 📊 Feature Importance")
        fi = state.get("feature_importance", {})
        if fi:
            import pandas as pd
            fi_df = pd.DataFrame(
                {"Feature": list(fi.keys()), "Importance": list(fi.values())}
            ).sort_values("Importance", ascending=False).head(15)
            try:
                import altair as alt
                chart = alt.Chart(fi_df).mark_bar(
                    color="#667eea", cornerRadiusTopRight=4, cornerRadiusBottomRight=4
                ).encode(
                    x=alt.X("Importance:Q", title="Importance Score"),
                    y=alt.Y("Feature:N", sort="-x", title=""),
                    tooltip=["Feature", "Importance"],
                ).properties(height=min(30 * len(fi_df) + 60, 500))
                st.altair_chart(chart, use_container_width=True)
            except ImportError:
                st.bar_chart(fi_df.set_index("Feature")["Importance"])

            # Overfitting indicator
            best_eval = next((r for r in state.get("eval_results", []) if r.get("is_best")), None)
            if best_eval:
                from config import PRIMARY_METRICS as PM
                pm = PM.get(state.get("task_type", "classification"), "f1_weighted")
                test_score = best_eval.get("metrics", {}).get(pm)
                cv_mean = best_eval.get("cv_mean")
                if test_score is not None and cv_mean is not None:
                    gap = abs(test_score - cv_mean)
                    if gap > 0.05:
                        st.warning(
                            f"⚠️ Possible overfit — Test {pm}={test_score:.4f} vs "
                            f"CV mean={cv_mean:.4f} (gap={gap:.4f})"
                        )
                    else:
                        st.success(
                            f"✅ Generalisation healthy — Test {pm}={test_score:.4f} ≈ "
                            f"CV mean={cv_mean:.4f} (gap={gap:.4f})"
                        )
        else:
            st.info("Feature importances will appear here after the pipeline runs.")

        st.markdown("---")

        # ── Calibration / Prediction Intervals (F4) ─────────────────────────
        with st.expander("🎯 Calibration & Confidence Intervals", expanded=False):
            task = state.get("task_type", "classification")
            if task == "classification":
                cal = state.get("calibration_data", {})
                if cal:
                    import pandas as pd
                    col_cal1, col_cal2 = st.columns([2, 1])
                    with col_cal1:
                        st.markdown("**Reliability Diagram** (fraction of positives vs mean predicted prob)")
                        cal_df = pd.DataFrame({
                            "Mean Predicted Prob": cal.get("mean_predicted_value", []),
                            "Fraction of Positives": cal.get("fraction_of_positives", []),
                        })
                        if not cal_df.empty:
                            st.line_chart(cal_df.set_index("Mean Predicted Prob"))
                    with col_cal2:
                        st.metric("ECE", f"{cal.get('ece', 'N/A'):.4f}", help="Expected Calibration Error — lower is better")
                        st.metric("Brier Score", f"{cal.get('brier_score', 'N/A'):.4f}", help="Brier Score — lower is better (max 1)")
                        if cal.get("ece", 1) < 0.05:
                            st.success("Well calibrated")
                        elif cal.get("ece", 1) < 0.15:
                            st.warning("Moderate calibration gap")
                        else:
                            st.error("Poor calibration — consider CalibratedClassifierCV")
                else:
                    st.info("Calibration data will appear here after the pipeline runs.")
            else:
                pi = state.get("prediction_intervals", {})
                if pi:
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.metric(f"{int(pi.get('confidence', 0.9)*100)}% CI Mean Width", f"{pi.get('mean_interval_width', 'N/A'):,.0f}")
                        st.metric("Empirical Coverage", f"{pi.get('empirical_coverage', 0):.1%}")
                    with col_p2:
                        st.metric("Bootstrap Samples", pi.get("n_bootstrap", "N/A"))
                        st.metric("Median CI Width", f"{pi.get('median_interval_width', 'N/A'):,.0f}")
                else:
                    st.info("Prediction intervals will appear after a regression run.")

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

            # ── Download buttons (F7) ─────────────────────────────────────
            dl_col1, dl_col2, dl_col3 = st.columns(3)

            with dl_col1:
                st.download_button(
                    label="⬇️ Download Report (Markdown)",
                    data=report_md,
                    file_name="automl_agent_report.md",
                    mime="text/markdown",
                    key="download_report_md",
                )

            with dl_col2:
                # PDF export via reportlab
                try:
                    from io import BytesIO
                    from reportlab.lib.pagesizes import A4
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib.units import cm
                    from reportlab.lib import colors
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
                    from reportlab.lib.enums import TA_LEFT

                    def _build_pdf(markdown_text: str, state_meta: dict) -> bytes:
                        buf = BytesIO()
                        doc = SimpleDocTemplate(
                            buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm,
                        )
                        styles = getSampleStyleSheet()
                        title_style = ParagraphStyle(
                            "Title", parent=styles["Heading1"],
                            fontSize=18, textColor=colors.HexColor("#2d3748"),
                            spaceAfter=12,
                        )
                        h2_style = ParagraphStyle(
                            "H2", parent=styles["Heading2"],
                            fontSize=13, textColor=colors.HexColor("#4a5568"),
                            spaceBefore=10, spaceAfter=6,
                        )
                        body_style = ParagraphStyle(
                            "Body", parent=styles["Normal"],
                            fontSize=10, leading=14, textColor=colors.HexColor("#1a202c"),
                        )
                        meta_style = ParagraphStyle(
                            "Meta", parent=styles["Normal"],
                            fontSize=9, textColor=colors.grey,
                        )

                        story = []
                        story.append(Paragraph("AutoML Agent — Pipeline Report", title_style))
                        story.append(HRFlowable(width="100%", color=colors.HexColor("#667eea")))
                        story.append(Spacer(1, 0.3*cm))

                        # Metadata block
                        meta_lines = [
                            f"Task type: {state_meta.get('task_type', '?')}",
                            f"Best model: {state_meta.get('_current_best_model_id', '?')}",
                            f"Stop reason: {state_meta.get('stop_reason', '?')}",
                            f"Iterations: {state_meta.get('iteration', '?')}",
                        ]
                        for line in meta_lines:
                            story.append(Paragraph(line, meta_style))
                        story.append(Spacer(1, 0.5*cm))
                        story.append(HRFlowable(width="100%", color=colors.lightgrey))
                        story.append(Spacer(1, 0.3*cm))

                        # Report body — convert markdown headings to reportlab paragraphs
                        for line in markdown_text.splitlines():
                            line = line.strip()
                            if not line:
                                story.append(Spacer(1, 0.2*cm))
                            elif line.startswith("## "):
                                story.append(Paragraph(line[3:], h2_style))
                            elif line.startswith("# "):
                                story.append(Paragraph(line[2:], title_style))
                            elif line.startswith("- ") or line.startswith("* "):
                                story.append(Paragraph(f"• {line[2:]}", body_style))
                            else:
                                # Escape HTML chars for reportlab
                                safe = (line.replace("&", "&amp;")
                                             .replace("<", "&lt;").replace(">", "&gt;"))
                                story.append(Paragraph(safe, body_style))

                        doc.build(story)
                        return buf.getvalue()

                    pdf_bytes = _build_pdf(report_md, state)
                    st.download_button(
                        label="📄 Download Report (PDF)",
                        data=pdf_bytes,
                        file_name="automl_agent_report.pdf",
                        mime="application/pdf",
                        key="download_report_pdf",
                    )
                except ImportError:
                    st.info("Install `reportlab` for PDF export.")
                except Exception as pdf_err:
                    st.warning(f"PDF generation failed: {pdf_err}")

            with dl_col3:
                import json as _json
                state_json = _json.dumps(
                    {k: v for k, v in state.items() if not k.startswith("_")},
                    indent=2, default=str,
                )
                st.download_button(
                    label="🗂️ Download State (JSON)",
                    data=state_json,
                    file_name="automl_agent_state.json",
                    mime="application/json",
                    key="download_state_json",
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
