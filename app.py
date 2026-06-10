import os
import sys
import json
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Incident Intelligence Agent",
    page_icon="🚨",
    layout="wide"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        background-color: #f6f6f6;
        border: 1px solid #e6e6e6;
        margin-bottom: 10px;
    }

    .agent-box {
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD COORDINATOR
# ============================================================

@st.cache_resource
def load_coordinator():
    from agents.coordinator import coordinator_agent
    return coordinator_agent



# ============================================================
# SAMPLE INCIDENTS
# ============================================================

sample_logs = {
    "Database Failure": """ERROR: Database timeout
CPU 95%
Memory 89%
Database connections exhausted""",

    "Security Incident": """CRITICAL: Unauthorized access attempt
150 failed logins
Multiple IP addresses detected""",

    "Memory Leak": """WARNING: Memory leak detected
Memory usage 92%
Application response time increased""",

    "GPU Inference Failure": """ERROR: GPU memory exhausted during inference
Model requests failing
High token load detected""",

    "Vector DB Failure": """ERROR: Vector database unavailable
Retrieval disabled
RAG quality degraded"""
}


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Incident Intelligence Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Hybrid multi-agent incident response system using Phi-3, ChromaDB RAG,
    RandomForest + IsolationForest, and a GRPO + LoRA fine-tuned Action Agent.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("System Architecture")

    st.markdown(
        """
        **Agents**
        - Analytics Agent: Phi-3
        - Prediction Agent: RandomForest + IsolationForest
        - Search Agent: ChromaDB RAG
        - Action Agent: GRPO + LoRA
        - Report Agent: Phi-3

        **Evaluation**
        - Valid JSON
        - Required action fields
        - Keyword alignment
        - Safety score
        """
    )

    st.divider()

    st.header("Demo Tips")
    st.markdown(
        """
        1. Select a sample incident.
        2. Click **Analyze Incident**.
        3. Show retrieved incidents.
        4. Show ML prediction.
        5. Show GRPO action plan.
        6. Show metrics.
        """
    )


# ============================================================
# INPUT SECTION
# ============================================================

left, right = st.columns([2, 1])

with left:
    selected_sample = st.selectbox(
        "Choose a sample incident",
        ["Custom"] + list(sample_logs.keys())
    )

    default_log = sample_logs.get(selected_sample, "")

    incident_log = st.text_area(
        "Incident Log",
        value=default_log,
        height=240,
        placeholder="Paste incident logs here..."
    )

with right:
    st.subheader("What this system does")

    st.markdown(
        """
        This workflow:
        1. Analyzes the incident.
        2. Predicts risk using ML.
        3. Retrieves similar incidents.
        4. Generates a remediation plan.
        5. Produces an executive report.
        """
    )


# ============================================================
# RUN BUTTON
# ============================================================

run_button = st.button(
    "Analyze Incident",
    type="primary",
    use_container_width=True
)


# ============================================================
# RUN WORKFLOW
# ============================================================

if run_button:

    if not incident_log.strip():
        st.error("Please enter an incident log first.")

    else:
        with st.spinner("Loading models and running multi-agent workflow..."):
            coordinator_agent = load_coordinator()
            result = coordinator_agent(incident_log)

        st.success("Analysis complete")

        metrics = result.get("metrics", {})

        # ====================================================
        # METRICS CARDS
        # ====================================================

        st.subheader("Performance Metrics")

        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric(
            "Total Latency",
            f"{metrics.get('total_latency', 0)}s"
        )

        m2.metric(
            "Prediction Latency",
            f"{metrics.get('prediction_latency', 0)}s"
        )

        m3.metric(
            "Action Latency",
            f"{metrics.get('action_latency', 0)}s"
        )

        m4.metric(
            "GPU Memory",
            f"{metrics.get('gpu_memory_gb', 0)} GB"
        )

        m5.metric(
            "RAG Top-K",
            metrics.get("rag_top_k", 0)
        )

        st.divider()

        # ====================================================
        # MAIN OUTPUT
        # ====================================================

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Executive Report",
                "Retrieved Incidents",
                "Analytics Agent",
                "Prediction Agent",
                "Action Agent"
            ]
        )

        with tab1:
            st.subheader("Executive Incident Report")
            st.write(result["report"]["response"])

        with tab2:
            st.subheader("Top Similar Historical Incidents")
            st.code(result["retrieved_incident"])

        with tab3:
            st.subheader("Analytics Agent Output")
            st.code(result["analytics"]["response"])

        with tab4:
            st.subheader("ML Prediction Agent Output")
            st.json(result["prediction"]["response"])

        with tab5:
            st.subheader("GRPO + LoRA Action Agent Output")
            st.code(result["actions"]["response"])

        st.divider()

        # ====================================================
        # FULL METRICS TABLE
        # ====================================================

        st.subheader("Full Metrics")

        metrics_df = pd.DataFrame(
            [
                {
                    "Metric": key,
                    "Value": value
                }
                for key, value in metrics.items()
            ]
        )

        st.dataframe(
            metrics_df,
            use_container_width=True
        )

        with st.expander("Raw result dictionary"):
            st.json(result)