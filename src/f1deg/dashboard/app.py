"""F1 Tyre Degradation Dashboard — entry point.

Launch with: streamlit run src/f1deg/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

from f1deg.dashboard.state import MODEL_LABELS, available_model_names, load_data

st.set_page_config(
    page_title="F1 Tyre Degradation",
    page_icon=":racing_car:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("F1 Tyre Degradation Dashboard")
st.markdown(
    "Interactive explorer for tire degradation models across compounds, "
    "circuits, and weather conditions."
)

# --- Quick stats ---
col1, col2, col3, col4 = st.columns(4)

df = load_data()
if not df.empty:
    col1.metric("Total Laps", f"{len(df):,}")
    col2.metric("Circuits", df["circuit_id"].nunique() if "circuit_id" in df.columns else "—")
    col3.metric("Drivers", df["driver_id"].nunique() if "driver_id" in df.columns else "—")
    col4.metric("Compounds", df["compound"].nunique() if "compound" in df.columns else "—")
else:
    st.info(
        "No processed data found. Run the data pipeline first:\n\n"
        "```\npython scripts/01_ingest.py\npython scripts/02_build_features.py\n```"
    )

# --- Available models ---
st.subheader("Available Models")

available = available_model_names()
all_models = list(MODEL_LABELS.keys())

cols = st.columns(len(all_models))
for col, name in zip(cols, all_models, strict=False):
    label = MODEL_LABELS[name]
    if name in available:
        col.success(f"**{label}**\n\nTrained")
    else:
        col.warning(f"**{label}**\n\nNot trained")

# --- Navigation hints ---
st.divider()
st.markdown(
    "**Pages** — Use the sidebar to navigate:\n\n"
    "1. **Degradation Explorer** — Interactive degradation curves with animation\n"
    "2. **Strategy Simulator** — Full race strategy comparison\n"
    "3. **Diagnostics** — Model error analysis\n"
    "4. **Data Explorer** — Browse the training dataset\n"
    "5. **Model Guide** — How each model works and when to use it\n"
    "6. **Pit Window Optimizer** — Optimal pit timing with SC-adjusted windows"
)
