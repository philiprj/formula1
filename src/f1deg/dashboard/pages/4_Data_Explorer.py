"""Page 5: Browse and explore the training dataset."""

from __future__ import annotations

import streamlit as st

from f1deg.dashboard.charts import (
    plot_data_scatter,
    plot_lap_time_distributions,
    plot_weather_scatter,
)
from f1deg.dashboard.state import load_data

st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("Data Explorer")
st.markdown(
    "Browse the processed training dataset used to fit the degradation models. "
    "Use the sidebar filters to narrow down by circuit, compound, or season."
)

df = load_data()
if df.empty:
    st.info("No processed data available. Run the data pipeline first.")
    st.stop()

# --- Sidebar filters ---
st.sidebar.header("Filters")

if "circuit_id" in df.columns:
    all_circuits = sorted(df["circuit_id"].unique())
    circuits = st.sidebar.multiselect("Circuits", all_circuits, default=[], key="data_circuits")
else:
    circuits = []

if "compound" in df.columns:
    all_compounds = sorted(df["compound"].unique())
    compounds = st.sidebar.multiselect("Compounds", all_compounds, default=[], key="data_compounds")
else:
    compounds = []

# Season filter (derive from race_id or a year column if available)
if "race_id" in df.columns:
    # Attempt to extract year from race_id
    try:
        df["_year"] = df["race_id"].str[:4].astype(int)
        all_years = sorted(df["_year"].unique())
        years = st.sidebar.multiselect("Seasons", all_years, default=[], key="data_years")
    except Exception:
        years = []
else:
    years = []

# Apply filters
filtered = df.copy()
if circuits:
    filtered = filtered[filtered["circuit_id"].isin(circuits)]
if compounds:
    filtered = filtered[filtered["compound"].isin(compounds)]
if years and "_year" in filtered.columns:
    filtered = filtered[filtered["_year"].isin(years)]

# --- Summary stats ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Laps", f"{len(filtered):,}")
col2.metric(
    "Circuits", filtered["circuit_id"].nunique() if "circuit_id" in filtered.columns else "—"
)
col3.metric("Drivers", filtered["driver_id"].nunique() if "driver_id" in filtered.columns else "—")
col4.metric(
    "Seasons",
    filtered["_year"].nunique() if "_year" in filtered.columns else "—",
)

# --- Charts ---
st.subheader("Lap Time vs Tyre Life")
st.caption(
    "Each point is a single lap. The x-axis shows how many laps old the tyres are, "
    "and the y-axis shows the lap time. The upward trend within each compound "
    "is tyre degradation in action — this is the signal the models learn to predict."
)
# Subsample for performance
MAX_SCATTER = 10_000
if len(filtered) > MAX_SCATTER:
    scatter_df = filtered.sample(MAX_SCATTER, random_state=42)
    st.caption(f"Scatter plots show a {MAX_SCATTER:,}-lap subsample.")
else:
    scatter_df = filtered

st.plotly_chart(
    plot_data_scatter(scatter_df),
    width="stretch",
)

col5, col6 = st.columns(2)
with col5:
    st.markdown("##### Lap Time Distributions")
    st.caption(
        "Overlaid histograms showing the spread of lap times for each compound. "
        "Softer compounds cluster at lower (faster) times but have wider tails "
        "from late-stint degradation. Harder compounds are slower but more consistent."
    )
    st.plotly_chart(
        plot_lap_time_distributions(filtered),
        width="stretch",
    )
with col6:
    st.markdown("##### Weather Conditions")
    st.caption(
        "Track temperature vs air temperature for each circuit in the training data. "
        "This shows the range of conditions the models were trained on — predictions "
        "outside these ranges are extrapolations and less reliable."
    )
    st.plotly_chart(
        plot_weather_scatter(scatter_df),
        width="stretch",
    )
