"""Page 4: Model diagnostics — residuals, predicted vs actual."""

from __future__ import annotations

import streamlit as st

from f1deg.dashboard.charts import (
    plot_predicted_vs_actual,
    plot_residual_histogram,
    plot_residuals_by_group,
    plot_residuals_scatter,
)
from f1deg.dashboard.components import model_selector
from f1deg.dashboard.state import load_data, load_model
from f1deg.eval.metrics import compute_all_metrics

st.set_page_config(page_title="Diagnostics", layout="wide")
st.title("Model Diagnostics")
st.markdown(
    "Evaluate how well the selected model fits the historical data. "
    "Good models have small, randomly distributed residuals and prediction "
    "intervals that cover close to 95% of actual values."
)

# --- Sidebar ---
model_name = model_selector(key="diag_model")

group_options = ["compound", "circuit_id", "driver_id"]
group_col = st.sidebar.selectbox("Group By", group_options, key="diag_group")

# --- Load data and model ---
df = load_data()
if df.empty:
    st.info("No processed data available. Run the data pipeline first.")
    st.stop()

model = load_model(model_name)
if model is None:
    st.error(f"Model '{model_name}' could not be loaded.")
    st.stop()

# Subsample for performance (Bayesian/LSTM can be slow on full data)
MAX_SAMPLES = 10_000
if len(df) > MAX_SAMPLES:
    df_sample = df.sample(MAX_SAMPLES, random_state=42)
    st.caption(f"Showing diagnostics on a {MAX_SAMPLES:,}-lap subsample for performance.")
else:
    df_sample = df


@st.cache_data(show_spinner="Running predictions...")
def _compute_predictions(_model_name: str, data_hash: int):
    """Compute predictions and metrics (cached by model name + data hash)."""
    y_pred = model.predict(df_sample)

    has_intervals = True
    try:
        lower, upper = model.predict_interval(df_sample)
    except Exception:
        lower, upper = None, None
        has_intervals = False

    y_true = df_sample["lap_time_seconds"].values

    metrics = compute_all_metrics(
        y_true,
        y_pred,
        lower,
        upper,
        df_sample if "stint_lap" in df_sample.columns else None,
    )
    return y_true, y_pred, lower, upper, has_intervals, metrics


# Use hash of index to cache by subsample
data_hash = hash(tuple(df_sample.index.tolist()[:100]))
y_true, y_pred, lower, upper, has_intervals, metrics = _compute_predictions(model_name, data_hash)

# --- Metrics summary ---
st.subheader("Summary Metrics")
st.caption(
    "**MAE** (Mean Absolute Error) — average prediction error in seconds; lower is better. "
    "**RMSE** (Root Mean Squared Error) — like MAE but penalises large errors more heavily. "
    "**PI Coverage** — fraction of actual lap times that fall within the 95% prediction interval; "
    "should be close to 95%. **PI Width** — average width of the interval; narrower is better "
    "given good coverage."
)
metric_cols = st.columns(4)
metric_cols[0].metric("MAE", f"{metrics['mae']:.4f}s")
metric_cols[1].metric("RMSE", f"{metrics['rmse']:.4f}s")
if has_intervals:
    metric_cols[2].metric("PI Coverage (95%)", f"{metrics.get('pi_coverage_95', 0):.1%}")
    metric_cols[3].metric("PI Width", f"{metrics.get('pi_width_mean', 0):.3f}s")

# --- Charts ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("##### Residuals vs Predicted")
    st.caption(
        "Each point is a lap. The y-axis shows the error (actual minus predicted). "
        "A well-calibrated model has residuals scattered randomly around zero with "
        "no visible patterns. Funnel shapes or curves indicate systematic bias."
    )
    st.plotly_chart(
        plot_residuals_scatter(y_true, y_pred),
        width="stretch",
    )

with col2:
    st.markdown("##### Residual Distribution")
    st.caption(
        "Histogram of all prediction errors. Ideally this is a tight, symmetric bell curve "
        "centred at zero. A mean far from zero indicates consistent over- or under-prediction. "
        "Heavy tails mean occasional large errors."
    )
    st.plotly_chart(
        plot_residual_histogram(y_true, y_pred),
        width="stretch",
    )

col3, col4 = st.columns(2)

with col3:
    st.markdown(f"##### Residuals by {group_col.replace('_', ' ').title()}")
    st.caption(
        "Box plots of errors grouped by the selected category. Each box shows the "
        "median error and spread. If certain groups (e.g. a specific compound or circuit) "
        "have boxes far from zero, the model struggles with those conditions."
    )
    if group_col in df_sample.columns:
        residuals = y_true - y_pred
        st.plotly_chart(
            plot_residuals_by_group(residuals, df_sample[group_col], group_col),
            width="stretch",
        )
    else:
        st.info(f"Column '{group_col}' not found in data.")

with col4:
    st.markdown("##### Predicted vs Actual")
    st.caption(
        "Each point compares the model's prediction (y-axis) against the true lap time "
        "(x-axis). Perfect predictions fall on the dashed diagonal line. Points above "
        "the line are over-predictions; points below are under-predictions."
    )
    st.plotly_chart(
        plot_predicted_vs_actual(y_true, y_pred),
        width="stretch",
    )
