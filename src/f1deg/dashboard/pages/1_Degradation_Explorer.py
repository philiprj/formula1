"""Page 1: Interactive degradation curve explorer with animation."""

from __future__ import annotations

import streamlit as st

from f1deg.dashboard.charts import (
    plot_degradation_animated,
    plot_degradation_curves,
    plot_lap_deltas,
)
from f1deg.dashboard.components import metric_card_row, model_selector, scenario_sidebar
from f1deg.dashboard.state import load_model

st.set_page_config(page_title="Degradation Explorer", layout="wide")
st.title("Degradation Explorer")
st.markdown(
    "Explore how tyre performance changes over a stint. Each compound degrades at a "
    "different rate — softer compounds are faster initially but lose grip more quickly. "
    "Use the sidebar to adjust the scenario and see how conditions affect degradation."
)

# --- Sidebar ---
model_name = model_selector(key="deg_model")
params = scenario_sidebar(key_prefix="deg_")

animate = st.sidebar.toggle("Animate build-up", value=False, key="deg_animate")

# --- Generate curves ---
model = load_model(model_name)
if model is None:
    st.error(f"Model '{model_name}' could not be loaded.")
    st.stop()

curves: dict = {}
for compound in params["compounds"]:
    curve = model.predict_degradation_curve(
        compound=compound,
        circuit=params["circuit"],
        n_laps=params["stint_length"],
        start_fuel_kg=params["start_fuel_kg"],
        burn_rate=params["burn_rate"],
        conditions=params["conditions"],
    )
    curves[compound] = curve

if not curves:
    st.info("Select at least one compound to see degradation curves.")
    st.stop()

# --- Summary metrics ---
st.subheader("Stint Summary")
st.caption(
    "**Total Deg** — how much slower the final lap is compared to the first lap on fresh tyres. "
    "**Avg Deg / Lap** — the mean time lost per lap due to tyre wear. "
    "**PI Width** — the width of the 95% prediction interval on the final lap; "
    "narrower means the model is more confident."
)
first_compound = next(iter(curves.keys()))
ref_curve = curves[first_compound]
total_deg = float(
    ref_curve["predicted_lap_time"].iloc[-1] - ref_curve["predicted_lap_time"].iloc[0]
)
avg_deg = total_deg / max(len(ref_curve) - 1, 1)
pi_width_final = float(ref_curve["upper_bound"].iloc[-1] - ref_curve["lower_bound"].iloc[-1])

metric_card_row(
    {
        f"Total Deg ({first_compound})": (f"{total_deg:+.3f}s", None),
        "Avg Deg / Lap": (f"{avg_deg:+.4f}s", None),
        f"PI Width (lap {len(ref_curve)})": (f"{pi_width_final:.3f}s", None),
    }
)

# --- Charts ---
st.subheader("Degradation Curves")
st.caption(
    "Each line shows the predicted lap time over the stint for a tyre compound. "
    "The shaded band is the model's 95% prediction interval — the region where "
    "the true lap time is expected to fall. An upward slope means the tyres are "
    "getting slower (degrading). Toggle **Animate build-up** in the sidebar to "
    "watch the curve build lap-by-lap."
)
if animate:
    st.plotly_chart(
        plot_degradation_animated(curves),
        width="stretch",
        key="deg_anim_chart",
    )
else:
    st.plotly_chart(
        plot_degradation_curves(curves),
        width="stretch",
        key="deg_static_chart",
    )

st.subheader("Lap-over-Lap Degradation Rate")
st.caption(
    "Each bar shows how much slower a lap is compared to the previous lap. "
    "Consistent bar heights indicate linear degradation; increasing heights "
    "mean the tyres are falling off a cliff — a critical sign for pit stop timing."
)
st.plotly_chart(
    plot_lap_deltas(curves),
    width="stretch",
    key="deg_delta_chart",
)
