"""Page 6: Pit Window Optimizer.

Interactive tool that uses the degradation model to recommend optimal pit
stop timing given current race conditions. Includes SC-adjusted pit windows.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from f1deg.dashboard.components import model_selector
from f1deg.dashboard.state import (
    CIRCUIT_TO_GP,
    get_circuits,
    get_config,
    get_race_defaults,
    load_data,
    load_model,
)
from f1deg.strategy import (
    compute_sc_adjusted_pit_windows,
    find_optimal_pit_lap,
)
from f1deg.viz.theme import ACCENT, COMPOUND, GRID, SLATE, SURFACE, TEXT, TEXT_DIM

st.set_page_config(page_title="Pit Window Optimizer", layout="wide")
st.title("Pit Window Optimizer")
st.markdown(
    "Given your current race position — lap, compound, tyre age, fuel — the model "
    "recommends when to pit and what compound to switch to. The SC-adjusted view "
    "accounts for circuit-specific safety car probability."
)

# ── Sidebar ──────────────────────────────────────────────────────────────
model_name = model_selector(key="opt_model")
config = get_config()
circuits = get_circuits(config)

circuit = st.sidebar.selectbox(
    "Circuit",
    circuits,
    key="opt_circuit",
    index=circuits.index("silverstone") if "silverstone" in circuits else 0,
)

# Load defaults
gp_name = CIRCUIT_TO_GP.get(circuit, circuit)
defaults = get_race_defaults(circuit)

default_laps = defaults["strategy"]["total_race_laps"] if defaults else 52
total_race_laps = st.sidebar.number_input(
    "Total Race Laps", min_value=10, max_value=80, value=default_laps, key="opt_total_laps"
)

current_compound = st.sidebar.selectbox(
    "Current Compound", ["SOFT", "MEDIUM", "HARD"], index=1, key="opt_compound"
)

current_tyre_age = st.sidebar.slider("Current Tyre Age (laps)", 0, 40, 0, key="opt_tyre_age")

current_lap = current_tyre_age  # Simplified: assume tyre age ≈ race lap in stint 1

fuel_config = config.get("fuel", {})
start_fuel = fuel_config.get("start_mass_kg", 110.0)
burn_rate_default = fuel_config.get("burn_rate_kg_per_lap", 1.5)
circuit_fuel = config.get("fuel_by_circuit", {})
burn_rate = circuit_fuel.get(circuit, burn_rate_default)

fuel_kg = max(0.0, start_fuel - burn_rate * current_tyre_age)
st.sidebar.metric("Est. Fuel Remaining", f"{fuel_kg:.1f} kg")

with st.sidebar.expander("Weather Conditions"):
    if defaults:
        w = defaults["weather"]
        air_temp = st.slider("Air Temp (C)", 0, 50, w["air_temp"], key="opt_air")
        track_temp = st.slider("Track Temp (C)", 5, 65, w["track_temp"], key="opt_track")
        humidity = st.slider("Humidity (%)", 10, 100, w["humidity"], key="opt_hum")
        wind_speed = st.slider("Wind Speed (m/s)", 0, 25, w["wind_speed"], key="opt_wind")
    else:
        air_temp = st.slider("Air Temp (C)", 0, 50, 25, key="opt_air")
        track_temp = st.slider("Track Temp (C)", 5, 65, 40, key="opt_track")
        humidity = st.slider("Humidity (%)", 10, 100, 50, key="opt_hum")
        wind_speed = st.slider("Wind Speed (m/s)", 0, 25, 2, key="opt_wind")

conditions = {
    "air_temp": float(air_temp),
    "track_temp": float(track_temp),
    "humidity": float(humidity),
    "wind_speed": float(wind_speed),
    "rainfall": False,
}

# SC probability toggle
use_sc_adjustment = st.sidebar.toggle("SC-Adjusted Windows", value=True, key="opt_sc")

# ── Load model ───────────────────────────────────────────────────────────
model = load_model(model_name)
if model is None:
    st.error(f"Model '{model_name}' could not be loaded.")
    st.stop()

# ── Compute SC rates ─────────────────────────────────────────────────────
sc_rates = None
sc_rate_display = 0.04
if use_sc_adjustment:
    try:
        from f1deg.strategy_sc import compute_sc_rates_per_circuit, get_sc_probability

        df = load_data()
        if not df.empty:
            sc_rates = compute_sc_rates_per_circuit(df)
            sc_rate_display = get_sc_probability(gp_name, sc_rates=sc_rates)
    except Exception:
        pass

    st.sidebar.metric("SC Probability/Lap", f"{sc_rate_display:.1%}")

# ── Run optimizer ────────────────────────────────────────────────────────
with st.spinner("Computing optimal pit windows..."):
    optimal = find_optimal_pit_lap(
        model=model,
        circuit=gp_name,
        total_laps=total_race_laps,
        current_compound=current_compound,
        current_tyre_age=current_tyre_age,
        fuel_kg=fuel_kg,
        burn_rate=burn_rate,
        conditions=conditions,
    )

    if use_sc_adjustment:
        sc_windows = compute_sc_adjusted_pit_windows(
            model=model,
            circuit=gp_name,
            total_laps=total_race_laps,
            current_compound=current_compound,
            current_tyre_age=current_tyre_age,
            fuel_kg=fuel_kg,
            burn_rate=burn_rate,
            conditions=conditions,
            sc_rates=sc_rates,
        )
    else:
        sc_windows = None

# ── Recommendation banner ────────────────────────────────────────────────
st.subheader("Recommendation")

if optimal["optimal_pit_lap"] is not None:
    rec_lap = optimal["optimal_pit_lap"]
    rec_compound = optimal["optimal_compound"]
    time_saved = optimal["time_saved_seconds"]
    crossover = optimal["crossover_lap"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optimal Pit Lap", f"Lap {rec_lap + current_tyre_age}")
    col2.metric("Switch To", rec_compound)
    col3.metric("Time Saved vs No-Stop", f"{time_saved:.1f}s")
    col4.metric("Crossover Lap", f"Lap {crossover + current_tyre_age}" if crossover else "N/A")

    if crossover and crossover < 3:
        st.success("Pit window is **open now** — pitting immediately is beneficial.")
    elif crossover:
        st.info(
            f"Pit window opens at lap {crossover + current_tyre_age}. "
            f"Optimal stop at lap {rec_lap + current_tyre_age}."
        )
    else:
        st.warning("No-stop strategy appears faster — consider staying out.")
else:
    st.warning("No beneficial pit stop found — staying out is fastest.")

# ── Pit window chart ─────────────────────────────────────────────────────
st.subheader("Pit Window Analysis")
st.caption(
    "Expected remaining race time for each pit lap / compound combination. "
    "The lowest line is the best strategy at each lap. "
    "Dashed horizontal line = no-stop baseline."
)

all_strategies = optimal["all_strategies"]
fig = go.Figure()

# No-stop baseline
no_stop = all_strategies[all_strategies["pit_lap"] == 0]
if len(no_stop) > 0:
    no_stop_time = no_stop.iloc[0]["expected_time"]
    fig.add_hline(
        y=no_stop_time,
        line_dash="dash",
        line_color=TEXT_DIM,
        annotation_text=f"No-stop: {no_stop_time:.1f}s",
        annotation_position="top right",
    )

# Plot each compound
pit_options = all_strategies[all_strategies["pit_lap"] > 0]
for compound in pit_options["strategy"].str.extract(r"-> (\w+)")[0].dropna().unique():
    mask = pit_options["strategy"].str.contains(f"-> {compound}")
    subset = pit_options[mask].sort_values("pit_lap")
    color = COMPOUND.get(compound.upper(), TEXT_DIM)

    fig.add_trace(
        go.Scatter(
            x=subset["pit_lap"] + current_tyre_age,
            y=subset["expected_time"],
            mode="lines+markers",
            name=f"Pit → {compound}",
            line={"color": color, "width": 2},
            marker={"size": 4},
            hovertemplate=(
                "Pit at lap %{x}<br>"
                "Total time: %{y:.1f}s<br>"
                f"Compound: {compound}<br>"
                "Delta: %{customdata:.1f}s"
                "<extra></extra>"
            ),
            customdata=subset["delta_vs_no_stop"],
        )
    )

    # Show uncertainty band
    if "lower_bound" in subset.columns:
        fig.add_trace(
            go.Scatter(
                x=list(subset["pit_lap"] + current_tyre_age)
                + list((subset["pit_lap"] + current_tyre_age)[::-1]),
                y=list(subset["upper_bound"]) + list(subset["lower_bound"][::-1]),
                fill="toself",
                fillcolor=color.replace(")", ", 0.1)").replace("rgb", "rgba"),
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )

# Mark optimal
if optimal["optimal_pit_lap"] is not None:
    best_row = pit_options.loc[pit_options["expected_time"].idxmin()]
    fig.add_trace(
        go.Scatter(
            x=[best_row["pit_lap"] + current_tyre_age],
            y=[best_row["expected_time"]],
            mode="markers",
            marker={"size": 14, "color": ACCENT, "symbol": "star"},
            name="Optimal",
            showlegend=True,
        )
    )

fig.update_layout(
    paper_bgcolor=SLATE,
    plot_bgcolor=SURFACE,
    font={"color": TEXT},
    xaxis={"title": "Pit Lap", "gridcolor": GRID},
    yaxis={"title": "Expected Remaining Race Time (s)", "gridcolor": GRID},
    legend={"bgcolor": SURFACE, "bordercolor": GRID},
    hovermode="x unified",
    height=500,
)

st.plotly_chart(fig, use_container_width=True)

# ── SC-adjusted comparison ───────────────────────────────────────────────
if sc_windows is not None:
    st.subheader("Safety Car Adjustment")
    st.caption(
        f"SC probability: **{sc_rate_display:.1%}** per lap for {gp_name}. "
        "The chart shows how expected SC savings shift the optimal pit window earlier "
        "(since pitting under SC is much cheaper)."
    )

    sc_pit = sc_windows[sc_windows["pit_lap"] > 0].copy()
    if len(sc_pit) > 0:
        fig_sc = go.Figure()

        # Green flag vs SC-adjusted for best compound
        for compound in sc_pit["strategy"].str.extract(r"-> (\w+)")[0].dropna().unique():
            mask = sc_pit["strategy"].str.contains(f"-> {compound}")
            subset = sc_pit[mask].sort_values("pit_lap")
            color = COMPOUND.get(compound.upper(), TEXT_DIM)

            fig_sc.add_trace(
                go.Scatter(
                    x=subset["pit_lap"] + current_tyre_age,
                    y=subset["expected_time"],
                    mode="lines",
                    name=f"{compound} (green flag)",
                    line={"color": color, "width": 1, "dash": "dot"},
                )
            )
            fig_sc.add_trace(
                go.Scatter(
                    x=subset["pit_lap"] + current_tyre_age,
                    y=subset["sc_adjusted_time"],
                    mode="lines+markers",
                    name=f"{compound} (SC-adjusted)",
                    line={"color": color, "width": 2},
                    marker={"size": 4},
                )
            )

        fig_sc.update_layout(
            paper_bgcolor=SLATE,
            plot_bgcolor=SURFACE,
            font={"color": TEXT},
            xaxis={"title": "Pit Lap", "gridcolor": GRID},
            yaxis={"title": "Expected Remaining Time (s)", "gridcolor": GRID},
            legend={"bgcolor": SURFACE, "bordercolor": GRID},
            hovermode="x unified",
            height=450,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ── SC rates table ───────────────────────────────────────────────────────
if sc_rates is not None:
    with st.expander("Circuit SC Rates"):
        display = sc_rates.copy()
        display["smoothed_rate"] = display["smoothed_rate"].map("{:.1%}".format)
        display["raw_rate"] = display["raw_rate"].map("{:.1%}".format)
        st.dataframe(
            display[["circuit_id", "total_races", "sc_laps", "raw_rate", "smoothed_rate"]],
            use_container_width=True,
            hide_index=True,
        )
