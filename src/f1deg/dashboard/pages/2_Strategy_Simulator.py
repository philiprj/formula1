"""Page 3: Full race strategy simulator."""

from __future__ import annotations

import streamlit as st

from f1deg.dashboard.charts import (
    plot_cumulative_time,
    plot_race_timeline,
    plot_strategy_comparison,
)
from f1deg.dashboard.components import COMPOUNDS, model_selector
from f1deg.dashboard.state import (
    CIRCUIT_TO_GP,
    get_circuits,
    get_config,
    get_pit_loss,
    get_race_defaults,
    load_model,
)

st.set_page_config(page_title="Strategy Simulator", layout="wide")
st.title("Strategy Simulator")
st.markdown(
    "Simulate a full race by choosing when to pit and which compounds to use. "
    "The model predicts lap times for each stint accounting for tyre degradation, "
    "fuel burn, and weather. Save multiple strategies and compare their total race times."
)

# --- Sidebar ---
model_name = model_selector(key="strat_model")
config = get_config()
circuits = get_circuits(config)

circuit = st.sidebar.selectbox(
    "Circuit",
    circuits,
    key="strat_circuit",
    index=circuits.index("silverstone") if "silverstone" in circuits else 0,
)

# Load 2025 race defaults when the circuit changes
_prev_key = "strat_prev_circuit"
if circuit != st.session_state.get(_prev_key):
    defaults = get_race_defaults(circuit)
    if defaults:
        w = defaults["weather"]
        st.session_state["strat_air"] = w["air_temp"]
        st.session_state["strat_track"] = w["track_temp"]
        st.session_state["strat_hum"] = w["humidity"]
        st.session_state["strat_wind"] = w["wind_speed"]
        st.session_state["strat_rain"] = w["rainfall"]

        s = defaults["strategy"]
        st.session_state["strat_total_laps"] = s["total_race_laps"]
        st.session_state["strat_num_stints"] = s["num_stints"]
        for i, stint in enumerate(s["stints"]):
            st.session_state[f"strat_compound_{i}"] = stint["compound"]
            if i < s["num_stints"] - 1:
                st.session_state[f"strat_laps_{i}"] = stint["laps"]
    st.session_state[_prev_key] = circuit
    st.rerun()

gp_name = CIRCUIT_TO_GP.get(circuit)
race_defaults = get_race_defaults(circuit)
if race_defaults:
    strat_summary = " / ".join(
        f"{s['compound']} ({s['laps']})" for s in race_defaults["strategy"]["stints"]
    )
    st.sidebar.caption(f"2025 {gp_name} winner: {race_defaults['winner_id']} ({strat_summary})")

total_race_laps = st.sidebar.number_input(
    "Total Race Laps",
    min_value=10,
    max_value=80,
    value=52,
    step=1,
    key="strat_total_laps",
)

num_stints = st.sidebar.slider("Number of Stints", 1, 4, 2, key="strat_num_stints")

# Weather (collapsed)
with st.sidebar.expander("Weather Conditions"):
    air_temp = st.slider("Air Temp (C)", 15, 45, 25, key="strat_air")
    track_temp = st.slider("Track Temp (C)", 10, 60, 40, key="strat_track")
    humidity = st.slider("Humidity (%)", 10, 100, 50, key="strat_hum")
    wind_speed = st.slider("Wind Speed (m/s)", 0, 15, 2, key="strat_wind")
    rainfall = st.toggle("Wet Conditions", value=False, key="strat_rain")

conditions = {
    "air_temp": float(air_temp),
    "track_temp": float(track_temp),
    "humidity": float(humidity),
    "wind_speed": float(wind_speed),
    "rainfall": rainfall,
}

# --- Stint configuration ---
st.subheader("Stint Configuration")

stint_compounds = []
stint_lengths = []
remaining = total_race_laps

stint_cols = st.columns(num_stints)
for i in range(num_stints):
    with stint_cols[i]:
        st.markdown(f"**Stint {i + 1}**")
        compound = st.selectbox(
            "Compound",
            COMPOUNDS,
            index=min(i, 2),  # cycle through S/M/H
            key=f"strat_compound_{i}",
        )
        stint_compounds.append(compound)

        if i < num_stints - 1:
            max_laps = remaining - (num_stints - i - 1)  # leave at least 1 per remaining stint
            default_laps = min(remaining // (num_stints - i), max_laps)
            laps = st.slider(
                "Laps",
                min_value=1,
                max_value=max(1, max_laps),
                value=max(1, default_laps),
                key=f"strat_laps_{i}",
            )
            stint_lengths.append(laps)
            remaining -= laps
        else:
            # Last stint gets the remaining laps
            stint_lengths.append(remaining)
            st.metric("Laps", remaining)

# Validate
if sum(stint_lengths) != total_race_laps:
    st.warning(f"Stint lengths sum to {sum(stint_lengths)} but race is {total_race_laps} laps.")
    st.stop()

# --- Generate predictions ---
model = load_model(model_name)
if model is None:
    st.error(f"Model '{model_name}' could not be loaded.")
    st.stop()

pit_loss = get_pit_loss(config, circuit)
fuel_config = config.get("fuel", {})
start_fuel = fuel_config.get("start_mass_kg", 110.0)
burn_rate = fuel_config.get("burn_rate_kg_per_lap", 1.5)

stint_curves = []
pit_laps = []
current_lap = 0
current_fuel = start_fuel

for i, (compound, length) in enumerate(zip(stint_compounds, stint_lengths, strict=False)):
    curve = model.predict_degradation_curve(
        compound=compound,
        circuit=circuit,
        n_laps=length,
        start_fuel_kg=current_fuel,
        burn_rate=burn_rate,
        conditions=conditions,
    )

    race_laps = list(range(current_lap + 1, current_lap + length + 1))
    stint_curves.append(
        {
            "compound": compound,
            "laps": race_laps,
            "times": curve["predicted_lap_time"].tolist(),
            "lower": curve["lower_bound"].tolist(),
            "upper": curve["upper_bound"].tolist(),
        }
    )

    current_lap += length
    current_fuel = max(0, current_fuel - burn_rate * length)

    if i < num_stints - 1:
        pit_laps.append(current_lap)

# --- Summary ---
st.subheader("Race Summary")
st.caption(
    "**Driving Time** — total time spent on track across all stints. "
    "**Pit Time** — time lost in the pit lane (stationary + speed limit). "
    "**Total Race Time** — the sum of both; the number to minimise."
)
total_driving_time = sum(sum(s["times"]) for s in stint_curves)
total_pit_time = pit_loss * len(pit_laps)
total_race_time = total_driving_time + total_pit_time

col1, col2, col3 = st.columns(3)
col1.metric("Driving Time", f"{total_driving_time / 60:.1f} min")
col2.metric("Pit Time", f"{total_pit_time:.1f}s ({len(pit_laps)} stops)")
col3.metric("Total Race Time", f"{total_race_time / 60:.1f} min")

# --- Charts ---
st.subheader("Race Lap Times")
st.caption(
    "Predicted lap time for every lap of the race. Each stint is coloured by "
    "compound — you can see the tyre degradation within each stint as an upward "
    "slope, and the drop in lap time after a pit stop onto fresh tyres. "
    "Dashed vertical lines mark pit stops. The shaded bands show prediction intervals."
)
st.plotly_chart(
    plot_race_timeline(stint_curves, pit_laps),
    width="stretch",
)

st.subheader("Cumulative Race Time")
st.caption(
    "Running total of elapsed race time. A steeper slope means slower laps. "
    "The step-ups at pit stops reflect the pit lane time loss. Comparing "
    "different strategies here shows where time is gained or lost over a full race."
)
pit_times = [pit_loss] * len(pit_laps)
st.plotly_chart(
    plot_cumulative_time(stint_curves, pit_times),
    width="stretch",
)

# --- Strategy comparison ---
st.divider()
st.subheader("Compare Strategies")
st.caption(
    "Save the current strategy and then adjust the stint configuration above to try "
    "a different approach. Each saved strategy appears as a bar — the shortest bar wins."
)

strategy_label = " / ".join(
    f"{c} ({n})" for c, n in zip(stint_compounds, stint_lengths, strict=False)
)

if "saved_strategies" not in st.session_state:
    st.session_state.saved_strategies = []

if st.button("Save Current Strategy", key="save_strat"):
    st.session_state.saved_strategies.append(
        {
            "name": strategy_label,
            "total_time": total_race_time,
        }
    )
    st.success(f"Saved: {strategy_label}")

if st.session_state.saved_strategies:
    st.plotly_chart(
        plot_strategy_comparison(st.session_state.saved_strategies),
        width="stretch",
    )

    if st.button("Clear All", key="clear_strats"):
        st.session_state.saved_strategies = []
        st.rerun()
