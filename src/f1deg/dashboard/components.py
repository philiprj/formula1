"""Shared UI components for the dashboard."""

from __future__ import annotations

import streamlit as st

from f1deg.dashboard.state import (
    MODEL_LABELS,
    available_model_names,
    get_circuits,
    get_config,
)

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def model_selector(*, multi: bool = False, key: str = "model_sel") -> str | list[str]:
    """Render a model selector dropdown. Returns selected model name(s)."""
    available = available_model_names()
    if not available:
        st.warning("No trained models found. Run `python scripts/03_train.py <model>` first.")
        st.stop()

    labels = [MODEL_LABELS.get(m, m) for m in available]
    label_to_name = dict(zip(labels, available, strict=False))

    if multi:
        selected_labels = st.sidebar.multiselect(
            "Models",
            labels,
            default=labels,
            key=key,
        )
        return [label_to_name[lb] for lb in selected_labels]
    else:
        selected_label = st.sidebar.selectbox("Model", labels, key=key)
        return label_to_name[selected_label]


def scenario_sidebar(*, key_prefix: str = "") -> dict:
    """Render the shared scenario controls in the sidebar.

    Returns a dict with keys:
        compounds, circuit, stint_length, start_fuel_kg, burn_rate, conditions
    """
    config = get_config()
    circuits = get_circuits(config)

    compounds = st.sidebar.multiselect(
        "Compounds",
        COMPOUNDS,
        default=DRY_COMPOUNDS,
        key=f"{key_prefix}compounds",
    )

    circuit = st.sidebar.selectbox(
        "Circuit",
        circuits,
        index=circuits.index("silverstone") if "silverstone" in circuits else 0,
        key=f"{key_prefix}circuit",
    )

    stint_length = st.sidebar.slider(
        "Stint Length (laps)",
        min_value=5,
        max_value=60,
        value=25,
        key=f"{key_prefix}stint_len",
    )

    # Fuel parameters (collapsed)
    with st.sidebar.expander("Fuel Parameters"):
        start_fuel = st.slider(
            "Start Fuel (kg)",
            min_value=50.0,
            max_value=110.0,
            value=110.0,
            step=1.0,
            key=f"{key_prefix}fuel",
        )
        burn_rate = st.slider(
            "Burn Rate (kg/lap)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            key=f"{key_prefix}burn",
        )

    # Weather conditions (collapsed)
    with st.sidebar.expander("Weather Conditions"):
        air_temp = st.slider("Air Temp (C)", 15, 45, 25, key=f"{key_prefix}air")
        track_temp = st.slider("Track Temp (C)", 20, 60, 40, key=f"{key_prefix}track")
        humidity = st.slider("Humidity (%)", 10, 100, 50, key=f"{key_prefix}hum")
        wind_speed = st.slider("Wind Speed (m/s)", 0, 15, 2, key=f"{key_prefix}wind")
        rainfall = st.toggle("Wet Conditions", value=False, key=f"{key_prefix}rain")

    conditions = {
        "air_temp": float(air_temp),
        "track_temp": float(track_temp),
        "humidity": float(humidity),
        "wind_speed": float(wind_speed),
        "rainfall": rainfall,
    }

    return {
        "compounds": compounds,
        "circuit": circuit,
        "stint_length": stint_length,
        "start_fuel_kg": start_fuel,
        "burn_rate": burn_rate,
        "conditions": conditions,
    }


def metric_card_row(metrics: dict[str, tuple[str, str | None]]) -> None:
    """Render a row of st.metric cards.

    Args:
        metrics: {label: (value, delta)} where delta can be None.
    """
    cols = st.columns(len(metrics))
    for col, (label, (value, delta)) in zip(cols, metrics.items(), strict=False):
        col.metric(label, value, delta)
