"""Cached data and model loading for the dashboard."""

from __future__ import annotations

import importlib
import logging

import numpy as np
import pandas as pd
import streamlit as st

from f1deg.config import DATA_DIR, load_config

logger = logging.getLogger(__name__)

# Mapping from config circuit slugs to FastF1 EventName values used in the
# processed data's ``circuit_id`` column.
CIRCUIT_TO_GP: dict[str, str] = {
    "bahrain": "Bahrain Grand Prix",
    "jeddah": "Saudi Arabian Grand Prix",
    "albert_park": "Australian Grand Prix",
    "suzuka": "Japanese Grand Prix",
    "shanghai": "Chinese Grand Prix",
    "miami": "Miami Grand Prix",
    "imola": "Emilia Romagna Grand Prix",
    "monaco": "Monaco Grand Prix",
    "montreal": "Canadian Grand Prix",
    "barcelona": "Spanish Grand Prix",
    "spielberg": "Austrian Grand Prix",
    "silverstone": "British Grand Prix",
    "hungaroring": "Hungarian Grand Prix",
    "spa": "Belgian Grand Prix",
    "zandvoort": "Dutch Grand Prix",
    "monza": "Italian Grand Prix",
    "baku": "Azerbaijan Grand Prix",
    "marina_bay": "Singapore Grand Prix",
    "cota": "United States Grand Prix",
    "mexico": "Mexico City Grand Prix",
    "interlagos": "São Paulo Grand Prix",
    "las_vegas": "Las Vegas Grand Prix",
    "lusail": "Qatar Grand Prix",
    "yas_marina": "Abu Dhabi Grand Prix",
}

MODEL_REGISTRY: dict[str, str] = {
    "linear": "f1deg.models.linear:LinearDegradationModel",
    "bayesian": "f1deg.models.bayesian:BayesianDegradationModel",
    "gbm": "f1deg.models.gbm:GBMDegradationModel",
}

MODEL_LABELS: dict[str, str] = {
    "linear": "Linear (Ridge)",
    "bayesian": "Bayesian Hierarchical",
    "gbm": "Gradient Boosted Trees",
}


def _get_model_class(model_name: str):
    """Import and return the model class by name."""
    module_path, class_name = MODEL_REGISTRY[model_name].rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@st.cache_data(show_spinner="Loading lap data...")
def load_data() -> pd.DataFrame:
    """Load the processed lap data from parquet."""
    path = DATA_DIR / "processed" / "laps_clean.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_name: str):
    """Load a trained model from disk. Returns None if not available."""
    model_dir = DATA_DIR / "models" / model_name
    if not model_dir.exists():
        return None
    try:
        model_cls = _get_model_class(model_name)
        return model_cls.load(model_dir)
    except Exception as e:
        logger.warning(f"Failed to load model '{model_name}': {e}")
        return None


def load_all_models() -> dict[str, object]:
    """Load all available models. Returns {name: model} for those that exist."""
    models = {}
    for name in MODEL_REGISTRY:
        model = load_model(name)
        if model is not None:
            models[name] = model
    return models


def available_model_names() -> list[str]:
    """Return names of models that have trained artifacts on disk."""
    return [name for name in MODEL_REGISTRY if (DATA_DIR / "models" / name).exists()]


@st.cache_data(show_spinner=False)
def get_config() -> dict:
    """Load the base project configuration."""
    return load_config()


def get_circuits(config: dict) -> list[str]:
    """Extract sorted circuit list from config."""
    return sorted(config.get("circuits", {}).keys())


def get_pit_loss(config: dict, circuit: str) -> float:
    """Get pit stop time loss for a circuit in seconds."""
    result: float = config.get("pit_loss", {}).get(circuit, 23.0)
    return result


@st.cache_data(show_spinner=False)
def get_race_defaults(circuit_slug: str) -> dict | None:
    """Return 2025 race weather and winner strategy for a circuit.

    Returns ``None`` if no 2025 data exists for the circuit.  Otherwise a dict::

        {
            "gp_name": str,
            "weather": {"air_temp": int, "track_temp": int, ...},
            "strategy": {"num_stints": int, "stints": [...], "total_race_laps": int},
            "winner_id": str,
        }
    """
    gp_name = CIRCUIT_TO_GP.get(circuit_slug)
    if gp_name is None:
        return None

    df = load_data()
    if df.empty:
        return None

    race_df = df[df["race_id"].str.startswith("2025") & (df["circuit_id"] == gp_name)]
    if race_df.empty:
        return None

    # Deduplicate in case of overlapping raw files
    dedup_cols = ["race_id", "driver_id", "lap_number"]
    present_cols = [c for c in dedup_cols if c in race_df.columns]
    if present_cols:
        race_df = race_df.drop_duplicates(subset=present_cols, keep="first")

    # --- Weather (median across all laps in the race) ---
    weather: dict[str, int | bool] = {}
    for col, default in [
        ("air_temp", 25),
        ("track_temp", 40),
        ("humidity", 50),
        ("wind_speed", 2),
    ]:
        if col in race_df.columns:
            med = race_df[col].dropna().median()
            weather[col] = default if np.isnan(med) else round(med)
        else:
            weather[col] = default

    if "rainfall" in race_df.columns:
        weather["rainfall"] = bool(race_df["rainfall"].any())
    else:
        weather["rainfall"] = False

    # --- Winner identification ---
    driver_stats = (
        race_df.groupby("driver_id")
        .agg(max_lap=("lap_number", "max"), total_time=("lap_time_seconds", "sum"))
        .sort_values(["max_lap", "total_time"], ascending=[False, True])
    )
    winner_id = driver_stats.index[0]

    # --- Winner's stint structure ---
    winner_laps = race_df[race_df["driver_id"] == winner_id]
    stint_info = (
        winner_laps.groupby("stint_number")
        .agg(compound=("compound", "first"), laps=("lap_number", "count"))
        .sort_index()
    )

    stints = [
        {"compound": row["compound"], "laps": int(row["laps"])} for _, row in stint_info.iterrows()
    ]
    # Cap at 4 stints (UI max)
    stints = stints[:4]

    total_race_laps = int(race_df["lap_number"].max())

    return {
        "gp_name": gp_name,
        "weather": weather,
        "strategy": {
            "num_stints": len(stints),
            "stints": stints,
            "total_race_laps": total_race_laps,
        },
        "winner_id": winner_id,
    }
