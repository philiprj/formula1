"""Cached data and model loading for the dashboard."""

from __future__ import annotations

import importlib
import logging

import pandas as pd
import streamlit as st

from f1deg.config import DATA_DIR, load_config

logger = logging.getLogger(__name__)

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
