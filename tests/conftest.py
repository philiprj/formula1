"""Shared test fixtures.

All fixtures use synthetic data — no API calls required.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_laps():
    """Synthetic raw lap data mimicking FastF1 output."""
    np.random.seed(42)
    n = 100

    compounds = ["SOFT", "MEDIUM", "HARD"]
    drivers = ["VER", "HAM", "LEC", "NOR"]
    teams = ["Red Bull", "Mercedes", "Ferrari", "McLaren"]

    # Create Time column (cumulative session time) for gap computation
    base_times = []
    for d in range(4):  # 4 drivers
        for lap in range(25):  # 25 laps each
            # Stagger drivers by ~1-3 seconds
            base_times.append(
                pd.Timedelta(seconds=90 * (lap + 2) + d * 1.5 + np.random.normal(0, 0.5))
            )

    data = {
        "LapNumber": np.tile(np.arange(2, 27), 4),  # Laps 2-26 for 4 drivers
        "LapTime_seconds": np.random.normal(90, 2, n),
        "TyreLife": np.tile(np.arange(1, 26), 4),
        "Compound": np.random.choice(compounds, n),
        "Driver": np.repeat(drivers, 25),
        "Team": np.repeat(teams, 25),
        "IsAccurate": [True] * 95 + [False] * 5,
        "TrackStatus": ["1"] * 85 + ["2"] * 5 + ["4"] * 5 + ["1"] * 5,
        "PitInTime": [pd.NaT] * 96 + [pd.Timestamp("2024-01-01 15:00:00")] * 4,
        "PitOutTime": [pd.NaT] * 96 + [pd.Timestamp("2024-01-01 15:00:25")] * 4,
        "Year": [2024] * n,
        "RoundNumber": [1] * n,
        "CircuitKey": ["Bahrain"] * n,
        "TotalLaps": [57] * n,
        "Position": np.tile(np.repeat([1, 2, 3, 4], 1), 25).flatten()[:n].astype(float),
        "Time": base_times,
        "AirTemp": np.random.normal(30, 2, n),
        "TrackTemp": np.random.normal(45, 3, n),
        "Humidity": np.random.normal(40, 5, n),
        "Pressure": np.random.normal(1013, 5, n),
        "WindSpeed": np.random.normal(3, 1, n),
        "WindDirection": np.random.uniform(0, 360, n),
        "Rainfall": [False] * n,
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_processed_laps():
    """Synthetic processed/feature-engineered lap data."""
    np.random.seed(42)
    n = 200

    compounds = ["SOFT", "MEDIUM", "HARD"]
    drivers = ["VER", "HAM", "LEC", "NOR"]
    circuits = ["Bahrain", "Jeddah"]

    data = {
        "race_id": np.repeat(["2024_01", "2024_02"], 100),
        "circuit_id": np.repeat(circuits, 100),
        "driver_id": np.tile(np.repeat(drivers, 25), 2),
        "constructor_id": np.tile(np.repeat(["Red Bull", "Mercedes", "Ferrari", "McLaren"], 25), 2),
        "lap_number": np.tile(np.arange(2, 27), 8),
        "tyre_life": np.tile(np.arange(1, 26).astype(float), 8),
        "compound": np.random.choice(compounds, n),
        "fuel_mass_kg": np.tile(np.linspace(108.5, 72.5, 25), 8),
        "air_temp": np.random.normal(30, 2, n),
        "track_temp": np.random.normal(45, 3, n),
        "humidity": np.random.normal(40, 5, n),
        "wind_speed": np.random.normal(3, 1, n),
        "rainfall": [False] * n,
        "stint_number": np.tile(np.repeat([1, 1, 1, 1], 25), 2),
        "stint_lap": np.tile(np.arange(1, 26), 8),
    }

    # Generate realistic lap times: base + fuel effect + tire deg + noise
    base_time = 90.0
    fuel_effect = -0.035 * data["fuel_mass_kg"]
    tyre_effect = 0.05 * data["tyre_life"] + 0.002 * data["tyre_life"] ** 2
    compound_effect = np.where(
        np.array(data["compound"]) == "SOFT",
        -1.5,
        np.where(np.array(data["compound"]) == "HARD", 1.0, 0.0),
    )
    noise = np.random.normal(0, 0.3, n)

    data["lap_time_seconds"] = base_time + fuel_effect + tyre_effect + compound_effect + noise
    data["tyre_life_sq"] = data["tyre_life"] ** 2

    return pd.DataFrame(data)


@pytest.fixture
def sample_config():
    """Minimal configuration for testing."""
    return {
        "seasons": [2024],
        "fuel": {"start_mass_kg": 110.0, "burn_rate_kg_per_lap": 1.5},
        "data_dir": "/tmp/f1deg_test",
        "features": {
            "filters": ["accurate", "track_status", "pit_laps", "first_lap", "outliers"],
            "track_status_exclude": ["4", "5", "6"],
            "outlier_iqr_multiplier": 3.0,
        },
        "model": {
            "alpha": 1.0,
            "features": [
                "tyre_life",
                "tyre_life_sq",
                "fuel_mass_kg",
                "track_temp",
                "air_temp",
                "rainfall",
            ],
            "categorical_features": ["compound", "circuit_id"],
        },
    }
