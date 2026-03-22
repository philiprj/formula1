"""Tests for feature engineering."""

import pandas as pd
import pytest

from f1deg.data.features import build_features, compute_fuel_mass, compute_stint_info


def test_compute_fuel_mass():
    laps = pd.Series([1, 10, 50, 75])
    fuel = compute_fuel_mass(laps, start_mass=110.0, burn_rate=1.5)

    assert fuel.iloc[0] == 110.0  # Lap 1: full tank
    assert fuel.iloc[1] == pytest.approx(96.5)  # 110 - 1.5 * 9
    assert fuel.iloc[2] == pytest.approx(36.5)  # 110 - 1.5 * 49
    assert fuel.iloc[3] == pytest.approx(0.0)  # 110 - 1.5 * 74 = -1 -> clamped to 0


def test_compute_fuel_mass_no_negative():
    laps = pd.Series([100])
    fuel = compute_fuel_mass(laps, start_mass=110.0, burn_rate=1.5)
    assert fuel.iloc[0] >= 0.0


def test_compute_stint_info():
    df = pd.DataFrame(
        {
            "race_id": ["r1"] * 10,
            "Driver": ["VER"] * 10,
            "LapNumber": list(range(1, 11)),
            "TyreLife": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],  # Pit at lap 6
        }
    )
    result = compute_stint_info(df)
    assert result["stint_number"].tolist() == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    assert result["stint_lap"].tolist() == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]


def test_build_features(sample_raw_laps, sample_config):
    features = build_features(sample_raw_laps, sample_config)
    assert "lap_time_seconds" in features.columns
    assert "tyre_life" in features.columns
    assert "fuel_mass_kg" in features.columns
    assert "compound" in features.columns
    assert "circuit_id" in features.columns
    assert len(features) > 0
    assert features["fuel_mass_kg"].min() >= 0
