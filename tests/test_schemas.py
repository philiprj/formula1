"""Tests for data schema validation."""

import pandas as pd
import pandera.pandas as pandera
import pytest

from f1deg.data.schemas import check_data_quality, validate_processed


def test_validate_processed_valid(sample_processed_laps):
    result = validate_processed(sample_processed_laps)
    assert len(result) == len(sample_processed_laps)


def test_validate_processed_invalid_lap_time():
    df = pd.DataFrame(
        {
            "race_id": ["2024_01"],
            "circuit_id": ["Bahrain"],
            "driver_id": ["VER"],
            "constructor_id": ["Red Bull"],
            "lap_number": [2],
            "lap_time_seconds": [5.0],  # Way too fast
            "tyre_life": [1.0],
            "tyre_life_sq": [1.0],
            "compound": ["SOFT"],
            "fuel_mass_kg": [110.0],
            "stint_number": [1],
            "stint_lap": [1],
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        validate_processed(df)


def test_validate_processed_invalid_compound():
    df = pd.DataFrame(
        {
            "race_id": ["2024_01"],
            "circuit_id": ["Bahrain"],
            "driver_id": ["VER"],
            "constructor_id": ["Red Bull"],
            "lap_number": [2],
            "lap_time_seconds": [90.0],
            "tyre_life": [1.0],
            "tyre_life_sq": [1.0],
            "compound": ["SUPER_SOFT"],  # Invalid compound
            "fuel_mass_kg": [110.0],
            "stint_number": [1],
            "stint_lap": [1],
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        validate_processed(df)


def test_check_data_quality(sample_processed_laps):
    report = check_data_quality(sample_processed_laps)
    assert report["total_laps"] == len(sample_processed_laps)
    assert "compound_distribution" in report
    assert isinstance(report["warnings"], list)
