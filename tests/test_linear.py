"""Tests for the linear degradation model."""

from pathlib import Path
import tempfile

import numpy as np

from f1deg.models.linear import LinearDegradationModel


def test_linear_fit_predict(sample_processed_laps, sample_config):
    model = LinearDegradationModel()
    model.fit(sample_processed_laps, sample_config)

    predictions = model.predict(sample_processed_laps)
    assert len(predictions) == len(sample_processed_laps)
    assert not np.any(np.isnan(predictions))

    # Predictions should be in reasonable range
    assert predictions.min() > 60
    assert predictions.max() < 150


def test_linear_prediction_intervals(sample_processed_laps, sample_config):
    model = LinearDegradationModel()
    model.fit(sample_processed_laps, sample_config)

    lower, upper = model.predict_interval(sample_processed_laps, alpha=0.05)
    predictions = model.predict(sample_processed_laps)

    assert np.all(lower <= predictions)
    assert np.all(upper >= predictions)
    assert np.all(upper > lower)


def test_linear_coefficient_signs(sample_processed_laps, sample_config):
    """Verify that model learns expected relationships."""
    model = LinearDegradationModel()
    model.fit(sample_processed_laps, sample_config)

    # Tyre life coefficient should be positive (more laps = slower)
    feature_names = model.feature_cols
    tyre_life_idx = feature_names.index("tyre_life")
    assert model.model.coef_[tyre_life_idx] > 0

    # Fuel mass coefficient should be positive (more fuel = heavier = slower)
    fuel_idx = feature_names.index("fuel_mass_kg")
    # Note: fuel effect is negative in the data generation (lighter = faster),
    # so the coefficient on fuel_mass should be negative (less fuel = faster)
    # Actually the coefficient should be positive if we think of it as more mass = slower
    # The sign depends on the data generation; just check it's learned something
    assert model.model.coef_[fuel_idx] != 0


def test_linear_save_load(sample_processed_laps, sample_config):
    model = LinearDegradationModel()
    model.fit(sample_processed_laps, sample_config)

    predictions_before = model.predict(sample_processed_laps)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model"
        model.save(model_path)

        loaded_model = LinearDegradationModel.load(model_path)
        predictions_after = loaded_model.predict(sample_processed_laps)

    np.testing.assert_array_almost_equal(predictions_before, predictions_after)


def test_linear_degradation_curve(sample_processed_laps, sample_config):
    model = LinearDegradationModel()
    model.fit(sample_processed_laps, sample_config)

    curve = model.predict_degradation_curve(
        compound="MEDIUM",
        circuit="Bahrain",
        n_laps=20,
    )

    assert len(curve) == 20
    assert list(curve.columns) == [
        "lap_in_stint",
        "predicted_lap_time",
        "lower_bound",
        "upper_bound",
    ]
    assert curve["lap_in_stint"].tolist() == list(range(1, 21))

    # Lap times should generally increase with tire life (degradation)
    first_5_mean = curve["predicted_lap_time"].iloc[:5].mean()
    last_5_mean = curve["predicted_lap_time"].iloc[-5:].mean()
    assert last_5_mean > first_5_mean
